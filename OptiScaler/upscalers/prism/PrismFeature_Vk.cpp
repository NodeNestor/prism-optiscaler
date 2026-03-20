#include "pch.h"
#include "PrismFeature_Vk.h"
#include "State.h"

#include <nvsdk_ngx_vk.h>

PrismFeatureVk::PrismFeatureVk(unsigned int InHandleId, NVSDK_NGX_Parameter* InParameters)
    : IFeature(InHandleId, InParameters),
      IFeature_Vk(InHandleId, InParameters),
      PrismFeature(InHandleId, InParameters)
{
    LOG_INFO("=== PRISM === PrismFeatureVk constructed (handle={})", InHandleId);
}

PrismFeatureVk::~PrismFeatureVk()
{
    _upscaler.reset();
    LOG_INFO("=== PRISM === PrismFeatureVk destroyed");
}

bool PrismFeatureVk::Init(VkInstance InInstance, VkPhysicalDevice InPD, VkDevice InDevice,
                           VkCommandBuffer InCmdList, PFN_vkGetInstanceProcAddr InGIPA,
                           PFN_vkGetDeviceProcAddr InGDPA, NVSDK_NGX_Parameter* InParameters)
{
    if (!_moduleLoaded)
        return false;

    Instance = InInstance;
    PhysicalDevice = InPD;
    Device = InDevice;
    GIPA = InGIPA;
    GDPA = InGDPA;
    _vkDevice = InDevice;

    if (!SetInitParameters(InParameters))
    {
        LOG_ERROR("=== PRISM === Vulkan SetInitParameters failed");
        return false;
    }

    LOG_INFO("=== PRISM === Vulkan Init ===");
    LOG_INFO("  VkInstance={}, VkDevice={}, VkPhysicalDevice={}", (void*)InInstance, (void*)InDevice, (void*)InPD);
    LOG_INFO("  Render: {}x{}", _renderWidth, _renderHeight);
    LOG_INFO("  Target: {}x{}", _targetWidth, _targetHeight);
    LOG_INFO("  Display: {}x{}", _displayWidth, _displayHeight);
    LOG_INFO("  HDR={} DepthInverted={} JitteredMV={} LowResMV={} AutoExposure={} Sharpen={}",
             IsHdr(), DepthInverted(), JitteredMV(), LowResMV(), AutoExposure(), SharpenEnabled());
    LOG_INFO("  Quality mode: {}", (int)PerfQualityValue());

    // Create the temporal upscale compute pipeline
    _upscaler = std::make_unique<PrismUpscale_Vk>("PrismUpscale", InDevice, InPD);

    if (!_upscaler || !_upscaler->CanRender())
    {
        LOG_ERROR("=== PRISM === Failed to create upscale pipeline");
        return false;
    }

    SetInit(true);
    _prismInited = true;

    LOG_INFO("=== PRISM === Vulkan temporal upscaler READY");
    return true;
}

bool PrismFeatureVk::Evaluate(VkCommandBuffer InCmdBuffer, NVSDK_NGX_Parameter* InParameters)
{
    if (!_prismInited || InCmdBuffer == VK_NULL_HANDLE || !_upscaler || !_upscaler->CanRender())
        return false;

    _frameCount++;

    // --- Extract Vulkan resources ---
    void* paramColor = nullptr;
    void* paramOutput = nullptr;
    void* paramDepth = nullptr;
    void* paramMV = nullptr;

    InParameters->Get(NVSDK_NGX_Parameter_Color, &paramColor);
    InParameters->Get(NVSDK_NGX_Parameter_Output, &paramOutput);
    InParameters->Get(NVSDK_NGX_Parameter_Depth, &paramDepth);
    InParameters->Get(NVSDK_NGX_Parameter_MotionVectors, &paramMV);

    if (!paramColor || !paramOutput)
    {
        LOG_ERROR("=== PRISM === Frame {}: missing color or output", _frameCount);
        return false;
    }

    auto* colorRes = (NVSDK_NGX_Resource_VK*)paramColor;
    auto* outputRes = (NVSDK_NGX_Resource_VK*)paramOutput;

    VkImage colorImage = colorRes->Resource.ImageViewInfo.Image;
    VkImageView colorView = colorRes->Resource.ImageViewInfo.ImageView;
    VkImage outputImage = outputRes->Resource.ImageViewInfo.Image;
    VkImageView outputView = outputRes->Resource.ImageViewInfo.ImageView;
    uint32_t outW = outputRes->Resource.ImageViewInfo.Width;
    uint32_t outH = outputRes->Resource.ImageViewInfo.Height;
    VkFormat outFmt = outputRes->Resource.ImageViewInfo.Format;

    VkImageView mvView = VK_NULL_HANDLE;
    VkImageView depthView = VK_NULL_HANDLE;

    if (paramMV)
        mvView = ((NVSDK_NGX_Resource_VK*)paramMV)->Resource.ImageViewInfo.ImageView;
    if (paramDepth)
        depthView = ((NVSDK_NGX_Resource_VK*)paramDepth)->Resource.ImageViewInfo.ImageView;

    // Extract scalar parameters
    float jitterX = 0, jitterY = 0, mvScaleX = 1, mvScaleY = 1, sharpness = 0;
    unsigned int reset = 0;

    InParameters->Get(NVSDK_NGX_Parameter_Jitter_Offset_X, &jitterX);
    InParameters->Get(NVSDK_NGX_Parameter_Jitter_Offset_Y, &jitterY);
    InParameters->Get(NVSDK_NGX_Parameter_MV_Scale_X, &mvScaleX);
    InParameters->Get(NVSDK_NGX_Parameter_MV_Scale_Y, &mvScaleY);
    InParameters->Get(NVSDK_NGX_Parameter_Sharpness, &sharpness);
    InParameters->Get(NVSDK_NGX_Parameter_Reset, &reset);

    unsigned int renderW = 0, renderH = 0;
    GetRenderResolution(InParameters, &renderW, &renderH);
    if (renderW == 0 || renderH == 0) { renderW = _renderWidth; renderH = _renderHeight; }

    // Log first few frames
    if (_frameCount <= 3 || (_frameCount % 600 == 0))
    {
        LOG_INFO("=== PRISM === Frame {} | {}x{} -> {}x{} | jitter=({:.3f},{:.3f}) | mv_scale=({},{}) | reset={}",
                 _frameCount, renderW, renderH, outW, outH, jitterX, jitterY, mvScaleX, mvScaleY, reset);
    }

    // Ensure history buffer exists
    if (!_upscaler->HasHistory() || !_upscaler->CreateHistoryBuffer(outW, outH, outFmt))
    {
        // Fallback: if no history, just force reset
        reset = 1;
        _upscaler->CreateHistoryBuffer(outW, outH, outFmt);
    }

    // --- Image layout transitions for compute shader ---
    _upscaler->SetImageLayout(InCmdBuffer, colorImage, VK_IMAGE_LAYOUT_GENERAL,
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    if (paramMV)
    {
        VkImage mvImage = ((NVSDK_NGX_Resource_VK*)paramMV)->Resource.ImageViewInfo.Image;
        _upscaler->SetImageLayout(InCmdBuffer, mvImage, VK_IMAGE_LAYOUT_GENERAL,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    if (paramDepth)
    {
        VkImage depthImage = ((NVSDK_NGX_Resource_VK*)paramDepth)->Resource.ImageViewInfo.Image;
        _upscaler->SetImageLayout(InCmdBuffer, depthImage, VK_IMAGE_LAYOUT_GENERAL,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    _upscaler->SetImageLayout(InCmdBuffer, outputImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);

    // --- Dispatch temporal upscale ---
    _upscaler->Dispatch(InCmdBuffer, colorView, mvView, depthView,
                         _upscaler->GetHistoryView(), outputView,
                         outW, outH,
                         (float)renderW, (float)renderH,
                         jitterX, jitterY, mvScaleX, mvScaleY,
                         (int)reset, sharpness);

    // --- Copy output to history for next frame ---
    _upscaler->CopyOutputToHistory(InCmdBuffer, outputImage, outW, outH);

    // --- Restore layouts ---
    _upscaler->SetImageLayout(InCmdBuffer, colorImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                               VK_IMAGE_LAYOUT_GENERAL);

    if (paramMV)
    {
        VkImage mvImage = ((NVSDK_NGX_Resource_VK*)paramMV)->Resource.ImageViewInfo.Image;
        _upscaler->SetImageLayout(InCmdBuffer, mvImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                   VK_IMAGE_LAYOUT_GENERAL);
    }

    if (paramDepth)
    {
        VkImage depthImage = ((NVSDK_NGX_Resource_VK*)paramDepth)->Resource.ImageViewInfo.Image;
        _upscaler->SetImageLayout(InCmdBuffer, depthImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                   VK_IMAGE_LAYOUT_GENERAL);
    }

    return true;
}
