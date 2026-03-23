// This file is compiled with VK_NO_PROTOTYPES so volk.h works.
// It wraps prism::PrismVulkan behind a C-style API that OptiScaler
// can call without conflicting Vulkan symbol definitions.

#define VK_NO_PROTOTYPES
#define VK_USE_PLATFORM_WIN32_KHR

#include "PrismEngineWrapper.h"
// Include with full relative paths to avoid include resolution issues
// prism_vulkan.h includes "deps/volk.h" which resolves relative to its location
#include "../../../prism-inference/vulkan_engine/prism_vulkan.h"

PrismEngineHandle PrismEngine_Create()
{
    return static_cast<PrismEngineHandle>(new prism::PrismVulkan());
}

void PrismEngine_Destroy(PrismEngineHandle handle)
{
    if (handle)
        delete static_cast<prism::PrismVulkan*>(handle);
}

bool PrismEngine_Init(PrismEngineHandle handle, const PrismEngineConfig* config)
{
    if (!handle || !config) return false;

    prism::PrismVulkanConfig cfg;
    cfg.channels = config->channels;
    cfg.n_blocks = config->blocks;
    cfg.scale = config->scale;
    cfg.render_w = config->render_w;
    cfg.render_h = config->render_h;
    if (config->shader_dir)
        cfg.shader_dir = config->shader_dir;

    return static_cast<prism::PrismVulkan*>(handle)->Init(cfg);
}

bool PrismEngine_LoadWeights(PrismEngineHandle handle, const char* path)
{
    if (!handle || !path) return false;
    return static_cast<prism::PrismVulkan*>(handle)->LoadWeights(path);
}

void PrismEngine_RecordCommandBuffer(PrismEngineHandle handle)
{
    if (handle)
        static_cast<prism::PrismVulkan*>(handle)->RecordCommandBuffer();
}

float PrismEngine_Infer(PrismEngineHandle handle, const void* input_fp16, void* output_fp16)
{
    if (!handle) return -1.0f;
    return static_cast<prism::PrismVulkan*>(handle)->Infer(input_fp16, output_fp16);
}

bool PrismEngine_IsInitialized(PrismEngineHandle handle)
{
    if (!handle) return false;
    return static_cast<prism::PrismVulkan*>(handle)->IsInitialized();
}

void PrismEngine_Shutdown(PrismEngineHandle handle)
{
    if (handle)
        static_cast<prism::PrismVulkan*>(handle)->Shutdown();
}

int PrismEngine_GetDisplayW(PrismEngineHandle handle)
{
    if (!handle) return 0;
    return static_cast<prism::PrismVulkan*>(handle)->GetDisplayW();
}

int PrismEngine_GetDisplayH(PrismEngineHandle handle)
{
    if (!handle) return 0;
    return static_cast<prism::PrismVulkan*>(handle)->GetDisplayH();
}
