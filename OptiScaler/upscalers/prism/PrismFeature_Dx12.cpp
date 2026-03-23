#include "pch.h"
#include "PrismFeature_Dx12.h"
#include "State.h"
#include "Config.h"

#include <d3dcompiler.h>
#include <d3dx/d3dx12.h>

static DXGI_FORMAT ResolveTypeless(DXGI_FORMAT fmt)
{
    switch (fmt)
    {
    case DXGI_FORMAT_R32G32B32A32_TYPELESS: return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS:  return DXGI_FORMAT_R10G10B10A2_UINT;
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:     return DXGI_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:     return DXGI_FORMAT_B8G8R8A8_UNORM;
    case DXGI_FORMAT_R16G16_TYPELESS:       return DXGI_FORMAT_R16G16_FLOAT;
    case DXGI_FORMAT_R32G32_TYPELESS:       return DXGI_FORMAT_R32G32_FLOAT;
    case DXGI_FORMAT_R32_TYPELESS:          return DXGI_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_R24G8_TYPELESS:        return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    default: return fmt;
    }
}

// ---------------------------------------------------------------------------
// Temporal jitter-aware upscaler
// Uses motion vectors + history buffer for temporal accumulation
// Neighborhood clamping for anti-ghosting, unsharp mask for sharpening
// ---------------------------------------------------------------------------
static const char* kTemporalUpscaleHLSL = R"(
cbuffer Constants : register(b0)
{
    float renderWidth;
    float renderHeight;
    float displayWidth;
    float displayHeight;
    float jitterX;
    float jitterY;
    float mvScaleX;
    float mvScaleY;
    int reset;
    float sharpness;
    float padding0;
    float padding1;
};

Texture2D<float4> InputColor : register(t0);
Texture2D<float> DepthBuffer : register(t1);
Texture2D<float2> MotionVectors : register(t2);
Texture2D<float4> HistoryBuffer : register(t3);
RWTexture2D<float4> Output : register(u0);
SamplerState LinearSampler : register(s0);
SamplerState PointSampler : register(s1);

// Neighborhood clamping — prevents ghosting by restricting reprojected history
// to the color range of the current frame's local neighborhood
float4 ClampToNeighborhood(float2 uv, float4 historySample)
{
    float2 texelSize = 1.0 / float2(renderWidth, renderHeight);

    float4 minColor = float4(1e10, 1e10, 1e10, 1e10);
    float4 maxColor = float4(-1e10, -1e10, -1e10, -1e10);

    [unroll]
    for (int y = -1; y <= 1; y++)
    {
        [unroll]
        for (int x = -1; x <= 1; x++)
        {
            float2 offset = float2(x, y) * texelSize;
            float4 s = InputColor.SampleLevel(PointSampler, uv + offset, 0);
            minColor = min(minColor, s);
            maxColor = max(maxColor, s);
        }
    }

    return clamp(historySample, minColor, maxColor);
}

[numthreads(16, 16, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= (uint)displayWidth || dtid.y >= (uint)displayHeight)
        return;

    // UV in display/output space
    float2 displayUV = (float2(dtid.xy) + 0.5) / float2(displayWidth, displayHeight);

    // UV in render/input space — compensate for subpixel jitter
    float2 jitterOffset = float2(jitterX, jitterY) / float2(renderWidth, renderHeight);
    float2 renderUV = displayUV - jitterOffset;

    // Sample current frame with bilinear (upscale from render to display res)
    float4 currentColor = InputColor.SampleLevel(LinearSampler, renderUV, 0);

    // Sample motion vectors and convert to display-space UV offset
    float2 mv = MotionVectors.SampleLevel(PointSampler, displayUV, 0);
    float2 mvNorm = mv / float2(mvScaleX + 0.0001, mvScaleY + 0.0001);

    // Reproject: where was this pixel in the previous frame?
    float2 historyUV = displayUV - mvNorm;

    bool validHistory = !reset &&
                        historyUV.x >= 0.0 && historyUV.x <= 1.0 &&
                        historyUV.y >= 0.0 && historyUV.y <= 1.0;

    float4 result;

    if (validHistory)
    {
        float4 historyColor = HistoryBuffer.SampleLevel(LinearSampler, historyUV, 0);
        historyColor = ClampToNeighborhood(renderUV, historyColor);

        // Temporal blend: 15% current, 85% history
        // This accumulates subpixel detail over multiple jittered frames
        result = lerp(historyColor, currentColor, 0.15);
    }
    else
    {
        result = currentColor;
    }

    // Unsharp mask sharpening
    if (sharpness > 0.0)
    {
        float2 texelSize = 1.0 / float2(renderWidth, renderHeight);
        float4 blur = InputColor.SampleLevel(LinearSampler, renderUV + float2(texelSize.x, 0), 0) * 0.25 +
                      InputColor.SampleLevel(LinearSampler, renderUV - float2(texelSize.x, 0), 0) * 0.25 +
                      InputColor.SampleLevel(LinearSampler, renderUV + float2(0, texelSize.y), 0) * 0.25 +
                      InputColor.SampleLevel(LinearSampler, renderUV - float2(0, texelSize.y), 0) * 0.25;

        float4 sharp = result + (result - blur) * sharpness;
        result = max(sharp, 0.0);
    }

    result.a = 1.0;
    Output[dtid.xy] = result;
}
)";

// ---------------------------------------------------------------------------
// FrameHeapData
// ---------------------------------------------------------------------------
bool PrismFeatureDx12::FrameHeapData::Init(ID3D12Device* device, UINT numDescriptors)
{
    D3D12_DESCRIPTOR_HEAP_DESC desc = {};
    desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    desc.NumDescriptors = numDescriptors;
    desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

    HRESULT hr = device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap));
    if (FAILED(hr))
        return false;

    incrementSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    return true;
}

void PrismFeatureDx12::FrameHeapData::Release()
{
    if (heap) { heap->Release(); heap = nullptr; }
}

D3D12_CPU_DESCRIPTOR_HANDLE PrismFeatureDx12::FrameHeapData::CpuHandle(UINT index) const
{
    auto h = heap->GetCPUDescriptorHandleForHeapStart();
    h.ptr += (SIZE_T)index * incrementSize;
    return h;
}

D3D12_GPU_DESCRIPTOR_HANDLE PrismFeatureDx12::FrameHeapData::GpuHandle(UINT index) const
{
    auto h = heap->GetGPUDescriptorHandleForHeapStart();
    h.ptr += (UINT64)index * incrementSize;
    return h;
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
PrismFeatureDx12::PrismFeatureDx12(unsigned int InHandleId, NVSDK_NGX_Parameter* InParameters)
    : IFeature(InHandleId, InParameters),
      IFeature_Dx12(InHandleId, InParameters),
      PrismFeature(InHandleId, InParameters)
{
    LOG_INFO("[Prism] PrismFeatureDx12 constructed");
}

PrismFeatureDx12::~PrismFeatureDx12()
{
    if (State::Instance().isShuttingDown)
        return;

    if (_pipelineState) { _pipelineState->Release(); _pipelineState = nullptr; }
    if (_rootSignature) { _rootSignature->Release(); _rootSignature = nullptr; }
    if (_constantBuffer) { _constantBuffer->Release(); _constantBuffer = nullptr; }
    if (_historyBuffer) { _historyBuffer->Release(); _historyBuffer = nullptr; }

    for (auto& h : _heaps)
        h.Release();

    LOG_INFO("[Prism] PrismFeatureDx12 destroyed");
}

// ---------------------------------------------------------------------------
// Compile temporal upscale shader
// ---------------------------------------------------------------------------
bool PrismFeatureDx12::CompileUpscaleShader(ID3D12Device* device)
{
    // Root signature: 1 table with SRV(t0-t3), UAV(u0), CBV(b0) + 2 static samplers
    CD3DX12_DESCRIPTOR_RANGE1 ranges[] = {
        CD3DX12_DESCRIPTOR_RANGE1(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 4, 0, 0), // t0-t3
        CD3DX12_DESCRIPTOR_RANGE1(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0), // u0
        CD3DX12_DESCRIPTOR_RANGE1(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0, 0), // b0
    };

    CD3DX12_ROOT_PARAMETER1 rootParam {};
    rootParam.InitAsDescriptorTable(std::size(ranges), ranges);

    // Two static samplers: linear (s0) and point (s1)
    CD3DX12_STATIC_SAMPLER_DESC samplers[2] {};
    samplers[0].Init(0, D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT);
    samplers[0].AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samplers[0].AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;

    samplers[1].Init(1, D3D12_FILTER_MIN_MAG_MIP_POINT);
    samplers[1].AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samplers[1].AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samplers[1].ShaderRegister = 1;

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSigDesc;
    rootSigDesc.Init_1_1(1, &rootParam, 2, samplers);

    ID3DBlob* sigBlob = nullptr;
    ID3DBlob* errBlob = nullptr;

    HRESULT hr = D3D12SerializeVersionedRootSignature(&rootSigDesc, &sigBlob, &errBlob);
    if (FAILED(hr))
    {
        LOG_ERROR("[Prism] D3D12SerializeVersionedRootSignature failed: {:X}", (UINT)hr);
        if (errBlob) { LOG_ERROR("  {}", (const char*)errBlob->GetBufferPointer()); errBlob->Release(); }
        return false;
    }

    hr = device->CreateRootSignature(0, sigBlob->GetBufferPointer(), sigBlob->GetBufferSize(),
                                     IID_PPV_ARGS(&_rootSignature));
    sigBlob->Release();
    if (errBlob) errBlob->Release();

    if (FAILED(hr))
    {
        LOG_ERROR("[Prism] CreateRootSignature failed: {:X}", (UINT)hr);
        return false;
    }

    // Compile HLSL
    ID3DBlob* csBlob = nullptr;
    ID3DBlob* csErr = nullptr;

    hr = D3DCompile(kTemporalUpscaleHLSL, strlen(kTemporalUpscaleHLSL), "PrismTemporalUpscale",
                    nullptr, nullptr, "CSMain", "cs_5_0",
                    D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &csBlob, &csErr);

    if (FAILED(hr))
    {
        LOG_ERROR("[Prism] Shader compile failed: {:X}", (UINT)hr);
        if (csErr) { LOG_ERROR("  {}", (const char*)csErr->GetBufferPointer()); csErr->Release(); }
        return false;
    }
    if (csErr) csErr->Release();

    // Pipeline state
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = _rootSignature;
    psoDesc.CS = CD3DX12_SHADER_BYTECODE(csBlob->GetBufferPointer(), csBlob->GetBufferSize());

    hr = device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&_pipelineState));
    csBlob->Release();

    if (FAILED(hr))
    {
        LOG_ERROR("[Prism] CreateComputePipelineState failed: {:X}", (UINT)hr);
        return false;
    }

    // Constant buffer (256-byte aligned)
    auto cbDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(PrismConstants));
    auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    hr = device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &cbDesc,
                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                         IID_PPV_ARGS(&_constantBuffer));
    if (FAILED(hr))
    {
        LOG_ERROR("[Prism] Constant buffer creation failed: {:X}", (UINT)hr);
        return false;
    }

    // Descriptor heaps: 4 SRVs + 1 UAV + 1 CBV = 6 descriptors
    for (auto& h : _heaps)
    {
        if (!h.Init(device, 6))
        {
            LOG_ERROR("[Prism] Descriptor heap init failed");
            return false;
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// History buffer management
// ---------------------------------------------------------------------------
bool PrismFeatureDx12::EnsureHistoryBuffer(ID3D12Device* device, UINT width, UINT height, DXGI_FORMAT format)
{
    if (_historyBuffer && _historyWidth == width && _historyHeight == height && _historyFormat == format)
        return true;

    if (_historyBuffer)
    {
        _historyBuffer->Release();
        _historyBuffer = nullptr;
    }

    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = D3D12_RESOURCE_FLAG_NONE; // SRV only, we copy into it

    auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);

    HRESULT hr = device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &desc,
                                                 D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                                 nullptr, IID_PPV_ARGS(&_historyBuffer));
    if (FAILED(hr))
    {
        LOG_ERROR("[Prism] History buffer creation failed: {:X}", (UINT)hr);
        return false;
    }

    _historyWidth = width;
    _historyHeight = height;
    _historyFormat = format;

    LOG_INFO("[Prism] History buffer created: {}x{} fmt={}", width, height, (int)format);
    return true;
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
bool PrismFeatureDx12::Init(ID3D12Device* InDevice, ID3D12GraphicsCommandList* InCommandList,
                             NVSDK_NGX_Parameter* InParameters)
{
    if (!_moduleLoaded)
        return false;

    Device = InDevice;

    if (!SetInitParameters(InParameters))
    {
        LOG_ERROR("[Prism] SetInitParameters failed");
        return false;
    }

    if (!CompileUpscaleShader(InDevice))
    {
        LOG_ERROR("[Prism] Shader pipeline setup failed");
        return false;
    }

    // Scan for neural models
    auto modelPath = Util::ExePath().parent_path() / Config::Instance()->PrismModelPath.value_or_default();
    PrismModelRegistry::Instance().ScanDirectory(modelPath);

    SetInit(true);
    _prismInited = true;

    int mode = Config::Instance()->PrismMode.value_or_default();
    LOG_INFO("[Prism] DX12 initialized: render={}x{} -> display={}x{}, mode={}",
             _renderWidth, _renderHeight, _displayWidth, _displayHeight,
             mode == 0 ? "Basic" : "Neural");

    return true;
}

// ---------------------------------------------------------------------------
// Evaluate — runs every frame
// ---------------------------------------------------------------------------
bool PrismFeatureDx12::Evaluate(ID3D12GraphicsCommandList* InCommandList, NVSDK_NGX_Parameter* InParameters)
{
    if (!_prismInited || !InCommandList)
        return false;

    _frameCount++;

    // Extract resources
    ID3D12Resource* paramColor = nullptr;
    ID3D12Resource* paramOutput = nullptr;
    ID3D12Resource* paramDepth = nullptr;
    ID3D12Resource* paramMV = nullptr;

    InParameters->Get(NVSDK_NGX_Parameter_Color, &paramColor);
    InParameters->Get(NVSDK_NGX_Parameter_Output, &paramOutput);
    InParameters->Get(NVSDK_NGX_Parameter_Depth, &paramDepth);
    InParameters->Get(NVSDK_NGX_Parameter_MotionVectors, &paramMV);

    if (!paramColor || !paramOutput)
    {
        LOG_ERROR("[Prism] Missing color or output resource");
        return false;
    }

    // Log on first frame
    if (_frameCount == 1)
    {
        auto colorDesc = paramColor->GetDesc();
        auto outDesc = paramOutput->GetDesc();
        LOG_INFO("[Prism] color={}x{} fmt={}, output={}x{} fmt={}, depth={}, mv={}",
                 colorDesc.Width, colorDesc.Height, (int)colorDesc.Format,
                 outDesc.Width, outDesc.Height, (int)outDesc.Format,
                 (void*)paramDepth, (void*)paramMV);
    }

    // Get render resolution
    unsigned int renderWidth = 0, renderHeight = 0;
    GetRenderResolution(InParameters, &renderWidth, &renderHeight);
    if (renderWidth == 0 || renderHeight == 0)
    {
        renderWidth = _renderWidth;
        renderHeight = _renderHeight;
    }

    auto outDesc = paramOutput->GetDesc();
    UINT outputWidth = (UINT)outDesc.Width;
    UINT outputHeight = outDesc.Height;
    DXGI_FORMAT outputFormat = ResolveTypeless(outDesc.Format);

    // Ensure history buffer
    if (!EnsureHistoryBuffer(Device, outputWidth, outputHeight, outputFormat))
        return false;

    // Get jitter and MV scale
    float jitterX = 0, jitterY = 0;
    float mvScaleX = 1, mvScaleY = 1;
    int resetFlag = 0;

    InParameters->Get(NVSDK_NGX_Parameter_Jitter_Offset_X, &jitterX);
    InParameters->Get(NVSDK_NGX_Parameter_Jitter_Offset_Y, &jitterY);
    InParameters->Get(NVSDK_NGX_Parameter_MV_Scale_X, &mvScaleX);
    InParameters->Get(NVSDK_NGX_Parameter_MV_Scale_Y, &mvScaleY);
    InParameters->Get(NVSDK_NGX_Parameter_Reset, &resetFlag);

    float sharpness = Config::Instance()->PrismSharpness.value_or_default();

    // Update constants
    PrismConstants constants = {};
    constants.srcWidth = (float)renderWidth;
    constants.srcHeight = (float)renderHeight;
    constants.dstWidth = (float)outputWidth;
    constants.dstHeight = (float)outputHeight;
    constants.jitterX = jitterX;
    constants.jitterY = jitterY;
    constants.mvScaleX = mvScaleX;
    constants.mvScaleY = mvScaleY;
    constants.reset = resetFlag;
    constants.sharpness = sharpness;

    UINT8* cbData = nullptr;
    CD3DX12_RANGE readRange(0, 0);
    _constantBuffer->Map(0, &readRange, reinterpret_cast<void**>(&cbData));
    memcpy(cbData, &constants, sizeof(constants));
    _constantBuffer->Unmap(0, nullptr);

    // Cycle heap
    _heapIndex = (_heapIndex + 1) % NUM_HEAPS;
    auto& heap = _heaps[_heapIndex];

    // Resource barriers
    ResourceBarrier(InCommandList, paramColor,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    if (paramDepth)
        ResourceBarrier(InCommandList, paramDepth,
                        D3D12_RESOURCE_STATE_DEPTH_WRITE,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    if (paramMV)
        ResourceBarrier(InCommandList, paramMV,
                        D3D12_RESOURCE_STATE_RENDER_TARGET,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    ResourceBarrier(InCommandList, paramOutput,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // Create SRVs: t0=color, t1=depth, t2=MV, t3=history
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = 1;

    // t0: color
    srvDesc.Format = ResolveTypeless(paramColor->GetDesc().Format);
    Device->CreateShaderResourceView(paramColor, &srvDesc, heap.CpuHandle(0));

    // t1: depth (may be null — use color as fallback)
    if (paramDepth)
    {
        srvDesc.Format = ResolveTypeless(paramDepth->GetDesc().Format);
        Device->CreateShaderResourceView(paramDepth, &srvDesc, heap.CpuHandle(1));
    }
    else
    {
        srvDesc.Format = ResolveTypeless(paramColor->GetDesc().Format);
        Device->CreateShaderResourceView(paramColor, &srvDesc, heap.CpuHandle(1));
    }

    // t2: motion vectors (may be null — use color as fallback)
    if (paramMV)
    {
        srvDesc.Format = ResolveTypeless(paramMV->GetDesc().Format);
        Device->CreateShaderResourceView(paramMV, &srvDesc, heap.CpuHandle(2));
    }
    else
    {
        srvDesc.Format = ResolveTypeless(paramColor->GetDesc().Format);
        Device->CreateShaderResourceView(paramColor, &srvDesc, heap.CpuHandle(2));
    }

    // t3: history buffer
    srvDesc.Format = outputFormat;
    Device->CreateShaderResourceView(_historyBuffer, &srvDesc, heap.CpuHandle(3));

    // u0: output UAV
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = outputFormat;
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    Device->CreateUnorderedAccessView(paramOutput, nullptr, &uavDesc, heap.CpuHandle(4));

    // b0: constants CBV
    D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc = {};
    cbvDesc.BufferLocation = _constantBuffer->GetGPUVirtualAddress();
    cbvDesc.SizeInBytes = sizeof(PrismConstants);
    Device->CreateConstantBufferView(&cbvDesc, heap.CpuHandle(5));

    // Dispatch
    ID3D12DescriptorHeap* heaps[] = { heap.heap };
    InCommandList->SetDescriptorHeaps(1, heaps);
    InCommandList->SetComputeRootSignature(_rootSignature);
    InCommandList->SetPipelineState(_pipelineState);
    InCommandList->SetComputeRootDescriptorTable(0, heap.GpuHandle(0));

    UINT dispatchX = (outputWidth + THREAD_GROUP_X - 1) / THREAD_GROUP_X;
    UINT dispatchY = (outputHeight + THREAD_GROUP_Y - 1) / THREAD_GROUP_Y;
    InCommandList->Dispatch(dispatchX, dispatchY, 1);

    // Copy output to history for next frame
    ResourceBarrier(InCommandList, paramOutput,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE);
    ResourceBarrier(InCommandList, _historyBuffer,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_COPY_DEST);

    InCommandList->CopyResource(_historyBuffer, paramOutput);

    // Restore states
    ResourceBarrier(InCommandList, _historyBuffer,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    ResourceBarrier(InCommandList, paramOutput,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_RENDER_TARGET);

    ResourceBarrier(InCommandList, paramColor,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_RENDER_TARGET);

    if (paramDepth)
        ResourceBarrier(InCommandList, paramDepth,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_DEPTH_WRITE);

    if (paramMV)
        ResourceBarrier(InCommandList, paramMV,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_RENDER_TARGET);

    return true;
}
