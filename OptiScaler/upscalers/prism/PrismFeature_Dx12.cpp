#include "pch.h"
#include "PrismFeature_Dx12.h"
#include "State.h"

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
    default: return fmt;
    }
}

// ---------------------------------------------------------------------------
// Bilinear upscale compute shader — the simplest useful upscaler.
// Replace this with your neural model (TensorRT dispatch, ONNX, custom HLSL).
// Inputs:  t0 = color (SRV), b0 = constants (CBV)
// Outputs: u0 = upscaled output (UAV)
// ---------------------------------------------------------------------------
static const char* kUpscaleShaderHLSL = R"(
cbuffer Constants : register(b0)
{
    float srcWidth;
    float srcHeight;
    float dstWidth;
    float dstHeight;
};

Texture2D<float4> InputColor : register(t0);
RWTexture2D<float4> Output   : register(u0);
SamplerState LinearSampler   : register(s0);

[numthreads(8, 8, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= (uint)dstWidth || dtid.y >= (uint)dstHeight)
        return;

    // Normalized UV for bilinear sampling
    float2 uv = (float2(dtid.xy) + 0.5) / float2(dstWidth, dstHeight);

    // Sample input color with bilinear filtering
    float4 color = InputColor.SampleLevel(LinearSampler, uv, 0);

    Output[dtid.xy] = color;
}
)";

// ---------------------------------------------------------------------------
// FrameHeapData — simple descriptor heap per frame-in-flight
// ---------------------------------------------------------------------------
bool PrismFeatureDx12::FrameHeapData::Init(ID3D12Device* device)
{
    D3D12_DESCRIPTOR_HEAP_DESC desc = {};
    desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    desc.NumDescriptors = 3; // SRV + UAV + CBV
    desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

    HRESULT hr = device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap));
    if (FAILED(hr))
        return false;

    incrementSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    return true;
}

void PrismFeatureDx12::FrameHeapData::Release()
{
    if (heap)
    {
        heap->Release();
        heap = nullptr;
    }
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
    LOG_INFO("PrismFeatureDx12 constructed");
}

PrismFeatureDx12::~PrismFeatureDx12()
{
    if (State::Instance().isShuttingDown)
        return;

    if (_pipelineState) { _pipelineState->Release(); _pipelineState = nullptr; }
    if (_rootSignature) { _rootSignature->Release(); _rootSignature = nullptr; }
    if (_constantBuffer) { _constantBuffer->Release(); _constantBuffer = nullptr; }

    for (auto& h : _heaps)
        h.Release();

    LOG_INFO("PrismFeatureDx12 destroyed");
}

// ---------------------------------------------------------------------------
// Compile the upscale compute shader at runtime
// ---------------------------------------------------------------------------
bool PrismFeatureDx12::CompileUpscaleShader(ID3D12Device* device)
{
    // --- Root signature: 1 table with SRV(t0), UAV(u0), CBV(b0) ---
    CD3DX12_DESCRIPTOR_RANGE1 ranges[] = {
        CD3DX12_DESCRIPTOR_RANGE1(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0),
        CD3DX12_DESCRIPTOR_RANGE1(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0),
        CD3DX12_DESCRIPTOR_RANGE1(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0, 0),
    };

    CD3DX12_ROOT_PARAMETER1 rootParam {};
    rootParam.InitAsDescriptorTable(std::size(ranges), ranges);

    // Static linear sampler at s0
    CD3DX12_STATIC_SAMPLER_DESC sampler {};
    sampler.Init(0, D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT);
    sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSigDesc;
    rootSigDesc.Init_1_1(1, &rootParam, 1, &sampler);

    ID3DBlob* sigBlob = nullptr;
    ID3DBlob* errBlob = nullptr;

    HRESULT hr = D3D12SerializeVersionedRootSignature(&rootSigDesc, &sigBlob, &errBlob);
    if (FAILED(hr))
    {
        LOG_ERROR("Prism: D3D12SerializeVersionedRootSignature failed: {:X}", (UINT)hr);
        if (errBlob) { LOG_ERROR("  {}", (const char*)errBlob->GetBufferPointer()); errBlob->Release(); }
        return false;
    }

    hr = device->CreateRootSignature(0, sigBlob->GetBufferPointer(), sigBlob->GetBufferSize(),
                                     IID_PPV_ARGS(&_rootSignature));
    sigBlob->Release();
    if (errBlob) errBlob->Release();

    if (FAILED(hr))
    {
        LOG_ERROR("Prism: CreateRootSignature failed: {:X}", (UINT)hr);
        return false;
    }

    // --- Compile HLSL ---
    ID3DBlob* csBlob = nullptr;
    ID3DBlob* csErr = nullptr;

    hr = D3DCompile(kUpscaleShaderHLSL, strlen(kUpscaleShaderHLSL), "PrismUpscale",
                    nullptr, nullptr, "CSMain", "cs_5_0",
                    D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &csBlob, &csErr);

    if (FAILED(hr))
    {
        LOG_ERROR("Prism: shader compile failed: {:X}", (UINT)hr);
        if (csErr) { LOG_ERROR("  {}", (const char*)csErr->GetBufferPointer()); csErr->Release(); }
        return false;
    }
    if (csErr) csErr->Release();

    // --- Pipeline state ---
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = _rootSignature;
    psoDesc.CS = CD3DX12_SHADER_BYTECODE(csBlob->GetBufferPointer(), csBlob->GetBufferSize());

    hr = device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&_pipelineState));
    csBlob->Release();

    if (FAILED(hr))
    {
        LOG_ERROR("Prism: CreateComputePipelineState failed: {:X}", (UINT)hr);
        return false;
    }

    // --- Constant buffer ---
    auto cbDesc = CD3DX12_RESOURCE_DESC::Buffer(256); // 256-byte aligned
    auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    hr = device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &cbDesc,
                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                         IID_PPV_ARGS(&_constantBuffer));
    if (FAILED(hr))
    {
        LOG_ERROR("Prism: constant buffer creation failed: {:X}", (UINT)hr);
        return false;
    }

    // --- Descriptor heaps ---
    for (auto& h : _heaps)
    {
        if (!h.Init(device))
        {
            LOG_ERROR("Prism: descriptor heap init failed");
            return false;
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Init — called once after construction
// ---------------------------------------------------------------------------
bool PrismFeatureDx12::Init(ID3D12Device* InDevice, ID3D12GraphicsCommandList* InCommandList,
                             NVSDK_NGX_Parameter* InParameters)
{
    if (!_moduleLoaded)
        return false;

    Device = InDevice;

    if (!SetInitParameters(InParameters))
    {
        LOG_ERROR("Prism: SetInitParameters failed");
        return false;
    }

    if (!CompileUpscaleShader(InDevice))
    {
        LOG_ERROR("Prism: shader pipeline setup failed");
        return false;
    }

    SetInit(true);
    _prismInited = true;

    LOG_INFO("Prism DX12 initialized: render={}x{} -> display={}x{}",
             _renderWidth, _renderHeight, _displayWidth, _displayHeight);

    return true;
}

// ---------------------------------------------------------------------------
// Evaluate — called every frame. This is where your model runs.
// ---------------------------------------------------------------------------
bool PrismFeatureDx12::Evaluate(ID3D12GraphicsCommandList* InCommandList, NVSDK_NGX_Parameter* InParameters)
{
    if (!_prismInited || !InCommandList)
        return false;

    _frameCount++;

    // --- Extract resources from parameters ---
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
        LOG_ERROR("Prism: missing color or output resource");
        return false;
    }

    // Log available inputs on first frame
    if (_frameCount == 1)
    {
        LOG_INFO("Prism inputs: color={} depth={} mv={} output={}",
                 (void*)paramColor, (void*)paramDepth, (void*)paramMV, (void*)paramOutput);

        auto colorDesc = paramColor->GetDesc();
        auto outDesc = paramOutput->GetDesc();
        LOG_INFO("Prism: color={}x{} fmt={}, output={}x{} fmt={}",
                 colorDesc.Width, colorDesc.Height, (int)colorDesc.Format,
                 outDesc.Width, outDesc.Height, (int)outDesc.Format);
    }

    // --- Get render resolution (may differ from resource dimensions) ---
    unsigned int renderWidth = 0, renderHeight = 0;
    GetRenderResolution(InParameters, &renderWidth, &renderHeight);
    if (renderWidth == 0 || renderHeight == 0)
    {
        renderWidth = _renderWidth;
        renderHeight = _renderHeight;
    }

    auto outDesc = paramOutput->GetDesc();
    unsigned int outputWidth = (unsigned int)outDesc.Width;
    unsigned int outputHeight = outDesc.Height;

    // --- Update constants ---
    PrismConstants constants = {
        (float)renderWidth,
        (float)renderHeight,
        (float)outputWidth,
        (float)outputHeight
    };

    UINT8* cbData = nullptr;
    CD3DX12_RANGE readRange(0, 0);
    _constantBuffer->Map(0, &readRange, reinterpret_cast<void**>(&cbData));
    memcpy(cbData, &constants, sizeof(constants));
    _constantBuffer->Unmap(0, nullptr);

    // --- Cycle heap ---
    _heapIndex = (_heapIndex + 1) % NUM_HEAPS;
    auto& heap = _heaps[_heapIndex];

    // --- Resource barriers: inputs -> SRV, output -> UAV ---
    ResourceBarrier(InCommandList, paramColor,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    ResourceBarrier(InCommandList, paramOutput,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // --- Create views ---
    auto colorDesc = paramColor->GetDesc();

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Format = ResolveTypeless(colorDesc.Format);
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = 1;
    Device->CreateShaderResourceView(paramColor, &srvDesc, heap.CpuHandle(0));

    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = ResolveTypeless(outDesc.Format);
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    uavDesc.Texture2D.MipSlice = 0;
    Device->CreateUnorderedAccessView(paramOutput, nullptr, &uavDesc, heap.CpuHandle(1));

    D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc = {};
    cbvDesc.BufferLocation = _constantBuffer->GetGPUVirtualAddress();
    cbvDesc.SizeInBytes = 256;
    Device->CreateConstantBufferView(&cbvDesc, heap.CpuHandle(2));

    // --- Dispatch ---
    ID3D12DescriptorHeap* heaps[] = { heap.heap };
    InCommandList->SetDescriptorHeaps(1, heaps);
    InCommandList->SetComputeRootSignature(_rootSignature);
    InCommandList->SetPipelineState(_pipelineState);
    InCommandList->SetComputeRootDescriptorTable(0, heap.GpuHandle(0));

    UINT dispatchX = (outputWidth + THREAD_GROUP_X - 1) / THREAD_GROUP_X;
    UINT dispatchY = (outputHeight + THREAD_GROUP_Y - 1) / THREAD_GROUP_Y;
    InCommandList->Dispatch(dispatchX, dispatchY, 1);

    // --- Restore barriers ---
    ResourceBarrier(InCommandList, paramColor,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_RENDER_TARGET);

    ResourceBarrier(InCommandList, paramOutput,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_RENDER_TARGET);

    return true;
}
