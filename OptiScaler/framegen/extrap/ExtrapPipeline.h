#pragma once
#include "SysUtils.h"
#include <shaders/Shader_Dx12Utils.h>

#include <d3d12.h>
#include <dxgi.h>

#define EXTRAP_NUM_LAYERS 5
#define EXTRAP_NUM_HEAPS 2

// Constant buffer structs — must be 256-byte aligned
struct alignas(256) LayerClassifyConstants
{
    UINT Width;
    UINT Height;
    float NearPlane;
    float FarPlane;
    float HUDThreshold;
    float SkyThreshold;
    float FarThreshold;
    float NearThreshold;
    UINT IsInvertedDepth;
    UINT Pad0, Pad1, Pad2;
};

struct alignas(256) ReprojConstants
{
    float MouseDeltaX;
    float MouseDeltaY;
    float MvScaleX;
    float MvScaleY;
    UINT Width;
    UINT Height;
    UINT CurrentLayer;
    float TimeFraction;
    float RotScale;
    float DepthScale;
    float NearPlane;
    float FarPlane;
    UINT IsInvertedDepth;
    UINT UseMVExtrapolation;
    UINT Pad0, Pad1;
};

struct alignas(256) GapFillConstants
{
    UINT Width;
    UINT Height;
    UINT MaxExtendPixels;
    UINT GapFillMode;
};

struct alignas(256) CompositeConstants
{
    UINT Width;
    UINT Height;
    UINT DebugLayers;
    UINT Pad;
};

// Configuration passed to pipeline dispatch
struct ExtrapConfig
{
    float nearPlane = 0.1f;
    float farPlane = 1000.0f;
    float hudThreshold = 0.001f;
    float skyThreshold = 0.99f;
    float farThreshold = 0.80f;
    float nearThreshold = 0.05f;
    bool invertedDepth = false;
    float rotScale = 0.001f;
    float depthScale = 1.0f;
    float timeFraction = 0.5f;
    bool useMVExtrapolation = true;
    int gapFillMode = 0;
    int maxExtendPixels = 32;
    bool debugLayers = false;
};

struct ExtrapPass
{
    ID3D12RootSignature* rootSignature = nullptr;
    ID3D12PipelineState* pipelineState = nullptr;
    ID3D12Resource* constantBuffer = nullptr;
    FrameDescriptorHeap heaps[EXTRAP_NUM_HEAPS];
    int counter = 0;
    bool init = false;
};

class ExtrapPipeline
{
  private:
    ID3D12Device* _device = nullptr;
    bool _init = false;
    std::string _name = "ExtrapPipeline";

    ExtrapPass _classifyPass;
    ExtrapPass _reprojPass;
    ExtrapPass _gapFillPass;
    ExtrapPass _compositePass;

    // Layer textures: color + validity per layer
    ID3D12Resource* _layerColor[EXTRAP_NUM_LAYERS] = {};
    ID3D12Resource* _layerValidity[EXTRAP_NUM_LAYERS] = {};
    ID3D12Resource* _layerMask = nullptr;
    ID3D12Resource* _compositeTarget = nullptr;

    // Gap fill needs a temp copy for read-while-write
    ID3D12Resource* _gapFillTemp = nullptr;

    UINT _width = 0;
    UINT _height = 0;
    DXGI_FORMAT _colorFormat = DXGI_FORMAT_R16G16B16A16_FLOAT;

    bool InitPass(ExtrapPass& pass, const char* shaderCode, UINT numSrv, UINT numUav, UINT numCbv, size_t cbSize);
    void ReleasePass(ExtrapPass& pass);
    bool EnsureLayerTextures(UINT width, UINT height, DXGI_FORMAT colorFormat);
    void ClearLayerTextures(ID3D12GraphicsCommandList* cmdList);

    void ResourceBarrier(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* resource,
                         D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after);

    template <typename T>
    void UpdateConstantBuffer(ID3D12Resource* cb, const T& data);

  public:
    bool Init(ID3D12Device* device);
    void Release();

    // Runs all 4 shader passes. Returns the composite target resource in UAV state.
    ID3D12Resource* Dispatch(ID3D12GraphicsCommandList* cmdList,
                             ID3D12Resource* color, D3D12_RESOURCE_STATES colorState,
                             ID3D12Resource* depth, D3D12_RESOURCE_STATES depthState,
                             ID3D12Resource* velocity, D3D12_RESOURCE_STATES velocityState,
                             float mouseDx, float mouseDy, float mvScaleX, float mvScaleY,
                             const ExtrapConfig& config);

    bool IsInit() const { return _init; }
    UINT GetWidth() const { return _width; }
    UINT GetHeight() const { return _height; }
};
