#pragma once
#include "PrismFeature.h"
#include "PrismModelRegistry.h"
#include "upscalers/IFeature_Dx12.h"

class PrismFeatureDx12 : public PrismFeature, public IFeature_Dx12
{
  private:
    static constexpr UINT NUM_HEAPS = 4;
    static constexpr UINT THREAD_GROUP_X = 16;
    static constexpr UINT THREAD_GROUP_Y = 16;

    ID3D12RootSignature* _rootSignature = nullptr;
    ID3D12PipelineState* _pipelineState = nullptr;
    ID3D12Resource* _constantBuffer = nullptr;

    // History buffer for temporal accumulation (display resolution)
    ID3D12Resource* _historyBuffer = nullptr;
    UINT _historyWidth = 0;
    UINT _historyHeight = 0;
    DXGI_FORMAT _historyFormat = DXGI_FORMAT_UNKNOWN;

    struct alignas(256) PrismConstants
    {
        float srcWidth;
        float srcHeight;
        float dstWidth;
        float dstHeight;
        float jitterX;
        float jitterY;
        float mvScaleX;
        float mvScaleY;
        int reset;
        float sharpness;
        float padding0;
        float padding1;
    };

    struct FrameHeapData
    {
        ID3D12DescriptorHeap* heap = nullptr;
        UINT incrementSize = 0;

        bool Init(ID3D12Device* device, UINT numDescriptors);
        void Release();

        D3D12_CPU_DESCRIPTOR_HANDLE CpuHandle(UINT index) const;
        D3D12_GPU_DESCRIPTOR_HANDLE GpuHandle(UINT index) const;
    };

    FrameHeapData _heaps[NUM_HEAPS] = {};
    UINT _heapIndex = 0;

    bool CompileUpscaleShader(ID3D12Device* device);
    bool EnsureHistoryBuffer(ID3D12Device* device, UINT width, UINT height, DXGI_FORMAT format);

  public:
    PrismFeatureDx12(unsigned int InHandleId, NVSDK_NGX_Parameter* InParameters);
    ~PrismFeatureDx12();

    bool Init(ID3D12Device* InDevice, ID3D12GraphicsCommandList* InCommandList,
              NVSDK_NGX_Parameter* InParameters) override;
    bool Evaluate(ID3D12GraphicsCommandList* InCommandList, NVSDK_NGX_Parameter* InParameters) override;

    bool IsWithDx12() override { return true; }
};
