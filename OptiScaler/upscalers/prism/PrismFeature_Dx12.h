#pragma once
#include "PrismFeature.h"
#include "upscalers/IFeature_Dx12.h"

// Prism DX12 backend.
// Uses a compute shader to upscale color from render res to display res.
// This is a working example — replace the shader with your neural model.

class PrismFeatureDx12 : public PrismFeature, public IFeature_Dx12
{
  private:
    static constexpr UINT NUM_HEAPS = 4;
    static constexpr UINT THREAD_GROUP_X = 8;
    static constexpr UINT THREAD_GROUP_Y = 8;

    ID3D12RootSignature* _rootSignature = nullptr;
    ID3D12PipelineState* _pipelineState = nullptr;
    ID3D12Resource* _constantBuffer = nullptr;

    struct PrismConstants
    {
        float srcWidth;
        float srcHeight;
        float dstWidth;
        float dstHeight;
    };

    struct FrameHeapData
    {
        ID3D12DescriptorHeap* heap = nullptr;
        UINT incrementSize = 0;

        bool Init(ID3D12Device* device);
        void Release();

        D3D12_CPU_DESCRIPTOR_HANDLE CpuHandle(UINT index) const;
        D3D12_GPU_DESCRIPTOR_HANDLE GpuHandle(UINT index) const;
    };

    FrameHeapData _heaps[NUM_HEAPS] = {};
    UINT _heapIndex = 0;

    bool CompileUpscaleShader(ID3D12Device* device);

  public:
    PrismFeatureDx12(unsigned int InHandleId, NVSDK_NGX_Parameter* InParameters);
    ~PrismFeatureDx12();

    bool Init(ID3D12Device* InDevice, ID3D12GraphicsCommandList* InCommandList,
              NVSDK_NGX_Parameter* InParameters) override;
    bool Evaluate(ID3D12GraphicsCommandList* InCommandList, NVSDK_NGX_Parameter* InParameters) override;

    bool IsWithDx12() override { return true; }
};
