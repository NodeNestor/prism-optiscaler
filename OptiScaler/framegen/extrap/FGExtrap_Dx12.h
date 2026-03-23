#pragma once
#include "SysUtils.h"
#include <framegen/IFGFeature_Dx12.h>
#include "MouseTracker.h"
#include "CadenceController.h"
#include "ExtrapPipeline.h"

#include <memory>

class FGExtrap_Dx12 : public virtual IFGFeature_Dx12
{
  private:
    std::unique_ptr<MouseTracker> _mouseTracker;
    std::unique_ptr<CadenceController> _cadenceController;
    std::unique_ptr<ExtrapPipeline> _pipeline;

    ID3D12GraphicsCommandList* _fgCommandList[BUFFER_COUNT] {};
    ID3D12CommandAllocator* _fgCommandAllocator[BUFFER_COUNT] {};

    bool _contextCreated = false;
    bool _waitingNewFrameData = false;

    // Auto-calibration state
    double _accumMvMag = 0.0;
    int _calibrationFrames = 0;
    float _calibratedSensitivity = 1.0f;
    bool _calibrationDone = false;

    bool DispatchSyntheticFrames();

  protected:
    void ReleaseObjects() override final;
    void CreateObjects(ID3D12Device* InDevice) override final;

  public:
    const char* Name() override final { return "FGExtrap"; }
    feature_version Version() override final { return { 1, 0, 0 }; }
    HWND Hwnd() override final { return _hwnd; }

    void* FrameGenerationContext() override final { return nullptr; }
    void* SwapchainContext() override final { return nullptr; }

    bool CreateSwapchain(IDXGIFactory* factory, ID3D12CommandQueue* cmdQueue,
                         DXGI_SWAP_CHAIN_DESC* desc, IDXGISwapChain** swapChain) override final;
    bool CreateSwapchain1(IDXGIFactory* factory, ID3D12CommandQueue* cmdQueue, HWND hwnd,
                          DXGI_SWAP_CHAIN_DESC1* desc, DXGI_SWAP_CHAIN_FULLSCREEN_DESC* pFullscreenDesc,
                          IDXGISwapChain1** swapChain) override final;
    bool ReleaseSwapchain(HWND hwnd) override final;

    void CreateContext(ID3D12Device* device, FG_Constants& fgConstants) override final;
    void EvaluateState(ID3D12Device* device, FG_Constants& fgConstants) override final;
    void Activate() override final;
    void Deactivate() override final;
    void DestroyFGContext() override final;
    bool Shutdown() override final;

    bool Present() override final;
    bool SetResource(Dx12Resource* inputResource) override final;
    void SetCommandQueue(FG_ResourceType type, ID3D12CommandQueue* queue) override final;
    bool SetInterpolatedFrameCount(UINT interpolatedFrameCount) override final;

    FGExtrap_Dx12();
    ~FGExtrap_Dx12();
};
