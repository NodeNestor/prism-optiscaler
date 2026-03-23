#pragma once
#include "SysUtils.h"
#include "PrismModelRegistry.h"
#include "PrismEngineWrapper.h"

#include <d3d12.h>
#include <string>

// PrismNeuralBridge — manages the Vulkan inference engine for neural upscaling
// Uses the C-style wrapper (PrismEngineWrapper) to avoid Vulkan symbol conflicts
class PrismNeuralBridge
{
  private:
    PrismEngineHandle _engine = nullptr;
    ID3D12Device* _dx12Device = nullptr;

    bool _initialized = false;
    int _renderW = 0, _renderH = 0;
    int _displayW = 0, _displayH = 0;
    int _scale = 2;

  public:
    PrismNeuralBridge() = default;
    ~PrismNeuralBridge();

    bool Init(ID3D12Device* dx12Device, const PrismModelInfo& model,
              int renderW, int renderH, const std::string& shaderDir);

    // Run inference — currently uses CPU staging path
    // Returns GPU inference time in ms, or -1 on failure
    float Evaluate(ID3D12GraphicsCommandList* cmdList,
                   ID3D12Resource* inputColor,
                   ID3D12Resource* depth,
                   ID3D12Resource* motionVectors,
                   ID3D12Resource* output);

    void Shutdown();
    bool IsInitialized() const { return _initialized; }
    int GetDisplayW() const { return _displayW; }
    int GetDisplayH() const { return _displayH; }
};
