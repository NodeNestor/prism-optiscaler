#include "pch.h"
#include "PrismNeuralBridge.h"

#include <d3dx/d3dx12.h>
#include <vector>

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

bool PrismNeuralBridge::Init(ID3D12Device* dx12Device, const PrismModelInfo& model,
                              int renderW, int renderH, const std::string& shaderDir)
{
    if (_initialized)
        Shutdown();

    _dx12Device = dx12Device;
    _renderW = renderW;
    _renderH = renderH;
    _scale = model.scale;
    _displayW = renderW * _scale;
    _displayH = renderH * _scale;

    LOG_INFO("[PrismNeural] Initializing: {}x{} -> {}x{} ({}ch, {}blocks, {}x)",
             renderW, renderH, _displayW, _displayH, model.channels, model.blocks, model.scale);

    // Create engine via C-style wrapper (avoids Vulkan symbol conflicts)
    _engine = PrismEngine_Create();
    if (!_engine)
    {
        LOG_ERROR("[PrismNeural] Failed to create engine");
        return false;
    }

    PrismEngineConfig cfg = {};
    cfg.channels = model.channels;
    cfg.blocks = model.blocks;
    cfg.scale = model.scale;
    cfg.render_w = renderW;
    cfg.render_h = renderH;
    cfg.shader_dir = shaderDir.c_str();

    if (!PrismEngine_Init(_engine, &cfg))
    {
        LOG_ERROR("[PrismNeural] Engine init failed");
        PrismEngine_Destroy(_engine);
        _engine = nullptr;
        return false;
    }

    // Load weights
    if (!PrismEngine_LoadWeights(_engine, model.weightsPath.c_str()))
    {
        LOG_ERROR("[PrismNeural] Failed to load weights: {}", model.weightsPath);
        PrismEngine_Destroy(_engine);
        _engine = nullptr;
        return false;
    }

    // Pre-record inference command buffer
    PrismEngine_RecordCommandBuffer(_engine);

    _initialized = true;
    LOG_INFO("[PrismNeural] Bridge initialized — model: {} ({}ch, {}blocks, {}x)",
             model.name, model.channels, model.blocks, model.scale);
    return true;
}

// ---------------------------------------------------------------------------
// Evaluate
// ---------------------------------------------------------------------------

float PrismNeuralBridge::Evaluate(ID3D12GraphicsCommandList* cmdList,
                                   ID3D12Resource* inputColor,
                                   ID3D12Resource* depth,
                                   ID3D12Resource* motionVectors,
                                   ID3D12Resource* output)
{
    if (!_initialized || !_engine || !PrismEngine_IsInitialized(_engine))
        return -1.0f;

    // The inference engine expects flat FP16 buffers (6 channels: RGB + depth + MV)
    // For now, we create a zero-filled input and run inference to verify the pipeline works.
    // Once the GPU-GPU texture-to-buffer copy shaders are implemented,
    // this will read directly from the game's DX12 textures via shared GPU memory.

    int pixels = _renderW * _renderH;
    int dPixels = _displayW * _displayH;

    std::vector<uint16_t> inputFP16(6 * pixels, 0x3800); // FP16 0.5 as placeholder
    std::vector<uint16_t> outputFP16(3 * dPixels, 0);

    float gpuMs = PrismEngine_Infer(_engine, inputFP16.data(), outputFP16.data());

    LOG_DEBUG("[PrismNeural] Inference: {:.2f}ms", gpuMs);
    return gpuMs;
}

// ---------------------------------------------------------------------------
// Shutdown
// ---------------------------------------------------------------------------

void PrismNeuralBridge::Shutdown()
{
    if (!_initialized)
        return;

    LOG_INFO("[PrismNeural] Shutting down");

    if (_engine)
    {
        PrismEngine_Shutdown(_engine);
        PrismEngine_Destroy(_engine);
        _engine = nullptr;
    }

    _dx12Device = nullptr;
    _initialized = false;
}

PrismNeuralBridge::~PrismNeuralBridge()
{
    Shutdown();
}
