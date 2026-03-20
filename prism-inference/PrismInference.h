#pragma once

/*
 * PrismInference — loads ONNX models and runs them via ONNX Runtime + TensorRT/CUDA.
 *
 * Drop a folder with model.onnx + config.json into the models/ directory.
 * Each folder becomes a selectable backend in OptiScaler.
 *
 * Config format (config.json):
 * {
 *   "name": "Prism Balanced v2",
 *   "version": "2.0",
 *   "input_channels": 6,       // color(3) + depth(1) + mv(2)
 *   "has_temporal": true,       // uses prev_output + prev_hidden
 *   "hidden_channels": 64,     // ConvGRU hidden state channels
 *   "supported_scales": [2, 3], // PixelShuffle paths available
 *   "default_scale": 2
 * }
 *
 * Usage in OptiScaler:
 *   PrismModelRegistry registry("models/");
 *   registry.scan();  // finds all model folders
 *   auto& model = registry.get("prism-v2-balanced");
 *   model.init(device);
 *   model.infer(color, depth, mv, prev_output, prev_hidden, output);
 */

#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <d3d12.h>

// Forward declarations — ONNX Runtime
struct OrtEnv;
struct OrtSession;
struct OrtSessionOptions;
struct OrtMemoryInfo;
struct OrtValue;

namespace prism {

// ============================================================================
// Model config loaded from config.json
// ============================================================================

struct ModelConfig {
    std::string name = "Unknown";
    std::string version = "0.0";
    std::string folder;              // folder path
    std::string onnx_path;           // full path to model.onnx

    int input_channels = 6;          // color(3) + depth(1) + mv(2)
    bool has_temporal = true;
    int hidden_channels = 64;
    std::vector<int> supported_scales = {2, 3};
    int default_scale = 2;

    bool valid = false;
};

// ============================================================================
// Single model instance — handles inference for one ONNX model
// ============================================================================

class PrismModel {
public:
    PrismModel() = default;
    ~PrismModel();

    // Load config + ONNX model
    bool Load(const std::filesystem::path& model_dir);

    // Initialize ONNX Runtime session on GPU
    bool Init(int gpu_device_id = 0);

    // Run inference
    // Inputs:  color [1,3,H,W], depth [1,1,H,W], mv [1,2,H,W] as GPU buffers
    // Output:  result [1,3,tH,tW] written to output buffer
    // Temporal: prev_output and prev_hidden are managed internally
    bool Infer(
        const float* color,          // [1, 3, renderH, renderW]
        const float* depth,          // [1, 1, renderH, renderW]
        const float* motion_vectors, // [1, 2, renderH, renderW]
        float* output,               // [1, 3, targetH, targetW] — written here
        int render_width, int render_height,
        int target_width, int target_height,
        bool reset_temporal = false
    );

    // Zero-copy GPU inference (for Vulkan/DX12 interop)
    bool InferGPU(
        void* color_gpu,             // GPU pointer to color texture
        void* depth_gpu,
        void* mv_gpu,
        void* output_gpu,
        int render_width, int render_height,
        int target_width, int target_height,
        bool reset_temporal = false
    );

    const ModelConfig& GetConfig() const { return _config; }
    bool IsLoaded() const { return _loaded; }
    bool IsInitialized() const { return _initialized; }
    const std::string& GetName() const { return _config.name; }

private:
    ModelConfig _config;
    bool _loaded = false;
    bool _initialized = false;

    // ONNX Runtime handles
    OrtEnv* _env = nullptr;
    OrtSession* _session = nullptr;
    OrtSessionOptions* _session_options = nullptr;
    OrtMemoryInfo* _memory_info = nullptr;

    // Temporal state (managed internally)
    std::vector<float> _prev_output;
    std::vector<float> _prev_hidden;
    int _prev_width = 0;
    int _prev_height = 0;
    bool _has_temporal_state = false;

    // Input/output names cached from model
    std::vector<std::string> _input_names;
    std::vector<std::string> _output_names;

    bool _LoadConfig(const std::filesystem::path& config_path);
};

// ============================================================================
// Model registry — scans a directory for model folders
// ============================================================================

class PrismModelRegistry {
public:
    PrismModelRegistry(const std::filesystem::path& models_dir);

    // Scan models/ directory for subfolders containing model.onnx + config.json
    int Scan();

    // Get model by folder name
    PrismModel* Get(const std::string& name);

    // Get all discovered models
    const std::vector<std::unique_ptr<PrismModel>>& GetAll() const { return _models; }

    // Get model names for UI
    std::vector<std::string> GetModelNames() const;

    // Number of models found
    size_t Count() const { return _models.size(); }

private:
    std::filesystem::path _models_dir;
    std::vector<std::unique_ptr<PrismModel>> _models;
};

} // namespace prism
