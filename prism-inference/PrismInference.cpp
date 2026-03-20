/*
 * PrismInference — ONNX Runtime inference engine for Prism neural upscaler.
 *
 * Uses ONNX Runtime C API with CUDA or TensorRT execution provider.
 * Handles model loading, session management, temporal state, and inference.
 */

#include "PrismInference.h"

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cassert>

// ONNX Runtime C API
#include <onnxruntime_c_api.h>

// JSON parsing (nlohmann single-header, already in OptiScaler deps)
#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace fs = std::filesystem;

// Helper: check ONNX Runtime status
#define ORT_CHECK(expr) do { \
    OrtStatus* _status = (expr); \
    if (_status != nullptr) { \
        const char* msg = Ort::GetApi().GetErrorMessage(_status); \
        std::cerr << "[PrismInference] ORT error: " << msg << std::endl; \
        Ort::GetApi().ReleaseStatus(_status); \
        return false; \
    } \
} while(0)

namespace prism {

// ============================================================================
// PrismModel
// ============================================================================

PrismModel::~PrismModel()
{
    if (_session) {
        Ort::GetApi().ReleaseSession(_session);
        _session = nullptr;
    }
    if (_session_options) {
        Ort::GetApi().ReleaseSessionOptions(_session_options);
        _session_options = nullptr;
    }
    if (_memory_info) {
        Ort::GetApi().ReleaseMemoryInfo(_memory_info);
        _memory_info = nullptr;
    }
    if (_env) {
        Ort::GetApi().ReleaseEnv(_env);
        _env = nullptr;
    }
}

bool PrismModel::_LoadConfig(const fs::path& config_path)
{
    std::ifstream f(config_path);
    if (!f.is_open()) return false;

    try {
        json j = json::parse(f);

        _config.name = j.value("name", "Unknown");
        _config.version = j.value("version", "0.0");
        _config.input_channels = j.value("input_channels", 6);
        _config.has_temporal = j.value("has_temporal", true);
        _config.hidden_channels = j.value("hidden_channels", 64);
        _config.default_scale = j.value("default_scale", 2);

        if (j.contains("supported_scales")) {
            _config.supported_scales.clear();
            for (auto& s : j["supported_scales"])
                _config.supported_scales.push_back(s.get<int>());
        }

        _config.valid = true;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[PrismInference] Config parse error: " << e.what() << std::endl;
        return false;
    }
}

bool PrismModel::Load(const fs::path& model_dir)
{
    _config.folder = model_dir.string();
    _config.onnx_path = (model_dir / "model.onnx").string();

    // Check files exist
    if (!fs::exists(model_dir / "model.onnx")) {
        std::cerr << "[PrismInference] No model.onnx in " << model_dir << std::endl;
        return false;
    }

    // Load config
    auto config_path = model_dir / "config.json";
    if (fs::exists(config_path)) {
        _LoadConfig(config_path);
    } else {
        // Use defaults
        _config.name = model_dir.filename().string();
        _config.valid = true;
    }

    _loaded = true;
    std::cout << "[PrismInference] Loaded model: " << _config.name
              << " v" << _config.version
              << " (temporal=" << (_config.has_temporal ? "yes" : "no")
              << ", scales=" << _config.supported_scales.size() << ")"
              << std::endl;

    return true;
}

bool PrismModel::Init(int gpu_device_id)
{
    if (!_loaded) return false;

    const auto& api = Ort::GetApi();

    // Create environment
    ORT_CHECK(api.CreateEnv(ORT_LOGGING_LEVEL_WARNING, "prism", &_env));

    // Session options
    ORT_CHECK(api.CreateSessionOptions(&_session_options));

    // Set graph optimization level
    ORT_CHECK(api.SetSessionGraphOptimizationLevel(_session_options, ORT_ENABLE_ALL));

    // Try TensorRT first, fall back to CUDA
    OrtStatus* trt_status = OrtSessionOptionsAppendExecutionProvider_TensorRT(
        _session_options, gpu_device_id);

    if (trt_status != nullptr) {
        api.ReleaseStatus(trt_status);
        std::cout << "[PrismInference] TensorRT not available, using CUDA EP" << std::endl;

        OrtCUDAProviderOptions cuda_opts = {};
        cuda_opts.device_id = gpu_device_id;
        cuda_opts.arena_extend_strategy = 0;
        cuda_opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
        cuda_opts.do_copy_in_default_stream = 1;
        ORT_CHECK(api.SessionOptionsAppendExecutionProvider_CUDA(_session_options, &cuda_opts));
    } else {
        std::cout << "[PrismInference] Using TensorRT EP" << std::endl;
    }

    // Enable memory pattern optimization
    ORT_CHECK(api.SetSessionExecutionMode(_session_options, ORT_SEQUENTIAL));

    // Create session
    #ifdef _WIN32
    std::wstring model_path_w(_config.onnx_path.begin(), _config.onnx_path.end());
    ORT_CHECK(api.CreateSession(_env, model_path_w.c_str(), _session_options, &_session));
    #else
    ORT_CHECK(api.CreateSession(_env, _config.onnx_path.c_str(), _session_options, &_session));
    #endif

    // Create memory info for CPU (we'll use IOBinding for GPU later)
    ORT_CHECK(api.CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &_memory_info));

    // Get input/output names
    size_t num_inputs = 0, num_outputs = 0;
    ORT_CHECK(api.SessionGetInputCount(_session, &num_inputs));
    ORT_CHECK(api.SessionGetOutputCount(_session, &num_outputs));

    OrtAllocator* allocator = nullptr;
    ORT_CHECK(api.GetAllocatorWithDefaultOptions(&allocator));

    for (size_t i = 0; i < num_inputs; i++) {
        char* name = nullptr;
        ORT_CHECK(api.SessionGetInputName(_session, i, allocator, &name));
        _input_names.push_back(name);
        api.AllocatorFree(allocator, name);
    }

    for (size_t i = 0; i < num_outputs; i++) {
        char* name = nullptr;
        ORT_CHECK(api.SessionGetOutputName(_session, i, allocator, &name));
        _output_names.push_back(name);
        api.AllocatorFree(allocator, name);
    }

    std::cout << "[PrismInference] Session created: "
              << num_inputs << " inputs, " << num_outputs << " outputs" << std::endl;
    for (auto& n : _input_names) std::cout << "  input: " << n << std::endl;
    for (auto& n : _output_names) std::cout << "  output: " << n << std::endl;

    _initialized = true;
    return true;
}

bool PrismModel::Infer(
    const float* color,
    const float* depth,
    const float* motion_vectors,
    float* output,
    int render_width, int render_height,
    int target_width, int target_height,
    bool reset_temporal)
{
    if (!_initialized) return false;

    const auto& api = Ort::GetApi();

    // Build input tensors
    int64_t color_shape[] = {1, 3, render_height, render_width};
    int64_t depth_shape[] = {1, 1, render_height, render_width};
    int64_t mv_shape[] = {1, 2, render_height, render_width};
    int64_t output_shape[] = {1, 3, target_height, target_width};

    int color_size = 3 * render_height * render_width;
    int depth_size = 1 * render_height * render_width;
    int mv_size = 2 * render_height * render_width;
    int output_size = 3 * target_height * target_width;

    // Create ORT values
    std::vector<OrtValue*> input_values;
    std::vector<const char*> input_name_ptrs;
    std::vector<const char*> output_name_ptrs;

    OrtValue* color_val = nullptr;
    ORT_CHECK(api.CreateTensorWithDataAsOrtValue(
        _memory_info, (void*)color, color_size * sizeof(float),
        color_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &color_val));

    OrtValue* depth_val = nullptr;
    ORT_CHECK(api.CreateTensorWithDataAsOrtValue(
        _memory_info, (void*)depth, depth_size * sizeof(float),
        depth_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &depth_val));

    OrtValue* mv_val = nullptr;
    ORT_CHECK(api.CreateTensorWithDataAsOrtValue(
        _memory_info, (void*)motion_vectors, mv_size * sizeof(float),
        mv_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &mv_val));

    input_values = {color_val, depth_val, mv_val};
    for (auto& n : _input_names) input_name_ptrs.push_back(n.c_str());
    for (auto& n : _output_names) output_name_ptrs.push_back(n.c_str());

    // Run inference
    OrtValue* output_values[2] = {nullptr, nullptr};  // output + hidden
    ORT_CHECK(api.Run(
        _session, nullptr,
        input_name_ptrs.data(), input_values.data(), input_values.size(),
        output_name_ptrs.data(), output_name_ptrs.size(), output_values));

    // Copy output
    float* output_data = nullptr;
    ORT_CHECK(api.GetTensorMutableData(output_values[0], (void**)&output_data));
    std::memcpy(output, output_data, output_size * sizeof(float));

    // Store hidden state for temporal
    if (_config.has_temporal && output_values[1] != nullptr) {
        float* hidden_data = nullptr;
        ORT_CHECK(api.GetTensorMutableData(output_values[1], (void**)&hidden_data));

        OrtTensorTypeAndShapeInfo* hidden_info = nullptr;
        ORT_CHECK(api.GetTensorTypeAndShape(output_values[1], &hidden_info));
        size_t hidden_count = 0;
        ORT_CHECK(api.GetTensorShapeElementCount(hidden_info, &hidden_count));
        api.ReleaseTensorTypeAndShapeInfo(hidden_info);

        _prev_hidden.resize(hidden_count);
        std::memcpy(_prev_hidden.data(), hidden_data, hidden_count * sizeof(float));
        _has_temporal_state = true;
    }

    // Store previous output for temporal warping
    _prev_output.resize(output_size);
    std::memcpy(_prev_output.data(), output, output_size * sizeof(float));
    _prev_width = target_width;
    _prev_height = target_height;

    // Cleanup
    for (auto v : input_values) api.ReleaseValue(v);
    for (auto v : output_values) if (v) api.ReleaseValue(v);

    if (reset_temporal) {
        _has_temporal_state = false;
        _prev_hidden.clear();
        _prev_output.clear();
    }

    return true;
}

bool PrismModel::InferGPU(
    void* color_gpu, void* depth_gpu, void* mv_gpu, void* output_gpu,
    int render_width, int render_height,
    int target_width, int target_height,
    bool reset_temporal)
{
    // TODO: Zero-copy GPU inference via IOBinding
    // For now, fall back to CPU path
    // This would use ORT's IOBinding API to bind CUDA pointers directly
    std::cerr << "[PrismInference] GPU inference not yet implemented, use Infer() for now" << std::endl;
    return false;
}

// ============================================================================
// PrismModelRegistry
// ============================================================================

PrismModelRegistry::PrismModelRegistry(const fs::path& models_dir)
    : _models_dir(models_dir)
{
}

int PrismModelRegistry::Scan()
{
    _models.clear();

    if (!fs::exists(_models_dir)) {
        std::cerr << "[PrismInference] Models directory not found: " << _models_dir << std::endl;
        return 0;
    }

    for (auto& entry : fs::directory_iterator(_models_dir)) {
        if (!entry.is_directory()) continue;

        auto model_path = entry.path() / "model.onnx";
        if (!fs::exists(model_path)) continue;

        auto model = std::make_unique<PrismModel>();
        if (model->Load(entry.path())) {
            _models.push_back(std::move(model));
        }
    }

    std::cout << "[PrismInference] Found " << _models.size() << " models in " << _models_dir << std::endl;
    return static_cast<int>(_models.size());
}

PrismModel* PrismModelRegistry::Get(const std::string& name)
{
    for (auto& m : _models) {
        if (m->GetName() == name || m->GetConfig().folder.find(name) != std::string::npos) {
            return m.get();
        }
    }
    return nullptr;
}

std::vector<std::string> PrismModelRegistry::GetModelNames() const
{
    std::vector<std::string> names;
    for (auto& m : _models) {
        names.push_back(m->GetName());
    }
    return names;
}

} // namespace prism
