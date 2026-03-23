#pragma once
// Thin C-style wrapper around prism::PrismVulkan
// This exists because prism-inference uses volk.h (VK_NO_PROTOTYPES)
// while OptiScaler uses standard vulkan.h — they can't coexist in one TU.
// This wrapper is compiled in its own .cpp with VK_NO_PROTOTYPES.

#include <cstdint>

// Opaque handle — hides PrismVulkan from OptiScaler's Vulkan headers
typedef void* PrismEngineHandle;

struct PrismEngineConfig
{
    int channels;
    int blocks;
    int scale;
    int render_w;
    int render_h;
    const char* shader_dir;
};

PrismEngineHandle PrismEngine_Create();
void PrismEngine_Destroy(PrismEngineHandle handle);

bool PrismEngine_Init(PrismEngineHandle handle, const PrismEngineConfig* config);
bool PrismEngine_LoadWeights(PrismEngineHandle handle, const char* path);
void PrismEngine_RecordCommandBuffer(PrismEngineHandle handle);
float PrismEngine_Infer(PrismEngineHandle handle, const void* input_fp16, void* output_fp16);
bool PrismEngine_IsInitialized(PrismEngineHandle handle);
void PrismEngine_Shutdown(PrismEngineHandle handle);

int PrismEngine_GetDisplayW(PrismEngineHandle handle);
int PrismEngine_GetDisplayH(PrismEngineHandle handle);
