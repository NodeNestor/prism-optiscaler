#pragma once
#include <d3dcompiler.h>
#include "SysUtils.h"

#include <string>

// Compile helper following project convention (Bias_Common.h pattern)
inline ID3DBlob* ExtrapCompileShader(const char* shaderCode, const char* entryPoint, const char* target)
{
    ID3DBlob* shaderBlob = nullptr;
    ID3DBlob* errorBlob = nullptr;

    HRESULT hr = D3DCompile(shaderCode, strlen(shaderCode), nullptr, nullptr, nullptr, entryPoint, target,
                            D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &shaderBlob, &errorBlob);

    if (FAILED(hr))
    {
        LOG_ERROR("[FGExtrap] Shader compile error");
        if (errorBlob)
        {
            LOG_ERROR("[FGExtrap] Shader compile error: {0}", (char*) errorBlob->GetBufferPointer());
            errorBlob->Release();
        }
        if (shaderBlob)
            shaderBlob->Release();
        return nullptr;
    }

    if (errorBlob)
        errorBlob->Release();

    return shaderBlob;
}

// ============================================================================
// Shader 1: LayerClassify
// Classifies each pixel into a depth layer based on linearized depth.
// Layer 0 = HUD (depth near 0 or 1 depending on format)
// Layer 1 = Sky (depth near far plane)
// Layer 2 = Far
// Layer 3 = Mid
// Layer 4 = Near/Foreground
// ============================================================================
inline static std::string extrapLayerClassifyCode = R"(
cbuffer Constants : register(b0)
{
    uint Width;
    uint Height;
    float NearPlane;
    float FarPlane;
    float HUDThreshold;
    float SkyThreshold;
    float FarThreshold;
    float NearThreshold;
    uint IsInvertedDepth;
    uint Pad0;
    uint Pad1;
    uint Pad2;
};

Texture2D<float> DepthTexture : register(t0);
RWTexture2D<uint> LayerMask : register(u0);

float LinearizeDepth(float d, float nearZ, float farZ)
{
    // Handle reversed-Z (common in modern games)
    if (IsInvertedDepth)
        d = 1.0 - d;

    // Standard perspective linearization
    if (d <= 0.0)
        return 0.0;
    if (d >= 1.0)
        return farZ;

    return (nearZ * farZ) / (farZ - d * (farZ - nearZ));
}

[numthreads(16, 16, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= Width || dtid.y >= Height)
        return;

    float rawDepth = DepthTexture[dtid.xy];
    float linearZ = LinearizeDepth(rawDepth, NearPlane, FarPlane);

    // Normalize to [0, 1] range
    float normalizedZ = saturate(linearZ / FarPlane);

    uint layer;

    // HUD: depth is essentially zero (or 1 in reversed-Z before inversion)
    if (normalizedZ < HUDThreshold)
        layer = 0;
    // Sky: depth is near the far plane
    else if (normalizedZ > SkyThreshold)
        layer = 1;
    // Far: behind the far threshold
    else if (normalizedZ > FarThreshold)
        layer = 2;
    // Near: closer than near threshold
    else if (normalizedZ < NearThreshold)
        layer = 4;
    // Mid: everything else
    else
        layer = 3;

    LayerMask[dtid.xy] = layer;
}
)";

// ============================================================================
// Shader 2: LayerReproject
// Reprojects pixels for one specific layer using mouse-derived camera shift
// and motion vector extrapolation.
// Writes to a per-layer color target and validity mask.
// Uses scatter (write to destination) approach via atomic-free overwrite.
// ============================================================================
inline static std::string extrapReprojCode = R"(
cbuffer Constants : register(b0)
{
    float MouseDeltaX;
    float MouseDeltaY;
    float MvScaleX;
    float MvScaleY;
    uint Width;
    uint Height;
    uint CurrentLayer;
    float TimeFraction;
    float RotScale;       // fov / resolution factor
    float DepthScale;     // parallax strength
    float NearPlane;
    float FarPlane;
    uint IsInvertedDepth;
    uint UseMVExtrapolation;
    uint Pad0;
    uint Pad1;
};

Texture2D<float4> SourceColor : register(t0);
Texture2D<float2> Velocity : register(t1);
Texture2D<uint> LayerMask : register(t2);
Texture2D<float> DepthTexture : register(t3);

RWTexture2D<float4> LayerColor : register(u0);
RWTexture2D<uint> LayerValidity : register(u1);

float LinearizeDepth(float d)
{
    if (IsInvertedDepth)
        d = 1.0 - d;
    if (d <= 0.0) return 0.0;
    if (d >= 1.0) return FarPlane;
    return (NearPlane * FarPlane) / (FarPlane - d * (FarPlane - NearPlane));
}

[numthreads(16, 16, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= Width || dtid.y >= Height)
        return;

    uint layer = LayerMask[dtid.xy];
    if (layer != CurrentLayer)
        return;

    float4 color = SourceColor[dtid.xy];
    float2 currentUV = (float2(dtid.xy) + 0.5) / float2(Width, Height);

    // Compute camera shift from mouse delta
    float2 cameraShift = float2(0.0, 0.0);

    if (layer == 0)
    {
        // HUD: no warp at all
        cameraShift = float2(0.0, 0.0);
    }
    else if (layer == 1)
    {
        // Sky: rotation only, no parallax
        cameraShift = float2(MouseDeltaX, MouseDeltaY) * RotScale;
    }
    else
    {
        // Depth-based parallax: closer objects shift more
        float z = LinearizeDepth(DepthTexture[dtid.xy]);
        float parallax = DepthScale / max(z, 0.001);
        cameraShift = float2(MouseDeltaX, MouseDeltaY) * RotScale * parallax;
    }

    // World object motion from motion vectors (extrapolated by time fraction)
    float2 worldShift = float2(0.0, 0.0);
    if (layer >= 2 && UseMVExtrapolation)
    {
        float2 mv = Velocity[dtid.xy] * float2(MvScaleX, MvScaleY);
        worldShift = mv * TimeFraction;
    }

    // Compute source UV (where this pixel came from / should go to)
    float2 destUV = currentUV + cameraShift + worldShift;

    // Convert to pixel coordinates
    int2 destPixel = int2(destUV * float2(Width, Height));

    // Bounds check
    if (destPixel.x >= 0 && destPixel.x < (int)Width &&
        destPixel.y >= 0 && destPixel.y < (int)Height)
    {
        LayerColor[destPixel] = color;
        LayerValidity[destPixel] = 1;
    }
}
)";

// ============================================================================
// Shader 3: GapFill
// Fills invalid (gap) pixels in a reprojected layer.
// Two modes:
//   Mode 0 (Extend): nearest valid neighbor extension with blur falloff
//   Mode 1 (Fourier): frequency-aware pattern continuation
// ============================================================================
inline static std::string extrapGapFillCode = R"(
cbuffer Constants : register(b0)
{
    uint Width;
    uint Height;
    uint MaxExtendPixels;
    uint GapFillMode;  // 0=extend, 1=fourier
};

Texture2D<float4> InputColor : register(t0);
Texture2D<uint> InputValidity : register(t1);

RWTexture2D<float4> OutputColor : register(u0);

// ---- Extend Mode ----
// For each gap pixel, search in 8 directions for the nearest valid pixel
// and extend its color with distance-based falloff.
float4 ExtendFill(uint2 pixel)
{
    float4 bestColor = float4(0, 0, 0, 0);
    float bestWeight = 0.0;

    // 8-directional search
    static const int2 dirs[8] = {
        int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1),
        int2(1, 1), int2(-1, 1), int2(1, -1), int2(-1, -1)
    };

    for (uint d = 0; d < 8; d++)
    {
        for (uint dist = 1; dist <= MaxExtendPixels; dist++)
        {
            int2 samplePos = int2(pixel) + dirs[d] * (int)dist;

            if (samplePos.x < 0 || samplePos.x >= (int)Width ||
                samplePos.y < 0 || samplePos.y >= (int)Height)
                break;

            if (InputValidity[samplePos] > 0)
            {
                // Weight decreases with distance, diagonal directions weighted less
                float dirWeight = (d < 4) ? 1.0 : 0.707;
                float weight = dirWeight / (float)dist;
                bestColor += InputColor[samplePos] * weight;
                bestWeight += weight;
                break; // Found nearest in this direction
            }
        }
    }

    if (bestWeight > 0.0)
        return bestColor / bestWeight;

    return float4(0, 0, 0, 0);
}

// ---- Fourier Mode ----
// For each gap pixel, sample a small patch from nearest valid edge,
// estimate dominant frequency, and synthesize fill via pattern continuation.
float4 FourierFill(uint2 pixel)
{
    // Step 1: Find nearest valid pixel and its neighborhood
    int2 nearestValid = int2(-1, -1);
    float nearestDist = 1e10;

    // Search in a small radius for the nearest valid pixel
    for (int dy = -8; dy <= 8; dy++)
    {
        for (int dx = -8; dx <= 8; dx++)
        {
            int2 sp = int2(pixel) + int2(dx, dy);
            if (sp.x < 0 || sp.x >= (int)Width || sp.y < 0 || sp.y >= (int)Height)
                continue;
            if (InputValidity[sp] > 0)
            {
                float d = (float)(dx * dx + dy * dy);
                if (d < nearestDist)
                {
                    nearestDist = d;
                    nearestValid = sp;
                }
            }
        }
    }

    if (nearestValid.x < 0)
        return float4(0, 0, 0, 0); // No valid neighbors at all

    // Step 2: Sample a patch around the nearest valid pixel
    float4 patchColors[16]; // 4x4 patch
    float4 patchMean = float4(0, 0, 0, 0);
    uint validCount = 0;

    for (int py = 0; py < 4; py++)
    {
        for (int px = 0; px < 4; px++)
        {
            int2 sp = nearestValid + int2(px - 2, py - 2);
            sp = clamp(sp, int2(0, 0), int2(Width - 1, Height - 1));

            if (InputValidity[sp] > 0)
            {
                patchColors[py * 4 + px] = InputColor[sp];
                patchMean += InputColor[sp];
                validCount++;
            }
            else
            {
                patchColors[py * 4 + px] = float4(0, 0, 0, 0);
            }
        }
    }

    if (validCount == 0)
        return float4(0, 0, 0, 0);

    patchMean /= (float)validCount;

    // Step 3: Estimate dominant horizontal and vertical gradients
    float4 gradH = float4(0, 0, 0, 0);
    float4 gradV = float4(0, 0, 0, 0);

    for (int py = 0; py < 4; py++)
    {
        for (int px = 0; px < 3; px++)
        {
            gradH += patchColors[py * 4 + px + 1] - patchColors[py * 4 + px];
        }
    }
    gradH /= 12.0;

    for (int py = 0; py < 3; py++)
    {
        for (int px = 0; px < 4; px++)
        {
            gradV += patchColors[(py + 1) * 4 + px] - patchColors[py * 4 + px];
        }
    }
    gradV /= 12.0;

    // Step 4: Estimate texture frequency from variance
    float4 variance = float4(0, 0, 0, 0);
    for (uint i = 0; i < 16; i++)
    {
        float4 diff = patchColors[i] - patchMean;
        variance += diff * diff;
    }
    variance /= 16.0;
    float amplitude = sqrt(dot(variance.rgb, float3(0.299, 0.587, 0.114)));

    // Estimate spatial frequency from gradient magnitude
    float gradMag = length(gradH.rgb) + length(gradV.rgb);
    float freq = clamp(gradMag * 6.283, 0.5, 12.0); // Map to reasonable frequency range

    // Step 5: Synthesize fill color
    float2 offset = float2(pixel) - float2(nearestValid);
    float dist = length(offset);

    // Base: edge color + gradient continuation
    float4 fillColor = patchMean + gradH * offset.x + gradV * offset.y;

    // Add sinusoidal texture continuation
    float phase = dot(offset, normalize(offset + 0.001)) * freq;
    float4 textureTerm = amplitude * sin(phase);
    fillColor.rgb += textureTerm.xxx * 0.3; // Subtle texture continuation

    // Fade with distance
    float falloff = exp(-dist * 0.15);
    fillColor = lerp(patchMean, fillColor, falloff);

    return saturate(fillColor);
}

[numthreads(16, 16, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= Width || dtid.y >= Height)
        return;

    // Pass through valid pixels unchanged
    if (InputValidity[dtid.xy] > 0)
    {
        OutputColor[dtid.xy] = InputColor[dtid.xy];
        return;
    }

    // Fill gap pixels
    if (GapFillMode == 0)
        OutputColor[dtid.xy] = ExtendFill(dtid.xy);
    else
        OutputColor[dtid.xy] = FourierFill(dtid.xy);
}
)";

// ============================================================================
// Shader 4: Composite
// Back-to-front compositing of all 5 depth layers into the final frame.
// Sky (1) → Far (2) → Mid (3) → Near (4) → HUD (0) on top.
// ============================================================================
inline static std::string extrapCompositeCode = R"(
cbuffer Constants : register(b0)
{
    uint Width;
    uint Height;
    uint DebugLayers;
    uint Pad;
};

// 5 layer color textures
Texture2D<float4> Layer0Color : register(t0); // HUD
Texture2D<float4> Layer1Color : register(t1); // Sky
Texture2D<float4> Layer2Color : register(t2); // Far
Texture2D<float4> Layer3Color : register(t3); // Mid
Texture2D<float4> Layer4Color : register(t4); // Near

// 5 layer validity masks
Texture2D<uint> Layer0Valid : register(t5);
Texture2D<uint> Layer1Valid : register(t6);
Texture2D<uint> Layer2Valid : register(t7);
Texture2D<uint> Layer3Valid : register(t8);
Texture2D<uint> Layer4Valid : register(t9);

RWTexture2D<float4> OutputFrame : register(u0);

// Debug tint colors per layer
static const float3 debugTints[5] = {
    float3(1.0, 0.0, 0.0),  // HUD = red
    float3(0.0, 0.0, 1.0),  // Sky = blue
    float3(0.0, 0.8, 0.0),  // Far = green
    float3(1.0, 1.0, 0.0),  // Mid = yellow
    float3(1.0, 0.5, 0.0),  // Near = orange
};

[numthreads(16, 16, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= Width || dtid.y >= Height)
        return;

    uint2 px = dtid.xy;
    float4 result = float4(0, 0, 0, 1);
    int topLayer = -1;

    // Back-to-front: Sky (1) → Far (2) → Mid (3) → Near (4) → HUD (0)

    // Layer 1: Sky (base layer, fills everything)
    if (Layer1Valid[px] > 0)
    {
        result = Layer1Color[px];
        topLayer = 1;
    }

    // Layer 2: Far
    if (Layer2Valid[px] > 0)
    {
        result = Layer2Color[px];
        topLayer = 2;
    }

    // Layer 3: Mid
    if (Layer3Valid[px] > 0)
    {
        result = Layer3Color[px];
        topLayer = 3;
    }

    // Layer 4: Near/Foreground
    if (Layer4Valid[px] > 0)
    {
        result = Layer4Color[px];
        topLayer = 4;
    }

    // Layer 0: HUD (always on top, unwarped)
    if (Layer0Valid[px] > 0)
    {
        result = Layer0Color[px];
        topLayer = 0;
    }

    // Debug visualization: tint each layer
    if (DebugLayers && topLayer >= 0)
    {
        result.rgb = lerp(result.rgb, debugTints[topLayer], 0.3);
    }

    OutputFrame[px] = result;
}
)";
