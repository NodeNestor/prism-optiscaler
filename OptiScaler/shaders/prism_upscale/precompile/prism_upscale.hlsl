// Prism Temporal Upscaler — example implementation
// Demonstrates jitter-compensated bilinear upscale with motion-vector-based
// temporal accumulation. Replace with your neural model.
//
// Bindings:
//   b0 = Constants (uniform buffer)
//   t0 = Color input (render resolution, sampled)
//   t1 = Motion vectors (render resolution, sampled)
//   t2 = Depth (render resolution, sampled)
//   t3 = History buffer (display resolution, sampled — previous frame output)
//   u0 = Output (display resolution, storage — write target)
//   u1 = History out (display resolution, storage — write target, same as output for feedback)

#ifdef VK_MODE
[[vk::binding(0, 0)]] cbuffer Constants : register(b0, space0) {
[[vk::binding(1, 0)]] Texture2D<float4> InputColor : register(t0, space0);
[[vk::binding(2, 0)]] Texture2D<float2> MotionVectors : register(t1, space0);
[[vk::binding(3, 0)]] Texture2D<float> DepthBuffer : register(t2, space0);
[[vk::binding(4, 0)]] Texture2D<float4> HistoryBuffer : register(t3, space0);
[[vk::binding(5, 0)]] RWTexture2D<float4> Output : register(u0, space0);
[[vk::binding(6, 0)]] SamplerState LinearSampler : register(s0, space0);
[[vk::binding(7, 0)]] SamplerState PointSampler : register(s1, space0);
#else
cbuffer Constants : register(b0) {
Texture2D<float4> InputColor : register(t0);
Texture2D<float2> MotionVectors : register(t1);
Texture2D<float> DepthBuffer : register(t2);
Texture2D<float4> HistoryBuffer : register(t3);
RWTexture2D<float4> Output : register(u0);
SamplerState LinearSampler : register(s0);
SamplerState PointSampler : register(s1);
#endif

    float renderWidth;
    float renderHeight;
    float displayWidth;
    float displayHeight;
    float jitterX;
    float jitterY;
    float mvScaleX;
    float mvScaleY;
    int reset;
    float sharpness;
    float padding0;
    float padding1;
};

// Neighborhood clamping for ghosting reduction
float4 ClampToNeighborhood(float2 uv, float4 historySample)
{
    float2 texelSize = 1.0 / float2(renderWidth, renderHeight);

    float4 minColor = float4(1e10, 1e10, 1e10, 1e10);
    float4 maxColor = float4(-1e10, -1e10, -1e10, -1e10);

    [unroll]
    for (int y = -1; y <= 1; y++)
    {
        [unroll]
        for (int x = -1; x <= 1; x++)
        {
            float2 offset = float2(x, y) * texelSize;
            float4 s = InputColor.SampleLevel(PointSampler, uv + offset, 0);
            minColor = min(minColor, s);
            maxColor = max(maxColor, s);
        }
    }

    return clamp(historySample, minColor, maxColor);
}

[numthreads(16, 16, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= (uint)displayWidth || dtid.y >= (uint)displayHeight)
        return;

    // UV in display/output space
    float2 displayUV = (float2(dtid.xy) + 0.5) / float2(displayWidth, displayHeight);

    // UV in render/input space — compensate for jitter
    // Jitter is in pixel space at render resolution, convert to normalized
    float2 jitterOffset = float2(jitterX, jitterY) / float2(renderWidth, renderHeight);
    float2 renderUV = displayUV - jitterOffset;

    // Sample current frame color with bilinear filtering (upscale)
    float4 currentColor = InputColor.SampleLevel(LinearSampler, renderUV, 0);

    // Sample motion vectors at this location
    float2 mv = MotionVectors.SampleLevel(PointSampler, displayUV, 0);

    // Motion vectors are in pixel space (scaled by mvScaleX/Y)
    // Convert to normalized display UV offset
    float2 mvNorm = mv / float2(mvScaleX, mvScaleY);

    // Reproject: where was this pixel in the previous frame?
    float2 historyUV = displayUV - mvNorm;

    // If reset or history UV is out of bounds, just use current color
    bool validHistory = !reset &&
                        historyUV.x >= 0.0 && historyUV.x <= 1.0 &&
                        historyUV.y >= 0.0 && historyUV.y <= 1.0;

    float4 result;

    if (validHistory)
    {
        // Sample reprojected history
        float4 historyColor = HistoryBuffer.SampleLevel(LinearSampler, historyUV, 0);

        // Clamp history to neighborhood of current frame to reduce ghosting
        historyColor = ClampToNeighborhood(renderUV, historyColor);

        // Blend: favor history for temporal stability, current for responsiveness
        // 0.1 = very stable but ghosty, 0.3 = responsive but noisy
        float blendFactor = 0.15;
        result = lerp(historyColor, currentColor, blendFactor);
    }
    else
    {
        result = currentColor;
    }

    // Optional: simple sharpening pass (unsharp mask)
    if (sharpness > 0.0)
    {
        float2 texelSize = 1.0 / float2(renderWidth, renderHeight);
        float4 blur = InputColor.SampleLevel(LinearSampler, renderUV + float2(texelSize.x, 0), 0) * 0.25 +
                      InputColor.SampleLevel(LinearSampler, renderUV - float2(texelSize.x, 0), 0) * 0.25 +
                      InputColor.SampleLevel(LinearSampler, renderUV + float2(0, texelSize.y), 0) * 0.25 +
                      InputColor.SampleLevel(LinearSampler, renderUV - float2(0, texelSize.y), 0) * 0.25;

        float4 sharp = result + (result - blur) * sharpness;
        result = max(sharp, 0.0);
    }

    result.a = 1.0;
    Output[dtid.xy] = result;
}
