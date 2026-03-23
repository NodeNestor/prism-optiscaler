#include "pch.h"
#include "CadenceController.h"

#include <algorithm>
#include <cmath>

CadenceController::CadenceController()
{
    QueryPerformanceFrequency(&_qpcFrequency);
    QueryPerformanceCounter(&_lastRealPresentQPC);
}

void CadenceController::SetTargetFPS(int fps)
{
    _targetFPS = std::max(fps, 1);
}

void CadenceController::OnRealPresent()
{
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);

    if (_initialized)
    {
        double elapsedMs = (double)(now.QuadPart - _lastRealPresentQPC.QuadPart) * 1000.0 / (double)_qpcFrequency.QuadPart;

        // Smooth the measurement to avoid jitter (EMA with alpha=0.3)
        _realFrameTimeMs = _realFrameTimeMs * 0.7 + elapsedMs * 0.3;
    }

    _lastRealPresentQPC = now;
    _initialized = true;

    // Compute how many synthetic frames to insert
    // realFPS = 1000 / realFrameTimeMs
    // totalFramesNeeded = targetFPS / realFPS = targetFPS * realFrameTimeMs / 1000
    // syntheticCount = totalFramesNeeded - 1 (the real frame counts as 1)
    double realFPS = 1000.0 / std::max(_realFrameTimeMs, 1.0);
    double totalFramesPerReal = (double)_targetFPS / realFPS;
    _syntheticCount = std::max((int)std::round(totalFramesPerReal) - 1, 0);

    // Cap at reasonable maximum (4 synthetic = 5x multiplier)
    _syntheticCount = std::min(_syntheticCount, 4);

    if (_syntheticCount > 0)
    {
        // Divide the real frame interval evenly among all frames (real + synthetic)
        _syntheticIntervalMs = _realFrameTimeMs / (double)(_syntheticCount + 1);
    }
    else
    {
        _syntheticIntervalMs = 0.0;
    }

    LOG_DEBUG("[FGExtrap] Cadence: realFPS={:.1f} target={} synth={} interval={:.2f}ms",
              realFPS, _targetFPS, _syntheticCount, _syntheticIntervalMs);
}

float CadenceController::GetTimeFraction(int i) const
{
    // i is 1-based: synthetic frame 1, 2, ..., N
    // Time fraction represents how far into the frame interval this synthetic frame is
    // For the real frame at the end, fraction would be 1.0
    if (_syntheticCount <= 0)
        return 0.5f;

    return (float)i / (float)(_syntheticCount + 1);
}

void CadenceController::WaitForSyntheticFrame(int i)
{
    if (_syntheticIntervalMs <= 0.0 || !_initialized)
        return;

    double targetOffsetMs = _syntheticIntervalMs * (double)i;

    // Convert to QPC ticks
    LONGLONG targetTicks = _lastRealPresentQPC.QuadPart +
                           (LONGLONG)(targetOffsetMs * (double)_qpcFrequency.QuadPart / 1000.0);

    // Hybrid wait: Sleep for most of the time, spin for the last bit
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);

    double remainingMs = (double)(targetTicks - now.QuadPart) * 1000.0 / (double)_qpcFrequency.QuadPart;

    // If we're already past the target, don't wait
    if (remainingMs <= 0.0)
        return;

    // Sleep for most of the wait (leave 1.5ms for spin)
    if (remainingMs > 2.0)
    {
        DWORD sleepMs = (DWORD)(remainingMs - 1.5);
        if (sleepMs > 0)
            Sleep(sleepMs);
    }

    // Spin-wait for precision
    for (;;)
    {
        QueryPerformanceCounter(&now);
        if (now.QuadPart >= targetTicks)
            break;
        YieldProcessor();
    }
}

double CadenceController::GetEffectiveFPS() const
{
    if (_realFrameTimeMs <= 0.0)
        return 0.0;
    return 1000.0 / _realFrameTimeMs * (double)(_syntheticCount + 1);
}
