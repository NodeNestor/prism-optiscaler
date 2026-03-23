#pragma once
#include "SysUtils.h"
#include <Windows.h>

class CadenceController
{
  private:
    double _realFrameTimeMs = 16.667; // assume 60fps initially
    LARGE_INTEGER _lastRealPresentQPC = {};
    LARGE_INTEGER _qpcFrequency = {};
    int _syntheticCount = 1;
    double _syntheticIntervalMs = 0.0;
    int _targetFPS = 120;
    bool _initialized = false;

  public:
    CadenceController();

    void SetTargetFPS(int fps);
    int GetTargetFPS() const { return _targetFPS; }

    // Called at each real game present. Measures frame time and computes cadence.
    void OnRealPresent();

    // How many synthetic frames to insert between this real frame and the next
    int GetSyntheticCount() const { return _syntheticCount; }

    // Time interval between synthetic frames (ms)
    double GetSyntheticIntervalMs() const { return _syntheticIntervalMs; }

    // Time fraction for the i-th synthetic frame (1-based). Returns value in (0, 1).
    // e.g., for 1:1 doubling: GetTimeFraction(1) = 0.5
    // e.g., for 1:2 tripling: GetTimeFraction(1) = 0.333, GetTimeFraction(2) = 0.667
    float GetTimeFraction(int i) const;

    // High-precision spin-wait until the target time for the i-th synthetic frame
    void WaitForSyntheticFrame(int i);

    // Get measured real frame time
    double GetRealFrameTimeMs() const { return _realFrameTimeMs; }
    double GetEffectiveFPS() const;
};
