#pragma once
#include "SysUtils.h"
#include <atomic>
#include <thread>
#include <Windows.h>

class MouseTracker
{
  private:
    std::atomic<long> _deltaX { 0 };
    std::atomic<long> _deltaY { 0 };
    std::atomic<bool> _running { false };
    HANDLE _stopEvent = NULL;
    std::thread _inputThread;

    // Calibration: accumulate mouse magnitude for auto-sensitivity
    std::atomic<double> _accumMouseMag { 0.0 };
    std::atomic<int> _accumMouseFrames { 0 };

    static LRESULT CALLBACK RawInputWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
    void InputThreadMain();

  public:
    MouseTracker() = default;
    ~MouseTracker();

    void Start();
    void Stop();
    bool IsRunning() const { return _running.load(); }

    // Atomically reads and resets accumulated deltas since last call
    void ConsumeDeltas(long& outDx, long& outDy);

    // Calibration helpers
    double GetAvgMouseMagnitude() const;
    void ResetCalibration();
};
