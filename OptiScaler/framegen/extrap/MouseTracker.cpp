#include "pch.h"
#include "MouseTracker.h"

#include <cmath>

static MouseTracker* g_activeTracker = nullptr;

LRESULT CALLBACK MouseTracker::RawInputWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (msg == WM_INPUT && g_activeTracker != nullptr)
    {
        UINT dwSize = 0;
        GetRawInputData((HRAWINPUT) lParam, RID_INPUT, NULL, &dwSize, sizeof(RAWINPUTHEADER));

        if (dwSize > 0 && dwSize <= 256)
        {
            BYTE buffer[256];
            if (GetRawInputData((HRAWINPUT) lParam, RID_INPUT, buffer, &dwSize, sizeof(RAWINPUTHEADER)) == dwSize)
            {
                RAWINPUT* raw = (RAWINPUT*) buffer;
                if (raw->header.dwType == RIM_TYPEMOUSE)
                {
                    // Only handle relative mouse movement (not absolute)
                    if ((raw->data.mouse.usFlags & MOUSE_MOVE_ABSOLUTE) == 0)
                    {
                        long dx = raw->data.mouse.lLastX;
                        long dy = raw->data.mouse.lLastY;

                        g_activeTracker->_deltaX.fetch_add(dx, std::memory_order_relaxed);
                        g_activeTracker->_deltaY.fetch_add(dy, std::memory_order_relaxed);

                        // Accumulate magnitude for calibration
                        double mag = std::sqrt((double)(dx * dx + dy * dy));
                        // Use relaxed since precision isn't critical here
                        double prev = g_activeTracker->_accumMouseMag.load(std::memory_order_relaxed);
                        g_activeTracker->_accumMouseMag.store(prev + mag, std::memory_order_relaxed);
                        g_activeTracker->_accumMouseFrames.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        }
    }

    return DefWindowProc(hwnd, msg, wParam, lParam);
}

void MouseTracker::InputThreadMain()
{
    // Create a hidden message-only window for RawInput
    WNDCLASSEX wc = {};
    wc.cbSize = sizeof(WNDCLASSEX);
    wc.lpfnWndProc = RawInputWndProc;
    wc.hInstance = GetModuleHandle(NULL);
    wc.lpszClassName = L"OptiScaler_FGExtrap_MouseTracker";

    if (!RegisterClassEx(&wc))
    {
        LOG_ERROR("[FGExtrap] MouseTracker: Failed to register window class");
        return;
    }

    HWND hwnd = CreateWindowEx(0, wc.lpszClassName, L"", 0, 0, 0, 0, 0, HWND_MESSAGE, NULL, wc.hInstance, NULL);
    if (hwnd == NULL)
    {
        LOG_ERROR("[FGExtrap] MouseTracker: Failed to create message window");
        UnregisterClass(wc.lpszClassName, wc.hInstance);
        return;
    }

    // Register for raw mouse input with INPUTSINK so we get input even when not focused
    RAWINPUTDEVICE rid = {};
    rid.usUsagePage = 0x01; // HID_USAGE_PAGE_GENERIC
    rid.usUsage = 0x02;     // HID_USAGE_GENERIC_MOUSE
    rid.dwFlags = RIDEV_INPUTSINK;
    rid.hwndTarget = hwnd;

    if (!RegisterRawInputDevices(&rid, 1, sizeof(rid)))
    {
        LOG_ERROR("[FGExtrap] MouseTracker: Failed to register RawInput device");
        DestroyWindow(hwnd);
        UnregisterClass(wc.lpszClassName, wc.hInstance);
        return;
    }

    LOG_INFO("[FGExtrap] MouseTracker: Started on dedicated thread");

    // Message pump
    while (_running.load(std::memory_order_relaxed))
    {
        DWORD result = MsgWaitForMultipleObjects(1, &_stopEvent, FALSE, 100, QS_RAWINPUT);

        if (result == WAIT_OBJECT_0)
            break; // Stop event signaled

        if (result == WAIT_OBJECT_0 + 1)
        {
            MSG msg;
            while (PeekMessage(&msg, hwnd, 0, 0, PM_REMOVE))
            {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
        }
    }

    // Unregister RawInput
    rid.dwFlags = RIDEV_REMOVE;
    rid.hwndTarget = NULL;
    RegisterRawInputDevices(&rid, 1, sizeof(rid));

    DestroyWindow(hwnd);
    UnregisterClass(wc.lpszClassName, wc.hInstance);

    LOG_INFO("[FGExtrap] MouseTracker: Stopped");
}

void MouseTracker::Start()
{
    if (_running.load())
        return;

    _running.store(true);
    _stopEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
    g_activeTracker = this;

    _inputThread = std::thread(&MouseTracker::InputThreadMain, this);
}

void MouseTracker::Stop()
{
    if (!_running.load())
        return;

    _running.store(false);

    if (_stopEvent != NULL)
    {
        SetEvent(_stopEvent);
    }

    if (_inputThread.joinable())
        _inputThread.join();

    if (_stopEvent != NULL)
    {
        CloseHandle(_stopEvent);
        _stopEvent = NULL;
    }

    g_activeTracker = nullptr;
}

void MouseTracker::ConsumeDeltas(long& outDx, long& outDy)
{
    outDx = _deltaX.exchange(0, std::memory_order_relaxed);
    outDy = _deltaY.exchange(0, std::memory_order_relaxed);
}

double MouseTracker::GetAvgMouseMagnitude() const
{
    int frames = _accumMouseFrames.load(std::memory_order_relaxed);
    if (frames == 0)
        return 0.0;
    return _accumMouseMag.load(std::memory_order_relaxed) / (double) frames;
}

void MouseTracker::ResetCalibration()
{
    _accumMouseMag.store(0.0, std::memory_order_relaxed);
    _accumMouseFrames.store(0, std::memory_order_relaxed);
}

MouseTracker::~MouseTracker()
{
    Stop();
}
