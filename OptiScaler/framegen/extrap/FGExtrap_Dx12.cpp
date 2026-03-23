#include "pch.h"
#include "FGExtrap_Dx12.h"

#include <Config.h>
#include <hooks/FG_Hooks.h>

#include <d3dx/d3dx12.h>
#include <cmath>

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

FGExtrap_Dx12::FGExtrap_Dx12() : IFGFeature_Dx12(), IFGFeature(1)
{
    _mouseTracker = std::make_unique<MouseTracker>();
    _cadenceController = std::make_unique<CadenceController>();
    _pipeline = std::make_unique<ExtrapPipeline>();
}

FGExtrap_Dx12::~FGExtrap_Dx12()
{
    Shutdown();
}

// ---------------------------------------------------------------------------
// Swapchain — we create the real swapchain directly, no proxy
// ---------------------------------------------------------------------------

bool FGExtrap_Dx12::CreateSwapchain(IDXGIFactory* factory, ID3D12CommandQueue* cmdQueue,
                                     DXGI_SWAP_CHAIN_DESC* desc, IDXGISwapChain** swapChain)
{
    LOG_INFO("[FGExtrap] CreateSwapchain");

    IDXGIFactory* realFactory = nullptr;
    ID3D12CommandQueue* realQueue = nullptr;

    if (!CheckForRealObject(__FUNCTION__, factory, (IUnknown**) &realFactory))
        realFactory = factory;
    if (!CheckForRealObject(__FUNCTION__, cmdQueue, (IUnknown**) &realQueue))
        realQueue = cmdQueue;

    // Create the real swapchain — no proxy needed
    HRESULT hr = realFactory->CreateSwapChain(realQueue, desc, swapChain);
    if (FAILED(hr))
    {
        LOG_ERROR("[FGExtrap] CreateSwapChain failed: {0:X}", hr);
        return false;
    }

    _gameCommandQueue = realQueue;
    _swapChain = *swapChain;
    _hwnd = desc->OutputWindow;

    LOG_INFO("[FGExtrap] Swapchain created successfully");
    return true;
}

bool FGExtrap_Dx12::CreateSwapchain1(IDXGIFactory* factory, ID3D12CommandQueue* cmdQueue, HWND hwnd,
                                      DXGI_SWAP_CHAIN_DESC1* desc,
                                      DXGI_SWAP_CHAIN_FULLSCREEN_DESC* pFullscreenDesc,
                                      IDXGISwapChain1** swapChain)
{
    LOG_INFO("[FGExtrap] CreateSwapchain1");

    IDXGIFactory2* realFactory = nullptr;
    ID3D12CommandQueue* realQueue = nullptr;

    IDXGIFactory2* factory2 = nullptr;
    if (factory->QueryInterface(IID_PPV_ARGS(&factory2)) != S_OK)
    {
        LOG_ERROR("[FGExtrap] Failed to get IDXGIFactory2");
        return false;
    }

    if (!CheckForRealObject(__FUNCTION__, factory2, (IUnknown**) &realFactory))
        realFactory = factory2;
    if (!CheckForRealObject(__FUNCTION__, cmdQueue, (IUnknown**) &realQueue))
        realQueue = cmdQueue;

    HRESULT hr;
    if (pFullscreenDesc != nullptr)
        hr = realFactory->CreateSwapChainForHwnd(realQueue, hwnd, desc, pFullscreenDesc, nullptr, swapChain);
    else
        hr = realFactory->CreateSwapChainForHwnd(realQueue, hwnd, desc, nullptr, nullptr, swapChain);

    factory2->Release();

    if (FAILED(hr))
    {
        LOG_ERROR("[FGExtrap] CreateSwapChainForHwnd failed: {0:X}", hr);
        return false;
    }

    _gameCommandQueue = realQueue;
    _swapChain = *swapChain;
    _hwnd = hwnd;

    LOG_INFO("[FGExtrap] Swapchain1 created successfully");
    return true;
}

bool FGExtrap_Dx12::ReleaseSwapchain(HWND hwnd)
{
    if (hwnd != _hwnd || _hwnd == NULL)
        return false;

    LOG_INFO("[FGExtrap] ReleaseSwapchain");

    if (Config::Instance()->FGUseMutexForSwapchain.value_or_default())
    {
        Mutex.lock(1);
    }

    DestroyFGContext();

    if (Config::Instance()->FGUseMutexForSwapchain.value_or_default())
    {
        Mutex.unlockThis(1);
    }

    State::Instance().currentFGSwapchain = nullptr;
    return true;
}

// ---------------------------------------------------------------------------
// Context management
// ---------------------------------------------------------------------------

void FGExtrap_Dx12::CreateObjects(ID3D12Device* InDevice)
{
    _device = InDevice;

    if (_fgCommandAllocator[0] != nullptr)
        return; // Already initialized

    for (int i = 0; i < BUFFER_COUNT; i++)
    {
        InDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                         IID_PPV_ARGS(&_fgCommandAllocator[i]));
        InDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                    _fgCommandAllocator[i], NULL,
                                    IID_PPV_ARGS(&_fgCommandList[i]));
        _fgCommandList[i]->Close();
    }

    LOG_INFO("[FGExtrap] D3D12 objects created");
}

void FGExtrap_Dx12::ReleaseObjects()
{
    for (int i = 0; i < BUFFER_COUNT; i++)
    {
        if (_fgCommandAllocator[i]) { _fgCommandAllocator[i]->Release(); _fgCommandAllocator[i] = nullptr; }
        if (_fgCommandList[i]) { _fgCommandList[i]->Release(); _fgCommandList[i] = nullptr; }
    }
}

void FGExtrap_Dx12::CreateContext(ID3D12Device* device, FG_Constants& fgConstants)
{
    LOG_INFO("[FGExtrap] CreateContext");

    CreateObjects(device);
    _constants = fgConstants;

    // Initialize compute pipeline
    if (!_pipeline->IsInit())
    {
        if (!_pipeline->Init(device))
        {
            LOG_ERROR("[FGExtrap] Failed to initialize compute pipeline");
            return;
        }
    }

    // Start mouse tracker
    if (!_mouseTracker->IsRunning())
        _mouseTracker->Start();

    // Configure cadence controller
    _cadenceController->SetTargetFPS(Config::Instance()->FGExtrapTargetFPS.value_or_default());

    _contextCreated = true;
    _isActive = true;

    LOG_INFO("[FGExtrap] Context created, target FPS: {}", _cadenceController->GetTargetFPS());
}

void FGExtrap_Dx12::EvaluateState(ID3D12Device* device, FG_Constants& fgConstants)
{
    if (!_contextCreated)
    {
        CreateContext(device, fgConstants);
    }
}

void FGExtrap_Dx12::Activate()
{
    _isActive = true;
    _waitingNewFrameData = false;
    LOG_INFO("[FGExtrap] Activated");
}

void FGExtrap_Dx12::Deactivate()
{
    _isActive = false;
    LOG_INFO("[FGExtrap] Deactivated");
}

void FGExtrap_Dx12::DestroyFGContext()
{
    LOG_INFO("[FGExtrap] DestroyFGContext");

    Deactivate();
    _mouseTracker->Stop();
    _pipeline->Release();
    ReleaseObjects();
    _contextCreated = false;
    _frameCount = 1;
}

bool FGExtrap_Dx12::Shutdown()
{
    LOG_INFO("[FGExtrap] Shutdown");
    DestroyFGContext();
    return true;
}

// ---------------------------------------------------------------------------
// Resource handling
// ---------------------------------------------------------------------------

bool FGExtrap_Dx12::SetResource(Dx12Resource* inputResource)
{
    if (inputResource == nullptr || inputResource->resource == nullptr)
        return false;

    if (inputResource->type != FG_ResourceType::UIColor && (!IsActive() || IsPaused()))
        return false;

    auto fIndex = inputResource->frameIndex;
    if (fIndex < 0)
        fIndex = GetIndex();

    auto& type = inputResource->type;
    std::unique_lock<std::shared_mutex> lock(_resourceMutex[fIndex]);

    // Skip if already have a valid resource of this type
    if (_frameResources[fIndex].contains(type) &&
        _frameResources[fIndex][type].validity == FG_ResourceValidity::ValidNow)
        return false;

    // Respect config disable flags
    if (type == FG_ResourceType::HudlessColor && Config::Instance()->FGDisableHudless.value_or_default())
        return false;
    if (type == FG_ResourceType::UIColor && Config::Instance()->FGDisableUI.value_or_default())
        return false;

    // Store the resource
    _frameResources[fIndex][type] = {};
    auto fResource = &_frameResources[fIndex][type];
    fResource->type = type;
    fResource->state = inputResource->state;
    fResource->validity = inputResource->validity;
    fResource->resource = inputResource->resource;
    fResource->width = inputResource->width;
    fResource->height = inputResource->height;
    fResource->cmdList = inputResource->cmdList;

    // Resource flipping for upscaler input
    auto willFlip = State::Instance().activeFgInput == FGInput::Upscaler &&
                    Config::Instance()->FGResourceFlip.value_or_default() &&
                    (type == FG_ResourceType::Velocity || type == FG_ResourceType::Depth);

    if (willFlip && _device != nullptr)
        FlipResource(fResource);

    // Adjust validity
    fResource->validity = (fResource->validity != FG_ResourceValidity::ValidNow || willFlip)
                              ? FG_ResourceValidity::UntilPresent
                              : FG_ResourceValidity::ValidNow;

    // Copy ValidNow resources for later use
    if (fResource->validity == FG_ResourceValidity::ValidNow && inputResource->cmdList != nullptr)
    {
        ID3D12Resource* copyOutput = nullptr;
        if (_resourceCopy[fIndex].contains(type))
            copyOutput = _resourceCopy[fIndex].at(type);

        if (CopyResource(inputResource->cmdList, inputResource->resource, &copyOutput, inputResource->state))
        {
            _resourceCopy[fIndex][type] = copyOutput;
            fResource->copy = copyOutput;
            fResource->state = D3D12_RESOURCE_STATE_COPY_DEST;
        }
    }

    // Auto-calibration: track MV magnitude
    if (type == FG_ResourceType::Velocity && Config::Instance()->FGExtrapAutoCalibrate.value_or_default())
    {
        // We can't easily read GPU resources on CPU here, so calibration
        // will be approximate based on MV scale factors
        _calibrationFrames++;
    }

    SetResourceReady(type, fIndex);

    if (_waitingNewFrameData &&
        HasResource(FG_ResourceType::Depth, fIndex) &&
        HasResource(FG_ResourceType::Velocity, fIndex))
    {
        _waitingNewFrameData = false;
        Activate();
    }

    return true;
}

void FGExtrap_Dx12::SetCommandQueue(FG_ResourceType type, ID3D12CommandQueue* queue)
{
    // FGExtrap uses the game command queue set during CreateSwapchain
}

bool FGExtrap_Dx12::SetInterpolatedFrameCount(UINT interpolatedFrameCount)
{
    return true;
}

// ---------------------------------------------------------------------------
// Present — main dispatch point
// ---------------------------------------------------------------------------

bool FGExtrap_Dx12::Present()
{
    if (!_contextCreated || !IsActive())
        return false;

    auto fIndex = GetIndexWillBeDispatched();

    // Handle UI overlay (same pattern as FSRFG/XeFG)
    if (Config::Instance()->FGDrawUIOverFG.value_or_default())
    {
        auto ui = GetResource(FG_ResourceType::UIColor, fIndex);
        if (ui != nullptr)
        {
            if (_renderUI.get() == nullptr)
            {
                _renderUI = std::make_unique<RUI_Dx12>("RenderUI", _device,
                    Config::Instance()->FGUIPremultipliedAlpha.value_or_default());
            }
            else if (_renderUI->IsInit())
            {
                auto commandList = GetSCCommandList(fIndex);
                if (commandList != nullptr)
                    _renderUI->Dispatch((IDXGISwapChain3*) _swapChain, commandList,
                                       ui->GetResource(), ui->state);
            }
        }
    }

    // Execute pending command lists
    if (_uiCommandListResetted[fIndex])
    {
        auto closeResult = _uiCommandList[fIndex]->Close();
        if (closeResult == S_OK)
            _gameCommandQueue->ExecuteCommandLists(1, (ID3D12CommandList**) &_uiCommandList[fIndex]);
        _uiCommandListResetted[fIndex] = false;
    }

    if (_scCommandListResetted[fIndex])
    {
        auto closeResult = _scCommandList[fIndex]->Close();
        if (closeResult == S_OK)
            _gameCommandQueue->ExecuteCommandLists(1, (ID3D12CommandList**) &_scCommandList[fIndex]);
        _scCommandListResetted[fIndex] = false;
    }

    // Stall detection
    if ((_fgFramePresentId - _lastFGFramePresentId) > 3 && IsActive() && !_waitingNewFrameData)
    {
        LOG_DEBUG("[FGExtrap] Pausing — stalled");
        Deactivate();
        _waitingNewFrameData = true;
        return false;
    }

    _fgFramePresentId++;

    return DispatchSyntheticFrames();
}

// ---------------------------------------------------------------------------
// Core dispatch — generates and presents N synthetic frames
// ---------------------------------------------------------------------------

bool FGExtrap_Dx12::DispatchSyntheticFrames()
{
    UINT64 willDispatchFrame = 0;
    auto fIndex = GetDispatchIndex(willDispatchFrame);
    if (fIndex < 0)
        return false;

    if (!IsActive() || IsPaused())
        return false;

    // Check required resources
    if (!IsResourceReady(FG_ResourceType::Depth, fIndex) ||
        !IsResourceReady(FG_ResourceType::Velocity, fIndex))
    {
        LOG_WARN("[FGExtrap] Depth or Velocity not ready, skipping");
        return false;
    }

    // We use HudlessColor as the main color source, fall back to UIColor
    auto color = GetResource(FG_ResourceType::HudlessColor, fIndex);
    if (color == nullptr)
    {
        LOG_WARN("[FGExtrap] No color resource available, skipping");
        return false;
    }

    auto depth = GetResource(FG_ResourceType::Depth, fIndex);
    auto velocity = GetResource(FG_ResourceType::Velocity, fIndex);

    if (!depth || !velocity)
        return false;

    // Update cadence controller
    _cadenceController->SetTargetFPS(Config::Instance()->FGExtrapTargetFPS.value_or_default());
    _cadenceController->OnRealPresent();

    int syntheticCount = _cadenceController->GetSyntheticCount();
    if (syntheticCount <= 0)
    {
        _lastFGFramePresentId = _fgFramePresentId;
        return true; // Nothing to generate
    }

    LOG_DEBUG("[FGExtrap] Generating {} synthetic frames", syntheticCount);

    // Build config from settings
    auto cfg = Config::Instance();
    ExtrapConfig extrapCfg;
    extrapCfg.nearPlane = _cameraNear[fIndex];
    extrapCfg.farPlane = _cameraFar[fIndex];
    extrapCfg.hudThreshold = cfg->FGExtrapHUDThreshold.value_or_default();
    extrapCfg.skyThreshold = cfg->FGExtrapSkyThreshold.value_or_default();
    extrapCfg.farThreshold = cfg->FGExtrapFarThreshold.value_or_default();
    extrapCfg.nearThreshold = cfg->FGExtrapNearThreshold.value_or_default();
    extrapCfg.invertedDepth = _constants.flags[FG_Flags::InvertedDepth];
    extrapCfg.depthScale = cfg->FGExtrapDepthScale.value_or_default();
    extrapCfg.useMVExtrapolation = cfg->FGExtrapMVExtrapolation.value_or_default();
    extrapCfg.gapFillMode = cfg->FGExtrapGapFillMode.value_or_default();
    extrapCfg.maxExtendPixels = 32;
    extrapCfg.debugLayers = cfg->FGExtrapDebugLayers.value_or_default();

    // Compute rotation scale from FOV and resolution
    float fov = cfg->FGExtrapFOV.value_or_default();
    float sensitivity = cfg->FGExtrapMouseSensitivity.value_or_default();
    if (_cameraVFov[fIndex] > 0.001f)
        fov = _cameraVFov[fIndex] * (180.0f / 3.14159f); // Convert from radians if available

    float resWidth = (float)(color->width > 0 ? color->width : _constants.displayWidth);
    extrapCfg.rotScale = (fov / resWidth) * sensitivity * 0.01f;

    // Generate each synthetic frame
    for (int i = 1; i <= syntheticCount; i++)
    {
        // Wait for proper timing
        _cadenceController->WaitForSyntheticFrame(i);

        // Read mouse deltas accumulated since last read
        long mouseDx = 0, mouseDy = 0;
        _mouseTracker->ConsumeDeltas(mouseDx, mouseDy);

        extrapCfg.timeFraction = _cadenceController->GetTimeFraction(i);

        // Reset and prepare command list
        auto allocIdx = fIndex; // Use the frame's command list slot
        auto hr = _fgCommandAllocator[allocIdx]->Reset();
        if (FAILED(hr))
        {
            LOG_ERROR("[FGExtrap] CommandAllocator Reset failed: {0:X}", hr);
            continue;
        }

        hr = _fgCommandList[allocIdx]->Reset(_fgCommandAllocator[allocIdx], nullptr);
        if (FAILED(hr))
        {
            LOG_ERROR("[FGExtrap] CommandList Reset failed: {0:X}", hr);
            continue;
        }

        auto cmdList = _fgCommandList[allocIdx];

        // Run compute pipeline
        auto compositeResult = _pipeline->Dispatch(
            cmdList,
            color->GetResource(), color->state,
            depth->GetResource(), depth->state,
            velocity->GetResource(), velocity->state,
            (float)mouseDx, (float)mouseDy,
            _mvScaleX[fIndex], _mvScaleY[fIndex],
            extrapCfg);

        if (compositeResult == nullptr)
        {
            LOG_ERROR("[FGExtrap] Pipeline dispatch failed");
            cmdList->Close();
            continue;
        }

        // Copy composite result to backbuffer
        IDXGISwapChain3* sc3 = nullptr;
        if (FAILED(_swapChain->QueryInterface(IID_PPV_ARGS(&sc3))))
        {
            LOG_ERROR("[FGExtrap] Failed to get IDXGISwapChain3");
            cmdList->Close();
            continue;
        }

        UINT backbufferIndex = sc3->GetCurrentBackBufferIndex();
        ID3D12Resource* backbuffer = nullptr;
        hr = sc3->GetBuffer(backbufferIndex, IID_PPV_ARGS(&backbuffer));
        if (FAILED(hr))
        {
            LOG_ERROR("[FGExtrap] GetBuffer failed: {0:X}", hr);
            sc3->Release();
            cmdList->Close();
            continue;
        }

        // Transition and copy
        ResourceBarrier(cmdList, backbuffer,
                        D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST);
        ResourceBarrier(cmdList, compositeResult,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);

        cmdList->CopyResource(backbuffer, compositeResult);

        ResourceBarrier(cmdList, backbuffer,
                        D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT);
        ResourceBarrier(cmdList, compositeResult,
                        D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        cmdList->Close();

        // Execute on game queue
        _gameCommandQueue->ExecuteCommandLists(1, (ID3D12CommandList**) &cmdList);

        backbuffer->Release();
        sc3->Release();

        // Present the synthetic frame via the hooks mechanism
        UINT syncInterval = 0;
        UINT flags = 0;
        if (State::Instance().SCAllowTearing && !State::Instance().realExclusiveFullscreen)
            flags |= DXGI_PRESENT_ALLOW_TEARING;

        FGHooks::PresentSynthetic(_swapChain, syncInterval, flags);

        LOG_DEBUG("[FGExtrap] Synthetic frame {}/{} presented (timeFrac={:.3f})",
                  i, syntheticCount, extrapCfg.timeFraction);
    }

    _lastFGFramePresentId = _fgFramePresentId;
    return true;
}
