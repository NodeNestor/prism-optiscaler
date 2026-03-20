#include "pch.h"
#include "PrismFeature_Dx11.h"
#include "State.h"

PrismFeatureDx11::PrismFeatureDx11(unsigned int InHandleId, NVSDK_NGX_Parameter* InParameters)
    : IFeature(InHandleId, InParameters),
      IFeature_Dx11(InHandleId, InParameters),
      PrismFeature(InHandleId, InParameters)
{
    LOG_INFO("PrismFeatureDx11 constructed");
}

PrismFeatureDx11::~PrismFeatureDx11()
{
    LOG_INFO("PrismFeatureDx11 destroyed");
}

bool PrismFeatureDx11::Init(ID3D11Device* InDevice, ID3D11DeviceContext* InContext,
                             NVSDK_NGX_Parameter* InParameters)
{
    if (!_moduleLoaded)
        return false;

    Device = InDevice;
    DeviceContext = InContext;

    if (!SetInitParameters(InParameters))
    {
        LOG_ERROR("Prism DX11: SetInitParameters failed");
        return false;
    }

    SetInit(true);
    _prismInited = true;

    LOG_INFO("Prism DX11 initialized: render={}x{} -> display={}x{}",
             _renderWidth, _renderHeight, _displayWidth, _displayHeight);

    return true;
}

bool PrismFeatureDx11::Evaluate(ID3D11DeviceContext* InContext, NVSDK_NGX_Parameter* InParameters)
{
    if (!_prismInited || !InContext)
        return false;

    _frameCount++;

    // Extract resources
    ID3D11Resource* paramColor = nullptr;
    ID3D11Resource* paramOutput = nullptr;

    InParameters->Get(NVSDK_NGX_Parameter_Color, (void**)&paramColor);
    InParameters->Get(NVSDK_NGX_Parameter_Output, (void**)&paramOutput);

    if (!paramColor || !paramOutput)
    {
        LOG_ERROR("Prism DX11: missing color or output");
        return false;
    }

    // Simple copy — replace with DX11 compute shader or bridge to DX12
    InContext->CopyResource(paramOutput, paramColor);

    return true;
}
