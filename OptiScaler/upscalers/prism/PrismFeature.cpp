#include "pch.h"
#include "PrismFeature.h"

PrismFeature::PrismFeature(unsigned int InHandleId, NVSDK_NGX_Parameter* InParameters)
    : IFeature(InHandleId, InParameters)
{
    _moduleLoaded = true;

    if (!SetInitParameters(InParameters))
    {
        LOG_ERROR("PrismFeature: SetInitParameters failed");
        return;
    }

    LOG_INFO("PrismFeature created: render={}x{} target={}x{} display={}x{}",
             _renderWidth, _renderHeight, _targetWidth, _targetHeight, _displayWidth, _displayHeight);
}
