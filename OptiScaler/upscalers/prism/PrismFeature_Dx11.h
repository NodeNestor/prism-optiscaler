#pragma once
#include "PrismFeature.h"
#include "upscalers/IFeature_Dx11.h"

// Prism DX11 backend — passthrough/copy for now.
// Extend with DX11 compute shader or bridge to DX12 for your model.

class PrismFeatureDx11 : public PrismFeature, public IFeature_Dx11
{
  public:
    PrismFeatureDx11(unsigned int InHandleId, NVSDK_NGX_Parameter* InParameters);
    ~PrismFeatureDx11();

    bool Init(ID3D11Device* InDevice, ID3D11DeviceContext* InContext,
              NVSDK_NGX_Parameter* InParameters) override;
    bool Evaluate(ID3D11DeviceContext* InContext, NVSDK_NGX_Parameter* InParameters) override;

    bool IsWithDx12() override { return false; }
};
