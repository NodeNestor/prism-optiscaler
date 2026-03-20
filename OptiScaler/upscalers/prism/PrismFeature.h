#pragma once
#include "upscalers/IFeature.h"

// Prism backend — custom neural upscaler placeholder.
// Currently implements a simple bilinear upscale as a working example.
// Replace Evaluate() with your own model (TensorRT, ONNX, compute shader, etc.)

class PrismFeature : public virtual IFeature
{
  protected:
    bool _prismInited = false;

  public:
    PrismFeature(unsigned int InHandleId, NVSDK_NGX_Parameter* InParameters);

    feature_version Version() override { return { 1, 0, 0 }; }
    std::string Name() const override { return "Prism"; }
};
