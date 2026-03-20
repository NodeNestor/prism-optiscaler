#pragma once
#include "PrismFeature.h"
#include "upscalers/IFeature_Vk.h"
#include <shaders/prism_upscale/PrismUpscale_Vk.h>

class PrismFeatureVk : public PrismFeature, public IFeature_Vk
{
  private:
    VkDevice _vkDevice = VK_NULL_HANDLE;
    std::unique_ptr<PrismUpscale_Vk> _upscaler;

  public:
    PrismFeatureVk(unsigned int InHandleId, NVSDK_NGX_Parameter* InParameters);
    ~PrismFeatureVk();

    bool Init(VkInstance InInstance, VkPhysicalDevice InPD, VkDevice InDevice, VkCommandBuffer InCmdList,
              PFN_vkGetInstanceProcAddr InGIPA, PFN_vkGetDeviceProcAddr InGDPA,
              NVSDK_NGX_Parameter* InParameters) override;
    bool Evaluate(VkCommandBuffer InCmdBuffer, NVSDK_NGX_Parameter* InParameters) override;
};
