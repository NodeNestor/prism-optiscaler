#pragma once

#include "SysUtils.h"
#include <shaders/Shader_Vk.h>

class PrismUpscale_Vk : public Shader_Vk
{
  public:
    PrismUpscale_Vk(std::string InName, VkDevice InDevice, VkPhysicalDevice InPhysicalDevice);
    ~PrismUpscale_Vk();

    bool Dispatch(VkCommandBuffer InCmdList, VkImageView colorView, VkImageView mvView,
                  VkImageView depthView, VkImageView historyView, VkImageView outputView,
                  uint32_t displayWidth, uint32_t displayHeight,
                  float renderWidth, float renderHeight,
                  float jitterX, float jitterY, float mvScaleX, float mvScaleY,
                  int reset, float sharpness);

    // History buffer management
    bool CreateHistoryBuffer(uint32_t width, uint32_t height, VkFormat format);
    void CopyOutputToHistory(VkCommandBuffer cmdList, VkImage outputImage, uint32_t width, uint32_t height);

    VkImageView GetHistoryView() const { return _historyImageView; }
    VkImage GetHistoryImage() const { return _historyImage; }
    bool HasHistory() const { return _historyImage != VK_NULL_HANDLE; }

    void SetImageLayout(VkCommandBuffer cmdBuffer, VkImage image, VkImageLayout oldLayout,
                        VkImageLayout newLayout);

    bool CanRender() const { return _init && _pipeline != VK_NULL_HANDLE; }

  private:
    struct alignas(256) PrismConstants
    {
        float renderWidth;
        float renderHeight;
        float displayWidth;
        float displayHeight;
        float jitterX;
        float jitterY;
        float mvScaleX;
        float mvScaleY;
        int reset;
        float sharpness;
        float padding0;
        float padding1;
    };

    VkBuffer _constantBuffer = VK_NULL_HANDLE;
    VkDeviceMemory _constantBufferMemory = VK_NULL_HANDLE;
    VkSampler _linearSampler = VK_NULL_HANDLE;
    VkSampler _pointSampler = VK_NULL_HANDLE;
    void* _mappedConstantBuffer = nullptr;

    VkDescriptorPool _descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> _descriptorSets;
    uint32_t _currentSetIndex = 0;
    static const int MAX_FRAMES_IN_FLIGHT = 3;

    // History buffer (display resolution, previous frame)
    VkImage _historyImage = VK_NULL_HANDLE;
    VkImageView _historyImageView = VK_NULL_HANDLE;
    VkDeviceMemory _historyMemory = VK_NULL_HANDLE;
    uint32_t _historyWidth = 0;
    uint32_t _historyHeight = 0;
    bool _historyInitialized = false;

    void CreateDescriptorSetLayout();
    void CreateDescriptorPool();
    void CreateDescriptorSets();
    void CreateConstantBuffer();
    void UpdateDescriptorSet(int setIndex, VkImageView colorView, VkImageView mvView,
                             VkImageView depthView, VkImageView historyView, VkImageView outputView);
    void ReleaseHistoryBuffer();
};
