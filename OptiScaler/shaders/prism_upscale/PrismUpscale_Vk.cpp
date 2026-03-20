#include "pch.h"
#include "PrismUpscale_Vk.h"
#include "precompile/PrismUpscale_Shader_Vk.h"

PrismUpscale_Vk::PrismUpscale_Vk(std::string InName, VkDevice InDevice, VkPhysicalDevice InPhysicalDevice)
    : Shader_Vk(InName, InDevice, InPhysicalDevice)
{
    if (InDevice == VK_NULL_HANDLE)
    {
        LOG_ERROR("PrismUpscale_Vk: device is null");
        return;
    }

    LOG_INFO("=== PRISM === Creating temporal upscale pipeline");

    CreateDescriptorSetLayout();
    CreateConstantBuffer();
    CreateDescriptorPool();
    CreateDescriptorSets();

    // Linear sampler for bilinear upscale
    VkSamplerCreateInfo linearInfo {};
    linearInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    linearInfo.magFilter = VK_FILTER_LINEAR;
    linearInfo.minFilter = VK_FILTER_LINEAR;
    linearInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    linearInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vkCreateSampler(_device, &linearInfo, nullptr, &_linearSampler);

    // Point sampler for motion vectors / depth
    VkSamplerCreateInfo pointInfo {};
    pointInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    pointInfo.magFilter = VK_FILTER_NEAREST;
    pointInfo.minFilter = VK_FILTER_NEAREST;
    pointInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    pointInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vkCreateSampler(_device, &pointInfo, nullptr, &_pointSampler);

    // Load SPIR-V and create pipeline
    std::vector<char> shaderCode(prism_upscale_spv, prism_upscale_spv + sizeof(prism_upscale_spv));
    if (!CreateComputePipeline(_device, _pipelineLayout, &_pipeline, shaderCode))
    {
        LOG_ERROR("=== PRISM === Failed to create compute pipeline");
        return;
    }

    _init = true;
    LOG_INFO("=== PRISM === Temporal upscale pipeline created successfully");
}

PrismUpscale_Vk::~PrismUpscale_Vk()
{
    if (_descriptorPool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(_device, _descriptorPool, nullptr);

    if (_descriptorSetLayout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(_device, _descriptorSetLayout, nullptr);

    if (_pipelineLayout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(_device, _pipelineLayout, nullptr);

    if (_constantBuffer != VK_NULL_HANDLE)
        vkDestroyBuffer(_device, _constantBuffer, nullptr);

    if (_constantBufferMemory != VK_NULL_HANDLE)
        vkFreeMemory(_device, _constantBufferMemory, nullptr);

    if (_linearSampler != VK_NULL_HANDLE)
        vkDestroySampler(_device, _linearSampler, nullptr);

    if (_pointSampler != VK_NULL_HANDLE)
        vkDestroySampler(_device, _pointSampler, nullptr);

    ReleaseHistoryBuffer();
}

void PrismUpscale_Vk::CreateDescriptorSetLayout()
{
    // b0: Constants
    VkDescriptorSetLayoutBinding cbBinding {};
    cbBinding.binding = 0;
    cbBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    cbBinding.descriptorCount = 1;
    cbBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // t0: Color (sampled)
    VkDescriptorSetLayoutBinding colorBinding {};
    colorBinding.binding = 1;
    colorBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    colorBinding.descriptorCount = 1;
    colorBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // t1: Motion vectors (sampled)
    VkDescriptorSetLayoutBinding mvBinding {};
    mvBinding.binding = 2;
    mvBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    mvBinding.descriptorCount = 1;
    mvBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // t2: Depth (sampled)
    VkDescriptorSetLayoutBinding depthBinding {};
    depthBinding.binding = 3;
    depthBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    depthBinding.descriptorCount = 1;
    depthBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // t3: History (sampled)
    VkDescriptorSetLayoutBinding histBinding {};
    histBinding.binding = 4;
    histBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    histBinding.descriptorCount = 1;
    histBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // u0: Output (storage)
    VkDescriptorSetLayoutBinding outBinding {};
    outBinding.binding = 5;
    outBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    outBinding.descriptorCount = 1;
    outBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // s0: Linear sampler (immutable)
    // s1: Point sampler (immutable)
    // These are set via the combined image samplers above

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        cbBinding, colorBinding, mvBinding, depthBinding, histBinding, outBinding
    };

    VkDescriptorSetLayoutCreateInfo layoutInfo {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = (uint32_t)bindings.size();
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(_device, &layoutInfo, nullptr, &_descriptorSetLayout) != VK_SUCCESS)
    {
        LOG_ERROR("=== PRISM === Failed to create descriptor set layout");
        return;
    }

    VkPipelineLayoutCreateInfo pipelineLayoutInfo {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &_descriptorSetLayout;

    if (vkCreatePipelineLayout(_device, &pipelineLayoutInfo, nullptr, &_pipelineLayout) != VK_SUCCESS)
        LOG_ERROR("=== PRISM === Failed to create pipeline layout");
}

void PrismUpscale_Vk::CreateDescriptorPool()
{
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, (uint32_t)MAX_FRAMES_IN_FLIGHT },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, (uint32_t)(4 * MAX_FRAMES_IN_FLIGHT) },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, (uint32_t)MAX_FRAMES_IN_FLIGHT },
    };

    VkDescriptorPoolCreateInfo poolInfo {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = (uint32_t)poolSizes.size();
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = (uint32_t)MAX_FRAMES_IN_FLIGHT;

    if (vkCreateDescriptorPool(_device, &poolInfo, nullptr, &_descriptorPool) != VK_SUCCESS)
        LOG_ERROR("=== PRISM === Failed to create descriptor pool");
}

void PrismUpscale_Vk::CreateDescriptorSets()
{
    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, _descriptorSetLayout);

    VkDescriptorSetAllocateInfo allocInfo {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = _descriptorPool;
    allocInfo.descriptorSetCount = (uint32_t)MAX_FRAMES_IN_FLIGHT;
    allocInfo.pSetLayouts = layouts.data();

    _descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    if (vkAllocateDescriptorSets(_device, &allocInfo, _descriptorSets.data()) != VK_SUCCESS)
        LOG_ERROR("=== PRISM === Failed to allocate descriptor sets");
}

void PrismUpscale_Vk::CreateConstantBuffer()
{
    VkDeviceSize size = sizeof(PrismConstants);

    if (!Shader_Vk::CreateBufferResource(_device, _physicalDevice, &_constantBuffer, &_constantBufferMemory,
                                          size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
    {
        LOG_ERROR("=== PRISM === Failed to create constant buffer");
        return;
    }

    vkMapMemory(_device, _constantBufferMemory, 0, size, 0, &_mappedConstantBuffer);
}

void PrismUpscale_Vk::UpdateDescriptorSet(int setIndex, VkImageView colorView, VkImageView mvView,
                                            VkImageView depthView, VkImageView historyView,
                                            VkImageView outputView)
{
    VkDescriptorSet ds = _descriptorSets[setIndex];

    // 0: UBO
    VkDescriptorBufferInfo bufInfo {};
    bufInfo.buffer = _constantBuffer;
    bufInfo.offset = 0;
    bufInfo.range = sizeof(PrismConstants);

    VkWriteDescriptorSet wUbo {};
    wUbo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wUbo.dstSet = ds;
    wUbo.dstBinding = 0;
    wUbo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    wUbo.descriptorCount = 1;
    wUbo.pBufferInfo = &bufInfo;

    // 1: Color
    VkDescriptorImageInfo colorInfo {};
    colorInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    colorInfo.imageView = colorView;
    colorInfo.sampler = _linearSampler;

    VkWriteDescriptorSet wColor {};
    wColor.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wColor.dstSet = ds;
    wColor.dstBinding = 1;
    wColor.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wColor.descriptorCount = 1;
    wColor.pImageInfo = &colorInfo;

    // 2: Motion vectors
    VkDescriptorImageInfo mvInfo {};
    mvInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    mvInfo.imageView = mvView ? mvView : colorView;
    mvInfo.sampler = _pointSampler;

    VkWriteDescriptorSet wMv {};
    wMv.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wMv.dstSet = ds;
    wMv.dstBinding = 2;
    wMv.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wMv.descriptorCount = 1;
    wMv.pImageInfo = &mvInfo;

    // 3: Depth
    VkDescriptorImageInfo depthInfo {};
    depthInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    depthInfo.imageView = depthView ? depthView : colorView;
    depthInfo.sampler = _pointSampler;

    VkWriteDescriptorSet wDepth {};
    wDepth.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wDepth.dstSet = ds;
    wDepth.dstBinding = 3;
    wDepth.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wDepth.descriptorCount = 1;
    wDepth.pImageInfo = &depthInfo;

    // 4: History
    VkDescriptorImageInfo histInfo {};
    histInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    histInfo.imageView = historyView;
    histInfo.sampler = _linearSampler;

    VkWriteDescriptorSet wHist {};
    wHist.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wHist.dstSet = ds;
    wHist.dstBinding = 4;
    wHist.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wHist.descriptorCount = 1;
    wHist.pImageInfo = &histInfo;

    // 5: Output (storage)
    VkDescriptorImageInfo outInfo {};
    outInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    outInfo.imageView = outputView;
    outInfo.sampler = VK_NULL_HANDLE;

    VkWriteDescriptorSet wOut {};
    wOut.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wOut.dstSet = ds;
    wOut.dstBinding = 5;
    wOut.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    wOut.descriptorCount = 1;
    wOut.pImageInfo = &outInfo;

    VkWriteDescriptorSet writes[] = { wUbo, wColor, wMv, wDepth, wHist, wOut };
    vkUpdateDescriptorSets(_device, 6, writes, 0, nullptr);
}

bool PrismUpscale_Vk::CreateHistoryBuffer(uint32_t width, uint32_t height, VkFormat format)
{
    if (_historyImage != VK_NULL_HANDLE && _historyWidth == width && _historyHeight == height)
        return true;

    ReleaseHistoryBuffer();

    _historyWidth = width;
    _historyHeight = height;

    VkImageCreateInfo imageInfo {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = { width, height, 1 };
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

    if (vkCreateImage(_device, &imageInfo, nullptr, &_historyImage) != VK_SUCCESS)
    {
        LOG_ERROR("=== PRISM === Failed to create history image");
        return false;
    }

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(_device, _historyImage, &memReq);

    VkMemoryAllocateInfo allocInfo {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = FindMemoryType(_physicalDevice, memReq.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(_device, &allocInfo, nullptr, &_historyMemory) != VK_SUCCESS)
    {
        LOG_ERROR("=== PRISM === Failed to allocate history memory");
        return false;
    }

    vkBindImageMemory(_device, _historyImage, _historyMemory, 0);

    VkImageViewCreateInfo viewInfo {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = _historyImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    if (vkCreateImageView(_device, &viewInfo, nullptr, &_historyImageView) != VK_SUCCESS)
    {
        LOG_ERROR("=== PRISM === Failed to create history image view");
        return false;
    }

    _historyInitialized = false;
    LOG_INFO("=== PRISM === Created history buffer {}x{}", width, height);
    return true;
}

void PrismUpscale_Vk::ReleaseHistoryBuffer()
{
    if (_historyImageView != VK_NULL_HANDLE)
    {
        vkDestroyImageView(_device, _historyImageView, nullptr);
        _historyImageView = VK_NULL_HANDLE;
    }
    if (_historyImage != VK_NULL_HANDLE)
    {
        vkDestroyImage(_device, _historyImage, nullptr);
        _historyImage = VK_NULL_HANDLE;
    }
    if (_historyMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(_device, _historyMemory, nullptr);
        _historyMemory = VK_NULL_HANDLE;
    }
}

void PrismUpscale_Vk::SetImageLayout(VkCommandBuffer cmdBuffer, VkImage image,
                                       VkImageLayout oldLayout, VkImageLayout newLayout)
{
    VkImageMemoryBarrier barrier {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;

    VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED)
    {
        barrier.srcAccessMask = 0;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_GENERAL)
    {
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }

    if (newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    {
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        dstStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
    else if (newLayout == VK_IMAGE_LAYOUT_GENERAL)
    {
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        dstStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }

    vkCmdPipelineBarrier(cmdBuffer, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void PrismUpscale_Vk::CopyOutputToHistory(VkCommandBuffer cmdList, VkImage outputImage,
                                            uint32_t width, uint32_t height)
{
    // Transition output to TRANSFER_SRC, history to TRANSFER_DST
    SetImageLayout(cmdList, outputImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    SetImageLayout(cmdList, _historyImage,
                   _historyInitialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    VkImageCopy region {};
    region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.extent = { width, height, 1 };

    vkCmdCopyImage(cmdList,
                   outputImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   _historyImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1, &region);

    // Transition back
    SetImageLayout(cmdList, outputImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    SetImageLayout(cmdList, _historyImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    _historyInitialized = true;
}

bool PrismUpscale_Vk::Dispatch(VkCommandBuffer InCmdList, VkImageView colorView, VkImageView mvView,
                                VkImageView depthView, VkImageView historyView, VkImageView outputView,
                                uint32_t displayWidth, uint32_t displayHeight,
                                float renderWidth, float renderHeight,
                                float jitterX, float jitterY, float mvScaleX, float mvScaleY,
                                int reset, float sharpness)
{
    if (!_init || InCmdList == VK_NULL_HANDLE)
        return false;

    // Update constants
    PrismConstants constants {};
    constants.renderWidth = renderWidth;
    constants.renderHeight = renderHeight;
    constants.displayWidth = (float)displayWidth;
    constants.displayHeight = (float)displayHeight;
    constants.jitterX = jitterX;
    constants.jitterY = jitterY;
    constants.mvScaleX = mvScaleX;
    constants.mvScaleY = mvScaleY;
    constants.reset = reset || !_historyInitialized ? 1 : 0;
    constants.sharpness = sharpness;

    if (_mappedConstantBuffer)
        memcpy(_mappedConstantBuffer, &constants, sizeof(PrismConstants));

    // Update descriptors
    _currentSetIndex = (_currentSetIndex + 1) % MAX_FRAMES_IN_FLIGHT;
    UpdateDescriptorSet(_currentSetIndex, colorView, mvView, depthView, historyView, outputView);

    // Bind and dispatch
    vkCmdBindPipeline(InCmdList, VK_PIPELINE_BIND_POINT_COMPUTE, _pipeline);
    vkCmdBindDescriptorSets(InCmdList, VK_PIPELINE_BIND_POINT_COMPUTE, _pipelineLayout,
                            0, 1, &_descriptorSets[_currentSetIndex], 0, nullptr);

    uint32_t groupX = (displayWidth + 15) / 16;
    uint32_t groupY = (displayHeight + 15) / 16;
    vkCmdDispatch(InCmdList, groupX, groupY, 1);

    return true;
}
