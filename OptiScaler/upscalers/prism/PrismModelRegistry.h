#pragma once
#include "SysUtils.h"

#include <string>
#include <vector>
#include <filesystem>

struct PrismModelInfo
{
    std::string name;           // Display name (from JSON or filename)
    std::string weightsPath;    // Full path to .weights file
    std::string configPath;     // Full path to .json config (may be empty)
    int channels = 64;
    int blocks = 4;
    int scale = 2;
};

class PrismModelRegistry
{
  private:
    std::vector<PrismModelInfo> _models;
    bool _scanned = false;

  public:
    // Scan a directory for model files (.weights + optional .json config)
    void ScanDirectory(const std::filesystem::path& modelDir);

    // Re-scan (e.g., after hot-reload)
    void Rescan(const std::filesystem::path& modelDir);

    const std::vector<PrismModelInfo>& GetModels() const { return _models; }
    int GetModelCount() const { return (int)_models.size(); }
    bool HasModels() const { return !_models.empty(); }
    bool IsScanned() const { return _scanned; }

    const PrismModelInfo* GetModel(int index) const
    {
        if (index >= 0 && index < (int)_models.size())
            return &_models[index];
        return nullptr;
    }

    // Singleton
    static PrismModelRegistry& Instance()
    {
        static PrismModelRegistry instance;
        return instance;
    }

  private:
    PrismModelRegistry() = default;
};
