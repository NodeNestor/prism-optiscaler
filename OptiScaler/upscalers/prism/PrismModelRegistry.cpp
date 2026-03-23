#include "pch.h"
#include "PrismModelRegistry.h"

#include <json.hpp>
#include <fstream>

using json = nlohmann::json;
namespace fs = std::filesystem;

void PrismModelRegistry::ScanDirectory(const fs::path& modelDir)
{
    _models.clear();
    _scanned = true;

    if (!fs::exists(modelDir) || !fs::is_directory(modelDir))
    {
        LOG_WARN("[Prism] Model directory not found: {}", modelDir.string());
        return;
    }

    LOG_INFO("[Prism] Scanning for models in: {}", modelDir.string());

    for (const auto& entry : fs::directory_iterator(modelDir))
    {
        if (!entry.is_regular_file())
            continue;

        auto ext = entry.path().extension().string();
        if (ext != ".weights" && ext != ".bin")
            continue;

        PrismModelInfo model;
        model.weightsPath = entry.path().string();

        // Look for accompanying .json config
        auto jsonPath = entry.path();
        jsonPath.replace_extension(".json");

        if (fs::exists(jsonPath))
        {
            model.configPath = jsonPath.string();

            try
            {
                std::ifstream f(jsonPath);
                json cfg = json::parse(f);

                if (cfg.contains("name"))
                    model.name = cfg["name"].get<std::string>();
                if (cfg.contains("channels"))
                    model.channels = cfg["channels"].get<int>();
                if (cfg.contains("blocks"))
                    model.blocks = cfg["blocks"].get<int>();
                if (cfg.contains("scale"))
                    model.scale = cfg["scale"].get<int>();
            }
            catch (const std::exception& e)
            {
                LOG_WARN("[Prism] Failed to parse {}: {}", jsonPath.string(), e.what());
            }
        }

        // Default name from filename if not in JSON
        if (model.name.empty())
            model.name = entry.path().stem().string();

        LOG_INFO("[Prism] Found model: {} ({}ch, {}blocks, {}x) @ {}",
                 model.name, model.channels, model.blocks, model.scale, model.weightsPath);

        _models.push_back(std::move(model));
    }

    LOG_INFO("[Prism] Found {} model(s)", _models.size());
}

void PrismModelRegistry::Rescan(const fs::path& modelDir)
{
    ScanDirectory(modelDir);
}
