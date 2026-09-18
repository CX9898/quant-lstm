#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "lstm/quant_config.h"

// 唯一 C++ resolver：严格解析 defaults/override，并生成字节稳定的 canonical resolved JSON。
namespace quant_lstm {

LstmOperatorQuantConfig parseResolvedQuantConfig(std::string_view json_text,
                                                 bool require_canonical = true);

LstmOperatorQuantConfig resolveQuantConfig(std::string_view default_json,
                                           std::string_view override_json);

LstmOperatorQuantConfig resolveQuantConfigFiles(
    const std::filesystem::path& default_path,
    const std::optional<std::filesystem::path>& override_path = std::nullopt);

std::string toCanonicalJson(const LstmOperatorQuantConfig& config);

std::string formatCanonicalFloat32Value(float value);
float parseCanonicalFloat32Value(std::string_view text);
std::string formatCanonicalFloat32(float value);
float parseCanonicalFloat32(std::string_view text);

}  // namespace quant_lstm
