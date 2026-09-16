#pragma once

#include "quantization/bit_width.h"

#include <cstdint>

// 外部量化参数只保存 standard scale/zp；执行编码不能反向覆盖本结构。
namespace quant_lstm::quantization {

enum class ScaleMode : std::uint8_t {
    Affine,
    Pot2,
};

struct QuantParam {
    float scale = 0.0F;
    std::int32_t zero_point = 0;

    void validate(const QuantizationType& type) const;
};

struct FixedPointScale {
    std::uint16_t multiplier = 0;
    std::int8_t shift = 0;
};

struct Pot2Rescale {
    std::int8_t shift = 0;
};

struct CalibrationDiagnostics {
    float original_min = 0.0F;
    float original_max = 0.0F;
    float adjusted_min = 0.0F;
    float adjusted_max = 0.0F;
    float minimum_scale = 0.0F;
    bool fallback_used = false;
};

struct CalibrationResult {
    QuantParam param;
    CalibrationDiagnostics diagnostics;
};

}  // namespace quant_lstm::quantization
