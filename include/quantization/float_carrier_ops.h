#pragma once

#include "quantization/quant_param.h"
#include "quantization/rounding.h"

#include <cmath>
#include <stdexcept>

// FP32 carrier 保存整数网格值，并模拟与整数 carrier 相同的 M+shift/POT2 编码。
namespace quant_lstm::quantization {

inline float applyRescale(float value, const FixedPointScale& encoded) {
    if (!std::isfinite(value) || encoded.multiplier < 32768U) {
        throw std::invalid_argument("FP carrier M+shift 输入或编码非法");
    }
    const float scaled =
        std::ldexp(value * static_cast<float>(encoded.multiplier), -encoded.shift);
    if (!std::isfinite(scaled)) {
        throw std::overflow_error("FP carrier M+shift 产生 Inf/NaN");
    }
    return roundToNearestEven(scaled);
}

inline float applyRescale(float value, const Pot2Rescale& encoded) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument("FP carrier POT2 输入必须有限");
    }
    const float scaled = std::ldexp(value, -encoded.shift);
    if (!std::isfinite(scaled)) {
        throw std::overflow_error("FP carrier POT2 产生 Inf/NaN");
    }
    return roundToNearestEven(scaled);
}

}  // namespace quant_lstm::quantization
