#pragma once

#include <cmath>
#include <stdexcept>

#include "quantization/quant_param.h"
#include "quantization/rounding.h"

#if defined(__CUDACC__)
#define QUANT_LSTM_FP_HOST_DEVICE __host__ __device__
#else
#define QUANT_LSTM_FP_HOST_DEVICE
#endif

// FP32 carrier 保存整数网格值，并模拟与整数 carrier 相同的 M+shift/POT2 编码。
namespace quant_lstm::quantization {

QUANT_LSTM_FP_HOST_DEVICE inline float floatCarrierMinimum(std::int32_t minimum) noexcept {
    const float endpoint = static_cast<float>(minimum);
    return static_cast<double>(endpoint) < static_cast<double>(minimum)
               ? ::nextafterf(endpoint, INFINITY)
               : endpoint;
}

QUANT_LSTM_FP_HOST_DEVICE inline float floatCarrierMaximum(std::int32_t maximum) noexcept {
    const float endpoint = static_cast<float>(maximum);
    return static_cast<double>(endpoint) > static_cast<double>(maximum)
               ? ::nextafterf(endpoint, -INFINITY)
               : endpoint;
}

QUANT_LSTM_FP_HOST_DEVICE inline float clampFloatCarrierCore(
    double value, std::int32_t minimum, std::int32_t maximum) noexcept {
    // The rounded float representation of INT32_MAX is outside its integer domain.
    const float lower = floatCarrierMinimum(minimum);
    const float upper = floatCarrierMaximum(maximum);
    return value <= static_cast<double>(lower)
               ? lower
               : (value >= static_cast<double>(upper) ? upper : static_cast<float>(value));
}

QUANT_LSTM_FP_HOST_DEVICE inline float centerFloatCarrierCore(
    float value, std::int32_t zero_point) noexcept {
    return static_cast<float>(static_cast<double>(value) - zero_point);
}

inline void validateFloatCarrierValue(float value, const QuantizationType& type) {
    const auto range = type.range();
    if (!std::isfinite(value) || roundToNearestEven(value) != value ||
        static_cast<double>(value) < static_cast<double>(range.minimum) ||
        static_cast<double>(value) > static_cast<double>(range.maximum)) {
        throw std::invalid_argument("FP carrier 输入必须位于合法整数网格");
    }
}

inline float quantizeFp(float value, const QuantParam& param, const QuantizationType& type) {
    param.validate(type);
    if (!std::isfinite(value)) {
        throw std::invalid_argument("量化输入必须有限");
    }
    const double translated = roundToNearestEven(static_cast<double>(value) / param.scale) +
                              static_cast<double>(param.zero_point);
    const auto range = type.range();
    return clampFloatCarrierCore(translated, range.minimum, range.maximum);
}

inline float dequantizeFp(float value, const QuantParam& param, const QuantizationType& type) {
    param.validate(type);
    validateFloatCarrierValue(value, type);
    const float result = static_cast<float>((static_cast<double>(value) - param.zero_point) *
                                             static_cast<double>(param.scale));
    if (!std::isfinite(result)) {
        throw std::overflow_error("反量化结果超出有限 float32");
    }
    return result;
}

namespace detail {

// 仅供已完成参数校验的 host/device 执行核心调用；不得传入 raw ratio。
QUANT_LSTM_FP_HOST_DEVICE inline float applyRescaleCore(float value,
                                                        const FixedPointScale& encoded) noexcept {
    return roundToNearestEven(
        ::ldexpf(value * static_cast<float>(encoded.multiplier), -static_cast<int>(encoded.shift)));
}

QUANT_LSTM_FP_HOST_DEVICE inline float applyRescaleCore(float value,
                                                        const Pot2Rescale& encoded) noexcept {
    return roundToNearestEven(::ldexpf(value, -static_cast<int>(encoded.shift)));
}

}  // namespace detail

inline float applyRescale(float value, const FixedPointScale& encoded) {
    if (!std::isfinite(value) || encoded.multiplier < 32768U) {
        throw std::invalid_argument("FP carrier M+shift 输入或编码非法");
    }
    const float scaled = detail::applyRescaleCore(value, encoded);
    if (!std::isfinite(scaled)) {
        throw std::overflow_error("FP carrier M+shift 产生 Inf/NaN");
    }
    return scaled;
}

inline float applyRescale(float value, const Pot2Rescale& encoded) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument("FP carrier POT2 输入必须有限");
    }
    const float scaled = detail::applyRescaleCore(value, encoded);
    if (!std::isfinite(scaled)) {
        throw std::overflow_error("FP carrier POT2 产生 Inf/NaN");
    }
    return scaled;
}

}  // namespace quant_lstm::quantization

#undef QUANT_LSTM_FP_HOST_DEVICE
