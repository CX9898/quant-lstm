#pragma once

#include "quantization/quant_param.h"
#include "quantization/rounding.h"

#include <cmath>
#include <stdexcept>

#if defined(__CUDACC__)
#define QUANT_LSTM_FP_HOST_DEVICE __host__ __device__
#else
#define QUANT_LSTM_FP_HOST_DEVICE
#endif

// FP32 carrier 保存整数网格值，并模拟与整数 carrier 相同的 M+shift/POT2 编码。
namespace quant_lstm::quantization {

namespace detail {

// 仅供已完成参数校验的 host/device 执行核心调用；不得传入 raw ratio。
QUANT_LSTM_FP_HOST_DEVICE inline float applyRescaleCore(
    float value, const FixedPointScale& encoded) noexcept {
    return roundToNearestEven(
        ::ldexpf(value * static_cast<float>(encoded.multiplier),
                 -static_cast<int>(encoded.shift)));
}

QUANT_LSTM_FP_HOST_DEVICE inline float applyRescaleCore(
    float value, const Pot2Rescale& encoded) noexcept {
    return roundToNearestEven(
        ::ldexpf(value, -static_cast<int>(encoded.shift)));
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
