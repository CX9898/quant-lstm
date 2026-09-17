#pragma once

#include "lstm/quantized_cell_fp.h"
#include "quantization/real_activation.h"
#include "quantization/rounding.h"

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define QUANT_LSTM_CUDA_HOST_DEVICE __host__ __device__
#else
#define QUANT_LSTM_CUDA_HOST_DEVICE
#endif

// CPU 包装与 CUDA pointwise kernel 共用的无异常纯计算入口。
// 调用方必须先验证量化点、范围、执行编码和输入整数网格。
namespace quant_lstm::cuda_detail {

QUANT_LSTM_CUDA_HOST_DEVICE inline float realSigmoid(float value) noexcept {
    if (value >= 0.0F) {
        const float exponential = ::expf(-value);
        return 1.0F / (1.0F + exponential);
    }
    const float exponential = ::expf(value);
    return exponential / (1.0F + exponential);
}

QUANT_LSTM_CUDA_HOST_DEVICE inline float realTanh(float value) noexcept {
    return ::tanhf(value);
}

QUANT_LSTM_CUDA_HOST_DEVICE inline float realActivationCore(
    float quantized_input, float input_scale, std::int32_t input_zero_point,
    float output_scale, std::int32_t output_zero_point, std::int32_t output_minimum,
    std::int32_t output_maximum, quantization::RealActivationKind kind) noexcept {
    // double 中间顺序与既有 dequantize/quantize host 边界一致；激活本身为 FP32。
    const float real_input = static_cast<float>(
        (static_cast<double>(quantized_input) -
         static_cast<double>(input_zero_point)) *
        static_cast<double>(input_scale));
    const float activated =
        kind == quantization::RealActivationKind::Tanh
            ? realTanh(real_input)
            : realSigmoid(real_input);
    // 这是激活输出的 standard-scale 量化边界，不是算术 rescale raw ratio。
    const double translated =
        static_cast<double>(activated) / static_cast<double>(output_scale) +
        static_cast<double>(output_zero_point);
    if (translated <= static_cast<double>(output_minimum)) {
        return static_cast<float>(output_minimum);
    }
    if (translated >= static_cast<double>(output_maximum)) {
        return static_cast<float>(output_maximum);
    }
    return static_cast<float>(
        quantization::roundToNearestEven(translated));
}

QUANT_LSTM_CUDA_HOST_DEVICE inline float applyExecutionRescaleCore(
    float value, const ExecutionRescale& encoded) noexcept {
    return detail::applyExecutionRescaleFpCore(value, encoded);
}

QUANT_LSTM_CUDA_HOST_DEVICE inline float applyEncodedRescaleCore(
    float value, bool is_m_shift, std::uint16_t multiplier,
    std::int8_t shift) noexcept {
    return quantization::roundToNearestEven(
        is_m_shift
            ? ::ldexpf(value * static_cast<float>(multiplier),
                       -static_cast<int>(shift))
            : ::ldexpf(value, -static_cast<int>(shift)));
}

using QuantizedCellFpCoreParams = detail::QuantizedCellFpCoreParams;
using QuantizedHiddenFpCoreParams = detail::QuantizedHiddenFpCoreParams;

QUANT_LSTM_CUDA_HOST_DEVICE inline QuantizedCellFpResult
computeQuantizedCellFpCore(float forget_gate, float old_cell, float input_gate,
                           float cell_gate,
                           const QuantizedCellFpCoreParams& params) noexcept {
    return detail::computeQuantizedCellFpCore(
        forget_gate, old_cell, input_gate, cell_gate, params);
}

QUANT_LSTM_CUDA_HOST_DEVICE inline QuantizedHiddenFpResult
computeQuantizedHiddenFpCore(
    float output_gate, float cell_tanh,
    const QuantizedHiddenFpCoreParams& params) noexcept {
    return detail::computeQuantizedHiddenFpCore(output_gate, cell_tanh, params);
}

QUANT_LSTM_CUDA_HOST_DEVICE inline QuantizedHiddenFpResult
computeQuantizedHiddenFpEncodedCore(
    float output_gate, float output_gate_zero_point, float cell_tanh,
    float cell_tanh_zero_point, bool is_m_shift, std::uint16_t multiplier,
    std::int8_t shift, float output_zero_point, float output_minimum,
    float output_maximum) noexcept {
    QuantizedHiddenFpResult result;
    result.diagnostics.raw_product =
        (output_gate - output_gate_zero_point) *
        (cell_tanh - cell_tanh_zero_point);
    const float centered = applyEncodedRescaleCore(
        result.diagnostics.raw_product, is_m_shift, multiplier, shift);
    result.value = detail::clampFpCore(
        centered + output_zero_point, output_minimum, output_maximum);
    return result;
}

}  // namespace quant_lstm::cuda_detail

#undef QUANT_LSTM_CUDA_HOST_DEVICE
