#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "lstm/lstm_execution_params.h"
#include "quantization/float_carrier_ops.h"
#include "quantization/rounding.h"

#if defined(__CUDACC__)
#define QUANT_LSTM_FP_CELL_HOST_DEVICE __host__ __device__
#else
#define QUANT_LSTM_FP_CELL_HOST_DEVICE
#endif

// FP32 carrier 保存整数网格值，并模拟与整数 reference 相同的执行编码。
namespace quant_lstm {

struct QuantizedCellFpDiagnostics {
    float p_forget = 0.0F;
    float p_input = 0.0F;
    float scaled_forget_contribution = 0.0F;
    float scaled_input_contribution = 0.0F;
    float pre_round_sum = 0.0F;
    bool forget_contribution_vanished = false;
    bool input_contribution_vanished = false;
};

struct QuantizedCellFpResult {
    float value = 0.0F;
    QuantizedCellFpDiagnostics diagnostics;
};

struct QuantizedHiddenFpDiagnostics {
    float raw_product = 0.0F;
};

struct QuantizedHiddenFpResult {
    float value = 0.0F;
    QuantizedHiddenFpDiagnostics diagnostics;
};

namespace detail {

struct QuantizedCellFpCoreParams {
    std::int32_t forget_zero_point;
    std::int32_t old_cell_zero_point;
    std::int32_t input_zero_point;
    std::int32_t cell_gate_zero_point;
    std::int64_t forget_multiplier;
    std::int64_t input_multiplier;
    std::int32_t new_cell_zero_point;
    float new_cell_minimum;
    float new_cell_maximum;
    double forget_mantissa = 0.0;
    double input_mantissa = 0.0;
    std::int16_t forget_exponent = 0;
    std::int16_t input_exponent = 0;
    bool forget_uses_fp_scale = false;
    bool input_uses_fp_scale = false;
};

struct QuantizedHiddenFpCoreParams {
    std::int32_t output_gate_zero_point;
    std::int32_t cell_tanh_zero_point;
    ExecutionRescale product_to_output;
    std::int32_t output_zero_point;
    float output_minimum;
    float output_maximum;
};

QUANT_LSTM_FP_CELL_HOST_DEVICE inline float clampFpCore(double value, float minimum,
                                                        float maximum) noexcept {
    return value < minimum ? minimum : (value > maximum ? maximum : static_cast<float>(value));
}

QUANT_LSTM_FP_CELL_HOST_DEVICE inline float applyExecutionRescaleFpCore(
    float value, const ExecutionRescale& encoded) noexcept {
    return encoded.kind == ExecutionRescaleKind::MShift
               ? quantization::detail::applyRescaleCore(value, encoded.m_shift)
               : quantization::detail::applyRescaleCore(value, encoded.pot2);
}

QUANT_LSTM_FP_CELL_HOST_DEVICE inline float scaleCellProductFpCore(
    float product, std::int64_t multiplier, double mantissa, std::int16_t exponent,
    bool uses_fp_scale) noexcept {
    if (uses_fp_scale) {
        return static_cast<float>(::ldexp(static_cast<double>(product) * mantissa, exponent));
    }
    const float ratio = ::ldexpf(static_cast<float>(multiplier),
                                -static_cast<int>(Q31Scale::kFractionalBits));
    return product * ratio;
}

// Q31 可表示的比例保持旧路径；扩展比例用尾数和指数避免先转 FP32 时下溢。
QUANT_LSTM_FP_CELL_HOST_DEVICE inline QuantizedCellFpResult computeQuantizedCellFpCore(
    float forget_gate, float old_cell, float input_gate, float cell_gate,
    const QuantizedCellFpCoreParams& params) noexcept {
    QuantizedCellFpResult result;
    result.diagnostics.p_forget =
        static_cast<float>(static_cast<double>(forget_gate) - params.forget_zero_point) *
        static_cast<float>(static_cast<double>(old_cell) - params.old_cell_zero_point);
    result.diagnostics.p_input =
        static_cast<float>(static_cast<double>(input_gate) - params.input_zero_point) *
        static_cast<float>(static_cast<double>(cell_gate) - params.cell_gate_zero_point);
    result.diagnostics.scaled_forget_contribution = scaleCellProductFpCore(
        result.diagnostics.p_forget, params.forget_multiplier, params.forget_mantissa,
        params.forget_exponent, params.forget_uses_fp_scale);
    result.diagnostics.scaled_input_contribution = scaleCellProductFpCore(
        result.diagnostics.p_input, params.input_multiplier, params.input_mantissa,
        params.input_exponent, params.input_uses_fp_scale);
    result.diagnostics.pre_round_sum = result.diagnostics.scaled_forget_contribution +
                                       result.diagnostics.scaled_input_contribution;
    result.diagnostics.forget_contribution_vanished =
        result.diagnostics.p_forget != 0.0F &&
        result.diagnostics.scaled_forget_contribution == 0.0F;
    result.diagnostics.input_contribution_vanished =
        result.diagnostics.p_input != 0.0F && result.diagnostics.scaled_input_contribution == 0.0F;
    const double translated =
        static_cast<double>(quantization::roundToNearestEven(result.diagnostics.pre_round_sum)) +
        params.new_cell_zero_point;
    result.value = clampFpCore(translated, params.new_cell_minimum, params.new_cell_maximum);
    return result;
}

// 已验证参数的 Hidden 核心；只消费 M+shift/POT2 编码，不接受 raw ratio。
QUANT_LSTM_FP_CELL_HOST_DEVICE inline QuantizedHiddenFpResult computeQuantizedHiddenFpCore(
    float output_gate, float cell_tanh, const QuantizedHiddenFpCoreParams& params) noexcept {
    QuantizedHiddenFpResult result;
    result.diagnostics.raw_product =
        static_cast<float>(static_cast<double>(output_gate) - params.output_gate_zero_point) *
        static_cast<float>(static_cast<double>(cell_tanh) - params.cell_tanh_zero_point);
    const float centered =
        applyExecutionRescaleFpCore(result.diagnostics.raw_product, params.product_to_output);
    result.value = clampFpCore(static_cast<double>(centered) + params.output_zero_point,
                               params.output_minimum,
                               params.output_maximum);
    return result;
}

inline float centeredValue(float value, const QuantizedPoint& point) {
    point.param.validate(point.type);
    quantization::validateFloatCarrierValue(value, point.type);
    return quantization::centerFloatCarrierCore(value, point.param.zero_point);
}

inline float checkedFinite(float value, const char* message) {
    if (!std::isfinite(value)) {
        throw std::overflow_error(message);
    }
    return value;
}

inline float addZeroPointAndClamp(float centered, const QuantizedPoint& target) {
    target.param.validate(target.type);
    const double translated = static_cast<double>(centered) + target.param.zero_point;
    if (!std::isfinite(translated)) {
        throw std::overflow_error("FP carrier 添加 zero point 后产生 Inf/NaN");
    }
    const auto range = target.type.range();
    return quantization::clampFloatCarrierCore(translated, range.minimum, range.maximum);
}

inline double decodeQ31ForFp(const Q31Scale& encoded) {
    if (encoded.uses_fp_scale) {
        const double ratio = std::ldexp(encoded.fp_mantissa, encoded.fp_exponent);
        if (!std::isfinite(ratio) || encoded.fp_mantissa < 0.5 || encoded.fp_mantissa >= 1.0) {
            throw std::invalid_argument("Cell FP scale 编码非法");
        }
        return ratio;
    }
    if (encoded.multiplier < 0) {
        throw std::invalid_argument("Cell Q31 multiplier 不能为负");
    }
    return checkedFinite(
        std::ldexp(static_cast<float>(encoded.multiplier), -Q31Scale::kFractionalBits),
        "FP carrier Q31 解码产生 Inf/NaN");
}

}  // namespace detail

// 两路 contribution 先以已编码比例融合，最外层只舍入一次。
inline QuantizedCellFpResult computeQuantizedCellFp(float forget_gate, float old_cell,
                                                    float input_gate, float cell_gate,
                                                    const CellExecutionParams& params) {
    detail::centeredValue(forget_gate, params.forget_gate);
    detail::centeredValue(old_cell, params.old_cell);
    detail::centeredValue(input_gate, params.input_gate);
    detail::centeredValue(cell_gate, params.cell_gate);
    detail::decodeQ31ForFp(params.forget_scale);
    detail::decodeQ31ForFp(params.input_scale);
    params.new_cell.param.validate(params.new_cell.type);
    const auto range = params.new_cell.type.range();
    const detail::QuantizedCellFpCoreParams core_params{
        params.forget_gate.param.zero_point,
        params.old_cell.param.zero_point,
        params.input_gate.param.zero_point,
        params.cell_gate.param.zero_point,
        params.forget_scale.multiplier,
        params.input_scale.multiplier,
        params.new_cell.param.zero_point,
        quantization::floatCarrierMinimum(range.minimum),
        quantization::floatCarrierMaximum(range.maximum),
        params.forget_scale.fp_mantissa,
        params.input_scale.fp_mantissa,
        params.forget_scale.fp_exponent,
        params.input_scale.fp_exponent,
        params.forget_scale.uses_fp_scale,
        params.input_scale.uses_fp_scale};
    auto result = detail::computeQuantizedCellFpCore(forget_gate, old_cell, input_gate, cell_gate,
                                                     core_params);
    detail::checkedFinite(result.diagnostics.p_forget, "FP carrier Cell forget 乘积产生 Inf/NaN");
    detail::checkedFinite(result.diagnostics.p_input, "FP carrier Cell input 乘积产生 Inf/NaN");
    detail::checkedFinite(result.diagnostics.scaled_forget_contribution,
                          "FP carrier Cell forget contribution 产生 Inf/NaN");
    detail::checkedFinite(result.diagnostics.scaled_input_contribution,
                          "FP carrier Cell input contribution 产生 Inf/NaN");
    detail::checkedFinite(result.diagnostics.pre_round_sum, "FP carrier Cell 融合和产生 Inf/NaN");
    if (!std::isfinite(
            static_cast<double>(quantization::roundToNearestEven(result.diagnostics.pre_round_sum)) +
            params.new_cell.param.zero_point)) {
        throw std::overflow_error("FP carrier 添加 zero point 后产生 Inf/NaN");
    }
    return result;
}

// Hidden 乘积复用 M+shift/POT2 编码，并在 output 边界完成唯一舍入。
inline QuantizedHiddenFpResult computeQuantizedHiddenFp(float output_gate, float cell_tanh,
                                                        const HiddenExecutionParams& params) {
    detail::centeredValue(output_gate, params.output_gate);
    detail::centeredValue(cell_tanh, params.cell_tanh);
    if (params.product_to_output.kind == ExecutionRescaleKind::MShift) {
        if (params.product_to_output.m_shift.multiplier < 32768U) {
            throw std::invalid_argument("FP carrier M+shift 输入或编码非法");
        }
    } else if (params.product_to_output.kind != ExecutionRescaleKind::Pot2) {
        throw std::invalid_argument("ExecutionRescaleKind 枚举值非法");
    }
    params.output.param.validate(params.output.type);
    const auto range = params.output.type.range();
    const detail::QuantizedHiddenFpCoreParams core_params{
        params.output_gate.param.zero_point,
        params.cell_tanh.param.zero_point,
        params.product_to_output,
        params.output.param.zero_point,
        quantization::floatCarrierMinimum(range.minimum),
        quantization::floatCarrierMaximum(range.maximum)};
    auto result = detail::computeQuantizedHiddenFpCore(output_gate, cell_tanh, core_params);
    detail::checkedFinite(result.diagnostics.raw_product, "FP carrier Hidden 原始乘积产生 Inf/NaN");
    const float centered = detail::applyExecutionRescaleFpCore(result.diagnostics.raw_product,
                                                               params.product_to_output);
    detail::checkedFinite(centered, params.product_to_output.kind == ExecutionRescaleKind::MShift
                                        ? "FP carrier M+shift 产生 Inf/NaN"
                                        : "FP carrier POT2 产生 Inf/NaN");
    if (!std::isfinite(static_cast<double>(centered) + params.output.param.zero_point)) {
        throw std::overflow_error("FP carrier 添加 zero point 后产生 Inf/NaN");
    }
    return result;
}

}  // namespace quant_lstm

#undef QUANT_LSTM_FP_CELL_HOST_DEVICE
