#pragma once

#include "lstm/lstm_execution_params.h"
#include "quantization/rounding.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

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

inline float centeredValue(float value, const QuantizedPoint& point) {
    point.param.validate(point.type);
    if (!std::isfinite(value) ||
        quantization::roundToNearestEven(value) != value) {
        throw std::invalid_argument("FP carrier q 值必须是有限整数");
    }
    point.type.validateValue(static_cast<std::int64_t>(value));
    return value - static_cast<float>(point.param.zero_point);
}

inline float checkedFinite(float value, const char* message) {
    if (!std::isfinite(value)) {
        throw std::overflow_error(message);
    }
    return value;
}

inline float addZeroPointAndClamp(float centered,
                                  const QuantizedPoint& target) {
    target.param.validate(target.type);
    const float translated = checkedFinite(
        centered + static_cast<float>(target.param.zero_point),
        "FP carrier 添加 zero point 后产生 Inf/NaN");
    const auto range = target.type.range();
    return std::clamp(translated, static_cast<float>(range.minimum),
                      static_cast<float>(range.maximum));
}

inline float decodeQ31ForFp(const Q31Scale& encoded) {
    if (encoded.multiplier < 0) {
        throw std::invalid_argument("Cell Q31 multiplier 不能为负");
    }
    return checkedFinite(
        std::ldexp(static_cast<float>(encoded.multiplier),
                   -Q31Scale::kFractionalBits),
        "FP carrier Q31 解码产生 Inf/NaN");
}

}  // namespace detail

// 两路 contribution 先以 Q31 已编码比例融合，最外层只舍入一次。
inline QuantizedCellFpResult computeQuantizedCellFp(
    float forget_gate, float old_cell, float input_gate, float cell_gate,
    const CellExecutionParams& params) {
    QuantizedCellFpResult result;
    result.diagnostics.p_forget = detail::checkedFinite(
        detail::centeredValue(forget_gate, params.forget_gate) *
            detail::centeredValue(old_cell, params.old_cell),
        "FP carrier Cell forget 乘积产生 Inf/NaN");
    result.diagnostics.p_input = detail::checkedFinite(
        detail::centeredValue(input_gate, params.input_gate) *
            detail::centeredValue(cell_gate, params.cell_gate),
        "FP carrier Cell input 乘积产生 Inf/NaN");
    const float forget_ratio = detail::decodeQ31ForFp(params.forget_scale);
    const float input_ratio = detail::decodeQ31ForFp(params.input_scale);
    result.diagnostics.scaled_forget_contribution = detail::checkedFinite(
        result.diagnostics.p_forget * forget_ratio,
        "FP carrier Cell forget contribution 产生 Inf/NaN");
    result.diagnostics.scaled_input_contribution = detail::checkedFinite(
        result.diagnostics.p_input * input_ratio,
        "FP carrier Cell input contribution 产生 Inf/NaN");
    result.diagnostics.pre_round_sum = detail::checkedFinite(
        result.diagnostics.scaled_forget_contribution +
            result.diagnostics.scaled_input_contribution,
        "FP carrier Cell 融合和产生 Inf/NaN");
    result.diagnostics.forget_contribution_vanished =
        result.diagnostics.p_forget != 0.0F &&
        params.forget_scale.multiplier == 0;
    result.diagnostics.input_contribution_vanished =
        result.diagnostics.p_input != 0.0F &&
        params.input_scale.multiplier == 0;

    const float centered =
        quantization::roundToNearestEven(result.diagnostics.pre_round_sum);
    result.value =
        detail::addZeroPointAndClamp(centered, params.new_cell);
    return result;
}

// Hidden 乘积复用 M+shift/POT2 编码，并在 output 边界完成唯一舍入。
inline QuantizedHiddenFpResult computeQuantizedHiddenFp(
    float output_gate, float cell_tanh,
    const HiddenExecutionParams& params) {
    QuantizedHiddenFpResult result;
    result.diagnostics.raw_product = detail::checkedFinite(
        detail::centeredValue(output_gate, params.output_gate) *
            detail::centeredValue(cell_tanh, params.cell_tanh),
        "FP carrier Hidden 原始乘积产生 Inf/NaN");
    const float centered =
        applyExecutionRescale(result.diagnostics.raw_product,
                              params.product_to_output);
    result.value =
        detail::addZeroPointAndClamp(centered, params.output);
    return result;
}

}  // namespace quant_lstm
