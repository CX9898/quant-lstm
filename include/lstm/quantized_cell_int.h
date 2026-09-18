#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "lstm/lstm_execution_params.h"
#include "quantization/fixed_point_ops.h"
#include "quantization/rounding.h"

// CPU int32 reference 的 Cell/Hidden 融合原语；中间过程不做隐式 Clamp。
namespace quant_lstm {

#if defined(__SIZEOF_INT128__)

struct QuantizedCellIntDiagnostics {
    std::int64_t p_forget = 0;
    std::int64_t p_input = 0;
    __int128 scaled_forget_contribution = 0;
    __int128 scaled_input_contribution = 0;
    __int128 pre_round_sum = 0;
    bool forget_contribution_vanished = false;
    bool input_contribution_vanished = false;
};

struct QuantizedCellIntResult {
    std::int32_t value = 0;
    QuantizedCellIntDiagnostics diagnostics;
};

struct QuantizedHiddenIntDiagnostics {
    std::int64_t raw_product = 0;
};

struct QuantizedHiddenIntResult {
    std::int32_t value = 0;
    QuantizedHiddenIntDiagnostics diagnostics;
};

namespace detail {

inline std::int64_t centeredValue(std::int32_t value, const QuantizedPoint& point) {
    point.param.validate(point.type);
    point.type.validateValue(value);
    return static_cast<std::int64_t>(value) - point.param.zero_point;
}

inline std::int64_t checkedProduct(std::int64_t lhs, std::int64_t rhs, const char* message) {
    const __int128 product = static_cast<__int128>(lhs) * static_cast<__int128>(rhs);
    if (product < std::numeric_limits<std::int64_t>::min() ||
        product > std::numeric_limits<std::int64_t>::max()) {
        throw std::overflow_error(message);
    }
    return static_cast<std::int64_t>(product);
}

inline std::int32_t addZeroPointAndClamp(std::int64_t centered, const QuantizedPoint& target) {
    target.param.validate(target.type);
    const __int128 translated = static_cast<__int128>(centered) + target.param.zero_point;
    if (translated < std::numeric_limits<std::int64_t>::min() ||
        translated > std::numeric_limits<std::int64_t>::max()) {
        throw std::overflow_error("量化边界添加 zero point 时溢出 int64");
    }
    return quantization::clampToRange(static_cast<std::int64_t>(translated), target.type);
}

}  // namespace detail

// 两路 Q31 contribution 在 int128 中合并，并且只执行一次 RoundShift31。
inline QuantizedCellIntResult computeQuantizedCellInt(std::int32_t forget_gate,
                                                      std::int32_t old_cell,
                                                      std::int32_t input_gate,
                                                      std::int32_t cell_gate,
                                                      const CellExecutionParams& params) {
    QuantizedCellIntResult result;
    const std::int64_t centered_forget = detail::centeredValue(forget_gate, params.forget_gate);
    const std::int64_t centered_old_cell = detail::centeredValue(old_cell, params.old_cell);
    const std::int64_t centered_input = detail::centeredValue(input_gate, params.input_gate);
    const std::int64_t centered_cell_gate = detail::centeredValue(cell_gate, params.cell_gate);
    result.diagnostics.p_forget =
        detail::checkedProduct(centered_forget, centered_old_cell, "Cell forget 乘积溢出 int64");
    result.diagnostics.p_input =
        detail::checkedProduct(centered_input, centered_cell_gate, "Cell input 乘积溢出 int64");
    if (params.forget_scale.multiplier < 0 || params.input_scale.multiplier < 0) {
        throw std::invalid_argument("Cell Q31 multiplier 不能为负");
    }
    result.diagnostics.scaled_forget_contribution =
        static_cast<__int128>(result.diagnostics.p_forget) * params.forget_scale.multiplier;
    result.diagnostics.scaled_input_contribution =
        static_cast<__int128>(result.diagnostics.p_input) * params.input_scale.multiplier;
    result.diagnostics.pre_round_sum = result.diagnostics.scaled_forget_contribution +
                                       result.diagnostics.scaled_input_contribution;
    result.diagnostics.forget_contribution_vanished =
        result.diagnostics.p_forget != 0 && params.forget_scale.multiplier == 0;
    result.diagnostics.input_contribution_vanished =
        result.diagnostics.p_input != 0 && params.input_scale.multiplier == 0;

    const __int128 rounded = quantization::roundShiftRight128(result.diagnostics.pre_round_sum,
                                                              Q31Scale::kFractionalBits);
    const std::int64_t centered = quantization::checkedInt128ToInt64(rounded);
    result.value = detail::addZeroPointAndClamp(centered, params.new_cell);
    return result;
}

// Hidden 单路乘积直接使用已派生编码对齐到 output 网格。
inline QuantizedHiddenIntResult computeQuantizedHiddenInt(std::int32_t output_gate,
                                                          std::int32_t cell_tanh,
                                                          const HiddenExecutionParams& params) {
    QuantizedHiddenIntResult result;
    result.diagnostics.raw_product = detail::checkedProduct(
        detail::centeredValue(output_gate, params.output_gate),
        detail::centeredValue(cell_tanh, params.cell_tanh), "Hidden 原始乘积溢出 int64");
    const std::int64_t centered =
        applyExecutionRescale(result.diagnostics.raw_product, params.product_to_output);
    result.value = detail::addZeroPointAndClamp(centered, params.output);
    return result;
}

#else

inline void requireInt128ForQuantizedCellInt() {
    throw std::runtime_error("当前编译器不支持 Cell 融合所需的 int128");
}

#endif

}  // namespace quant_lstm
