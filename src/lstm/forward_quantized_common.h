#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "lstm/forward_cpu.h"
#include "lstm/gate_layout.h"
#include "lstm/lstm_execution_params.h"
#include "quantization/rounding.h"

namespace quant_lstm::reference_detail {

inline std::size_t checkedSize(std::int64_t value, const char* name) {
    if (value <= 0 || static_cast<std::uint64_t>(value) >
                          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument(std::string(name) + " 必须是可表示的正数");
    }
    return static_cast<std::size_t>(value);
}

inline std::size_t operatorIndex(QuantOperator id) { return static_cast<std::size_t>(id); }

inline const QuantizedPoint& pointAt(const LinearExecutionParams& linear, bool weight,
                                     std::size_t channel) {
    return weight ? linear.weights.at(channel) : linear.biases.at(channel);
}

inline std::int32_t clampInt(std::int64_t value, const QuantizedPoint& point) {
    const auto range = point.type.range();
    return static_cast<std::int32_t>(std::clamp<std::int64_t>(value, range.minimum, range.maximum));
}

inline float clampFp(float value, const QuantizedPoint& point) {
    if (!std::isfinite(value)) {
        throw std::overflow_error("FP carrier 边界值产生 Inf/NaN");
    }
    const auto range = point.type.range();
    return std::clamp(value, static_cast<float>(range.minimum), static_cast<float>(range.maximum));
}

inline void requireGrid(float value, const QuantizedPoint& point) {
    const auto range = point.type.range();
    if (!std::isfinite(value) || quantization::roundToNearestEven(value) != value ||
        value < static_cast<float>(range.minimum) || value > static_cast<float>(range.maximum)) {
        throw std::invalid_argument("FP carrier 输入必须位于合法整数网格");
    }
}

inline std::int64_t checkedAdd(std::int64_t lhs, std::int64_t rhs) {
    if ((rhs > 0 && lhs > std::numeric_limits<std::int64_t>::max() - rhs) ||
        (rhs < 0 && lhs < std::numeric_limits<std::int64_t>::min() - rhs)) {
        throw std::overflow_error("reference int64 累加溢出");
    }
    return lhs + rhs;
}

#if defined(__SIZEOF_INT128__)
inline std::string int128ToString(__int128 value) {
    if (value == 0) {
        return "0";
    }
    using Unsigned128 = unsigned __int128;
    const bool negative = value < 0;
    const Unsigned128 raw = static_cast<Unsigned128>(value);
    Unsigned128 magnitude = negative ? Unsigned128{0} - raw : raw;
    std::string result;
    while (magnitude != 0) {
        result.push_back(static_cast<char>('0' + static_cast<int>(magnitude % 10U)));
        magnitude /= 10U;
    }
    if (negative) {
        result.push_back('-');
    }
    std::reverse(result.begin(), result.end());
    return result;
}
#endif

inline void validateCommon(const LstmShape& shape, bool bias_enabled, const void* weight_ih,
                           const void* weight_hh, const void* bias_ih, const void* bias_hh,
                           const void* input, const void* initial_hidden, const void* initial_cell,
                           const void* output, const void* final_hidden, const void* final_cell,
                           const LstmOperatorQuantConfig& config,
                           const LstmQuantParams& quant_params,
                           const LstmExecutionParams& execution_params) {
    checkedSize(shape.sequence_length, "sequence_length");
    checkedSize(shape.batch_size, "batch_size");
    checkedSize(shape.input_size, "input_size");
    checkedSize(shape.hidden_size, "hidden_size");
    if (weight_ih == nullptr || weight_hh == nullptr || input == nullptr || output == nullptr ||
        final_hidden == nullptr || final_cell == nullptr) {
        throw std::invalid_argument("量化 reference 必需指针不能为空");
    }
    if ((initial_hidden == nullptr) != (initial_cell == nullptr)) {
        throw std::invalid_argument("initial_hidden/initial_cell 必须同时提供或省略");
    }
    if (bias_enabled && (bias_ih == nullptr || bias_hh == nullptr)) {
        throw std::invalid_argument("启用 bias 时 bias 指针不能为空");
    }
    if (!bias_enabled && (bias_ih != nullptr || bias_hh != nullptr)) {
        throw std::invalid_argument("禁用 bias 时 bias 指针必须为空");
    }
    quant_params.validate(config);
    if (shape.hidden_size != quant_params.hidden_size ||
        shape.hidden_size != execution_params.hidden_size ||
        shape.input_size != execution_params.input_size ||
        bias_enabled != quant_params.bias_enabled) {
        throw std::invalid_argument("shape、量化参数与执行参数不一致");
    }
}

}  // namespace quant_lstm::reference_detail
