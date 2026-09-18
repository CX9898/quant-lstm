#include "lstm/lstm_execution_params.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "lstm/gate_layout.h"
#include "quantization/fixed_point_ops.h"
#include "quantization/float_carrier_ops.h"
#include "quantization/rounding.h"
#include "quantization/scale_encoding.h"

namespace quant_lstm {
namespace {

constexpr std::array<QuantOperator, kGateCount> kGateInputs{
    QuantOperator::InputGateInput, QuantOperator::ForgetGateInput, QuantOperator::CellGateInput,
    QuantOperator::OutputGateInput};
constexpr std::array<QuantOperator, kGateCount> kGateOutputs{
    QuantOperator::InputGateOutput, QuantOperator::ForgetGateOutput, QuantOperator::CellGateOutput,
    QuantOperator::OutputGateOutput};

std::uint64_t magnitude(std::int64_t value) {
    const auto raw = static_cast<std::uint64_t>(value);
    return value < 0 ? std::uint64_t{0} - raw : raw;
}

std::uint64_t centeredMaximum(const QuantizedPoint& point) {
    point.param.validate(point.type);
    const auto range = point.type.range();
    return std::max(magnitude(static_cast<std::int64_t>(range.minimum) - point.param.zero_point),
                    magnitude(static_cast<std::int64_t>(range.maximum) - point.param.zero_point));
}

QuantizedPoint point(const LstmOperatorQuantConfig& config, const LstmQuantParams& params,
                     QuantOperator id, std::size_t channel = 0) {
    const auto& values = params.at(id).values;
    if (values.empty()) {
        throw std::invalid_argument("执行参数引用了缺失的量化点");
    }
    return {values.at(isParameterOperator(id) ? channel : 0), config.at(id).type};
}

float productScale(float lhs, float rhs) {
    const float result = static_cast<float>(static_cast<double>(lhs) * rhs);
    if (!std::isfinite(result) || result <= 0.0F) {
        throw std::overflow_error("执行比例的 scale 乘积非法");
    }
    return result;
}

Q31Scale encodeQ31(double ratio) {
    if (!std::isfinite(ratio) || ratio <= 0.0) {
        throw std::invalid_argument("Cell Q31 ratio 必须是有限正数");
    }
    return {
        quantization::roundToInteger<std::int64_t>(std::ldexp(ratio, Q31Scale::kFractionalBits))};
}

ExecutionRescale encode(quantization::ScaleMode mode, float source, float destination) {
    ExecutionRescale result;
    if (mode == quantization::ScaleMode::Affine) {
        result.kind = ExecutionRescaleKind::MShift;
        result.m_shift = quantization::encodeMShift(static_cast<double>(source) / destination);
    } else if (mode == quantization::ScaleMode::Pot2) {
        result.kind = ExecutionRescaleKind::Pot2;
        result.pot2 = quantization::encodePot2Rescale(source, destination);
    } else {
        throw std::invalid_argument("ScaleMode 枚举值非法");
    }
    return result;
}

quantization::EncodedRescaleSafetyInput safetyEncoding(const ExecutionRescale& value,
                                                       std::uint64_t source) {
    if (value.kind == ExecutionRescaleKind::MShift) {
        return {quantization::SafetyRescaleEncoding::MShift, source, value.m_shift.multiplier,
                value.m_shift.shift};
    }
    if (value.kind == ExecutionRescaleKind::Pot2) {
        return {quantization::SafetyRescaleEncoding::Pot2, source, 0, value.pot2.shift};
    }
    throw std::invalid_argument("ExecutionRescaleKind 枚举值非法");
}

std::uint64_t checkedMul(std::uint64_t lhs, std::uint64_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        throw std::overflow_error("执行上界乘法超出 uint64");
    }
    return lhs * rhs;
}

std::uint64_t checkedAdd(std::uint64_t lhs, std::uint64_t rhs) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw std::overflow_error("执行上界加法超出 uint64");
    }
    return lhs + rhs;
}

std::uint64_t cellFpBound(std::uint64_t product, std::int64_t forget_multiplier,
                          std::int64_t input_multiplier) {
    if (forget_multiplier < 0 || input_multiplier < 0) {
        throw std::invalid_argument("Cell Q31 multiplier 不能为负");
    }
    const long double multiplier_sum =
        static_cast<long double>(forget_multiplier) + static_cast<long double>(input_multiplier);
    const long double bound =
        std::ldexp(static_cast<long double>(product) * multiplier_sum, -Q31Scale::kFractionalBits);
    if (!std::isfinite(bound) ||
        bound > static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return static_cast<std::uint64_t>(std::ceil(bound));
}

std::uint64_t rescaleBound(std::uint64_t source, const ExecutionRescale& value) {
    if (source > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error("执行 rescale 上界超出 int64");
    }
    const auto result = applyExecutionRescale(static_cast<std::int64_t>(source), value);
    if (result < 0) {
        throw std::runtime_error("正数上界 rescale 不应产生负数");
    }
    return static_cast<std::uint64_t>(result);
}

quantization::NumericSafetyReport enforce(const quantization::NumericSafetyInput& input,
                                          bool exact) {
    auto report = quantization::buildNumericSafetyReport(input);
    quantization::enforceNumericSafety(report, exact);
    return report;
}

void buildLinear(const LstmOperatorQuantConfig& config, const LstmQuantParams& params,
                 QuantOperator input_id, QuantOperator weight_id, QuantOperator bias_id,
                 QuantOperator output_id, std::uint64_t reduction, bool exact,
                 LinearExecutionParams* result,
                 std::vector<quantization::NumericSafetyReport>* reports) {
    result->input = point(config, params, input_id);
    result->output = point(config, params, output_id);
    const std::size_t channels = params.at(weight_id).values.size();
    if (channels == 0 || (params.bias_enabled && params.at(bias_id).values.size() != channels)) {
        throw std::invalid_argument("Linear weight/bias 参数长度不一致");
    }
    result->weights.reserve(channels);
    result->biases.reserve(params.bias_enabled ? channels : 0);
    result->bias_to_accumulator.reserve(params.bias_enabled ? channels : 0);
    result->accumulator_to_output.reserve(channels);
    reports->reserve(channels);
    for (std::size_t channel = 0; channel < channels; ++channel) {
        result->weights.push_back(point(config, params, weight_id, channel));
        const float accumulator_scale =
            productScale(result->input.param.scale, result->weights.back().param.scale);
        result->accumulator_to_output.push_back(
            encode(config.scale_mode, accumulator_scale, result->output.param.scale));

        quantization::NumericSafetyInput safety;
        safety.gemm_reduction = reduction;
        safety.gemm_lhs_maximum = centeredMaximum(result->input);
        safety.gemm_rhs_maximum = centeredMaximum(result->weights.back());
        std::uint64_t bound =
            checkedMul(checkedMul(reduction, safety.gemm_lhs_maximum), safety.gemm_rhs_maximum);
        if (params.bias_enabled) {
            result->biases.push_back(point(config, params, bias_id, channel));
            result->bias_to_accumulator.push_back(
                encode(config.scale_mode, result->biases.back().param.scale, accumulator_scale));
            const auto bias_bound = centeredMaximum(result->biases.back());
            safety.bias_rescale = safetyEncoding(result->bias_to_accumulator.back(), bias_bound);
            bound = checkedAdd(bound, rescaleBound(bias_bound, result->bias_to_accumulator.back()));
        }
        safety.linear_output_rescale = safetyEncoding(result->accumulator_to_output.back(), 0);
        safety.linear_target_zero_point_maximum = magnitude(result->output.param.zero_point);
        safety.fp32_accumulator_maximum =
            std::max(bound, rescaleBound(bound, result->accumulator_to_output.back()));
        reports->push_back(enforce(safety, exact));
    }
}

}  // namespace

std::int64_t applyExecutionRescale(std::int64_t value, const ExecutionRescale& encoded) {
    if (encoded.kind == ExecutionRescaleKind::MShift) {
        return quantization::applyRescale(value, encoded.m_shift);
    }
    if (encoded.kind == ExecutionRescaleKind::Pot2) {
        return quantization::applyRescale(value, encoded.pot2);
    }
    throw std::invalid_argument("ExecutionRescaleKind 枚举值非法");
}

float applyExecutionRescale(float value, const ExecutionRescale& encoded) {
    if (encoded.kind == ExecutionRescaleKind::MShift) {
        return quantization::applyRescale(value, encoded.m_shift);
    }
    if (encoded.kind == ExecutionRescaleKind::Pot2) {
        return quantization::applyRescale(value, encoded.pot2);
    }
    throw std::invalid_argument("ExecutionRescaleKind 枚举值非法");
}

LstmExecutionParams deriveLstmExecutionParams(const LstmOperatorQuantConfig& config,
                                              const LstmQuantParams& params,
                                              std::int64_t input_size,
                                              bool require_exact_accumulation) {
#if !defined(__SIZEOF_INT128__)
    (void)config;
    (void)params;
    (void)input_size;
    (void)require_exact_accumulation;
    throw std::runtime_error("阶段 3 执行参数要求编译器支持 int128");
#else
    if (input_size <= 0) {
        throw std::invalid_argument("input_size 必须为正数");
    }
    config.validate();
    params.validate(config);
    LstmExecutionParams result;
    result.input_size = input_size;
    result.hidden_size = params.hidden_size;
    buildLinear(config, params, QuantOperator::Input, QuantOperator::WeightInputHidden,
                QuantOperator::BiasInputHidden, QuantOperator::WeightInputHiddenLinear,
                static_cast<std::uint64_t>(input_size), require_exact_accumulation,
                &result.input_hidden_linear, &result.diagnostics.input_hidden_linear);
    buildLinear(config, params, QuantOperator::Output, QuantOperator::WeightHiddenHidden,
                QuantOperator::BiasHiddenHidden, QuantOperator::WeightHiddenHiddenLinear,
                static_cast<std::uint64_t>(params.hidden_size), require_exact_accumulation,
                &result.hidden_hidden_linear, &result.diagnostics.hidden_hidden_linear);

    for (std::size_t gate = 0; gate < kGateCount; ++gate) {
        auto& output = result.gates[gate];
        output.input_hidden_linear = result.input_hidden_linear.output;
        output.hidden_hidden_linear = result.hidden_hidden_linear.output;
        output.gate_input = point(config, params, kGateInputs[gate]);
        output.gate_output = point(config, params, kGateOutputs[gate]);
        output.input_hidden_to_gate =
            encode(config.scale_mode, output.input_hidden_linear.param.scale,
                   output.gate_input.param.scale);
        output.hidden_hidden_to_gate =
            encode(config.scale_mode, output.hidden_hidden_linear.param.scale,
                   output.gate_input.param.scale);
        const auto lhs = centeredMaximum(output.input_hidden_linear);
        const auto rhs = centeredMaximum(output.hidden_hidden_linear);
        quantization::NumericSafetyInput safety;
        safety.gate_lhs_rescale = safetyEncoding(output.input_hidden_to_gate, lhs);
        safety.gate_rhs_rescale = safetyEncoding(output.hidden_hidden_to_gate, rhs);
        safety.gate_target_zero_point_maximum = magnitude(output.gate_input.param.zero_point);
        safety.fp32_accumulator_maximum =
            checkedAdd(rescaleBound(lhs, output.input_hidden_to_gate),
                       rescaleBound(rhs, output.hidden_hidden_to_gate));
        result.diagnostics.gates[gate] = enforce(safety, require_exact_accumulation);
    }

    result.cell.forget_gate = result.gates[1].gate_output;
    result.cell.old_cell = point(config, params, QuantOperator::CellState);
    result.cell.input_gate = result.gates[0].gate_output;
    result.cell.cell_gate = result.gates[2].gate_output;
    result.cell.new_cell = result.cell.old_cell;
    result.cell.forget_scale =
        encodeQ31(static_cast<double>(result.cell.forget_gate.param.scale) *
                  result.cell.old_cell.param.scale / result.cell.new_cell.param.scale);
    result.cell.input_scale =
        encodeQ31(static_cast<double>(result.cell.input_gate.param.scale) *
                  result.cell.cell_gate.param.scale / result.cell.new_cell.param.scale);
    quantization::NumericSafetyInput cell_safety;
    cell_safety.cell_lhs_maximum =
        std::max(centeredMaximum(result.cell.forget_gate), centeredMaximum(result.cell.input_gate));
    cell_safety.cell_rhs_maximum =
        std::max(centeredMaximum(result.cell.old_cell), centeredMaximum(result.cell.cell_gate));
    cell_safety.q31_multiplier_maximum = static_cast<std::uint64_t>(
        std::max(result.cell.forget_scale.multiplier, result.cell.input_scale.multiplier));
    cell_safety.cell_fractional_bits = Q31Scale::kFractionalBits;
    cell_safety.cell_target_zero_point_maximum = magnitude(result.cell.new_cell.param.zero_point);
    cell_safety.fp32_accumulator_maximum =
        cellFpBound(checkedMul(cell_safety.cell_lhs_maximum, cell_safety.cell_rhs_maximum),
                    result.cell.forget_scale.multiplier, result.cell.input_scale.multiplier);
    result.diagnostics.cell = enforce(cell_safety, require_exact_accumulation);

    result.hidden.output_gate = result.gates[3].gate_output;
    result.hidden.cell_tanh = point(config, params, QuantOperator::CellTanhOutput);
    result.hidden.output = point(config, params, QuantOperator::Output);
    result.hidden.product_to_output = encode(
        config.scale_mode,
        productScale(result.hidden.output_gate.param.scale, result.hidden.cell_tanh.param.scale),
        result.hidden.output.param.scale);
    quantization::NumericSafetyInput hidden_safety;
    hidden_safety.hidden_lhs_maximum = centeredMaximum(result.hidden.output_gate);
    hidden_safety.hidden_rhs_maximum = centeredMaximum(result.hidden.cell_tanh);
    const auto hidden_product =
        checkedMul(hidden_safety.hidden_lhs_maximum, hidden_safety.hidden_rhs_maximum);
    hidden_safety.hidden_rescale = safetyEncoding(result.hidden.product_to_output, 0);
    hidden_safety.hidden_target_zero_point_maximum =
        magnitude(result.hidden.output.param.zero_point);
    hidden_safety.fp32_accumulator_maximum =
        std::max(hidden_product, rescaleBound(hidden_product, result.hidden.product_to_output));
    result.diagnostics.hidden = enforce(hidden_safety, require_exact_accumulation);
    return result;
#endif
}

}  // namespace quant_lstm
