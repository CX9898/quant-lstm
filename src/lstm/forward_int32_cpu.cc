#include "lstm/forward_cpu.h"

#include "lstm/gate_layout.h"
#include "lstm/lstm_execution_params.h"
#include "lstm/quantized_cell_int.h"
#include "quantization/real_activation.h"
#include "forward_quantized_common.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace quant_lstm {
namespace {

using reference_detail::checkedAdd;
using reference_detail::clampInt;

std::int32_t linearValue(const std::int32_t* vector,
                         const std::int32_t* weights,
                         const std::int32_t* bias, std::size_t width,
                         std::size_t channel,
                         const LinearExecutionParams& params) {
    const QuantizedPoint& weight_point = params.weights.at(channel);
    std::int64_t accumulator = 0;
    for (std::size_t column = 0; column < width; ++column) {
        weight_point.type.validateValue(
            weights[channel * width + column]);
        params.input.type.validateValue(vector[column]);
        const std::int64_t centered_weight =
            static_cast<std::int64_t>(weights[channel * width + column]) -
            weight_point.param.zero_point;
        const std::int64_t centered_input =
            static_cast<std::int64_t>(vector[column]) -
            params.input.param.zero_point;
        accumulator = checkedAdd(accumulator,
                                 centered_weight * centered_input);
    }
    if (bias != nullptr) {
        params.biases.at(channel).type.validateValue(bias[channel]);
        const std::int64_t centered_bias =
            static_cast<std::int64_t>(bias[channel]) -
            params.biases.at(channel).param.zero_point;
        accumulator = checkedAdd(
            accumulator,
            applyExecutionRescale(centered_bias,
                                  params.bias_to_accumulator.at(channel)));
    }
    const std::int64_t output =
        checkedAdd(applyExecutionRescale(
                       accumulator,
                       params.accumulator_to_output.at(channel)),
                   params.output.param.zero_point);
    return clampInt(output, params.output);
}

void prepareTrace(LstmInt32ReferenceTrace* trace, std::size_t steps,
                  std::size_t batch, std::size_t hidden) {
    if (trace == nullptr) {
        return;
    }
    *trace = {};
    const std::size_t gates = steps * batch * kGateCount * hidden;
    const std::size_t states = steps * batch * hidden;
    trace->weight_ih_linear.resize(gates);
    trace->weight_hh_linear.resize(gates);
    trace->gate_inputs.resize(gates);
    trace->gate_outputs.resize(gates);
    trace->cell_states.resize(states);
    trace->cell_tanh_outputs.resize(states);
    trace->hidden_outputs.resize(states);
    trace->p_forget.reserve(states);
    trace->p_input.reserve(states);
    trace->hidden_products.reserve(states);
    trace->scaled_forget_contributions.reserve(states);
    trace->scaled_input_contributions.reserve(states);
    trace->cell_wide_sums.reserve(states);
}

}  // namespace

void lstmForwardInt32CpuReference(
    const LstmShape& shape, const LstmInt32Weights& weights,
    const std::int32_t* input, const std::int32_t* initial_hidden,
    const std::int32_t* initial_cell, const LstmOperatorQuantConfig& config,
    const LstmQuantParams& quant_params,
    const LstmExecutionParams& execution_params, std::int32_t* output,
    std::int32_t* final_hidden, std::int32_t* final_cell,
    LstmInt32ReferenceTrace* trace) {
    reference_detail::validateCommon(
        shape, quant_params.bias_enabled, weights.weight_ih, weights.weight_hh,
        weights.bias_ih, weights.bias_hh, input, initial_hidden, initial_cell,
        output, final_hidden, final_cell, config, quant_params,
        execution_params);
#if !defined(__SIZEOF_INT128__)
    throw std::runtime_error("int32 Cell reference 要求编译器支持 __int128");
#else
    const std::size_t steps =
        reference_detail::checkedSize(shape.sequence_length, "sequence_length");
    const std::size_t batch =
        reference_detail::checkedSize(shape.batch_size, "batch_size");
    const std::size_t input_size =
        reference_detail::checkedSize(shape.input_size, "input_size");
    const std::size_t hidden =
        reference_detail::checkedSize(shape.hidden_size, "hidden_size");
    const std::size_t channels = kGateCount * hidden;
    prepareTrace(trace, steps, batch, hidden);

    std::vector<std::int32_t> hidden_state(batch * hidden);
    std::vector<std::int32_t> cell_state(batch * hidden);
    std::vector<std::int32_t> linear_ih(batch * channels);
    std::vector<std::int32_t> linear_hh(batch * channels);
    std::vector<std::int32_t> gate_inputs(batch * channels);
    std::vector<std::int32_t> gate_outputs(batch * channels);
    const std::int32_t hidden_zero =
        execution_params.hidden.output.param.zero_point;
    const std::int32_t cell_zero =
        execution_params.cell.old_cell.param.zero_point;
    for (std::size_t index = 0; index < batch * hidden; ++index) {
        hidden_state[index] =
            initial_hidden == nullptr ? hidden_zero : initial_hidden[index];
        cell_state[index] =
            initial_cell == nullptr ? cell_zero : initial_cell[index];
        execution_params.hidden.output.type.validateValue(hidden_state[index]);
        execution_params.cell.old_cell.type.validateValue(cell_state[index]);
    }

    for (std::size_t step = 0; step < steps; ++step) {
        for (std::size_t row = 0; row < batch; ++row) {
            const std::int32_t* input_row =
                input + (step * batch + row) * input_size;
            const std::int32_t* hidden_row =
                hidden_state.data() + row * hidden;
            for (std::size_t channel = 0; channel < channels; ++channel) {
                const std::size_t index = row * channels + channel;
                linear_ih[index] = linearValue(
                    input_row, weights.weight_ih, weights.bias_ih, input_size,
                    channel, execution_params.input_hidden_linear);
                linear_hh[index] = linearValue(
                    hidden_row, weights.weight_hh, weights.bias_hh, hidden,
                    channel, execution_params.hidden_hidden_linear);
                const std::size_t gate = channel / hidden;
                const GateExecutionParams& gate_params =
                    execution_params.gates.at(gate);
                const std::int64_t lhs = applyExecutionRescale(
                    static_cast<std::int64_t>(linear_ih[index]) -
                        gate_params.input_hidden_linear.param.zero_point,
                    gate_params.input_hidden_to_gate);
                const std::int64_t rhs = applyExecutionRescale(
                    static_cast<std::int64_t>(linear_hh[index]) -
                        gate_params.hidden_hidden_linear.param.zero_point,
                    gate_params.hidden_hidden_to_gate);
                gate_inputs[index] = clampInt(
                    checkedAdd(checkedAdd(lhs, rhs),
                               gate_params.gate_input.param.zero_point),
                    gate_params.gate_input);
                const auto activation =
                    gate == static_cast<std::size_t>(GateKind::Cell)
                        ? quantization::RealActivationKind::Tanh
                        : quantization::RealActivationKind::Sigmoid;
                gate_outputs[index] = quantization::realActivation(
                    gate_inputs[index], gate_params.gate_input.param,
                    gate_params.gate_input.type,
                    gate_params.gate_output.param,
                    gate_params.gate_output.type, activation);
            }
        }

        for (std::size_t row = 0; row < batch; ++row) {
            for (std::size_t column = 0; column < hidden; ++column) {
                const std::size_t state_index = row * hidden + column;
                const auto gateIndex = [&](GateKind gate) {
                    return row * channels + gateOffset(gate, hidden) + column;
                };
                const auto centeredGate = [&](GateKind gate,
                                              const QuantizedPoint& point) {
                    return static_cast<std::int64_t>(
                               gate_outputs[gateIndex(gate)]) -
                           point.param.zero_point;
                };
                const std::int64_t p_forget =
                    centeredGate(GateKind::Forget,
                                 execution_params.cell.forget_gate) *
                    (static_cast<std::int64_t>(cell_state[state_index]) -
                     execution_params.cell.old_cell.param.zero_point);
                const std::int64_t p_input =
                    centeredGate(GateKind::Input,
                                 execution_params.cell.input_gate) *
                    centeredGate(GateKind::Cell,
                                 execution_params.cell.cell_gate);
                const auto cell_result = computeQuantizedCellInt(
                    gate_outputs[gateIndex(GateKind::Forget)],
                    cell_state[state_index],
                    gate_outputs[gateIndex(GateKind::Input)],
                    gate_outputs[gateIndex(GateKind::Cell)],
                    execution_params.cell);
                cell_state[state_index] = cell_result.value;
                const std::int32_t cell_tanh =
                    quantization::realActivation(
                        cell_state[state_index],
                        execution_params.cell.new_cell.param,
                        execution_params.cell.new_cell.type,
                        execution_params.hidden.cell_tanh.param,
                        execution_params.hidden.cell_tanh.type,
                        quantization::RealActivationKind::Tanh);
                const auto hidden_result = computeQuantizedHiddenInt(
                    gate_outputs[gateIndex(GateKind::Output)], cell_tanh,
                    execution_params.hidden);
                const std::int64_t hidden_product =
                    hidden_result.diagnostics.raw_product;
                hidden_state[state_index] = hidden_result.value;
                output[(step * batch + row) * hidden + column] =
                    hidden_state[state_index];

                if (trace != nullptr) {
                    const std::size_t trace_state =
                        (step * batch + row) * hidden + column;
                    trace->cell_states[trace_state] = cell_state[state_index];
                    trace->cell_tanh_outputs[trace_state] = cell_tanh;
                    trace->hidden_outputs[trace_state] =
                        hidden_state[state_index];
                    trace->p_forget.push_back(p_forget);
                    trace->p_input.push_back(p_input);
                    trace->hidden_products.push_back(hidden_product);
                    trace->scaled_forget_contributions.push_back(
                        reference_detail::int128ToString(
                            cell_result.diagnostics
                                .scaled_forget_contribution));
                    trace->scaled_input_contributions.push_back(
                        reference_detail::int128ToString(
                            cell_result.diagnostics
                                .scaled_input_contribution));
                    trace->cell_wide_sums.push_back(
                        reference_detail::int128ToString(
                            cell_result.diagnostics.pre_round_sum));
                }
            }
        }
        if (trace != nullptr) {
            const std::size_t offset = step * batch * channels;
            std::copy(linear_ih.begin(), linear_ih.end(),
                      trace->weight_ih_linear.begin() + offset);
            std::copy(linear_hh.begin(), linear_hh.end(),
                      trace->weight_hh_linear.begin() + offset);
            std::copy(gate_inputs.begin(), gate_inputs.end(),
                      trace->gate_inputs.begin() + offset);
            std::copy(gate_outputs.begin(), gate_outputs.end(),
                      trace->gate_outputs.begin() + offset);
        }
    }
    std::copy(hidden_state.begin(), hidden_state.end(), final_hidden);
    std::copy(cell_state.begin(), cell_state.end(), final_cell);
#endif
}

}  // namespace quant_lstm
