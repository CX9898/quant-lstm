#include "lstm/forward_cpu.h"

#include "lstm/gate_layout.h"
#include "lstm/lstm_execution_params.h"
#include "lstm/quantized_cell_fp.h"
#include "quantization/real_activation.h"
#include "forward_quantized_common.h"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace quant_lstm {
namespace {

using reference_detail::clampFp;
using reference_detail::requireGrid;

float linearValue(const float* vector, const float* weights, const float* bias,
                  std::size_t width, std::size_t channel,
                  const LinearExecutionParams& params) {
    const QuantizedPoint& weight_point = params.weights.at(channel);
    float accumulator = 0.0F;
    for (std::size_t column = 0; column < width; ++column) {
        const float weight = weights[channel * width + column];
        const float value = vector[column];
        requireGrid(weight, weight_point);
        requireGrid(value, params.input);
        accumulator +=
            (weight - static_cast<float>(weight_point.param.zero_point)) *
            (value - static_cast<float>(params.input.param.zero_point));
    }
    if (!std::isfinite(accumulator)) {
        throw std::overflow_error("FP carrier GEMM 产生 Inf/NaN");
    }
    if (bias != nullptr) {
        requireGrid(bias[channel], params.biases.at(channel));
        const float centered_bias =
            bias[channel] -
            static_cast<float>(params.biases.at(channel).param.zero_point);
        accumulator += applyExecutionRescale(
            centered_bias, params.bias_to_accumulator.at(channel));
    }
    const float output =
        applyExecutionRescale(
            accumulator, params.accumulator_to_output.at(channel)) +
        static_cast<float>(params.output.param.zero_point);
    return clampFp(output, params.output);
}

void prepareTrace(LstmFpReferenceTrace* trace, std::size_t steps,
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
    trace->scaled_forget_contributions.reserve(states);
    trace->scaled_input_contributions.reserve(states);
    trace->cell_wide_sums.reserve(states);
    trace->hidden_products.reserve(states);
}

}  // namespace

void lstmForwardQuantizedFpCpuReference(
    const LstmShape& shape, const LstmFpCarrierWeights& weights,
    const float* input, const float* initial_hidden, const float* initial_cell,
    const LstmOperatorQuantConfig& config,
    const LstmQuantParams& quant_params,
    const LstmExecutionParams& execution_params, float* output,
    float* final_hidden, float* final_cell, LstmFpReferenceTrace* trace) {
    reference_detail::validateCommon(
        shape, quant_params.bias_enabled, weights.weight_ih, weights.weight_hh,
        weights.bias_ih, weights.bias_hh, input, initial_hidden, initial_cell,
        output, final_hidden, final_cell, config, quant_params,
        execution_params);
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

    std::vector<float> hidden_state(batch * hidden);
    std::vector<float> cell_state(batch * hidden);
    std::vector<float> linear_ih(batch * channels);
    std::vector<float> linear_hh(batch * channels);
    std::vector<float> gate_inputs(batch * channels);
    std::vector<float> gate_outputs(batch * channels);
    const float hidden_zero =
        static_cast<float>(execution_params.hidden.output.param.zero_point);
    const float cell_zero =
        static_cast<float>(execution_params.cell.old_cell.param.zero_point);
    for (std::size_t index = 0; index < batch * hidden; ++index) {
        hidden_state[index] =
            initial_hidden == nullptr ? hidden_zero : initial_hidden[index];
        cell_state[index] =
            initial_cell == nullptr ? cell_zero : initial_cell[index];
        requireGrid(hidden_state[index], execution_params.hidden.output);
        requireGrid(cell_state[index], execution_params.cell.old_cell);
    }

    for (std::size_t step = 0; step < steps; ++step) {
        for (std::size_t row = 0; row < batch; ++row) {
            const float* input_row =
                input + (step * batch + row) * input_size;
            const float* hidden_row =
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
                const float lhs = applyExecutionRescale(
                    linear_ih[index] -
                        static_cast<float>(
                            gate_params.input_hidden_linear.param.zero_point),
                    gate_params.input_hidden_to_gate);
                const float rhs = applyExecutionRescale(
                    linear_hh[index] -
                        static_cast<float>(
                            gate_params.hidden_hidden_linear.param.zero_point),
                    gate_params.hidden_hidden_to_gate);
                gate_inputs[index] = clampFp(
                    lhs + rhs +
                        static_cast<float>(
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
                    return gate_outputs[gateIndex(gate)] -
                           static_cast<float>(point.param.zero_point);
                };
                const float p_forget =
                    centeredGate(GateKind::Forget,
                                 execution_params.cell.forget_gate) *
                    (cell_state[state_index] -
                     static_cast<float>(
                         execution_params.cell.old_cell.param.zero_point));
                const float p_input =
                    centeredGate(GateKind::Input,
                                 execution_params.cell.input_gate) *
                    centeredGate(GateKind::Cell,
                                 execution_params.cell.cell_gate);
                const auto cell_result = computeQuantizedCellFp(
                    gate_outputs[gateIndex(GateKind::Forget)],
                    cell_state[state_index],
                    gate_outputs[gateIndex(GateKind::Input)],
                    gate_outputs[gateIndex(GateKind::Cell)],
                    execution_params.cell);
                cell_state[state_index] = cell_result.value;
                const float cell_tanh = quantization::realActivation(
                    cell_state[state_index],
                    execution_params.cell.new_cell.param,
                    execution_params.cell.new_cell.type,
                    execution_params.hidden.cell_tanh.param,
                    execution_params.hidden.cell_tanh.type,
                    quantization::RealActivationKind::Tanh);
                const auto hidden_result = computeQuantizedHiddenFp(
                    gate_outputs[gateIndex(GateKind::Output)], cell_tanh,
                    execution_params.hidden);
                const float hidden_product =
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
                    trace->scaled_forget_contributions.push_back(
                        cell_result.diagnostics
                            .scaled_forget_contribution);
                    trace->scaled_input_contributions.push_back(
                        cell_result.diagnostics
                            .scaled_input_contribution);
                    trace->cell_wide_sums.push_back(
                        cell_result.diagnostics.pre_round_sum);
                    trace->hidden_products.push_back(hidden_product);
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
}

}  // namespace quant_lstm
