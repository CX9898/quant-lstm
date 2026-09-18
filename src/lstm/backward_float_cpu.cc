#include "lstm/backward_float_cpu.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "lstm/gate_layout.h"

namespace quant_lstm {
namespace {

std::size_t checkedProduct(std::initializer_list<std::int64_t> factors, const char* name) {
    std::size_t result = 1;
    for (const std::int64_t factor : factors) {
        if (factor <= 0 ||
            static_cast<std::uint64_t>(factor) >
                static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
            result > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(factor)) {
            throw std::invalid_argument(std::string(name) + " 元素数量溢出");
        }
        result *= static_cast<std::size_t>(factor);
    }
    return result;
}

void validateArguments(const LstmShape& shape, const LstmFloatWeights& weights, const float* input,
                       const float* initial_hidden, const float* initial_cell,
                       const LstmFloatCpuBackwardTrace& trace, const float* grad_output,
                       const float* grad_final_hidden, const float* grad_final_cell,
                       const LstmFloatCpuGradients& gradients) {
    checkedProduct({shape.sequence_length, shape.batch_size, shape.input_size, shape.hidden_size},
                   "LSTM CPU backward shape");
    if (weights.weight_ih == nullptr || weights.weight_hh == nullptr || input == nullptr ||
        trace.gate_outputs == nullptr || trace.cell_states == nullptr ||
        trace.cell_tanh_outputs == nullptr || trace.hidden_outputs == nullptr ||
        grad_output == nullptr || grad_final_hidden == nullptr || grad_final_cell == nullptr ||
        gradients.input == nullptr || gradients.weight_ih == nullptr ||
        gradients.weight_hh == nullptr || gradients.bias_ih == nullptr ||
        gradients.bias_hh == nullptr || gradients.initial_hidden == nullptr ||
        gradients.initial_cell == nullptr) {
        throw std::invalid_argument("LSTM CPU backward 必需张量指针不能为空");
    }
    if ((initial_hidden == nullptr) != (initial_cell == nullptr)) {
        throw std::invalid_argument("h_0 和 c_0 必须同时提供或同时省略");
    }
}

}  // namespace

void lstmBackwardFloatCpu(const LstmShape& shape, const LstmFloatWeights& weights,
                          const float* input, const float* initial_hidden,
                          const float* initial_cell, const LstmFloatCpuBackwardTrace& trace,
                          const float* grad_output, const float* grad_final_hidden,
                          const float* grad_final_cell, const LstmFloatCpuGradients& gradients) {
    validateArguments(shape, weights, input, initial_hidden, initial_cell, trace, grad_output,
                      grad_final_hidden, grad_final_cell, gradients);

    const std::size_t steps = static_cast<std::size_t>(shape.sequence_length);
    const std::size_t batch = static_cast<std::size_t>(shape.batch_size);
    const std::size_t input_size = static_cast<std::size_t>(shape.input_size);
    const std::size_t hidden = static_cast<std::size_t>(shape.hidden_size);
    const std::size_t channels = kGateCount * hidden;
    const std::size_t state_elements = batch * hidden;

    std::fill_n(gradients.input, steps * batch * input_size, 0.0F);
    std::fill_n(gradients.weight_ih, channels * input_size, 0.0F);
    std::fill_n(gradients.weight_hh, channels * hidden, 0.0F);
    std::fill_n(gradients.bias_ih, channels, 0.0F);
    std::fill_n(gradients.bias_hh, channels, 0.0F);

    std::vector<float> grad_hidden(grad_final_hidden, grad_final_hidden + state_elements);
    std::vector<float> grad_cell(grad_final_cell, grad_final_cell + state_elements);
    std::vector<float> next_grad_hidden(state_elements, 0.0F);
    std::vector<float> grad_gates(batch * channels, 0.0F);

    for (std::size_t reverse_time = steps; reverse_time > 0; --reverse_time) {
        const std::size_t time = reverse_time - 1;
        const std::size_t state_offset = time * state_elements;
        const std::size_t gate_offset = time * batch * channels;
        const float* previous_cell =
            time == 0 ? initial_cell : trace.cell_states + (time - 1) * state_elements;

        for (std::size_t row = 0; row < batch; ++row) {
            for (std::size_t column = 0; column < hidden; ++column) {
                const std::size_t state_index = row * hidden + column;
                const std::size_t gate_base = row * channels + column;
                const std::size_t trace_base = gate_offset + gate_base;
                const float input_gate = trace.gate_outputs[trace_base];
                const float forget_gate = trace.gate_outputs[trace_base + hidden];
                const float cell_gate = trace.gate_outputs[trace_base + 2 * hidden];
                const float output_gate = trace.gate_outputs[trace_base + 3 * hidden];
                const float cell_tanh = trace.cell_tanh_outputs[state_offset + state_index];
                const float dh = grad_hidden[state_index] + grad_output[state_offset + state_index];
                const float dc =
                    grad_cell[state_index] + dh * output_gate * (1.0F - cell_tanh * cell_tanh);
                const float previous_cell_value =
                    previous_cell == nullptr ? 0.0F : previous_cell[state_index];

                const float grad_input_gate = dc * cell_gate;
                const float grad_forget_gate = dc * previous_cell_value;
                const float grad_cell_gate = dc * input_gate;
                const float grad_output_gate = dh * cell_tanh;
                grad_gates[gate_base] = grad_input_gate * input_gate * (1.0F - input_gate);
                grad_gates[gate_base + hidden] =
                    grad_forget_gate * forget_gate * (1.0F - forget_gate);
                grad_gates[gate_base + 2 * hidden] =
                    grad_cell_gate * (1.0F - cell_gate * cell_gate);
                grad_gates[gate_base + 3 * hidden] =
                    grad_output_gate * output_gate * (1.0F - output_gate);
                grad_cell[state_index] = dc * forget_gate;
            }
        }

        std::fill(next_grad_hidden.begin(), next_grad_hidden.end(), 0.0F);
        for (std::size_t row = 0; row < batch; ++row) {
            const float* input_row = input + (time * batch + row) * input_size;
            const float* previous_hidden =
                time == 0 ? initial_hidden : trace.hidden_outputs + (time - 1) * state_elements;
            if (previous_hidden != nullptr) {
                previous_hidden += row * hidden;
            }
            float* grad_input_row = gradients.input + (time * batch + row) * input_size;
            for (std::size_t channel = 0; channel < channels; ++channel) {
                const float gradient = grad_gates[row * channels + channel];
                gradients.bias_ih[channel] += gradient;
                gradients.bias_hh[channel] += gradient;
                for (std::size_t index = 0; index < input_size; ++index) {
                    grad_input_row[index] +=
                        gradient * weights.weight_ih[channel * input_size + index];
                    gradients.weight_ih[channel * input_size + index] +=
                        gradient * input_row[index];
                }
                for (std::size_t index = 0; index < hidden; ++index) {
                    next_grad_hidden[row * hidden + index] +=
                        gradient * weights.weight_hh[channel * hidden + index];
                    gradients.weight_hh[channel * hidden + index] +=
                        gradient * (previous_hidden == nullptr ? 0.0F : previous_hidden[index]);
                }
            }
        }
        grad_hidden.swap(next_grad_hidden);
    }

    std::copy(grad_hidden.begin(), grad_hidden.end(), gradients.initial_hidden);
    std::copy(grad_cell.begin(), grad_cell.end(), gradients.initial_cell);
}

}  // namespace quant_lstm
