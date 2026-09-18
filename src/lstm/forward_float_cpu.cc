#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "lstm/forward_float.h"
#include "lstm/gate_layout.h"

// CPU 标量实现是阶段 1 的浮点语义参考；仅允许 FP32 运算，不包含量化逻辑。
namespace quant_lstm {
namespace {

std::int64_t checkedProduct(std::initializer_list<std::int64_t> factors) {
    std::int64_t result = 1;
    for (const std::int64_t factor : factors) {
        if (factor <= 0 || result > std::numeric_limits<std::int64_t>::max() / factor) {
            throw std::invalid_argument("LSTM shape 非法或元素数量溢出");
        }
        result *= factor;
    }
    return result;
}

float sigmoid(float value) {
    if (value >= 0.0F) {
        const float exp_value = std::exp(-value);
        return 1.0F / (1.0F + exp_value);
    }
    const float exp_value = std::exp(value);
    return exp_value / (1.0F + exp_value);
}

}  // namespace

void validateLstmFloatArguments(const LstmShape& shape, const LstmFloatWeights& weights,
                                const float* input, const float* initial_hidden,
                                const float* initial_cell, const float* output,
                                const float* final_hidden, const float* final_cell) {
    checkedProduct({shape.sequence_length, shape.batch_size, shape.input_size, shape.hidden_size});
    checkedProduct({kGateCount, shape.hidden_size, shape.input_size});
    checkedProduct({kGateCount, shape.hidden_size, shape.hidden_size});

    if (weights.weight_ih == nullptr || weights.weight_hh == nullptr || input == nullptr ||
        output == nullptr || final_hidden == nullptr || final_cell == nullptr) {
        throw std::invalid_argument("LSTM 必需张量指针不能为空");
    }
    if ((weights.bias_ih == nullptr) != (weights.bias_hh == nullptr)) {
        throw std::invalid_argument("bias_ih 和 bias_hh 必须同时提供或同时省略");
    }
    if ((initial_hidden == nullptr) != (initial_cell == nullptr)) {
        throw std::invalid_argument("h_0 和 c_0 必须同时提供或同时省略");
    }
}

void lstmForwardFloatCpu(const LstmShape& shape, const LstmFloatWeights& weights,
                         const float* input, const float* initial_hidden, const float* initial_cell,
                         float* output, float* final_hidden, float* final_cell,
                         LstmFloatReferenceTrace* trace) {
    validateLstmFloatArguments(shape, weights, input, initial_hidden, initial_cell, output,
                               final_hidden, final_cell);

    const auto batch_size = static_cast<std::size_t>(shape.batch_size);
    const auto input_size = static_cast<std::size_t>(shape.input_size);
    const auto hidden_size = static_cast<std::size_t>(shape.hidden_size);
    const auto state_elements = batch_size * hidden_size;
    const auto gate_elements =
        static_cast<std::size_t>(shape.sequence_length) * batch_size * kGateCount * hidden_size;
    const auto recurrent_elements =
        static_cast<std::size_t>(shape.sequence_length) * state_elements;
    const bool has_bias = weights.bias_ih != nullptr;

    if (trace != nullptr) {
        trace->weight_input_hidden_linear.assign(gate_elements, 0.0F);
        trace->weight_hidden_hidden_linear.assign(gate_elements, 0.0F);
        trace->gate_inputs.assign(gate_elements, 0.0F);
        trace->gate_outputs.assign(gate_elements, 0.0F);
        trace->cell_states.assign(recurrent_elements, 0.0F);
        trace->cell_tanh_outputs.assign(recurrent_elements, 0.0F);
        trace->hidden_outputs.assign(recurrent_elements, 0.0F);
    }

    if (initial_hidden == nullptr) {
        std::fill_n(final_hidden, state_elements, 0.0F);
        std::fill_n(final_cell, state_elements, 0.0F);
    } else {
        std::copy_n(initial_hidden, state_elements, final_hidden);
        std::copy_n(initial_cell, state_elements, final_cell);
    }

    std::vector<float> gates(kGateCount * hidden_size);
    for (std::int64_t time = 0; time < shape.sequence_length; ++time) {
        for (std::size_t batch = 0; batch < batch_size; ++batch) {
            const float* input_row =
                input + (static_cast<std::size_t>(time) * batch_size + batch) * input_size;
            float* hidden_row = final_hidden + batch * hidden_size;
            float* cell_row = final_cell + batch * hidden_size;
            const std::size_t gate_trace_offset =
                (static_cast<std::size_t>(time) * batch_size + batch) * kGateCount * hidden_size;
            const std::size_t state_trace_offset =
                (static_cast<std::size_t>(time) * batch_size + batch) * hidden_size;

            for (std::size_t channel = 0; channel < kGateCount * hidden_size; ++channel) {
                float input_linear = has_bias ? weights.bias_ih[channel] : 0.0F;
                float hidden_linear = has_bias ? weights.bias_hh[channel] : 0.0F;
                const float* weight_ih_row = weights.weight_ih + channel * input_size;
                const float* weight_hh_row = weights.weight_hh + channel * hidden_size;
                for (std::size_t index = 0; index < input_size; ++index) {
                    input_linear += weight_ih_row[index] * input_row[index];
                }
                for (std::size_t index = 0; index < hidden_size; ++index) {
                    hidden_linear += weight_hh_row[index] * hidden_row[index];
                }
                gates[channel] = input_linear + hidden_linear;
                if (trace != nullptr) {
                    const std::size_t index = gate_trace_offset + channel;
                    trace->weight_input_hidden_linear[index] = input_linear;
                    trace->weight_hidden_hidden_linear[index] = hidden_linear;
                    trace->gate_inputs[index] = gates[channel];
                }
            }

            for (std::size_t hidden = 0; hidden < hidden_size; ++hidden) {
                const float input_gate =
                    sigmoid(gates[gateOffset(GateKind::Input, hidden_size) + hidden]);
                const float forget_gate =
                    sigmoid(gates[gateOffset(GateKind::Forget, hidden_size) + hidden]);
                const float cell_gate =
                    std::tanh(gates[gateOffset(GateKind::Cell, hidden_size) + hidden]);
                const float output_gate =
                    sigmoid(gates[gateOffset(GateKind::Output, hidden_size) + hidden]);

                const float next_cell = forget_gate * cell_row[hidden] + input_gate * cell_gate;
                const float cell_tanh = std::tanh(next_cell);
                const float next_hidden = output_gate * cell_tanh;
                cell_row[hidden] = next_cell;
                hidden_row[hidden] = next_hidden;
                output[(static_cast<std::size_t>(time) * batch_size + batch) * hidden_size +
                       hidden] = next_hidden;
                if (trace != nullptr) {
                    const std::array<float, kGateCount> gate_values{input_gate, forget_gate,
                                                                    cell_gate, output_gate};
                    for (std::size_t gate = 0; gate < kGateCount; ++gate) {
                        trace->gate_outputs[gate_trace_offset + gate * hidden_size + hidden] =
                            gate_values[gate];
                    }
                    trace->cell_states[state_trace_offset + hidden] = next_cell;
                    trace->cell_tanh_outputs[state_trace_offset + hidden] = cell_tanh;
                    trace->hidden_outputs[state_trace_offset + hidden] = next_hidden;
                }
            }
        }
    }
}

}  // namespace quant_lstm
