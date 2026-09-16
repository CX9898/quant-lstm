#include "lstm/forward_float.h"

#include "lstm/gate_layout.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

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
    checkedProduct(
        {shape.sequence_length, shape.batch_size, shape.input_size, shape.hidden_size});
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
                         const float* input, const float* initial_hidden,
                         const float* initial_cell, float* output, float* final_hidden,
                         float* final_cell) {
    validateLstmFloatArguments(shape, weights, input, initial_hidden, initial_cell, output,
                               final_hidden, final_cell);

    const auto batch_size = static_cast<std::size_t>(shape.batch_size);
    const auto input_size = static_cast<std::size_t>(shape.input_size);
    const auto hidden_size = static_cast<std::size_t>(shape.hidden_size);
    const auto state_elements = batch_size * hidden_size;
    const bool has_bias = weights.bias_ih != nullptr;

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

            for (std::size_t channel = 0; channel < kGateCount * hidden_size; ++channel) {
                float value = has_bias ? weights.bias_ih[channel] + weights.bias_hh[channel] : 0.0F;
                const float* weight_ih_row = weights.weight_ih + channel * input_size;
                const float* weight_hh_row = weights.weight_hh + channel * hidden_size;
                for (std::size_t index = 0; index < input_size; ++index) {
                    value += weight_ih_row[index] * input_row[index];
                }
                for (std::size_t index = 0; index < hidden_size; ++index) {
                    value += weight_hh_row[index] * hidden_row[index];
                }
                gates[channel] = value;
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
                const float next_hidden = output_gate * std::tanh(next_cell);
                cell_row[hidden] = next_cell;
                hidden_row[hidden] = next_hidden;
                output[(static_cast<std::size_t>(time) * batch_size + batch) * hidden_size +
                       hidden] = next_hidden;
            }
        }
    }
}

}  // namespace quant_lstm
