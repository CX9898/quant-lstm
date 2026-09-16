#include "lstm/forward_float.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

bool near(float actual, float expected, float tolerance = 1.0e-6F) {
    return std::abs(actual - expected) <= tolerance;
}

}  // namespace

int main() {
    const quant_lstm::LstmShape shape{2, 1, 1, 1};
    const float input[2] = {0.25F, -0.5F};
    const float weight_ih[4] = {};
    const float weight_hh[4] = {};
    const quant_lstm::LstmFloatWeights weights{weight_ih, weight_hh, nullptr, nullptr};

    float output[2] = {};
    float final_hidden[1] = {};
    float final_cell[1] = {};
    quant_lstm::lstmForwardFloatCpu(shape, weights, input, nullptr, nullptr, output, final_hidden,
                                    final_cell);
    if (!near(output[0], 0.0F) || !near(output[1], 0.0F) || !near(final_hidden[0], 0.0F) ||
        !near(final_cell[0], 0.0F)) {
        std::cerr << "省略初始状态的零参数结果不为零\n";
        return EXIT_FAILURE;
    }

    const float initial_hidden[1] = {0.75F};
    const float initial_cell[1] = {1.0F};
    quant_lstm::lstmForwardFloatCpu(shape, weights, input, initial_hidden, initial_cell, output,
                                    final_hidden, final_cell);
    const float expected_cell_0 = 0.5F;
    const float expected_hidden_0 = 0.5F * std::tanh(expected_cell_0);
    const float expected_cell_1 = 0.25F;
    const float expected_hidden_1 = 0.5F * std::tanh(expected_cell_1);
    if (!near(output[0], expected_hidden_0) || !near(output[1], expected_hidden_1) ||
        !near(final_hidden[0], expected_hidden_1) || !near(final_cell[0], expected_cell_1)) {
        std::cerr << "CPU LSTM 状态递推结果错误\n";
        return EXIT_FAILURE;
    }

    bool rejected_partial_state = false;
    try {
        quant_lstm::lstmForwardFloatCpu(shape, weights, input, initial_hidden, nullptr, output,
                                        final_hidden, final_cell);
    } catch (const std::invalid_argument&) {
        rejected_partial_state = true;
    }
    if (!rejected_partial_state) {
        std::cerr << "未拒绝不完整的初始状态\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
