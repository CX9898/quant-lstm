#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "lstm/backward_float_cpu.h"
#include "lstm/forward_float.h"

namespace {

bool near(float actual, float expected, float tolerance = 1.0e-6F) {
    return std::abs(actual - expected) <= tolerance;
}

float evaluateLoss(const quant_lstm::LstmShape& shape, const quant_lstm::LstmFloatWeights& weights,
                   const float* input, const float* initial_hidden, const float* initial_cell,
                   const float* grad_output, const float* grad_final_hidden,
                   const float* grad_final_cell) {
    float output[2] = {};
    float final_hidden[1] = {};
    float final_cell[1] = {};
    quant_lstm::lstmForwardFloatCpu(shape, weights, input, initial_hidden, initial_cell, output,
                                    final_hidden, final_cell);
    return output[0] * grad_output[0] + output[1] * grad_output[1] +
           final_hidden[0] * grad_final_hidden[0] + final_cell[0] * grad_final_cell[0];
}

}  // namespace

int main() {
    const quant_lstm::LstmShape shape{2, 1, 1, 1};
    float input[2] = {0.25F, -0.5F};
    float weight_ih[4] = {};
    float weight_hh[4] = {};
    float bias_ih[4] = {};
    float bias_hh[4] = {};
    const quant_lstm::LstmFloatWeights weights{weight_ih, weight_hh, bias_ih, bias_hh};

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

    float initial_hidden[1] = {0.75F};
    float initial_cell[1] = {1.0F};
    quant_lstm::LstmFloatReferenceTrace trace;
    quant_lstm::lstmForwardFloatCpu(shape, weights, input, initial_hidden, initial_cell, output,
                                    final_hidden, final_cell, &trace);
    const float expected_cell_0 = 0.5F;
    const float expected_hidden_0 = 0.5F * std::tanh(expected_cell_0);
    const float expected_cell_1 = 0.25F;
    const float expected_hidden_1 = 0.5F * std::tanh(expected_cell_1);
    if (!near(output[0], expected_hidden_0) || !near(output[1], expected_hidden_1) ||
        !near(final_hidden[0], expected_hidden_1) || !near(final_cell[0], expected_cell_1)) {
        std::cerr << "CPU LSTM 状态递推结果错误\n";
        return EXIT_FAILURE;
    }

    const float grad_output[2] = {0.4F, -0.2F};
    const float grad_final_hidden[1] = {0.3F};
    const float grad_final_cell[1] = {-0.1F};
    float grad_input[2] = {};
    float grad_weight_ih[4] = {};
    float grad_weight_hh[4] = {};
    float grad_bias_ih[4] = {};
    float grad_bias_hh[4] = {};
    float grad_initial_hidden[1] = {};
    float grad_initial_cell[1] = {};
    const quant_lstm::LstmFloatCpuBackwardTrace backward_trace{
        trace.gate_outputs.data(), trace.cell_states.data(), trace.cell_tanh_outputs.data(),
        output};
    const quant_lstm::LstmFloatCpuGradients gradients{
        grad_input,   grad_weight_ih,      grad_weight_hh,   grad_bias_ih,
        grad_bias_hh, grad_initial_hidden, grad_initial_cell};
    quant_lstm::lstmBackwardFloatCpu(shape, weights, input, initial_hidden, initial_cell,
                                     backward_trace, grad_output, grad_final_hidden,
                                     grad_final_cell, gradients);

    const auto check_finite_difference = [&](const char* name, float* value, float actual) {
        constexpr float epsilon = 1.0e-3F;
        const float original = *value;
        *value = original + epsilon;
        const float positive = evaluateLoss(shape, weights, input, initial_hidden, initial_cell,
                                            grad_output, grad_final_hidden, grad_final_cell);
        *value = original - epsilon;
        const float negative = evaluateLoss(shape, weights, input, initial_hidden, initial_cell,
                                            grad_output, grad_final_hidden, grad_final_cell);
        *value = original;
        const float expected = (positive - negative) / (2.0F * epsilon);
        if (!near(actual, expected, 2.0e-4F)) {
            std::cerr << name << " CPU backward 有限差分不匹配: actual=" << actual
                      << " expected=" << expected << '\n';
            return false;
        }
        return true;
    };
    for (int index = 0; index < 2; ++index) {
        if (!check_finite_difference("input", &input[index], grad_input[index])) {
            return EXIT_FAILURE;
        }
    }
    for (int index = 0; index < 4; ++index) {
        if (!check_finite_difference("weight_ih", &weight_ih[index], grad_weight_ih[index]) ||
            !check_finite_difference("weight_hh", &weight_hh[index], grad_weight_hh[index]) ||
            !check_finite_difference("bias_ih", &bias_ih[index], grad_bias_ih[index]) ||
            !check_finite_difference("bias_hh", &bias_hh[index], grad_bias_hh[index])) {
            return EXIT_FAILURE;
        }
    }
    if (!check_finite_difference("initial_hidden", initial_hidden, grad_initial_hidden[0]) ||
        !check_finite_difference("initial_cell", initial_cell, grad_initial_cell[0])) {
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
