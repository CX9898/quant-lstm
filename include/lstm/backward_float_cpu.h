#pragma once

#include "lstm/forward_float.h"

namespace quant_lstm {

struct LstmFloatCpuBackwardTrace {
    const float* gate_outputs;
    const float* cell_states;
    const float* cell_tanh_outputs;
    const float* hidden_outputs;
};

struct LstmFloatCpuGradients {
    float* input;
    float* weight_ih;
    float* weight_hh;
    float* bias_ih;
    float* bias_hh;
    float* initial_hidden;
    float* initial_cell;
};

/// 单层、单向 FP32 LSTM CPU backward。所有张量使用 time-major row-major 布局。
void lstmBackwardFloatCpu(const LstmShape& shape, const LstmFloatWeights& weights,
                          const float* input, const float* initial_hidden,
                          const float* initial_cell, const LstmFloatCpuBackwardTrace& trace,
                          const float* grad_output, const float* grad_final_hidden,
                          const float* grad_final_cell, const LstmFloatCpuGradients& gradients);

}  // namespace quant_lstm
