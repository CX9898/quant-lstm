#pragma once

#include <array>
#include <cstdint>
#include <vector>

// 本模块定义单层、单向 FP32 LSTM 的载体无关公共数据契约。
// 输入和输出均为 row-major time-major；权重使用 PyTorch 的 [4H,I]/[4H,H] 布局。
namespace quant_lstm {

struct LstmShape {
    std::int64_t sequence_length;
    std::int64_t batch_size;
    std::int64_t input_size;
    std::int64_t hidden_size;
};

struct LstmFloatWeights {
    const float* weight_ih;
    const float* weight_hh;
    const float* bias_ih;
    const float* bias_hh;
};

// 正式 FP32 reference 的逐时间步 checkpoint。门张量布局为 [T,B,4,H]，
// 状态张量布局为 [T,B,H]；门顺序固定为 (i,f,g,o)。
struct LstmFloatReferenceTrace {
    std::vector<float> weight_input_hidden_linear;
    std::vector<float> weight_hidden_hidden_linear;
    std::vector<float> gate_inputs;
    std::vector<float> gate_outputs;
    std::vector<float> cell_states;
    std::vector<float> cell_tanh_outputs;
    std::vector<float> hidden_outputs;
};

/// 校验 shape、权重和状态指针；失败时抛出 std::invalid_argument。
void validateLstmFloatArguments(const LstmShape& shape, const LstmFloatWeights& weights,
                                const float* input, const float* initial_hidden,
                                const float* initial_cell, const float* output,
                                const float* final_hidden, const float* final_cell);

/// 执行 CPU FP32 单层单向 LSTM。
///
/// initial_hidden/initial_cell 可同时为 nullptr，此时使用全零状态；二者必须同时省略或提供。
void lstmForwardFloatCpu(const LstmShape& shape, const LstmFloatWeights& weights,
                         const float* input, const float* initial_hidden, const float* initial_cell,
                         float* output, float* final_hidden, float* final_cell,
                         LstmFloatReferenceTrace* trace = nullptr);

}  // namespace quant_lstm
