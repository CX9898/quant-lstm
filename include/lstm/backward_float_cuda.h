#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cstdint>

#include "lstm/forward_float.h"

// 单层、单向 FP32 LSTM CUDA backward。checkpoint 使用 time-major row-major 布局。
namespace quant_lstm {

struct LstmFloatCudaBackwardTrace {
    const float* gate_outputs;
    const float* cell_states;
    const float* cell_tanh_outputs;
    const float* hidden_outputs;
};

struct LstmFloatCudaGradients {
    float* input;
    float* weight_ih;
    float* weight_hh;
    float* bias_ih;
    float* bias_hh;
    float* initial_hidden;
    float* initial_cell;
};

// 所有 mask 与对应 forward 张量同形，1 表示发生 Clamp，反向梯度必须置零。
struct LstmFloatCudaBackwardMasks {
    const std::uint8_t* input = nullptr;
    const std::uint8_t* weight_ih = nullptr;
    const std::uint8_t* weight_hh = nullptr;
    const std::uint8_t* bias_ih = nullptr;
    const std::uint8_t* bias_hh = nullptr;
    const std::uint8_t* initial_hidden = nullptr;
    const std::uint8_t* initial_cell = nullptr;
    const std::uint8_t* weight_ih_linear = nullptr;
    const std::uint8_t* weight_hh_linear = nullptr;
    const std::uint8_t* gate_inputs = nullptr;
    const std::uint8_t* gate_outputs = nullptr;
    const std::uint8_t* cell_states = nullptr;
    const std::uint8_t* cell_tanh_outputs = nullptr;
    const std::uint8_t* hidden_outputs = nullptr;
};

/// 执行完整 FP32 backward。grad_output 为 [T,B,H]，最终状态梯度为 [B,H]。
/// initial_hidden/initial_cell 可同时为空，此时按全零初始状态处理。
void lstmBackwardFloatCuda(const LstmShape& shape, const LstmFloatWeights& weights,
                           const float* input, const float* initial_hidden,
                           const float* initial_cell, const LstmFloatCudaBackwardTrace& trace,
                           const float* grad_output, const float* grad_final_hidden,
                           const float* grad_final_cell, const LstmFloatCudaGradients& gradients,
                           cublasHandle_t handle, cudaStream_t stream, float* workspace = nullptr,
                           const LstmFloatCudaBackwardMasks* masks = nullptr);

/// 返回 backward 临时存储所需的 float 元素数。
std::int64_t cudaBackwardWorkspaceElementCount(const LstmShape& shape,
                                               bool split_linear_gradients = false);

}  // namespace quant_lstm
