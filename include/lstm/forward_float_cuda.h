#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "lstm/forward_float.h"

// 本模块声明 CUDA FP32 LSTM 前向；所有数据指针均须位于同一 CUDA 设备。
namespace quant_lstm {

/// 使用调用方持有的 cuBLAS handle 和 stream 执行前向。
///
/// workspace 可为 nullptr；若非空，其容量至少为 cudaWorkspaceElementCount(shape) 个 float。
/// 函数不拥有 handle、stream、输入、输出或 workspace。
void lstmForwardFloatCuda(const LstmShape& shape, const LstmFloatWeights& weights,
                          const float* input, const float* initial_hidden,
                          const float* initial_cell, float* output, float* final_hidden,
                          float* final_cell, cublasHandle_t handle, cudaStream_t stream,
                          float* workspace = nullptr);

/// 返回 CUDA 前向所需 workspace 的 float 元素数。
std::int64_t cudaWorkspaceElementCount(const LstmShape& shape);

}  // namespace quant_lstm
