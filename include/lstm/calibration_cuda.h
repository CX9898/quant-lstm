#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "lstm/calibration.h"

// CUDA 校准前向与设备端统计。原始 input/parameter/checkpoint 不离开设备；
// 只有每组 min/max 和可选 histogram bins 会回传给公共 finalization 会话。
namespace quant_lstm {

void lstmForwardCalibrateCuda(const LstmShape& shape, const LstmFloatWeights& weights,
                              const float* input, const float* initial_hidden,
                              const float* initial_cell, float* output, float* final_hidden,
                              float* final_cell, cublasHandle_t handle, cudaStream_t stream,
                              LstmCalibrationSession& session);

}  // namespace quant_lstm
