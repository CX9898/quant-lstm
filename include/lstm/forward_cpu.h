#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "lstm/forward_float.h"
#include "lstm/quant_params.h"

// 本模块声明两套单层、单向量化 CPU reference；输入输出均为 time-major row-major q 网格。
namespace quant_lstm {

struct LstmExecutionParams;

struct LstmInt32Weights {
    const std::int32_t* weight_ih;
    const std::int32_t* weight_hh;
    const std::int32_t* bias_ih;
    const std::int32_t* bias_hh;
};

struct LstmFpCarrierWeights {
    const float* weight_ih;
    const float* weight_hh;
    const float* bias_ih;
    const float* bias_hh;
};

struct LstmInt32ReferenceTrace {
    std::vector<std::int32_t> weight_ih_linear;
    std::vector<std::int32_t> weight_hh_linear;
    std::vector<std::int32_t> gate_inputs;
    std::vector<std::int32_t> gate_outputs;
    std::vector<std::int32_t> cell_states;
    std::vector<std::int32_t> cell_tanh_outputs;
    std::vector<std::int32_t> hidden_outputs;
    std::vector<std::int64_t> p_forget;
    std::vector<std::int64_t> p_input;
    std::vector<std::int64_t> hidden_products;
    std::vector<std::string> scaled_forget_contributions;
    std::vector<std::string> scaled_input_contributions;
    std::vector<std::string> cell_wide_sums;
};

struct LstmFpReferenceTrace {
    std::vector<float> weight_ih_linear;
    std::vector<float> weight_hh_linear;
    std::vector<float> gate_inputs;
    std::vector<float> gate_outputs;
    std::vector<float> cell_states;
    std::vector<float> cell_tanh_outputs;
    std::vector<float> hidden_outputs;
    std::vector<float> p_forget;
    std::vector<float> p_input;
    std::vector<float> scaled_forget_contributions;
    std::vector<float> scaled_input_contributions;
    std::vector<float> cell_wide_sums;
    std::vector<float> hidden_products;
};

/// 执行 int32 carrier reference；初始状态须同时提供或同时省略，省略时使用各自 zero point。
void lstmForwardInt32CpuReference(const LstmShape& shape, const LstmInt32Weights& weights,
                                  const std::int32_t* input, const std::int32_t* initial_hidden,
                                  const std::int32_t* initial_cell,
                                  const LstmOperatorQuantConfig& config,
                                  const LstmQuantParams& quant_params,
                                  const LstmExecutionParams& execution_params, std::int32_t* output,
                                  std::int32_t* final_hidden, std::int32_t* final_cell,
                                  LstmInt32ReferenceTrace* trace = nullptr);

/// 执行 FP32 q-carrier 标量 reference；所有 float q 输入必须位于合法整数网格。
void lstmForwardQuantizedFpCpuReference(
    const LstmShape& shape, const LstmFpCarrierWeights& weights, const float* input,
    const float* initial_hidden, const float* initial_cell, const LstmOperatorQuantConfig& config,
    const LstmQuantParams& quant_params, const LstmExecutionParams& execution_params, float* output,
    float* final_hidden, float* final_cell, LstmFpReferenceTrace* trace = nullptr);

}  // namespace quant_lstm
