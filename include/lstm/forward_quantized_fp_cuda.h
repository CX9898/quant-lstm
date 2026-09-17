#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "lstm/forward_float.h"
#include "lstm/lstm_execution_params.h"
#include "lstm/quant_params.h"

// CUDA FP32 q-carrier 主路径。公开边界只接受/返回 real-domain master float；
// q 值仅存在于调用内部及可选 device checkpoints 中。
namespace quant_lstm {

enum class LstmQuantizedFpCudaMathMode : std::uint8_t {
    Pedantic,
    Tf32,
};

/// 可复用且完全由调用方拥有的 CUDA 执行资源。
///
/// handles/streams[0] 用于递推 GEMM、pointwise 和最终输出；
/// handles/streams[1] 用于一次性的 all-time input GEMM。
/// events[0] 表示 input GEMM ready，events[1] 表示整个调用完成。
/// device_execution_params 可为空；非空时由调用方分配/释放，函数按签名缓存
/// 已展开的 4H 参数。device_static_parameters 可为空；非空且 forward 提供非零
/// static_parameter_cache_key 时缓存量化后的 W/R/bias 和 weight sums。调用方必须在
/// master 参数内容变化时更换 key。复用同一 context 前必须等待 events[1]。
struct LstmQuantizedFpCudaContext {
    std::array<cublasHandle_t, 2> handles{};
    std::array<cudaStream_t, 2> streams{};
    std::array<cudaEvent_t, 2> events{};
    void* device_execution_params = nullptr;
    std::size_t device_execution_params_bytes = 0;
    std::uint64_t cached_execution_signature = 0;
    void* device_static_parameters = nullptr;
    std::size_t device_static_parameters_bytes = 0;
    std::uint64_t cached_static_parameter_signature = 0;
};

struct LstmQuantizedFpCudaWorkspace {
    void* data = nullptr;
    std::size_t bytes = 0;
};

struct LstmQuantizedFpCudaWorkspaceBreakdown {
    std::size_t quantized_input_bytes = 0;
    std::size_t quantized_weight_ih_bytes = 0;
    std::size_t quantized_weight_hh_bytes = 0;
    std::size_t quantized_bias_bytes = 0;
    std::size_t quantized_state_bytes = 0;
    std::size_t input_linear_bytes = 0;
    std::size_t recurrent_linear_bytes = 0;
    std::size_t weight_sum_bytes = 0;
    std::size_t device_parameter_bytes = 0;
    std::size_t alignment_padding_bytes = 0;
    std::size_t total_bytes = 0;
};

/// 所有 checkpoint 都是可选 device float 指针，保存 q 网格值。
/// 非空张量分别采用 [T,B,4H]、[T,B,4H]、[T,B,4H]、[T,B,4H]、
/// [T,B,H]、[T,B,H]、[T,B,H] 的连续 row-major 布局。
struct LstmQuantizedFpCudaCheckpoints {
    float* weight_ih_linear = nullptr;
    float* weight_hh_linear = nullptr;
    float* gate_inputs = nullptr;
    float* gate_outputs = nullptr;
    float* cell_states = nullptr;
    float* cell_tanh_outputs = nullptr;
    float* hidden_outputs = nullptr;

    // 每个 mask 与对应 checkpoint 同形，1 表示该真实量化边界发生 Clamp。
    // mask 只描述 Clamp，不把融合乘法临时值伪装为量化点。
    std::uint8_t* weight_ih_linear_clamped = nullptr;
    std::uint8_t* weight_hh_linear_clamped = nullptr;
    std::uint8_t* gate_inputs_clamped = nullptr;
    std::uint8_t* gate_outputs_clamped = nullptr;
    std::uint8_t* cell_states_clamped = nullptr;
    std::uint8_t* cell_tanh_outputs_clamped = nullptr;
    std::uint8_t* hidden_outputs_clamped = nullptr;
};

/// 计数在 enqueue 返回前即可读取，不要求同步 GPU。
struct LstmQuantizedFpCudaStats {
    std::uint64_t input_gemm_calls = 0;
    std::uint64_t recurrent_gemm_calls = 0;
    std::size_t workspace_bytes = 0;
    bool used_internal_workspace = false;
    bool execution_parameter_cache_hit = false;
    bool static_parameter_cache_hit = false;
};

/// 可选的调用方计时事件；函数仅记录、不创建或销毁。
/// input/recurrent quantized 分别标记双 stream 的边界，core_complete 已包含
/// 双 SGEMM 与 pointwise，complete 还包含最终反量化。
struct LstmQuantizedFpCudaTimingEvents {
    cudaEvent_t start = nullptr;
    cudaEvent_t input_quantized = nullptr;
    cudaEvent_t recurrent_quantized = nullptr;
    cudaEvent_t quantization_complete = nullptr;
    cudaEvent_t core_complete = nullptr;
    cudaEvent_t complete = nullptr;
};

LstmQuantizedFpCudaWorkspaceBreakdown
lstmQuantizedFpCudaWorkspaceBreakdown(const LstmShape& shape,
                                      bool bias_enabled);

std::size_t lstmQuantizedFpCudaWorkspaceBytes(const LstmShape& shape,
                                              bool bias_enabled);

std::size_t lstmQuantizedFpCudaStaticParameterBytes(
    const LstmShape& shape, bool bias_enabled);

/// 异步执行单层、单向 LSTM。
///
/// input/weights/bias/initial states 是同一当前 CUDA device 上的 real-domain
/// master float，本函数在调用内量化。output/final_hidden/final_cell 写入反量化
/// real-domain float。initial_hidden/initial_cell 必须同时提供或同时为空。
///
/// workspace.data 为空时在 streams[0] 上使用 cudaMallocAsync/cudaFreeAsync；
/// 非空时 bytes 必须不小于 lstmQuantizedFpCudaWorkspaceBytes()。函数返回时工作
/// 已入队，events[1] 已在 streams[0] 记录；调用方不得在该事件完成前复用输出、
/// checkpoint、workspace 或 context 资源。
void lstmForwardQuantizedFpCuda(
    const LstmShape& shape, const LstmFloatWeights& master_weights,
    const float* input, const float* initial_hidden,
    const float* initial_cell, const LstmOperatorQuantConfig& resolved_config,
    const LstmQuantParams& quant_params,
    const LstmExecutionParams& execution_params, float* output,
    float* final_hidden, float* final_cell,
    LstmQuantizedFpCudaContext& context,
    LstmQuantizedFpCudaMathMode math_mode,
    LstmQuantizedFpCudaWorkspace workspace = {},
    const LstmQuantizedFpCudaCheckpoints* checkpoints = nullptr,
    LstmQuantizedFpCudaStats* stats = nullptr,
    const LstmQuantizedFpCudaTimingEvents* timing_events = nullptr,
    std::uint64_t static_parameter_cache_key = 0);

}  // namespace quant_lstm
