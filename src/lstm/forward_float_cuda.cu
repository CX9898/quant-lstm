#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "cuda/cuda_common.cuh"
#include "cuda/quantized_fp_math.cuh"
#include "lstm/forward_float_cuda.h"
#include "lstm/gate_layout.h"

// CUDA 基准将两路 Linear 交给 cuBLAS，逐时间步 kernel 完成 (i,f,g,o) 与状态更新。
namespace quant_lstm {
namespace {

__global__ void updateLstmState(std::int64_t batch_size, std::int64_t hidden_size,
                                const float* input_linear, const float* recurrent_linear,
                                const float* bias_ih, const float* bias_hh, float* hidden,
                                float* cell, float* output, float* gate_outputs, float* cell_states,
                                float* cell_tanh_outputs, float* traced_input_linear,
                                float* traced_recurrent_linear, float* gate_inputs) {
    const std::int64_t count = batch_size * hidden_size;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t element = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         element < count; element += stride) {
        const std::int64_t batch = element / hidden_size;
        const std::int64_t hidden_index = element % hidden_size;
        const std::int64_t gate_stride = 4 * hidden_size;
        const std::int64_t base = batch * gate_stride + hidden_index;

        float gate_values[4];
#pragma unroll
        for (int gate = 0; gate < 4; ++gate) {
            const std::int64_t gate_index = base + static_cast<std::int64_t>(gate) * hidden_size;
            const std::int64_t bias_index =
                static_cast<std::int64_t>(gate) * hidden_size + hidden_index;
            float value = input_linear[gate_index] + recurrent_linear[gate_index];
            if (bias_ih != nullptr) {
                value += bias_ih[bias_index] + bias_hh[bias_index];
            }
            if (gate_inputs != nullptr) {
                traced_input_linear[gate_index] =
                    input_linear[gate_index] + (bias_ih == nullptr ? 0.0F : bias_ih[bias_index]);
                traced_recurrent_linear[gate_index] =
                    recurrent_linear[gate_index] +
                    (bias_hh == nullptr ? 0.0F : bias_hh[bias_index]);
                gate_inputs[gate_index] = value;
            }
            gate_values[gate] = value;
        }

        const float input_gate = cuda_detail::realSigmoid(gate_values[0]);
        const float forget_gate = cuda_detail::realSigmoid(gate_values[1]);
        const float cell_gate = cuda_detail::realTanh(gate_values[2]);
        const float output_gate = cuda_detail::realSigmoid(gate_values[3]);
        const float next_cell = forget_gate * cell[element] + input_gate * cell_gate;
        const float cell_tanh = cuda_detail::realTanh(next_cell);
        const float next_hidden = output_gate * cell_tanh;
        cell[element] = next_cell;
        hidden[element] = next_hidden;
        output[element] = next_hidden;
        if (gate_outputs != nullptr) {
            gate_outputs[base] = input_gate;
            gate_outputs[base + hidden_size] = forget_gate;
            gate_outputs[base + 2 * hidden_size] = cell_gate;
            gate_outputs[base + 3 * hidden_size] = output_gate;
            cell_states[element] = next_cell;
            cell_tanh_outputs[element] = cell_tanh;
        }
    }
}

}  // namespace

std::int64_t cudaWorkspaceElementCount(const LstmShape& shape) {
    if (shape.sequence_length <= 0 || shape.batch_size <= 0 || shape.hidden_size <= 0) {
        throw std::invalid_argument("LSTM shape 必须为正数");
    }
    constexpr std::int64_t gate_count = static_cast<std::int64_t>(kGateCount);
    if (shape.sequence_length > std::numeric_limits<std::int64_t>::max() / shape.batch_size /
                                    gate_count / shape.hidden_size) {
        throw std::invalid_argument("CUDA workspace 元素数量溢出");
    }
    const std::int64_t input_linear =
        shape.sequence_length * shape.batch_size * gate_count * shape.hidden_size;
    const std::int64_t recurrent_linear = shape.batch_size * gate_count * shape.hidden_size;
    if (input_linear > std::numeric_limits<std::int64_t>::max() - recurrent_linear) {
        throw std::invalid_argument("CUDA workspace 元素数量溢出");
    }
    return input_linear + recurrent_linear;
}

void lstmForwardFloatCuda(const LstmShape& shape, const LstmFloatWeights& weights,
                          const float* input, const float* initial_hidden,
                          const float* initial_cell, float* output, float* final_hidden,
                          float* final_cell, cublasHandle_t handle, cudaStream_t stream,
                          float* workspace, LstmFloatCudaTrace* trace) {
    validateLstmFloatArguments(shape, weights, input, initial_hidden, initial_cell, output,
                               final_hidden, final_cell);
    if (handle == nullptr) {
        throw std::invalid_argument("cuBLAS handle 不能为空");
    }
    if (trace != nullptr && (trace->gate_outputs == nullptr || trace->cell_states == nullptr ||
                             trace->cell_tanh_outputs == nullptr)) {
        throw std::invalid_argument("CUDA float recurrent trace 指针必须全部提供");
    }
    if (trace != nullptr) {
        const bool has_calibration_trace = trace->gate_inputs != nullptr;
        if (has_calibration_trace != (trace->weight_input_hidden_linear != nullptr) ||
            has_calibration_trace != (trace->weight_hidden_hidden_linear != nullptr)) {
            throw std::invalid_argument("CUDA float calibration trace 指针必须全部提供或全部省略");
        }
    }

    const int sequence_length =
        cuda_detail::checkedCublasInt(shape.sequence_length, "sequence_length");
    const int batch_size = cuda_detail::checkedCublasInt(shape.batch_size, "batch_size");
    const int input_size = cuda_detail::checkedCublasInt(shape.input_size, "input_size");
    const int hidden_size = cuda_detail::checkedCublasInt(shape.hidden_size, "hidden_size");
    if (shape.batch_size > std::numeric_limits<int>::max() / shape.sequence_length) {
        throw std::invalid_argument("sequence_length * batch_size 超出 cuBLAS int 范围");
    }
    const int sequence_batch = sequence_length * batch_size;
    if (hidden_size > std::numeric_limits<int>::max() / 4) {
        throw std::invalid_argument("4 * hidden_size 超出 cuBLAS int 范围");
    }
    const int gate_size = 4 * hidden_size;
    const std::int64_t state_elements = shape.batch_size * shape.hidden_size;

    cuda_detail::OwnedWorkspace owned_workspace(
        workspace == nullptr ? cudaWorkspaceElementCount(shape) : 0, stream);
    if (workspace == nullptr) {
        workspace = owned_workspace.get();
    }
    float* input_linear = workspace;
    float* recurrent_linear = input_linear + static_cast<std::int64_t>(sequence_batch) * gate_size;

    if (initial_hidden == nullptr) {
        cuda_detail::checkCuda(
            cudaMemsetAsync(final_hidden, 0,
                            static_cast<std::size_t>(state_elements) * sizeof(float), stream),
            "cudaMemsetAsync h_0");
        cuda_detail::checkCuda(
            cudaMemsetAsync(final_cell, 0, static_cast<std::size_t>(state_elements) * sizeof(float),
                            stream),
            "cudaMemsetAsync c_0");
    } else {
        if (initial_hidden != final_hidden) {
            cuda_detail::checkCuda(
                cudaMemcpyAsync(final_hidden, initial_hidden,
                                static_cast<std::size_t>(state_elements) * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream),
                "cudaMemcpyAsync h_0");
        }
        if (initial_cell != final_cell) {
            cuda_detail::checkCuda(
                cudaMemcpyAsync(final_cell, initial_cell,
                                static_cast<std::size_t>(state_elements) * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream),
                "cudaMemcpyAsync c_0");
        }
    }

    cuda_detail::ScopedCublasSettings settings(handle, stream,
                                               cuda_detail::CublasMathMode::Pedantic);
    cuda_detail::runGemm(handle, sequence_batch, gate_size, input_size, input, weights.weight_ih,
                         input_linear);

    constexpr int threads = 256;
    constexpr std::int64_t max_blocks = 65535;
    const auto blocks =
        static_cast<unsigned int>(std::min(max_blocks, (state_elements + threads - 1) / threads));
    for (int time = 0; time < sequence_length; ++time) {
        float* gate_outputs =
            trace == nullptr
                ? nullptr
                : trace->gate_outputs + static_cast<std::int64_t>(time) * batch_size * gate_size;
        float* cell_states =
            trace == nullptr
                ? nullptr
                : trace->cell_states + static_cast<std::int64_t>(time) * state_elements;
        float* cell_tanh_outputs =
            trace == nullptr
                ? nullptr
                : trace->cell_tanh_outputs + static_cast<std::int64_t>(time) * state_elements;
        float* traced_input_linear =
            trace == nullptr || trace->gate_inputs == nullptr
                ? nullptr
                : trace->weight_input_hidden_linear +
                      static_cast<std::int64_t>(time) * batch_size * gate_size;
        float* traced_recurrent_linear =
            trace == nullptr || trace->gate_inputs == nullptr
                ? nullptr
                : trace->weight_hidden_hidden_linear +
                      static_cast<std::int64_t>(time) * batch_size * gate_size;
        float* gate_inputs =
            trace == nullptr || trace->gate_inputs == nullptr
                ? nullptr
                : trace->gate_inputs + static_cast<std::int64_t>(time) * batch_size * gate_size;
        cuda_detail::runGemm(handle, batch_size, gate_size, hidden_size, final_hidden,
                             weights.weight_hh, recurrent_linear);
        updateLstmState<<<blocks, threads, 0, stream>>>(
            shape.batch_size, shape.hidden_size,
            input_linear + static_cast<std::int64_t>(time) * batch_size * gate_size,
            recurrent_linear, weights.bias_ih, weights.bias_hh, final_hidden, final_cell,
            output + static_cast<std::int64_t>(time) * state_elements, gate_outputs, cell_states,
            cell_tanh_outputs, traced_input_linear, traced_recurrent_linear, gate_inputs);
        cuda_detail::checkCuda(cudaGetLastError(), "updateLstmState kernel");
    }
    settings.restore();
    owned_workspace.release();
}

}  // namespace quant_lstm
