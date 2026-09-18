#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "cuda/cuda_common.cuh"
#include "lstm/backward_float_cuda.h"
#include "lstm/gate_layout.h"

namespace quant_lstm {
namespace {

std::int64_t checkedProduct(std::initializer_list<std::int64_t> factors, const char* name) {
    std::int64_t result = 1;
    for (const std::int64_t factor : factors) {
        if (factor <= 0 || result > std::numeric_limits<std::int64_t>::max() / factor) {
            throw std::invalid_argument(std::string(name) + " 元素数量溢出");
        }
        result *= factor;
    }
    return result;
}

void validateBackwardArguments(const LstmShape& shape, const LstmFloatWeights& weights,
                               const float* input, const float* initial_hidden,
                               const float* initial_cell, const LstmFloatCudaBackwardTrace& trace,
                               const float* grad_output, const float* grad_final_hidden,
                               const float* grad_final_cell,
                               const LstmFloatCudaGradients& gradients, cublasHandle_t handle) {
    checkedProduct({shape.sequence_length, shape.batch_size, shape.input_size, shape.hidden_size},
                   "LSTM backward shape");
    if (handle == nullptr) {
        throw std::invalid_argument("cuBLAS handle 不能为空");
    }
    if (weights.weight_ih == nullptr || weights.weight_hh == nullptr || input == nullptr ||
        trace.gate_outputs == nullptr || trace.cell_states == nullptr ||
        trace.cell_tanh_outputs == nullptr || trace.hidden_outputs == nullptr ||
        grad_output == nullptr || grad_final_hidden == nullptr || grad_final_cell == nullptr ||
        gradients.input == nullptr || gradients.weight_ih == nullptr ||
        gradients.weight_hh == nullptr || gradients.bias_ih == nullptr ||
        gradients.bias_hh == nullptr || gradients.initial_hidden == nullptr ||
        gradients.initial_cell == nullptr) {
        throw std::invalid_argument("LSTM backward 必需张量指针不能为空");
    }
    if ((initial_hidden == nullptr) != (initial_cell == nullptr)) {
        throw std::invalid_argument("h_0 和 c_0 必须同时提供或同时省略");
    }
}

__global__ void lstmBackwardPointwise(
    std::int64_t batch_size, std::int64_t hidden_size, const float* gate_outputs,
    const float* cell_tanh_outputs, const float* previous_cell, const float* grad_output,
    float* grad_hidden, float* grad_cell, float* grad_input_linear, float* grad_recurrent_linear,
    const std::uint8_t* weight_ih_linear_mask, const std::uint8_t* weight_hh_linear_mask,
    const std::uint8_t* gate_input_mask, const std::uint8_t* gate_output_mask,
    const std::uint8_t* cell_state_mask, const std::uint8_t* cell_tanh_output_mask,
    const std::uint8_t* hidden_output_mask) {
    const std::int64_t count = batch_size * hidden_size;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t element = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         element < count; element += stride) {
        const std::int64_t batch = element / hidden_size;
        const std::int64_t hidden = element % hidden_size;
        const std::int64_t gate_base = batch * 4 * hidden_size + hidden;
        const float input_gate = gate_outputs[gate_base];
        const float forget_gate = gate_outputs[gate_base + hidden_size];
        const float cell_gate = gate_outputs[gate_base + 2 * hidden_size];
        const float output_gate = gate_outputs[gate_base + 3 * hidden_size];
        const float cell_tanh = cell_tanh_outputs[element];

        const auto keep = [](const std::uint8_t* mask, std::int64_t index) -> float {
            return mask == nullptr || mask[index] == 0 ? 1.0F : 0.0F;
        };
        const float dh =
            (grad_hidden[element] + grad_output[element]) * keep(hidden_output_mask, element);
        const float grad_output_gate = dh * cell_tanh;
        const float grad_cell_tanh = dh * output_gate * keep(cell_tanh_output_mask, element);
        const float dc = grad_cell[element] + grad_cell_tanh * (1.0F - cell_tanh * cell_tanh);
        const float masked_dc = dc * keep(cell_state_mask, element);
        const float grad_forget_gate =
            masked_dc * (previous_cell == nullptr ? 0.0F : previous_cell[element]);
        const float grad_input_gate = masked_dc * cell_gate;
        const float grad_cell_gate = masked_dc * input_gate;
        const float gate_output_gradients[4]{
            grad_input_gate * keep(gate_output_mask, gate_base),
            grad_forget_gate * keep(gate_output_mask, gate_base + hidden_size),
            grad_cell_gate * keep(gate_output_mask, gate_base + 2 * hidden_size),
            grad_output_gate * keep(gate_output_mask, gate_base + 3 * hidden_size)};
        const float gate_input_gradients[4]{
            gate_output_gradients[0] * input_gate * (1.0F - input_gate),
            gate_output_gradients[1] * forget_gate * (1.0F - forget_gate),
            gate_output_gradients[2] * (1.0F - cell_gate * cell_gate),
            gate_output_gradients[3] * output_gate * (1.0F - output_gate)};
#pragma unroll
        for (int gate = 0; gate < 4; ++gate) {
            const std::int64_t gate_index =
                gate_base + static_cast<std::int64_t>(gate) * hidden_size;
            const float gradient = gate_input_gradients[gate] * keep(gate_input_mask, gate_index);
            grad_input_linear[gate_index] = gradient * keep(weight_ih_linear_mask, gate_index);
            grad_recurrent_linear[gate_index] = gradient * keep(weight_hh_linear_mask, gate_index);
        }
        grad_cell[element] = masked_dc * forget_gate;
    }
}

__global__ void reduceBiasGradients(std::int64_t rows, std::int64_t columns,
                                    const float* grad_input_linear,
                                    const float* grad_recurrent_linear, float* grad_bias_ih,
                                    float* grad_bias_hh) {
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t column = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         column < columns; column += stride) {
        float input_sum = 0.0F;
        float recurrent_sum = 0.0F;
        for (std::int64_t row = 0; row < rows; ++row) {
            input_sum += grad_input_linear[row * columns + column];
            recurrent_sum += grad_recurrent_linear[row * columns + column];
        }
        grad_bias_ih[column] = input_sum;
        grad_bias_hh[column] = recurrent_sum;
    }
}

__global__ void applyClampedMask(float* values, const std::uint8_t* mask, std::int64_t count) {
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        if (mask[index] != 0) {
            values[index] = 0.0F;
        }
    }
}

void launchClampedMask(float* values, const std::uint8_t* mask, std::int64_t count,
                       cudaStream_t stream, const char* name) {
    if (mask == nullptr) {
        return;
    }
    constexpr int threads = 256;
    constexpr std::int64_t max_blocks = 65535;
    const auto blocks =
        static_cast<unsigned int>(std::min(max_blocks, (count + threads - 1) / threads));
    applyClampedMask<<<blocks, threads, 0, stream>>>(values, mask, count);
    cuda_detail::checkCuda(cudaGetLastError(), name);
}

// row-major C[M,N] = A[M,K] * B[K,N].
void runRightGemm(cublasHandle_t handle, int rows, int columns, int reduction, const float* left,
                  const float* right, float* output) {
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    cuda_detail::checkCublas(
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, columns, rows, reduction, &alpha, right,
                    columns, left, reduction, &beta, output, columns),
        "cublasSgemm backward input");
}

// row-major C[K,N] = A[M,K]^T * B[M,N].
void runWeightGradientGemm(cublasHandle_t handle, int rows, int columns, int reduction,
                           const float* grad_gate_inputs, const float* activations, float* output) {
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    cuda_detail::checkCublas(
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, columns, reduction, rows, &alpha, activations,
                    columns, grad_gate_inputs, reduction, &beta, output, columns),
        "cublasSgemm backward weight");
}

}  // namespace

std::int64_t cudaBackwardWorkspaceElementCount(const LstmShape& shape,
                                               bool split_linear_gradients) {
    const std::int64_t gate_gradients =
        checkedProduct({shape.sequence_length, shape.batch_size, 4, shape.hidden_size},
                       "CUDA backward gate workspace");
    const std::int64_t previous_hidden =
        checkedProduct({shape.sequence_length, shape.batch_size, shape.hidden_size},
                       "CUDA backward hidden workspace");
    const std::int64_t linear_gradients =
        split_linear_gradients
            ? checkedProduct({2, gate_gradients}, "CUDA backward split gate workspace")
            : gate_gradients;
    if (linear_gradients > std::numeric_limits<std::int64_t>::max() - previous_hidden) {
        throw std::invalid_argument("CUDA backward workspace 元素数量溢出");
    }
    return linear_gradients + previous_hidden;
}

void lstmBackwardFloatCuda(const LstmShape& shape, const LstmFloatWeights& weights,
                           const float* input, const float* initial_hidden,
                           const float* initial_cell, const LstmFloatCudaBackwardTrace& trace,
                           const float* grad_output, const float* grad_final_hidden,
                           const float* grad_final_cell, const LstmFloatCudaGradients& gradients,
                           cublasHandle_t handle, cudaStream_t stream, float* workspace,
                           const LstmFloatCudaBackwardMasks* masks) {
    validateBackwardArguments(shape, weights, input, initial_hidden, initial_cell, trace,
                              grad_output, grad_final_hidden, grad_final_cell, gradients, handle);

    const int steps = cuda_detail::checkedCublasInt(shape.sequence_length, "sequence_length");
    const int batch = cuda_detail::checkedCublasInt(shape.batch_size, "batch_size");
    const int input_size = cuda_detail::checkedCublasInt(shape.input_size, "input_size");
    const int hidden = cuda_detail::checkedCublasInt(shape.hidden_size, "hidden_size");
    if (hidden > std::numeric_limits<int>::max() / 4 ||
        batch > std::numeric_limits<int>::max() / steps) {
        throw std::invalid_argument("CUDA backward shape 超出 cuBLAS int 范围");
    }
    const int gates = 4 * hidden;
    const int sequence_batch = steps * batch;
    const std::int64_t state_elements = shape.batch_size * shape.hidden_size;
    const std::int64_t gate_elements = static_cast<std::int64_t>(sequence_batch) * gates;

    cuda_detail::OwnedWorkspace owned_workspace(
        workspace == nullptr ? cudaBackwardWorkspaceElementCount(shape, masks != nullptr) : 0,
        stream);
    if (workspace == nullptr) {
        workspace = owned_workspace.get();
    }
    float* grad_input_linear = workspace;
    float* grad_recurrent_linear =
        masks == nullptr ? grad_input_linear : grad_input_linear + gate_elements;
    float* previous_hidden = grad_recurrent_linear + gate_elements;

    if (initial_hidden == nullptr) {
        cuda_detail::checkCuda(
            cudaMemsetAsync(previous_hidden, 0,
                            static_cast<std::size_t>(state_elements) * sizeof(float), stream),
            "cudaMemsetAsync previous h_0");
    } else {
        cuda_detail::checkCuda(
            cudaMemcpyAsync(previous_hidden, initial_hidden,
                            static_cast<std::size_t>(state_elements) * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpyAsync previous h_0");
    }
    if (steps > 1) {
        cuda_detail::checkCuda(
            cudaMemcpyAsync(previous_hidden + state_elements, trace.hidden_outputs,
                            static_cast<std::size_t>(steps - 1) *
                                static_cast<std::size_t>(state_elements) * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpyAsync previous hidden outputs");
    }
    cuda_detail::checkCuda(cudaMemcpyAsync(gradients.initial_hidden, grad_final_hidden,
                                           static_cast<std::size_t>(state_elements) * sizeof(float),
                                           cudaMemcpyDeviceToDevice, stream),
                           "cudaMemcpyAsync grad h_n");
    cuda_detail::checkCuda(cudaMemcpyAsync(gradients.initial_cell, grad_final_cell,
                                           static_cast<std::size_t>(state_elements) * sizeof(float),
                                           cudaMemcpyDeviceToDevice, stream),
                           "cudaMemcpyAsync grad c_n");

    cuda_detail::ScopedCublasSettings settings(handle, stream,
                                               cuda_detail::CublasMathMode::Pedantic);
    constexpr int threads = 256;
    constexpr std::int64_t max_blocks = 65535;
    const auto state_blocks =
        static_cast<unsigned int>(std::min(max_blocks, (state_elements + threads - 1) / threads));

    for (int time = steps - 1; time >= 0; --time) {
        const std::int64_t state_offset = static_cast<std::int64_t>(time) * state_elements;
        const std::int64_t gate_offset = static_cast<std::int64_t>(time) * batch * gates;
        const float* previous_cell =
            time == 0 ? initial_cell : trace.cell_states + state_offset - state_elements;
        lstmBackwardPointwise<<<state_blocks, threads, 0, stream>>>(
            shape.batch_size, shape.hidden_size, trace.gate_outputs + gate_offset,
            trace.cell_tanh_outputs + state_offset, previous_cell, grad_output + state_offset,
            gradients.initial_hidden, gradients.initial_cell, grad_input_linear + gate_offset,
            grad_recurrent_linear + gate_offset,
            masks == nullptr || masks->weight_ih_linear == nullptr
                ? nullptr
                : masks->weight_ih_linear + gate_offset,
            masks == nullptr || masks->weight_hh_linear == nullptr
                ? nullptr
                : masks->weight_hh_linear + gate_offset,
            masks == nullptr || masks->gate_inputs == nullptr ? nullptr
                                                              : masks->gate_inputs + gate_offset,
            masks == nullptr || masks->gate_outputs == nullptr ? nullptr
                                                               : masks->gate_outputs + gate_offset,
            masks == nullptr || masks->cell_states == nullptr ? nullptr
                                                              : masks->cell_states + state_offset,
            masks == nullptr || masks->cell_tanh_outputs == nullptr
                ? nullptr
                : masks->cell_tanh_outputs + state_offset,
            masks == nullptr || masks->hidden_outputs == nullptr
                ? nullptr
                : masks->hidden_outputs + state_offset);
        cuda_detail::checkCuda(cudaGetLastError(), "lstmBackwardPointwise kernel");
        runRightGemm(handle, batch, hidden, gates, grad_recurrent_linear + gate_offset,
                     weights.weight_hh, gradients.initial_hidden);
    }

    runRightGemm(handle, sequence_batch, input_size, gates, grad_input_linear, weights.weight_ih,
                 gradients.input);
    runWeightGradientGemm(handle, sequence_batch, input_size, gates, grad_input_linear, input,
                          gradients.weight_ih);
    runWeightGradientGemm(handle, sequence_batch, hidden, gates, grad_recurrent_linear,
                          previous_hidden, gradients.weight_hh);

    const auto bias_blocks = static_cast<unsigned int>(std::min<std::int64_t>(
        max_blocks, (static_cast<std::int64_t>(gates) + threads - 1) / threads));
    reduceBiasGradients<<<bias_blocks, threads, 0, stream>>>(
        sequence_batch, gates, grad_input_linear, grad_recurrent_linear, gradients.bias_ih,
        gradients.bias_hh);
    cuda_detail::checkCuda(cudaGetLastError(), "reduceBiasGradients kernel");

    if (masks != nullptr) {
        launchClampedMask(gradients.input, masks->input,
                          static_cast<std::int64_t>(sequence_batch) * input_size, stream,
                          "apply input clamp mask");
        launchClampedMask(gradients.weight_ih, masks->weight_ih,
                          static_cast<std::int64_t>(gates) * input_size, stream,
                          "apply weight_ih clamp mask");
        launchClampedMask(gradients.weight_hh, masks->weight_hh,
                          static_cast<std::int64_t>(gates) * hidden, stream,
                          "apply weight_hh clamp mask");
        launchClampedMask(gradients.bias_ih, masks->bias_ih, gates, stream,
                          "apply bias_ih clamp mask");
        launchClampedMask(gradients.bias_hh, masks->bias_hh, gates, stream,
                          "apply bias_hh clamp mask");
        launchClampedMask(gradients.initial_hidden, masks->initial_hidden, state_elements, stream,
                          "apply initial hidden clamp mask");
        launchClampedMask(gradients.initial_cell, masks->initial_cell, state_elements, stream,
                          "apply initial cell clamp mask");
    }

    settings.restore();
    owned_workspace.release();
}

}  // namespace quant_lstm
