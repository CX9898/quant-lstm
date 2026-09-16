#include "lstm/forward_float_cuda.h"

#include "lstm/gate_layout.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

// CUDA 基准将两路 Linear 交给 cuBLAS，逐时间步 kernel 完成 (i,f,g,o) 与状态更新。
namespace quant_lstm {
namespace {

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void checkCublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " 失败，cuBLAS status=" +
                                 std::to_string(static_cast<int>(status)));
    }
}

class OwnedWorkspace {
   public:
    OwnedWorkspace(std::int64_t elements, cudaStream_t stream) : stream_(stream) {
        if (elements == 0) {
            return;
        }
        if (elements < 0 ||
            static_cast<std::uint64_t>(elements) >
                std::numeric_limits<std::size_t>::max() / sizeof(float)) {
            throw std::invalid_argument("CUDA workspace 字节数溢出");
        }
        checkCuda(cudaMallocAsync(reinterpret_cast<void**>(&data_),
                                  static_cast<std::size_t>(elements) * sizeof(float), stream_),
                  "cudaMallocAsync workspace");
    }

    ~OwnedWorkspace() {
        if (data_ != nullptr) {
            cudaFreeAsync(data_, stream_);
        }
    }

    OwnedWorkspace(const OwnedWorkspace&) = delete;
    OwnedWorkspace& operator=(const OwnedWorkspace&) = delete;

    float* get() const noexcept { return data_; }
    void release() {
        if (data_ != nullptr) {
            checkCuda(cudaFreeAsync(data_, stream_), "cudaFreeAsync workspace");
            data_ = nullptr;
        }
    }

   private:
    float* data_ = nullptr;
    cudaStream_t stream_ = nullptr;
};

class ScopedCublasSettings {
   public:
    ScopedCublasSettings(cublasHandle_t handle, cudaStream_t stream) : handle_(handle) {
        checkCublas(cublasGetStream(handle_, &old_stream_), "cublasGetStream");
        checkCublas(cublasGetMathMode(handle_, &old_math_mode_), "cublasGetMathMode");
        checkCublas(cublasGetPointerMode(handle_, &old_pointer_mode_), "cublasGetPointerMode");
        try {
            checkCublas(cublasSetStream(handle_, stream), "cublasSetStream");
            stream_changed_ = true;
            checkCublas(cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode");
            math_changed_ = true;
            checkCublas(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST),
                        "cublasSetPointerMode");
            pointer_mode_changed_ = true;
        } catch (...) {
            restoreNoThrow();
            throw;
        }
    }

    ~ScopedCublasSettings() { restoreNoThrow(); }

    void restore() {
        if (pointer_mode_changed_) {
            checkCublas(cublasSetPointerMode(handle_, old_pointer_mode_),
                        "恢复 cublas pointer mode");
            pointer_mode_changed_ = false;
        }
        if (math_changed_) {
            checkCublas(cublasSetMathMode(handle_, old_math_mode_), "恢复 cublas math mode");
            math_changed_ = false;
        }
        if (stream_changed_) {
            checkCublas(cublasSetStream(handle_, old_stream_), "恢复 cublas stream");
            stream_changed_ = false;
        }
    }

    ScopedCublasSettings(const ScopedCublasSettings&) = delete;
    ScopedCublasSettings& operator=(const ScopedCublasSettings&) = delete;

   private:
    void restoreNoThrow() noexcept {
        if (pointer_mode_changed_ &&
            cublasSetPointerMode(handle_, old_pointer_mode_) == CUBLAS_STATUS_SUCCESS) {
            pointer_mode_changed_ = false;
        }
        if (math_changed_ &&
            cublasSetMathMode(handle_, old_math_mode_) == CUBLAS_STATUS_SUCCESS) {
            math_changed_ = false;
        }
        if (stream_changed_ && cublasSetStream(handle_, old_stream_) == CUBLAS_STATUS_SUCCESS) {
            stream_changed_ = false;
        }
    }

    cublasHandle_t handle_;
    cudaStream_t old_stream_ = nullptr;
    cublasMath_t old_math_mode_ = CUBLAS_DEFAULT_MATH;
    cublasPointerMode_t old_pointer_mode_ = CUBLAS_POINTER_MODE_HOST;
    bool stream_changed_ = false;
    bool math_changed_ = false;
    bool pointer_mode_changed_ = false;
};

__device__ float sigmoid(float value) {
    return 1.0F / (1.0F + expf(-value));
}

__global__ void updateLstmState(std::int64_t batch_size, std::int64_t hidden_size,
                                const float* input_linear, const float* recurrent_linear,
                                const float* bias_ih, const float* bias_hh, float* hidden,
                                float* cell, float* output) {
    const std::int64_t count = batch_size * hidden_size;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t element =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         element < count; element += stride) {
        const std::int64_t batch = element / hidden_size;
        const std::int64_t hidden_index = element % hidden_size;
        const std::int64_t gate_stride = 4 * hidden_size;
        const std::int64_t base = batch * gate_stride + hidden_index;

        float gate_values[4];
#pragma unroll
        for (int gate = 0; gate < 4; ++gate) {
            const std::int64_t gate_index =
                base + static_cast<std::int64_t>(gate) * hidden_size;
            float value = input_linear[gate_index] + recurrent_linear[gate_index];
            if (bias_ih != nullptr) {
                const std::int64_t bias_index =
                    static_cast<std::int64_t>(gate) * hidden_size + hidden_index;
                value += bias_ih[bias_index] + bias_hh[bias_index];
            }
            gate_values[gate] = value;
        }

        const float input_gate = sigmoid(gate_values[0]);
        const float forget_gate = sigmoid(gate_values[1]);
        const float cell_gate = tanhf(gate_values[2]);
        const float output_gate = sigmoid(gate_values[3]);
        const float next_cell = forget_gate * cell[element] + input_gate * cell_gate;
        const float next_hidden = output_gate * tanhf(next_cell);
        cell[element] = next_cell;
        hidden[element] = next_hidden;
        output[element] = next_hidden;
    }
}

void runGemm(cublasHandle_t handle, int rows, int columns, int reduction,
             const float* row_major_left, const float* row_major_right,
             float* row_major_output) {
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    // row-major C = A * B^T 等价为 column-major C^T = B * A^T。
    checkCublas(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, columns, rows, reduction, &alpha,
                            row_major_right, reduction, row_major_left, reduction, &beta,
                            row_major_output, columns),
                "cublasSgemm");
}

int checkedInt(std::int64_t value, const char* name) {
    if (value <= 0 || value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string(name) + " 超出 cuBLAS int 范围");
    }
    return static_cast<int>(value);
}

}  // namespace

std::int64_t cudaWorkspaceElementCount(const LstmShape& shape) {
    if (shape.sequence_length <= 0 || shape.batch_size <= 0 || shape.hidden_size <= 0) {
        throw std::invalid_argument("LSTM shape 必须为正数");
    }
    constexpr std::int64_t gate_count = static_cast<std::int64_t>(kGateCount);
    if (shape.sequence_length >
        std::numeric_limits<std::int64_t>::max() / shape.batch_size / gate_count /
            shape.hidden_size) {
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
                          float* workspace) {
    validateLstmFloatArguments(shape, weights, input, initial_hidden, initial_cell, output,
                               final_hidden, final_cell);
    if (handle == nullptr) {
        throw std::invalid_argument("cuBLAS handle 不能为空");
    }

    const int sequence_length = checkedInt(shape.sequence_length, "sequence_length");
    const int batch_size = checkedInt(shape.batch_size, "batch_size");
    const int input_size = checkedInt(shape.input_size, "input_size");
    const int hidden_size = checkedInt(shape.hidden_size, "hidden_size");
    if (shape.batch_size > std::numeric_limits<int>::max() / shape.sequence_length) {
        throw std::invalid_argument("sequence_length * batch_size 超出 cuBLAS int 范围");
    }
    const int sequence_batch = sequence_length * batch_size;
    if (hidden_size > std::numeric_limits<int>::max() / 4) {
        throw std::invalid_argument("4 * hidden_size 超出 cuBLAS int 范围");
    }
    const int gate_size = 4 * hidden_size;
    const std::int64_t state_elements = shape.batch_size * shape.hidden_size;

    OwnedWorkspace owned_workspace(workspace == nullptr ? cudaWorkspaceElementCount(shape) : 0,
                                   stream);
    if (workspace == nullptr) {
        workspace = owned_workspace.get();
    }
    float* input_linear = workspace;
    float* recurrent_linear =
        input_linear + static_cast<std::int64_t>(sequence_batch) * gate_size;

    if (initial_hidden == nullptr) {
        checkCuda(cudaMemsetAsync(final_hidden, 0,
                                  static_cast<std::size_t>(state_elements) * sizeof(float), stream),
                  "cudaMemsetAsync h_0");
        checkCuda(cudaMemsetAsync(final_cell, 0,
                                  static_cast<std::size_t>(state_elements) * sizeof(float), stream),
                  "cudaMemsetAsync c_0");
    } else {
        if (initial_hidden != final_hidden) {
            checkCuda(cudaMemcpyAsync(final_hidden, initial_hidden,
                                      static_cast<std::size_t>(state_elements) * sizeof(float),
                                      cudaMemcpyDeviceToDevice, stream),
                      "cudaMemcpyAsync h_0");
        }
        if (initial_cell != final_cell) {
            checkCuda(cudaMemcpyAsync(final_cell, initial_cell,
                                      static_cast<std::size_t>(state_elements) * sizeof(float),
                                      cudaMemcpyDeviceToDevice, stream),
                      "cudaMemcpyAsync c_0");
        }
    }

    ScopedCublasSettings settings(handle, stream);
    runGemm(handle, sequence_batch, gate_size, input_size, input, weights.weight_ih,
            input_linear);

    constexpr int threads = 256;
    constexpr std::int64_t max_blocks = 65535;
    const auto blocks = static_cast<unsigned int>(
        std::min(max_blocks, (state_elements + threads - 1) / threads));
    for (int time = 0; time < sequence_length; ++time) {
        runGemm(handle, batch_size, gate_size, hidden_size, final_hidden, weights.weight_hh,
                recurrent_linear);
        updateLstmState<<<blocks, threads, 0, stream>>>(
            shape.batch_size, shape.hidden_size,
            input_linear + static_cast<std::int64_t>(time) * batch_size * gate_size,
            recurrent_linear, weights.bias_ih, weights.bias_hh, final_hidden, final_cell,
            output + static_cast<std::int64_t>(time) * state_elements);
        checkCuda(cudaGetLastError(), "updateLstmState kernel");
    }
    settings.restore();
    owned_workspace.release();
}

}  // namespace quant_lstm
