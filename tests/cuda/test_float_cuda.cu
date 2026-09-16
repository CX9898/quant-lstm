#include "lstm/forward_float.h"
#include "lstm/forward_float_cuda.h"
#include "common/deterministic_rng.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
   public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)), "cudaMalloc");
    }
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return data_; }
    void copyFrom(const std::vector<T>& source) {
        if (source.size() != count_) {
            throw std::invalid_argument("host/device 元素数量不匹配");
        }
        checkCuda(cudaMemcpy(data_, source.data(), count_ * sizeof(T), cudaMemcpyHostToDevice),
                  "cudaMemcpy H2D");
    }
    std::vector<T> copyToHost() const {
        std::vector<T> result(count_);
        checkCuda(cudaMemcpy(result.data(), data_, count_ * sizeof(T), cudaMemcpyDeviceToHost),
                  "cudaMemcpy D2H");
        return result;
    }

   private:
    T* data_ = nullptr;
    std::size_t count_;
};

}  // namespace

int main() {
    try {
        constexpr quant_lstm::LstmShape shape{3, 2, 5, 7};
        const std::size_t input_count = shape.sequence_length * shape.batch_size * shape.input_size;
        const std::size_t state_count = shape.batch_size * shape.hidden_size;
        const std::size_t weight_ih_count = 4 * shape.hidden_size * shape.input_size;
        const std::size_t weight_hh_count = 4 * shape.hidden_size * shape.hidden_size;
        const std::size_t bias_count = 4 * shape.hidden_size;
        const std::size_t output_count = shape.sequence_length * state_count;

        std::vector<float> input(input_count);
        std::vector<float> initial_hidden(state_count);
        std::vector<float> initial_cell(state_count);
        std::vector<float> weight_ih(weight_ih_count);
        std::vector<float> weight_hh(weight_hh_count);
        std::vector<float> bias_ih(bias_count);
        std::vector<float> bias_hh(bias_count);
        using quant_lstm::test::TensorStream;
        quant_lstm::test::fillNormalLike(input.data(), input.size(), 0, TensorStream::Input);
        quant_lstm::test::fillNormalLike(initial_hidden.data(), initial_hidden.size(), 0,
                                         TensorStream::InitialHidden);
        quant_lstm::test::fillNormalLike(initial_cell.data(), initial_cell.size(), 0,
                                         TensorStream::InitialCell);
        quant_lstm::test::fillLstmParameter(weight_ih.data(), weight_ih.size(), shape.hidden_size,
                                             3004, TensorStream::WeightInputHidden);
        quant_lstm::test::fillLstmParameter(weight_hh.data(), weight_hh.size(), shape.hidden_size,
                                             3004, TensorStream::WeightHiddenHidden);
        quant_lstm::test::fillLstmParameter(bias_ih.data(), bias_ih.size(), shape.hidden_size, 3004,
                                             TensorStream::BiasInputHidden);
        quant_lstm::test::fillLstmParameter(bias_hh.data(), bias_hh.size(), shape.hidden_size, 3004,
                                             TensorStream::BiasHiddenHidden);

        std::vector<float> expected_output(output_count);
        std::vector<float> expected_hidden(state_count);
        std::vector<float> expected_cell(state_count);
        const quant_lstm::LstmFloatWeights cpu_weights{
            weight_ih.data(), weight_hh.data(), bias_ih.data(), bias_hh.data()};
        quant_lstm::lstmForwardFloatCpu(shape, cpu_weights, input.data(), initial_hidden.data(),
                                        initial_cell.data(), expected_output.data(),
                                        expected_hidden.data(), expected_cell.data());

        DeviceBuffer<float> device_input(input_count);
        DeviceBuffer<float> device_initial_hidden(state_count);
        DeviceBuffer<float> device_initial_cell(state_count);
        DeviceBuffer<float> device_weight_ih(weight_ih_count);
        DeviceBuffer<float> device_weight_hh(weight_hh_count);
        DeviceBuffer<float> device_bias_ih(bias_count);
        DeviceBuffer<float> device_bias_hh(bias_count);
        DeviceBuffer<float> device_output(output_count);
        DeviceBuffer<float> device_hidden(state_count);
        DeviceBuffer<float> device_cell(state_count);
        device_input.copyFrom(input);
        device_initial_hidden.copyFrom(initial_hidden);
        device_initial_cell.copyFrom(initial_cell);
        device_weight_ih.copyFrom(weight_ih);
        device_weight_hh.copyFrom(weight_hh);
        device_bias_ih.copyFrom(bias_ih);
        device_bias_hh.copyFrom(bias_hh);

        cublasHandle_t handle = nullptr;
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasCreate 失败");
        }
        const quant_lstm::LstmFloatWeights cuda_weights{device_weight_ih.get(),
                                                         device_weight_hh.get(),
                                                         device_bias_ih.get(),
                                                         device_bias_hh.get()};
        quant_lstm::lstmForwardFloatCuda(
            shape, cuda_weights, device_input.get(), device_initial_hidden.get(),
            device_initial_cell.get(), device_output.get(), device_hidden.get(), device_cell.get(),
            handle, nullptr);
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        cublasDestroy(handle);

        const auto actual_output = device_output.copyToHost();
        const auto actual_hidden = device_hidden.copyToHost();
        const auto actual_cell = device_cell.copyToHost();
        const auto check = [](const std::vector<float>& actual, const std::vector<float>& expected) {
            for (std::size_t index = 0; index < actual.size(); ++index) {
                if (std::abs(actual[index] - expected[index]) > 1.0e-5F) {
                    return false;
                }
            }
            return true;
        };
        if (!check(actual_output, expected_output) || !check(actual_hidden, expected_hidden) ||
            !check(actual_cell, expected_cell)) {
            std::cerr << "CUDA 与 CPU FP32 reference 不一致\n";
            return EXIT_FAILURE;
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
