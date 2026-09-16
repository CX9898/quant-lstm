#include "quantization/rounding.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdlib>
#include <iostream>

namespace {

__global__ void roundValues(const float* input, float* output, int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count) {
        output[index] = quant_lstm::quantization::roundToNearestEven(input[index]);
    }
}

}  // namespace

int main() {
    constexpr std::array<float, 10> input{
        -3.5F, -2.5F, -1.5F, -0.5F, 0.5F, 1.5F, 2.5F, 3.5F, 4.49F, 4.51F};
    constexpr std::array<float, 10> expected{
        -4.0F, -2.0F, -2.0F, 0.0F, 0.0F, 2.0F, 2.0F, 4.0F, 4.0F, 5.0F};
    float* device_input = nullptr;
    float* device_output = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_input), sizeof(input)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&device_output), sizeof(input)) != cudaSuccess ||
        cudaMemcpy(device_input, input.data(), sizeof(input), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        std::cerr << "CUDA test allocation/copy failed\n";
        cudaFree(device_input);
        cudaFree(device_output);
        return EXIT_FAILURE;
    }
    roundValues<<<1, 32>>>(device_input, device_output, static_cast<int>(input.size()));
    std::array<float, input.size()> actual{};
    const cudaError_t status =
        cudaMemcpy(actual.data(), device_output, sizeof(actual), cudaMemcpyDeviceToHost);
    cudaFree(device_input);
    cudaFree(device_output);
    if (status != cudaSuccess || actual != expected) {
        std::cerr << "CUDA round-to-nearest-even mismatch\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
