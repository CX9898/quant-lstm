#pragma once

#include "lstm/forward_quantized_fp_cuda.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace quant_lstm::test {

inline void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

inline void checkCublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) +
                                 " 失败，cuBLAS status=" +
                                 std::to_string(static_cast<int>(status)));
    }
}

template <typename T>
class DeviceBuffer {
   public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count_ != 0) {
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&data_),
                                 count_ * sizeof(T)),
                      "cudaMalloc");
        }
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          count_(std::exchange(other.count_, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) {
                cudaFree(data_);
            }
            data_ = std::exchange(other.data_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }

    T* get() const noexcept { return data_; }
    std::size_t size() const noexcept { return count_; }
    std::size_t bytes() const noexcept { return count_ * sizeof(T); }

    void copyFrom(const std::vector<T>& source) {
        if (source.size() != count_) {
            throw std::invalid_argument("host/device 元素数量不匹配");
        }
        if (count_ != 0) {
            checkCuda(cudaMemcpy(data_, source.data(), bytes(),
                                 cudaMemcpyHostToDevice),
                      "cudaMemcpy H2D");
        }
    }

    std::vector<T> copyToHost() const {
        std::vector<T> result(count_);
        if (count_ != 0) {
            checkCuda(cudaMemcpy(result.data(), data_, bytes(),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy D2H");
        }
        return result;
    }

   private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

class QuantizedCudaContextOwner {
   public:
    explicit QuantizedCudaContextOwner(
        std::size_t execution_parameter_bytes = 0) {
        try {
            for (cudaStream_t& stream : context_.streams) {
                checkCuda(cudaStreamCreateWithFlags(
                              &stream, cudaStreamNonBlocking),
                          "cudaStreamCreateWithFlags");
            }
            for (cublasHandle_t& handle : context_.handles) {
                checkCublas(cublasCreate(&handle), "cublasCreate");
            }
            for (cudaEvent_t& event : context_.events) {
                checkCuda(cudaEventCreateWithFlags(
                              &event, cudaEventDisableTiming),
                          "cudaEventCreateWithFlags");
            }
            if (execution_parameter_bytes != 0) {
                checkCuda(cudaMalloc(
                              &context_.device_execution_params,
                              execution_parameter_bytes),
                          "cudaMalloc execution parameter cache");
                context_.device_execution_params_bytes =
                    execution_parameter_bytes;
            }
        } catch (...) {
            release();
            throw;
        }
    }

    ~QuantizedCudaContextOwner() { release(); }

    QuantizedCudaContextOwner(const QuantizedCudaContextOwner&) = delete;
    QuantizedCudaContextOwner& operator=(const QuantizedCudaContextOwner&) =
        delete;

    LstmQuantizedFpCudaContext& get() noexcept { return context_; }
    const LstmQuantizedFpCudaContext& get() const noexcept {
        return context_;
    }

    void synchronize() const {
        checkCuda(cudaEventSynchronize(context_.events[1]),
                  "cudaEventSynchronize forward complete");
    }

   private:
    void release() noexcept {
        if (context_.device_execution_params != nullptr) {
            cudaFree(context_.device_execution_params);
            context_.device_execution_params = nullptr;
            context_.device_execution_params_bytes = 0;
            context_.cached_execution_signature = 0;
        }
        for (cudaEvent_t& event : context_.events) {
            if (event != nullptr) {
                cudaEventDestroy(event);
                event = nullptr;
            }
        }
        for (cublasHandle_t& handle : context_.handles) {
            if (handle != nullptr) {
                cublasDestroy(handle);
                handle = nullptr;
            }
        }
        for (cudaStream_t& stream : context_.streams) {
            if (stream != nullptr) {
                cudaStreamDestroy(stream);
                stream = nullptr;
            }
        }
    }

    LstmQuantizedFpCudaContext context_{};
};

}  // namespace quant_lstm::test
