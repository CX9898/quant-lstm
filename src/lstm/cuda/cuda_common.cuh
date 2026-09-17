#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

// CUDA LSTM 后端共用的资源管理与 row-major SGEMM 适配。
namespace quant_lstm::cuda_detail {

inline void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

inline void checkCublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " 失败，cuBLAS status=" +
            std::to_string(static_cast<int>(status)));
    }
}

// 该枚举要求调用点明确选择正确性或性能语义，禁止依赖 handle 的遗留状态。
enum class CublasMathMode : std::uint8_t {
    Pedantic,
    Tf32,
};

inline cublasMath_t toCublasMathMode(CublasMathMode mode) {
    switch (mode) {
        case CublasMathMode::Pedantic:
            return CUBLAS_PEDANTIC_MATH;
        case CublasMathMode::Tf32:
            return CUBLAS_TF32_TENSOR_OP_MATH;
    }
    throw std::invalid_argument("CublasMathMode 枚举值非法");
}

// 临时覆盖 stream、math mode 与 pointer mode，并在离开作用域时恢复调用方状态。
class ScopedCublasSettings {
   public:
    ScopedCublasSettings(cublasHandle_t handle, cudaStream_t stream,
                         CublasMathMode math_mode)
        : handle_(handle) {
        if (handle_ == nullptr) {
            throw std::invalid_argument("cuBLAS handle 不能为空");
        }
        checkCublas(cublasGetStream(handle_, &old_stream_), "cublasGetStream");
        checkCublas(cublasGetMathMode(handle_, &old_math_mode_),
                    "cublasGetMathMode");
        checkCublas(cublasGetPointerMode(handle_, &old_pointer_mode_),
                    "cublasGetPointerMode");
        try {
            checkCublas(cublasSetStream(handle_, stream), "cublasSetStream");
            stream_changed_ = true;
            checkCublas(cublasSetMathMode(handle_, toCublasMathMode(math_mode)),
                        "cublasSetMathMode");
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

    ScopedCublasSettings(const ScopedCublasSettings&) = delete;
    ScopedCublasSettings& operator=(const ScopedCublasSettings&) = delete;

    void restore() {
        if (pointer_mode_changed_) {
            checkCublas(cublasSetPointerMode(handle_, old_pointer_mode_),
                        "恢复 cublas pointer mode");
            pointer_mode_changed_ = false;
        }
        if (math_changed_) {
            checkCublas(cublasSetMathMode(handle_, old_math_mode_),
                        "恢复 cublas math mode");
            math_changed_ = false;
        }
        if (stream_changed_) {
            checkCublas(cublasSetStream(handle_, old_stream_),
                        "恢复 cublas stream");
            stream_changed_ = false;
        }
    }

   private:
    void restoreNoThrow() noexcept {
        if (pointer_mode_changed_ &&
            cublasSetPointerMode(handle_, old_pointer_mode_) ==
                CUBLAS_STATUS_SUCCESS) {
            pointer_mode_changed_ = false;
        }
        if (math_changed_ &&
            cublasSetMathMode(handle_, old_math_mode_) ==
                CUBLAS_STATUS_SUCCESS) {
            math_changed_ = false;
        }
        if (stream_changed_ &&
            cublasSetStream(handle_, old_stream_) == CUBLAS_STATUS_SUCCESS) {
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

// 以 float 元素数分配异步 workspace；析构只做不抛异常的兜底释放。
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
        checkCuda(cudaMallocAsync(
                      reinterpret_cast<void**>(&data_),
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

// 两个 stream 共用的字节 workspace；异常析构也先建立 secondary->primary
// 依赖，避免异步释放早于 secondary 上已提交的工作。
class OwnedMultiStreamWorkspace {
   public:
    OwnedMultiStreamWorkspace(std::size_t bytes, cudaStream_t primary,
                              cudaStream_t secondary,
                              cudaEvent_t join_event)
        : primary_(primary),
          secondary_(secondary),
          join_event_(join_event) {
        if (bytes != 0) {
            checkCuda(cudaMallocAsync(&data_, bytes, primary_),
                      "cudaMallocAsync multi-stream workspace");
        }
    }

    ~OwnedMultiStreamWorkspace() {
        if (data_ != nullptr &&
            cudaEventRecord(join_event_, secondary_) == cudaSuccess &&
            cudaStreamWaitEvent(primary_, join_event_, 0) == cudaSuccess) {
            cudaFreeAsync(data_, primary_);
        }
    }

    OwnedMultiStreamWorkspace(const OwnedMultiStreamWorkspace&) = delete;
    OwnedMultiStreamWorkspace& operator=(
        const OwnedMultiStreamWorkspace&) = delete;

    void* get() const noexcept { return data_; }

    void release() {
        if (data_ != nullptr) {
            checkCuda(cudaFreeAsync(data_, primary_),
                      "cudaFreeAsync multi-stream workspace");
            data_ = nullptr;
        }
    }

   private:
    cudaStream_t primary_ = nullptr;
    cudaStream_t secondary_ = nullptr;
    cudaEvent_t join_event_ = nullptr;
    void* data_ = nullptr;
};

inline int checkedCublasInt(std::int64_t value, const char* name) {
    if (value <= 0 || value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string(name) +
                                    " 超出 cuBLAS int 范围");
    }
    return static_cast<int>(value);
}

// row-major C[rows, columns] = A[rows, reduction] * B[columns, reduction]^T。
inline void runGemm(cublasHandle_t handle, int rows, int columns, int reduction,
                    const float* row_major_left,
                    const float* row_major_right, float* row_major_output) {
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    checkCublas(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, columns, rows,
                            reduction, &alpha, row_major_right, reduction,
                            row_major_left, reduction, &beta, row_major_output,
                            columns),
                "cublasSgemm");
}

}  // namespace quant_lstm::cuda_detail
