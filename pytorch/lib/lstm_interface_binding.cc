#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <array>
#include <cstdint>
#include <optional>
#include <tuple>

#include "lstm/forward_float.h"
#include "lstm/forward_float_cuda.h"

// Torch binding 仅处理张量契约和布局；LSTM 数学全部位于公共 C++/CUDA 核心。
namespace {

void checkFloatTensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.defined(), name, " 未定义");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " 必须为 float32");
    TORCH_CHECK(tensor.layout() == torch::kStrided, name, " 必须为 strided tensor");
}

void checkSameDevice(const torch::Tensor& reference, const torch::Tensor& tensor,
                     const char* name) {
    TORCH_CHECK(tensor.device() == reference.device(), name, " 与 input 不在同一设备");
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> lstmForward(
    const torch::Tensor& input, const torch::Tensor& weight_ih,
    const torch::Tensor& weight_hh, const std::optional<torch::Tensor>& bias_ih,
    const std::optional<torch::Tensor>& bias_hh, const std::optional<torch::Tensor>& initial_hidden,
    const std::optional<torch::Tensor>& initial_cell, bool batch_first) {
    checkFloatTensor(input, "input");
    checkFloatTensor(weight_ih, "weight_ih");
    checkFloatTensor(weight_hh, "weight_hh");
    TORCH_CHECK(input.dim() == 3, "input 必须是 3-D tensor");
    TORCH_CHECK(weight_ih.dim() == 2 && weight_hh.dim() == 2, "weight 必须是 2-D tensor");
    TORCH_CHECK(bias_ih.has_value() == bias_hh.has_value(),
                "bias_ih 和 bias_hh 必须同时提供或同时省略");
    TORCH_CHECK(initial_hidden.has_value() == initial_cell.has_value(),
                "h_0 和 c_0 必须同时提供或同时省略");

    const torch::Tensor time_major = batch_first ? input.transpose(0, 1).contiguous()
                                                 : input.contiguous();
    TORCH_CHECK(time_major.device().is_cpu() || time_major.is_cuda(),
                "阶段 1 仅支持 CPU 或 CUDA tensor");
    const std::int64_t sequence_length = time_major.size(0);
    const std::int64_t batch_size = time_major.size(1);
    const std::int64_t input_size = time_major.size(2);
    TORCH_CHECK(weight_ih.size(1) == input_size, "weight_ih input_size 不匹配");
    TORCH_CHECK(weight_ih.size(0) > 0 && weight_ih.size(0) % 4 == 0,
                "weight_ih 第一维必须为正的 4H");
    const std::int64_t hidden_size = weight_ih.size(0) / 4;
    TORCH_CHECK(weight_hh.sizes() == torch::IntArrayRef({4 * hidden_size, hidden_size}),
                "weight_hh shape 必须为 [4H,H]");

    checkSameDevice(time_major, weight_ih, "weight_ih");
    checkSameDevice(time_major, weight_hh, "weight_hh");
    auto weight_ih_contiguous = weight_ih.contiguous();
    auto weight_hh_contiguous = weight_hh.contiguous();

    std::optional<torch::Tensor> bias_ih_contiguous;
    std::optional<torch::Tensor> bias_hh_contiguous;
    if (bias_ih.has_value()) {
        checkFloatTensor(*bias_ih, "bias_ih");
        checkFloatTensor(*bias_hh, "bias_hh");
        checkSameDevice(time_major, *bias_ih, "bias_ih");
        checkSameDevice(time_major, *bias_hh, "bias_hh");
        TORCH_CHECK(bias_ih->sizes() == torch::IntArrayRef({4 * hidden_size}) &&
                        bias_hh->sizes() == torch::IntArrayRef({4 * hidden_size}),
                    "bias shape 必须为 [4H]");
        bias_ih_contiguous = bias_ih->contiguous();
        bias_hh_contiguous = bias_hh->contiguous();
    }

    std::optional<torch::Tensor> hidden_contiguous;
    std::optional<torch::Tensor> cell_contiguous;
    if (initial_hidden.has_value()) {
        checkFloatTensor(*initial_hidden, "h_0");
        checkFloatTensor(*initial_cell, "c_0");
        checkSameDevice(time_major, *initial_hidden, "h_0");
        checkSameDevice(time_major, *initial_cell, "c_0");
        const std::array<std::int64_t, 3> expected_state_shape{1, batch_size, hidden_size};
        TORCH_CHECK(initial_hidden->sizes() == torch::IntArrayRef(expected_state_shape) &&
                        initial_cell->sizes() == torch::IntArrayRef(expected_state_shape),
                    "h_0/c_0 shape 必须为 [1,B,H]");
        hidden_contiguous = initial_hidden->contiguous();
        cell_contiguous = initial_cell->contiguous();
    }

    auto output = torch::empty({sequence_length, batch_size, hidden_size}, time_major.options());
    auto final_hidden = torch::empty({1, batch_size, hidden_size}, time_major.options());
    auto final_cell = torch::empty({1, batch_size, hidden_size}, time_major.options());
    const quant_lstm::LstmShape shape{sequence_length, batch_size, input_size, hidden_size};
    const quant_lstm::LstmFloatWeights weights{
        weight_ih_contiguous.data_ptr<float>(), weight_hh_contiguous.data_ptr<float>(),
        bias_ih_contiguous.has_value() ? bias_ih_contiguous->data_ptr<float>() : nullptr,
        bias_hh_contiguous.has_value() ? bias_hh_contiguous->data_ptr<float>() : nullptr};
    const float* hidden_data =
        hidden_contiguous.has_value() ? hidden_contiguous->data_ptr<float>() : nullptr;
    const float* cell_data =
        cell_contiguous.has_value() ? cell_contiguous->data_ptr<float>() : nullptr;

    if (time_major.is_cuda()) {
        c10::cuda::CUDAGuard device_guard(time_major.device());
        const cudaStream_t stream =
            c10::cuda::getCurrentCUDAStream(time_major.get_device()).stream();
        cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
        auto workspace =
            torch::empty({quant_lstm::cudaWorkspaceElementCount(shape)}, time_major.options());
        quant_lstm::lstmForwardFloatCuda(
            shape, weights, time_major.data_ptr<float>(), hidden_data, cell_data,
            output.data_ptr<float>(), final_hidden.data_ptr<float>(), final_cell.data_ptr<float>(),
            handle, stream, workspace.data_ptr<float>());
    } else {
        quant_lstm::lstmForwardFloatCpu(
            shape, weights, time_major.data_ptr<float>(), hidden_data, cell_data,
            output.data_ptr<float>(), final_hidden.data_ptr<float>(), final_cell.data_ptr<float>());
    }

    if (batch_first) {
        output = output.transpose(0, 1);
    }
    return {output, final_hidden, final_cell};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("lstm_forward", &lstmForward, "单层单向 FP32 LSTM 前向", pybind11::arg("input"),
               pybind11::arg("weight_ih"), pybind11::arg("weight_hh"),
               pybind11::arg("bias_ih") = std::nullopt,
               pybind11::arg("bias_hh") = std::nullopt,
               pybind11::arg("initial_hidden") = std::nullopt,
               pybind11::arg("initial_cell") = std::nullopt,
               pybind11::arg("batch_first") = false);
}
