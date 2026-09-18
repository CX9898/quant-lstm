#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "lstm/backward_float_cuda.h"
#include "lstm/calibration.h"
#include "lstm/forward_float.h"
#include "lstm/forward_float_cuda.h"
#include "lstm/forward_quantized_fp_cuda.h"
#include "lstm/quant_config_loader.h"
#include "lstm/quant_params_io.h"

namespace py = pybind11;

// Torch binding 只处理张量、配置和资源生命周期；LSTM 数学位于公共核心。
namespace {

void checkCuda(cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": " +
                                 cudaGetErrorString(status));
    }
}

void checkCublas(cublasStatus_t status, const char* context) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(context) +
                                 ": cuBLAS status=" +
                                 std::to_string(static_cast<int>(status)));
    }
}

void checkFloatTensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.defined(), name, " 未定义");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name,
                " 必须为 float32");
    TORCH_CHECK(tensor.layout() == torch::kStrided, name,
                " 必须为 strided tensor");
}

void checkSameDevice(const torch::Tensor& reference,
                     const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.device() == reference.device(), name,
                " 与 input 不在同一设备");
}

struct PreparedForward {
    torch::Tensor time_major;
    torch::Tensor weight_ih;
    torch::Tensor weight_hh;
    std::optional<torch::Tensor> bias_ih;
    std::optional<torch::Tensor> bias_hh;
    std::optional<torch::Tensor> initial_hidden;
    std::optional<torch::Tensor> initial_cell;
    quant_lstm::LstmShape shape;
};

PreparedForward prepareForward(
    const torch::Tensor& input, const torch::Tensor& weight_ih,
    const torch::Tensor& weight_hh,
    const std::optional<torch::Tensor>& bias_ih,
    const std::optional<torch::Tensor>& bias_hh,
    const std::optional<torch::Tensor>& initial_hidden,
    const std::optional<torch::Tensor>& initial_cell, bool batch_first) {
    checkFloatTensor(input, "input");
    checkFloatTensor(weight_ih, "weight_ih");
    checkFloatTensor(weight_hh, "weight_hh");
    TORCH_CHECK(input.dim() == 3, "input 必须是 3-D tensor");
    TORCH_CHECK(weight_ih.dim() == 2 && weight_hh.dim() == 2,
                "weight 必须是 2-D tensor");
    TORCH_CHECK(bias_ih.has_value() == bias_hh.has_value(),
                "bias_ih 和 bias_hh 必须同时提供或同时省略");
    TORCH_CHECK(initial_hidden.has_value() == initial_cell.has_value(),
                "h_0 和 c_0 必须同时提供或同时省略");

    PreparedForward result;
    result.time_major =
        batch_first ? input.transpose(0, 1).contiguous()
                    : input.contiguous();
    TORCH_CHECK(result.time_major.device().is_cpu() ||
                    result.time_major.is_cuda(),
                "仅支持 CPU 或 CUDA tensor");
    const std::int64_t sequence_length = result.time_major.size(0);
    const std::int64_t batch_size = result.time_major.size(1);
    const std::int64_t input_size = result.time_major.size(2);
    TORCH_CHECK(weight_ih.size(1) == input_size,
                "weight_ih input_size 不匹配");
    TORCH_CHECK(weight_ih.size(0) > 0 && weight_ih.size(0) % 4 == 0,
                "weight_ih 第一维必须为正的 4H");
    const std::int64_t hidden_size = weight_ih.size(0) / 4;
    TORCH_CHECK(
        weight_hh.sizes() ==
            torch::IntArrayRef({4 * hidden_size, hidden_size}),
        "weight_hh shape 必须为 [4H,H]");

    checkSameDevice(result.time_major, weight_ih, "weight_ih");
    checkSameDevice(result.time_major, weight_hh, "weight_hh");
    result.weight_ih = weight_ih.contiguous();
    result.weight_hh = weight_hh.contiguous();

    if (bias_ih.has_value()) {
        checkFloatTensor(*bias_ih, "bias_ih");
        checkFloatTensor(*bias_hh, "bias_hh");
        checkSameDevice(result.time_major, *bias_ih, "bias_ih");
        checkSameDevice(result.time_major, *bias_hh, "bias_hh");
        TORCH_CHECK(
            bias_ih->sizes() == torch::IntArrayRef({4 * hidden_size}) &&
                bias_hh->sizes() ==
                    torch::IntArrayRef({4 * hidden_size}),
            "bias shape 必须为 [4H]");
        result.bias_ih = bias_ih->contiguous();
        result.bias_hh = bias_hh->contiguous();
    }

    if (initial_hidden.has_value()) {
        checkFloatTensor(*initial_hidden, "h_0");
        checkFloatTensor(*initial_cell, "c_0");
        checkSameDevice(result.time_major, *initial_hidden, "h_0");
        checkSameDevice(result.time_major, *initial_cell, "c_0");
        const std::array<std::int64_t, 3> expected{
            1, batch_size, hidden_size};
        TORCH_CHECK(
            initial_hidden->sizes() == torch::IntArrayRef(expected) &&
                initial_cell->sizes() == torch::IntArrayRef(expected),
            "h_0/c_0 shape 必须为 [1,B,H]");
        result.initial_hidden = initial_hidden->contiguous();
        result.initial_cell = initial_cell->contiguous();
    }

    result.shape = {sequence_length, batch_size, input_size, hidden_size};
    return result;
}

const float* optionalData(
    const std::optional<torch::Tensor>& tensor) {
    return tensor.has_value() ? tensor->data_ptr<float>() : nullptr;
}

quant_lstm::LstmFloatWeights preparedWeights(
    const PreparedForward& prepared) {
    return {
        prepared.weight_ih.data_ptr<float>(),
        prepared.weight_hh.data_ptr<float>(),
        optionalData(prepared.bias_ih),
        optionalData(prepared.bias_hh),
    };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> lstmForward(
    const torch::Tensor& input, const torch::Tensor& weight_ih,
    const torch::Tensor& weight_hh,
    const std::optional<torch::Tensor>& bias_ih,
    const std::optional<torch::Tensor>& bias_hh,
    const std::optional<torch::Tensor>& initial_hidden,
    const std::optional<torch::Tensor>& initial_cell, bool batch_first) {
    PreparedForward prepared =
        prepareForward(input, weight_ih, weight_hh, bias_ih, bias_hh,
                       initial_hidden, initial_cell, batch_first);
    auto output = torch::empty(
        {prepared.shape.sequence_length, prepared.shape.batch_size,
         prepared.shape.hidden_size},
        prepared.time_major.options());
    auto final_hidden = torch::empty(
        {1, prepared.shape.batch_size, prepared.shape.hidden_size},
        prepared.time_major.options());
    auto final_cell = torch::empty_like(final_hidden);

    if (prepared.time_major.is_cuda()) {
        c10::cuda::CUDAGuard guard(prepared.time_major.device());
        const cudaStream_t stream =
            c10::cuda::getCurrentCUDAStream(
                prepared.time_major.get_device()).stream();
        cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
        auto workspace = torch::empty(
            {static_cast<std::int64_t>(
                quant_lstm::cudaWorkspaceElementCount(prepared.shape))},
            prepared.time_major.options());
        quant_lstm::lstmForwardFloatCuda(
            prepared.shape, preparedWeights(prepared),
            prepared.time_major.data_ptr<float>(),
            optionalData(prepared.initial_hidden),
            optionalData(prepared.initial_cell), output.data_ptr<float>(),
            final_hidden.data_ptr<float>(), final_cell.data_ptr<float>(),
            handle, stream, workspace.data_ptr<float>());
    } else {
        quant_lstm::lstmForwardFloatCpu(
            prepared.shape, preparedWeights(prepared),
            prepared.time_major.data_ptr<float>(),
            optionalData(prepared.initial_hidden),
            optionalData(prepared.initial_cell), output.data_ptr<float>(),
            final_hidden.data_ptr<float>(), final_cell.data_ptr<float>());
    }

    if (batch_first) {
        output = output.transpose(0, 1);
    }
    return {output, final_hidden, final_cell};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
lstmForwardTraining(
    const torch::Tensor& input, const torch::Tensor& weight_ih,
    const torch::Tensor& weight_hh,
    const std::optional<torch::Tensor>& bias_ih,
    const std::optional<torch::Tensor>& bias_hh,
    const std::optional<torch::Tensor>& initial_hidden,
    const std::optional<torch::Tensor>& initial_cell, bool batch_first) {
    PreparedForward prepared =
        prepareForward(input, weight_ih, weight_hh, bias_ih, bias_hh,
                       initial_hidden, initial_cell, batch_first);
    TORCH_CHECK(prepared.time_major.is_cuda(),
                "训练态 native checkpoint 只支持 CUDA tensor");
    const auto options = prepared.time_major.options();
    auto output = torch::empty(
        {prepared.shape.sequence_length, prepared.shape.batch_size,
         prepared.shape.hidden_size},
        options);
    auto final_hidden = torch::empty(
        {1, prepared.shape.batch_size, prepared.shape.hidden_size}, options);
    auto final_cell = torch::empty_like(final_hidden);
    auto gate_outputs = torch::empty(
        {prepared.shape.sequence_length, prepared.shape.batch_size,
         4 * prepared.shape.hidden_size},
        options);
    auto cell_states = torch::empty_like(output);
    auto cell_tanh_outputs = torch::empty_like(output);

    c10::cuda::CUDAGuard guard(prepared.time_major.device());
    const cudaStream_t stream =
        c10::cuda::getCurrentCUDAStream(
            prepared.time_major.get_device()).stream();
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    auto workspace = torch::empty(
        {static_cast<std::int64_t>(
            quant_lstm::cudaWorkspaceElementCount(prepared.shape))},
        options);
    quant_lstm::LstmFloatCudaTrace trace{
        gate_outputs.data_ptr<float>(), cell_states.data_ptr<float>(),
        cell_tanh_outputs.data_ptr<float>()};
    quant_lstm::lstmForwardFloatCuda(
        prepared.shape, preparedWeights(prepared),
        prepared.time_major.data_ptr<float>(),
        optionalData(prepared.initial_hidden),
        optionalData(prepared.initial_cell), output.data_ptr<float>(),
        final_hidden.data_ptr<float>(), final_cell.data_ptr<float>(), handle,
        stream, workspace.data_ptr<float>(), &trace);

    auto hidden_outputs = output;
    if (batch_first) {
        output = output.transpose(0, 1);
    }
    return {output,          final_hidden,      final_cell, gate_outputs,
            cell_states,    cell_tanh_outputs, hidden_outputs};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
lstmBackwardFloat(
    const torch::Tensor& input, const torch::Tensor& weight_ih,
    const torch::Tensor& weight_hh,
    const std::optional<torch::Tensor>& bias_ih,
    const std::optional<torch::Tensor>& bias_hh,
    const std::optional<torch::Tensor>& initial_hidden,
    const std::optional<torch::Tensor>& initial_cell, bool batch_first,
    const torch::Tensor& gate_outputs, const torch::Tensor& cell_states,
    const torch::Tensor& cell_tanh_outputs,
    const torch::Tensor& hidden_outputs, const torch::Tensor& grad_output,
    const torch::Tensor& grad_final_hidden,
    const torch::Tensor& grad_final_cell) {
    PreparedForward prepared =
        prepareForward(input, weight_ih, weight_hh, bias_ih, bias_hh,
                       initial_hidden, initial_cell, batch_first);
    TORCH_CHECK(prepared.time_major.is_cuda(),
                "native float backward 只支持 CUDA tensor");
    const auto& reference = prepared.time_major;
    const std::array<std::int64_t, 3> gate_shape{
        prepared.shape.sequence_length, prepared.shape.batch_size,
        4 * prepared.shape.hidden_size};
    const std::array<std::int64_t, 3> state_trace_shape{
        prepared.shape.sequence_length, prepared.shape.batch_size,
        prepared.shape.hidden_size};
    const std::array<std::int64_t, 2> state_gradient_shape{
        prepared.shape.batch_size, prepared.shape.hidden_size};
    const auto prepare_tensor = [&](const torch::Tensor& tensor,
                                    const char* name,
                                    torch::IntArrayRef expected) {
        checkFloatTensor(tensor, name);
        checkSameDevice(reference, tensor, name);
        TORCH_CHECK(tensor.sizes() == expected, name, " shape 不匹配");
        return tensor.contiguous();
    };
    auto gates = prepare_tensor(gate_outputs, "gate_outputs", gate_shape);
    auto cells = prepare_tensor(cell_states, "cell_states", state_trace_shape);
    auto cell_tanh = prepare_tensor(cell_tanh_outputs,
                                    "cell_tanh_outputs", state_trace_shape);
    auto hidden = prepare_tensor(hidden_outputs, "hidden_outputs",
                                 state_trace_shape);
    auto grad_output_time =
        prepare_tensor(grad_output, "grad_output", state_trace_shape);
    auto grad_hidden = prepare_tensor(grad_final_hidden, "grad_final_hidden",
                                      state_gradient_shape);
    auto grad_cell = prepare_tensor(grad_final_cell, "grad_final_cell",
                                    state_gradient_shape);

    auto grad_input_time = torch::empty_like(prepared.time_major);
    auto grad_weight_ih = torch::empty_like(prepared.weight_ih);
    auto grad_weight_hh = torch::empty_like(prepared.weight_hh);
    auto grad_bias_ih = torch::empty(
        {4 * prepared.shape.hidden_size}, reference.options());
    auto grad_bias_hh = torch::empty_like(grad_bias_ih);
    auto grad_initial_hidden = torch::empty(state_gradient_shape,
                                            reference.options());
    auto grad_initial_cell = torch::empty_like(grad_initial_hidden);

    c10::cuda::CUDAGuard guard(reference.device());
    const cudaStream_t stream =
        c10::cuda::getCurrentCUDAStream(reference.get_device()).stream();
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    auto workspace = torch::empty(
        {quant_lstm::cudaBackwardWorkspaceElementCount(prepared.shape)},
        reference.options());
    const quant_lstm::LstmFloatCudaBackwardTrace trace{
        gates.data_ptr<float>(), cells.data_ptr<float>(),
        cell_tanh.data_ptr<float>(), hidden.data_ptr<float>()};
    const quant_lstm::LstmFloatCudaGradients gradients{
        grad_input_time.data_ptr<float>(), grad_weight_ih.data_ptr<float>(),
        grad_weight_hh.data_ptr<float>(), grad_bias_ih.data_ptr<float>(),
        grad_bias_hh.data_ptr<float>(), grad_initial_hidden.data_ptr<float>(),
        grad_initial_cell.data_ptr<float>()};
    quant_lstm::lstmBackwardFloatCuda(
        prepared.shape, preparedWeights(prepared),
        prepared.time_major.data_ptr<float>(),
        optionalData(prepared.initial_hidden),
        optionalData(prepared.initial_cell), trace,
        grad_output_time.data_ptr<float>(), grad_hidden.data_ptr<float>(),
        grad_cell.data_ptr<float>(), gradients, handle, stream,
        workspace.data_ptr<float>());

    torch::Tensor grad_input = grad_input_time;
    if (batch_first) {
        grad_input = grad_input_time.transpose(0, 1);
    }
    return {grad_input,          grad_weight_ih,       grad_weight_hh,
            grad_bias_ih,        grad_bias_hh,         grad_initial_hidden,
            grad_initial_cell};
}

quant_lstm::CalibrationMethod parseCalibrationMethod(
    const std::string& method) {
    if (method == "minmax") {
        return quant_lstm::CalibrationMethod::MinMax;
    }
    if (method == "sqnr") {
        return quant_lstm::CalibrationMethod::Sqnr;
    }
    if (method == "percentile") {
        return quant_lstm::CalibrationMethod::Percentile;
    }
    throw std::invalid_argument(
        "calibration_method 必须是 minmax、sqnr 或 percentile");
}

const char* accumulationClassName(
    quant_lstm::quantization::Fp32AccumulationClass value) {
    using Class =
        quant_lstm::quantization::Fp32AccumulationClass;
    switch (value) {
        case Class::ExactIntegerRange:
            return "exact_integer_range";
        case Class::PrecisionRisk:
            return "precision_risk";
        case Class::UnsafeNonFinite:
            return "unsafe_non_finite";
    }
    return "unknown";
}

py::dict safetySummary(
    const quant_lstm::ExecutionSafetyDiagnostics& diagnostics) {
    py::list entries;
    std::int64_t exact_count = 0;
    std::int64_t risk_count = 0;
    std::int64_t unsafe_count = 0;
    const auto append =
        [&](const std::string& name,
            const quant_lstm::quantization::NumericSafetyReport& report) {
            py::dict entry;
            entry["name"] = name;
            entry["fp32_accumulation"] =
                accumulationClassName(report.fp32_accumulation);
            entry["safe"] = report.safe();
            entries.append(std::move(entry));
            using Class =
                quant_lstm::quantization::Fp32AccumulationClass;
            if (report.fp32_accumulation == Class::ExactIntegerRange) {
                ++exact_count;
            } else if (report.fp32_accumulation ==
                       Class::PrecisionRisk) {
                ++risk_count;
            } else {
                ++unsafe_count;
            }
        };

    for (std::size_t index = 0;
         index < diagnostics.input_hidden_linear.size(); ++index) {
        append("weight_ih_linear[" + std::to_string(index) + "]",
               diagnostics.input_hidden_linear[index]);
    }
    for (std::size_t index = 0;
         index < diagnostics.hidden_hidden_linear.size(); ++index) {
        append("weight_hh_linear[" + std::to_string(index) + "]",
               diagnostics.hidden_hidden_linear[index]);
    }
    constexpr std::array<const char*, 4> gate_names{
        "input_gate_input", "forget_gate_input", "cell_gate_input",
        "output_gate_input"};
    for (std::size_t index = 0; index < gate_names.size(); ++index) {
        append(gate_names[index], diagnostics.gates[index]);
    }
    append("cell_state", diagnostics.cell);
    append("output", diagnostics.hidden);

    py::dict result;
    result["exact_integer_range_count"] = exact_count;
    result["precision_risk_count"] = risk_count;
    result["unsafe_non_finite_count"] = unsafe_count;
    result["has_precision_risk"] = risk_count != 0;
    result["entries"] = std::move(entries);
    return result;
}

class QuantizedCudaContextOwner {
   public:
    explicit QuantizedCudaContextOwner(
        std::size_t execution_parameter_bytes) {
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
    QuantizedCudaContextOwner& operator=(
        const QuantizedCudaContextOwner&) = delete;

    quant_lstm::LstmQuantizedFpCudaContext& get() noexcept {
        return context_;
    }

    void waitFor(cudaStream_t producer) {
        cudaEvent_t ready = nullptr;
        checkCuda(cudaEventCreateWithFlags(&ready,
                                           cudaEventDisableTiming),
                  "cudaEventCreate current-stream ready");
        try {
            checkCuda(cudaEventRecord(ready, producer),
                      "cudaEventRecord current-stream ready");
            for (cudaStream_t stream : context_.streams) {
                checkCuda(cudaStreamWaitEvent(stream, ready, 0),
                          "cudaStreamWaitEvent current-stream ready");
            }
        } catch (...) {
            cudaEventDestroy(ready);
            throw;
        }
        checkCuda(cudaEventDestroy(ready),
                  "cudaEventDestroy current-stream ready");
    }

    void synchronize() const {
        checkCuda(cudaEventSynchronize(context_.events[1]),
                  "cudaEventSynchronize quantized forward");
    }

   private:
    void release() noexcept {
        if (context_.device_execution_params != nullptr) {
            cudaFree(context_.device_execution_params);
            context_.device_execution_params = nullptr;
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

    quant_lstm::LstmQuantizedFpCudaContext context_{};
};

torch::Tensor byteTensor(const torch::Tensor& reference,
                         std::int64_t elements) {
    return torch::empty(
        {elements},
        reference.options().dtype(torch::kUInt8));
}

py::tuple lstmForwardQuantized(
    const torch::Tensor& input, const torch::Tensor& weight_ih,
    const torch::Tensor& weight_hh,
    const std::optional<torch::Tensor>& bias_ih,
    const std::optional<torch::Tensor>& bias_hh,
    const std::optional<torch::Tensor>& initial_hidden,
    const std::optional<torch::Tensor>& initial_cell, bool batch_first,
    const std::string& bundle_json, const std::string& math_mode,
    bool require_exact_accumulation, bool save_checkpoints) {
    PreparedForward prepared =
        prepareForward(input, weight_ih, weight_hh, bias_ih, bias_hh,
                       initial_hidden, initial_cell, batch_first);
    TORCH_CHECK(prepared.time_major.is_cuda(),
                "量化 FP 载体主路径只支持 CUDA tensor");
    const auto bundle =
        quant_lstm::importQuantParamsBundle(bundle_json, false);
    TORCH_CHECK(bundle.input_size == prepared.shape.input_size,
                "量化参数 input_size 与模块不匹配");
    TORCH_CHECK(bundle.quant_params.hidden_size ==
                    prepared.shape.hidden_size,
                "量化参数 hidden_size 与模块不匹配");
    TORCH_CHECK(bundle.quant_params.bias_enabled ==
                    prepared.bias_ih.has_value(),
                "量化参数 bias_enabled 与模块不匹配");
    const auto execution = quant_lstm::auditQuantParamsBundle(
        bundle, require_exact_accumulation);
    quant_lstm::LstmQuantizedFpCudaMathMode selected_math_mode;
    if (math_mode == "pedantic") {
        selected_math_mode =
            quant_lstm::LstmQuantizedFpCudaMathMode::Pedantic;
    } else if (math_mode == "tf32") {
        selected_math_mode =
            quant_lstm::LstmQuantizedFpCudaMathMode::Tf32;
    } else {
        throw std::invalid_argument(
            "cublas_math_mode 必须是 pedantic 或 tf32");
    }

    c10::cuda::CUDAGuard guard(prepared.time_major.device());
    auto output = torch::empty(
        {prepared.shape.sequence_length, prepared.shape.batch_size,
         prepared.shape.hidden_size},
        prepared.time_major.options());
    auto final_hidden = torch::empty(
        {1, prepared.shape.batch_size, prepared.shape.hidden_size},
        prepared.time_major.options());
    auto final_cell = torch::empty_like(final_hidden);
    const auto breakdown =
        quant_lstm::lstmQuantizedFpCudaWorkspaceBreakdown(
            prepared.shape, bundle.quant_params.bias_enabled);
    QuantizedCudaContextOwner context(
        breakdown.device_parameter_bytes);
    const cudaStream_t current_stream =
        c10::cuda::getCurrentCUDAStream(
            prepared.time_major.get_device()).stream();
    context.waitFor(current_stream);
    auto workspace = byteTensor(
        prepared.time_major,
        static_cast<std::int64_t>(breakdown.total_bytes));

    py::dict checkpoint_result;
    quant_lstm::LstmQuantizedFpCudaCheckpoints checkpoints{};
    std::array<torch::Tensor, 14> saved;
    if (save_checkpoints) {
        const std::int64_t linears =
            prepared.shape.sequence_length *
            prepared.shape.batch_size * 4 *
            prepared.shape.hidden_size;
        const std::int64_t states =
            prepared.shape.sequence_length *
            prepared.shape.batch_size *
            prepared.shape.hidden_size;
        for (std::size_t index = 0; index < 4; ++index) {
            saved[index] = torch::empty(
                {prepared.shape.sequence_length,
                 prepared.shape.batch_size,
                 4 * prepared.shape.hidden_size},
                prepared.time_major.options());
            saved[index + 7] =
                byteTensor(prepared.time_major, linears).reshape(
                    {prepared.shape.sequence_length,
                     prepared.shape.batch_size,
                     4 * prepared.shape.hidden_size});
        }
        for (std::size_t index = 4; index < 7; ++index) {
            saved[index] = torch::empty(
                {prepared.shape.sequence_length,
                 prepared.shape.batch_size,
                 prepared.shape.hidden_size},
                prepared.time_major.options());
            saved[index + 7] =
                byteTensor(prepared.time_major, states).reshape(
                    {prepared.shape.sequence_length,
                     prepared.shape.batch_size,
                     prepared.shape.hidden_size});
        }
        checkpoints = {
            saved[0].data_ptr<float>(), saved[1].data_ptr<float>(),
            saved[2].data_ptr<float>(), saved[3].data_ptr<float>(),
            saved[4].data_ptr<float>(), saved[5].data_ptr<float>(),
            saved[6].data_ptr<float>(), saved[7].data_ptr<std::uint8_t>(),
            saved[8].data_ptr<std::uint8_t>(),
            saved[9].data_ptr<std::uint8_t>(),
            saved[10].data_ptr<std::uint8_t>(),
            saved[11].data_ptr<std::uint8_t>(),
            saved[12].data_ptr<std::uint8_t>(),
            saved[13].data_ptr<std::uint8_t>()};
    }

    quant_lstm::lstmForwardQuantizedFpCuda(
        prepared.shape, preparedWeights(prepared),
        prepared.time_major.data_ptr<float>(),
        optionalData(prepared.initial_hidden),
        optionalData(prepared.initial_cell), bundle.config,
        bundle.quant_params, execution, output.data_ptr<float>(),
        final_hidden.data_ptr<float>(), final_cell.data_ptr<float>(),
        context.get(), selected_math_mode,
        {workspace.data_ptr(), breakdown.total_bytes},
        save_checkpoints ? &checkpoints : nullptr);
    context.synchronize();

    if (save_checkpoints) {
        constexpr std::array<const char*, 7> names{
            "weight_ih_linear", "weight_hh_linear", "gate_inputs",
            "gate_outputs", "cell_states", "cell_tanh_outputs",
            "hidden_outputs"};
        py::dict values;
        py::dict masks;
        for (std::size_t index = 0; index < names.size(); ++index) {
            values[names[index]] = saved[index];
            masks[names[index]] = saved[index + 7];
        }
        checkpoint_result["values"] = std::move(values);
        checkpoint_result["clamp_masks"] = std::move(masks);
    }

    if (batch_first) {
        output = output.transpose(0, 1);
    }
    return py::make_tuple(
        output, final_hidden, final_cell, checkpoint_result,
        safetySummary(execution.diagnostics));
}

class CalibrationSessionBinding {
   public:
    CalibrationSessionBinding(const std::string& resolved_config_json,
                              std::int64_t input_size,
                              std::int64_t hidden_size,
                              bool bias_enabled,
                              const std::string& method)
        : config_(quant_lstm::parseResolvedQuantConfig(
              resolved_config_json, true)),
          session_(config_, input_size, hidden_size, bias_enabled,
                   parseCalibrationMethod(method)) {}

    void collect(
        const torch::Tensor& input, const torch::Tensor& weight_ih,
        const torch::Tensor& weight_hh,
        const std::optional<torch::Tensor>& bias_ih,
        const std::optional<torch::Tensor>& bias_hh,
        const std::optional<torch::Tensor>& initial_hidden,
        const std::optional<torch::Tensor>& initial_cell,
        bool batch_first) {
        PreparedForward prepared =
            prepareForward(input, weight_ih, weight_hh, bias_ih,
                           bias_hh, initial_hidden, initial_cell,
                           batch_first);
        TORCH_CHECK(prepared.time_major.device().is_cpu(),
                    "校准 binding 只接受 CPU tensor");
        TORCH_CHECK(prepared.shape.input_size ==
                        session_.collector().inputSize() &&
                        prepared.shape.hidden_size ==
                        session_.collector().hiddenSize(),
                    "校准 tensor shape 与 session 不匹配");
        TORCH_CHECK(prepared.bias_ih.has_value() ==
                        session_.collector().biasEnabled(),
                    "校准 bias 状态与 session 不匹配");
        py::gil_scoped_release release;
        session_.collect(
            prepared.shape, preparedWeights(prepared),
            prepared.time_major.data_ptr<float>(),
            optionalData(prepared.initial_hidden),
            optionalData(prepared.initial_cell));
    }

    py::dict finalize(bool require_exact_accumulation) {
        const quant_lstm::FinalizedLstmCalibration* finalized = nullptr;
        {
            py::gil_scoped_release release;
            finalized =
                &session_.finalize(require_exact_accumulation);
        }
        const quant_lstm::LstmQuantParamsBundle bundle{
            1, session_.collector().inputSize(), config_,
            finalized->quant_params};
        py::dict result;
        result["bundle_json"] =
            quant_lstm::exportQuantParamsBundle(bundle);
        result["resolved_config_json"] =
            quant_lstm::toCanonicalJson(config_);
        result["safety"] =
            safetySummary(finalized->execution_params.diagnostics);
        result["batch_count"] = finalized->report.batch_count;
        result["method"] =
            quant_lstm::calibrationMethodName(
                finalized->report.method);
        return result;
    }

    void reset() { session_.reset(); }

    std::string state() const {
        return quant_lstm::calibrationStateName(session_.state());
    }

    std::uint64_t batchCount() const {
        return session_.collector().batchCount();
    }

   private:
    quant_lstm::LstmOperatorQuantConfig config_;
    quant_lstm::LstmCalibrationSession session_;
};

py::dict auditBundle(const std::string& json_text,
                     bool require_exact_accumulation) {
    const auto bundle =
        quant_lstm::importQuantParamsBundle(json_text, false);
    const auto execution = quant_lstm::auditQuantParamsBundle(
        bundle, require_exact_accumulation);
    py::dict result;
    result["bundle_json"] =
        quant_lstm::exportQuantParamsBundle(bundle);
    result["resolved_config_json"] =
        quant_lstm::toCanonicalJson(bundle.config);
    result["safety"] = safetySummary(execution.diagnostics);
    result["input_size"] = bundle.input_size;
    result["hidden_size"] = bundle.quant_params.hidden_size;
    result["bias_enabled"] = bundle.quant_params.bias_enabled;
    return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "lstm_forward", &lstmForward,
        "单层单向 FP32 LSTM 前向", py::arg("input"),
        py::arg("weight_ih"), py::arg("weight_hh"),
        py::arg("bias_ih") = std::nullopt,
        py::arg("bias_hh") = std::nullopt,
        py::arg("initial_hidden") = std::nullopt,
        py::arg("initial_cell") = std::nullopt,
        py::arg("batch_first") = false);
    module.def(
        "lstm_forward_training", &lstmForwardTraining,
        "单层单向 CUDA FP32 LSTM 训练前向与 checkpoint",
        py::arg("input"), py::arg("weight_ih"), py::arg("weight_hh"),
        py::arg("bias_ih") = std::nullopt,
        py::arg("bias_hh") = std::nullopt,
        py::arg("initial_hidden") = std::nullopt,
        py::arg("initial_cell") = std::nullopt,
        py::arg("batch_first") = false);
    module.def(
        "lstm_backward_float", &lstmBackwardFloat,
        "单层单向 CUDA FP32 LSTM 原生反向",
        py::arg("input"), py::arg("weight_ih"), py::arg("weight_hh"),
        py::arg("bias_ih"), py::arg("bias_hh"),
        py::arg("initial_hidden"), py::arg("initial_cell"),
        py::arg("batch_first"), py::arg("gate_outputs"),
        py::arg("cell_states"), py::arg("cell_tanh_outputs"),
        py::arg("hidden_outputs"), py::arg("grad_output"),
        py::arg("grad_final_hidden"), py::arg("grad_final_cell"));
    module.def(
        "lstm_forward_quantized", &lstmForwardQuantized,
        "单层单向 CUDA FP32 q-carrier LSTM 前向",
        py::arg("input"), py::arg("weight_ih"),
        py::arg("weight_hh"), py::arg("bias_ih") = std::nullopt,
        py::arg("bias_hh") = std::nullopt,
        py::arg("initial_hidden") = std::nullopt,
        py::arg("initial_cell") = std::nullopt,
        py::arg("batch_first") = false,
        py::arg("bundle_json") = std::string(),
        py::arg("math_mode") = "pedantic",
        py::arg("require_exact_accumulation") = false,
        py::arg("save_checkpoints") = false);
    module.def(
        "resolve_quant_config",
        [](const std::string& defaults,
           const std::string& override_json) {
            return quant_lstm::toCanonicalJson(
                quant_lstm::resolveQuantConfig(
                    defaults, override_json));
        },
        "通过唯一 C++ resolver 合并量化配置",
        py::arg("defaults_json"), py::arg("override_json"));
    module.def(
        "validate_resolved_quant_config",
        [](const std::string& resolved) {
            return quant_lstm::toCanonicalJson(
                quant_lstm::parseResolvedQuantConfig(
                    resolved, false));
        },
        "严格校验并规范化 resolved config",
        py::arg("resolved_json"));
    module.def(
        "audit_quant_params_bundle", &auditBundle,
        "规范化参数包并重新派生执行编码",
        py::arg("bundle_json"),
        py::arg("require_exact_accumulation") = false);

    py::class_<CalibrationSessionBinding>(
        module, "CalibrationSession")
        .def(py::init<const std::string&, std::int64_t,
                      std::int64_t, bool, const std::string&>(),
             py::arg("resolved_config_json"),
             py::arg("input_size"), py::arg("hidden_size"),
             py::arg("bias_enabled"),
             py::arg("method") = "minmax")
        .def("collect", &CalibrationSessionBinding::collect,
             py::arg("input"), py::arg("weight_ih"),
             py::arg("weight_hh"),
             py::arg("bias_ih") = std::nullopt,
             py::arg("bias_hh") = std::nullopt,
             py::arg("initial_hidden") = std::nullopt,
             py::arg("initial_cell") = std::nullopt,
             py::arg("batch_first") = false)
        .def("finalize", &CalibrationSessionBinding::finalize,
             py::arg("require_exact_accumulation") = false)
        .def("reset", &CalibrationSessionBinding::reset)
        .def_property_readonly("state",
                               &CalibrationSessionBinding::state)
        .def_property_readonly(
            "batch_count",
            &CalibrationSessionBinding::batchCount);
}
