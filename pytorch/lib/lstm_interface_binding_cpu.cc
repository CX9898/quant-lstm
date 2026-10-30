#include <pybind11/stl.h>
#include <torch/extension.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include "lstm/calibration.h"
#include "lstm/forward_cpu.h"
#include "lstm/quant_config_loader.h"
#include "lstm/quant_params_io.h"
#include "quantization/float_carrier_ops.h"

namespace py = pybind11;

namespace {

struct PreparedForward {
    torch::Tensor input;
    torch::Tensor weight_ih;
    torch::Tensor weight_hh;
    std::optional<torch::Tensor> bias_ih;
    std::optional<torch::Tensor> bias_hh;
    std::optional<torch::Tensor> initial_hidden;
    std::optional<torch::Tensor> initial_cell;
    quant_lstm::LstmShape shape;
};

void checkTensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.defined() && tensor.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32 && tensor.layout() == torch::kStrided,
                name, " must be a strided float32 tensor");
}

PreparedForward prepareForward(
    const torch::Tensor& input, const torch::Tensor& weight_ih, const torch::Tensor& weight_hh,
    const std::optional<torch::Tensor>& bias_ih, const std::optional<torch::Tensor>& bias_hh,
    const std::optional<torch::Tensor>& initial_hidden,
    const std::optional<torch::Tensor>& initial_cell, bool batch_first) {
    checkTensor(input, "input");
    checkTensor(weight_ih, "weight_ih");
    checkTensor(weight_hh, "weight_hh");
    TORCH_CHECK(input.dim() == 3, "input must be rank 3");
    TORCH_CHECK(weight_ih.dim() == 2 && weight_hh.dim() == 2, "weights must be rank 2");
    TORCH_CHECK(bias_ih.has_value() == bias_hh.has_value(), "both biases must be supplied together");
    TORCH_CHECK(initial_hidden.has_value() == initial_cell.has_value(),
                "h_0 and c_0 must be supplied together");
    auto time_major = batch_first ? input.transpose(0, 1).contiguous() : input.contiguous();
    const auto hidden = weight_hh.size(1);
    TORCH_CHECK(hidden > 0 && weight_ih.size(0) == 4 * hidden &&
                    weight_ih.size(1) == time_major.size(2) && weight_hh.size(0) == 4 * hidden,
                "weights must have shapes [4H,I] and [4H,H]");
    TORCH_CHECK(time_major.size(0) > 0 && time_major.size(1) > 0 && time_major.size(2) > 0,
                "input dimensions must be positive");
    PreparedForward result{time_major, weight_ih.contiguous(), weight_hh.contiguous(),
                           std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                           {time_major.size(0), time_major.size(1), time_major.size(2), hidden}};
    if (bias_ih) {
        checkTensor(*bias_ih, "bias_ih");
        checkTensor(*bias_hh, "bias_hh");
        TORCH_CHECK(bias_ih->sizes() == torch::IntArrayRef({4 * hidden}) &&
                        bias_hh->sizes() == torch::IntArrayRef({4 * hidden}),
                    "biases must have shape [4H]");
        result.bias_ih = bias_ih->contiguous();
        result.bias_hh = bias_hh->contiguous();
    }
    if (initial_hidden) {
        checkTensor(*initial_hidden, "h_0");
        checkTensor(*initial_cell, "c_0");
        const std::array<std::int64_t, 3> expected{1, time_major.size(1), hidden};
        TORCH_CHECK(initial_hidden->sizes() == torch::IntArrayRef(expected) &&
                        initial_cell->sizes() == torch::IntArrayRef(expected),
                    "h_0 and c_0 must have shape [1,B,H]");
        result.initial_hidden = initial_hidden->contiguous();
        result.initial_cell = initial_cell->contiguous();
    }
    return result;
}

const float* data(const std::optional<torch::Tensor>& value) {
    return value ? value->data_ptr<float>() : nullptr;
}

quant_lstm::LstmFloatWeights weights(const PreparedForward& value) {
    return {value.weight_ih.data_ptr<float>(), value.weight_hh.data_ptr<float>(),
            data(value.bias_ih), data(value.bias_hh)};
}

using ForwardResult = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>;

ForwardResult allocateOutputs(const PreparedForward& value) {
    auto options = value.input.options().requires_grad(false);
    auto output = torch::empty({value.shape.sequence_length, value.shape.batch_size,
                                value.shape.hidden_size}, options);
    auto hidden = torch::empty({1, value.shape.batch_size, value.shape.hidden_size}, options);
    return {output, hidden, torch::empty_like(hidden)};
}

ForwardResult layoutOutputs(ForwardResult result, bool batch_first) {
    if (batch_first) {
        std::get<0>(result) = std::get<0>(result).transpose(0, 1);
    }
    return result;
}

ForwardResult lstmForward(
    const torch::Tensor& input, const torch::Tensor& weight_ih, const torch::Tensor& weight_hh,
    const std::optional<torch::Tensor>& bias_ih, const std::optional<torch::Tensor>& bias_hh,
    const std::optional<torch::Tensor>& initial_hidden,
    const std::optional<torch::Tensor>& initial_cell, bool batch_first) {
    const auto prepared = prepareForward(input, weight_ih, weight_hh, bias_ih, bias_hh,
                                         initial_hidden, initial_cell, batch_first);
    auto result = allocateOutputs(prepared);
    {
        py::gil_scoped_release release;
        quant_lstm::lstmForwardFloatCpu(
            prepared.shape, weights(prepared), prepared.input.data_ptr<float>(),
            data(prepared.initial_hidden), data(prepared.initial_cell),
            std::get<0>(result).data_ptr<float>(), std::get<1>(result).data_ptr<float>(),
            std::get<2>(result).data_ptr<float>());
    }
    return layoutOutputs(std::move(result), batch_first);
}

const char* accumulationClassName(quant_lstm::quantization::Fp32AccumulationClass value) {
    using Class = quant_lstm::quantization::Fp32AccumulationClass;
    switch (value) {
        case Class::ExactIntegerRange: return "exact_integer_range";
        case Class::PrecisionRisk: return "precision_risk";
        case Class::UnsafeNonFinite: return "unsafe_non_finite";
    }
    return "unknown";
}

py::dict safetySummary(const quant_lstm::ExecutionSafetyDiagnostics& diagnostics) {
    py::list entries;
    std::int64_t exact = 0, risk = 0, unsafe = 0;
    const auto append = [&](const std::string& name,
                            const quant_lstm::quantization::NumericSafetyReport& report) {
        py::dict entry;
        entry["name"] = name;
        entry["fp32_accumulation"] = accumulationClassName(report.fp32_accumulation);
        entry["safe"] = report.fpSafe();
        entries.append(std::move(entry));
        using Class = quant_lstm::quantization::Fp32AccumulationClass;
        if (report.fp32_accumulation == Class::ExactIntegerRange) ++exact;
        else if (report.fp32_accumulation == Class::PrecisionRisk) ++risk;
        else ++unsafe;
    };
    for (std::size_t index = 0; index < diagnostics.input_hidden_linear.size(); ++index)
        append("weight_ih_linear[" + std::to_string(index) + "]", diagnostics.input_hidden_linear[index]);
    for (std::size_t index = 0; index < diagnostics.hidden_hidden_linear.size(); ++index)
        append("weight_hh_linear[" + std::to_string(index) + "]", diagnostics.hidden_hidden_linear[index]);
    constexpr std::array<const char*, 4> gate_names{
        "input_gate_input", "forget_gate_input", "cell_gate_input", "output_gate_input"};
    for (std::size_t index = 0; index < gate_names.size(); ++index)
        append(gate_names[index], diagnostics.gates[index]);
    append("cell_state", diagnostics.cell);
    append("output", diagnostics.hidden);
    py::dict result;
    result["exact_integer_range_count"] = exact;
    result["precision_risk_count"] = risk;
    result["unsafe_non_finite_count"] = unsafe;
    result["has_precision_risk"] = risk != 0;
    result["entries"] = std::move(entries);
    return result;
}

torch::Tensor quantize(const torch::Tensor& source, quant_lstm::QuantOperator id,
                        const quant_lstm::LstmQuantParamsBundle& bundle,
                        std::size_t row_width = 0) {
    auto result = torch::empty_like(source);
    const auto& points = bundle.quant_params.at(id).values;
    const auto& type = bundle.config.at(id).type;
    const float* input = source.data_ptr<float>();
    float* output = result.data_ptr<float>();
    for (std::int64_t index = 0; index < source.numel(); ++index) {
        const auto parameter = row_width == 0 ? 0 : static_cast<std::size_t>(index) / row_width;
        output[index] = quant_lstm::quantization::quantizeFp(
            input[index], points.at(parameter), type);
    }
    return result;
}

void dequantize(torch::Tensor& value, quant_lstm::QuantOperator id,
                  const quant_lstm::LstmQuantParamsBundle& bundle) {
    const auto& point = bundle.quant_params.at(id).values.front();
    const auto& type = bundle.config.at(id).type;
    float* values = value.data_ptr<float>();
    for (std::int64_t index = 0; index < value.numel(); ++index) {
        values[index] = quant_lstm::quantization::dequantizeFp(values[index], point, type);
    }
}

py::tuple lstmForwardQuantized(
    const torch::Tensor& input, const torch::Tensor& weight_ih, const torch::Tensor& weight_hh,
    const std::optional<torch::Tensor>& bias_ih, const std::optional<torch::Tensor>& bias_hh,
    const std::optional<torch::Tensor>& initial_hidden,
    const std::optional<torch::Tensor>& initial_cell, bool batch_first,
    const std::string& bundle_json, const std::string& math_mode,
    bool require_exact_accumulation, bool save_checkpoints) {
    TORCH_CHECK(!save_checkpoints, "CPU QuantLSTM supports inference and PTQ only; QAT is unsupported");
    TORCH_CHECK(math_mode == "pedantic" || math_mode == "tf32", "invalid cublas_math_mode");
    const auto prepared = prepareForward(input, weight_ih, weight_hh, bias_ih, bias_hh,
                                         initial_hidden, initial_cell, batch_first);
    const auto bundle = quant_lstm::importQuantParamsBundle(bundle_json, false);
    TORCH_CHECK(bundle.input_size == prepared.shape.input_size &&
                    bundle.quant_params.hidden_size == prepared.shape.hidden_size &&
                    bundle.quant_params.bias_enabled == prepared.bias_ih.has_value(),
                "quantization parameters do not match input, hidden size or bias");
    const auto execution = quant_lstm::auditQuantParamsBundle(bundle, require_exact_accumulation);
    auto result = allocateOutputs(prepared);
    {
        py::gil_scoped_release release;
        using Op = quant_lstm::QuantOperator;
        auto q_input = quantize(prepared.input, Op::Input, bundle);
        auto q_weight_ih = quantize(prepared.weight_ih, Op::WeightInputHidden, bundle,
                                    prepared.shape.input_size);
        auto q_weight_hh = quantize(prepared.weight_hh, Op::WeightHiddenHidden, bundle,
                                    prepared.shape.hidden_size);
        std::optional<torch::Tensor> q_bias_ih, q_bias_hh, q_hidden, q_cell;
        if (prepared.bias_ih) {
            q_bias_ih = quantize(*prepared.bias_ih, Op::BiasInputHidden, bundle, 1);
            q_bias_hh = quantize(*prepared.bias_hh, Op::BiasHiddenHidden, bundle, 1);
        }
        if (prepared.initial_hidden) {
            q_hidden = quantize(*prepared.initial_hidden, Op::Output, bundle);
            q_cell = quantize(*prepared.initial_cell, Op::CellState, bundle);
        }
        quant_lstm::lstmForwardQuantizedFpCpuReference(
            prepared.shape, {q_weight_ih.data_ptr<float>(), q_weight_hh.data_ptr<float>(),
                              data(q_bias_ih), data(q_bias_hh)},
            q_input.data_ptr<float>(), data(q_hidden), data(q_cell), bundle.config,
            bundle.quant_params, execution, std::get<0>(result).data_ptr<float>(),
            std::get<1>(result).data_ptr<float>(), std::get<2>(result).data_ptr<float>());
        dequantize(std::get<0>(result), Op::Output, bundle);
        dequantize(std::get<1>(result), Op::Output, bundle);
        dequantize(std::get<2>(result), Op::CellState, bundle);
    }
    result = layoutOutputs(std::move(result), batch_first);
    return py::make_tuple(std::get<0>(result), std::get<1>(result), std::get<2>(result),
                          py::dict(), safetySummary(execution.diagnostics));
}

quant_lstm::CalibrationMethod calibrationMethod(const std::string& method) {
    if (method == "minmax") return quant_lstm::CalibrationMethod::MinMax;
    if (method == "sqnr") return quant_lstm::CalibrationMethod::Sqnr;
    if (method == "percentile") return quant_lstm::CalibrationMethod::Percentile;
    throw std::invalid_argument("calibration_method must be minmax, sqnr or percentile");
}

quant_lstm::quantization::HistogramCalibrationOptions histogramOptions(float percentile) {
    TORCH_CHECK(std::isfinite(percentile) && percentile > 0.0F && percentile <= 100.0F,
                "percentile must be in (0, 100]");
    quant_lstm::quantization::HistogramCalibrationOptions options;
    options.percentile = percentile;
    return options;
}

class CalibrationSessionBinding {
   public:
    CalibrationSessionBinding(const std::string& config, std::int64_t input_size,
                              std::int64_t hidden_size, bool bias, const std::string& method,
                              float percentile)
        : config_(quant_lstm::parseResolvedQuantConfig(config, true)),
          session_(config_, input_size, hidden_size, bias, calibrationMethod(method),
                   histogramOptions(percentile)) {}

    ForwardResult collect(
        const torch::Tensor& input, const torch::Tensor& weight_ih, const torch::Tensor& weight_hh,
        const std::optional<torch::Tensor>& bias_ih, const std::optional<torch::Tensor>& bias_hh,
        const std::optional<torch::Tensor>& initial_hidden,
        const std::optional<torch::Tensor>& initial_cell, bool batch_first) {
        const auto prepared = prepareForward(input, weight_ih, weight_hh, bias_ih, bias_hh,
                                             initial_hidden, initial_cell, batch_first);
        auto result = allocateOutputs(prepared);
        {
            py::gil_scoped_release release;
            session_.collect(prepared.shape, weights(prepared), prepared.input.data_ptr<float>(),
                             data(prepared.initial_hidden), data(prepared.initial_cell),
                             std::get<0>(result).data_ptr<float>(),
                             std::get<1>(result).data_ptr<float>(),
                             std::get<2>(result).data_ptr<float>());
        }
        return layoutOutputs(std::move(result), batch_first);
    }

    py::dict finalize(bool require_exact_accumulation) {
        const quant_lstm::FinalizedLstmCalibration* finalized;
        {
            py::gil_scoped_release release;
            finalized = &session_.finalize(require_exact_accumulation);
        }
        const quant_lstm::LstmQuantParamsBundle bundle{
            1, session_.collector().inputSize(), config_, finalized->quant_params};
        py::dict result;
        result["bundle_json"] = quant_lstm::exportQuantParamsBundle(bundle);
        result["resolved_config_json"] = quant_lstm::toCanonicalJson(config_);
        result["safety"] = safetySummary(finalized->execution_params.diagnostics);
        result["batch_count"] = finalized->report.batch_count;
        result["method"] = quant_lstm::calibrationMethodName(finalized->report.method);
        py::dict ranges;
        for (const auto& op : finalized->report.operators) {
            if (!quant_lstm::isParameterOperator(op.id) && !op.groups.empty()) {
                const auto& diagnostics = op.groups.front().diagnostics;
                ranges[py::str(quant_lstm::quantOperatorName(op.id))] =
                    py::make_tuple(diagnostics.adjusted_min, diagnostics.adjusted_max);
            }
        }
        result["activation_ranges"] = ranges;
        return result;
    }
    void reset() { session_.reset(); }
    std::string state() const { return quant_lstm::calibrationStateName(session_.state()); }
    std::uint64_t batchCount() const { return session_.collector().batchCount(); }

   private:
    quant_lstm::LstmOperatorQuantConfig config_;
    quant_lstm::LstmCalibrationSession session_;
};

py::dict auditBundle(const std::string& json, bool require_exact_accumulation) {
    const auto bundle = quant_lstm::importQuantParamsBundle(json, false);
    const auto execution = quant_lstm::auditQuantParamsBundle(bundle, require_exact_accumulation);
    py::dict result;
    result["bundle_json"] = quant_lstm::exportQuantParamsBundle(bundle);
    result["resolved_config_json"] = quant_lstm::toCanonicalJson(bundle.config);
    result["safety"] = safetySummary(execution.diagnostics);
    result["input_size"] = bundle.input_size;
    result["hidden_size"] = bundle.quant_params.hidden_size;
    result["bias_enabled"] = bundle.quant_params.bias_enabled;
    return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.attr("supports_cuda") = false;
    module.def("lstm_forward", &lstmForward, py::arg("input"), py::arg("weight_ih"),
               py::arg("weight_hh"), py::arg("bias_ih") = std::nullopt,
               py::arg("bias_hh") = std::nullopt, py::arg("initial_hidden") = std::nullopt,
               py::arg("initial_cell") = std::nullopt, py::arg("batch_first") = false);
    module.def("lstm_forward_quantized", &lstmForwardQuantized, py::arg("input"),
               py::arg("weight_ih"), py::arg("weight_hh"), py::arg("bias_ih") = std::nullopt,
               py::arg("bias_hh") = std::nullopt, py::arg("initial_hidden") = std::nullopt,
               py::arg("initial_cell") = std::nullopt, py::arg("batch_first") = false,
               py::arg("bundle_json") = std::string(), py::arg("math_mode") = "pedantic",
               py::arg("require_exact_accumulation") = false, py::arg("save_checkpoints") = false);
    module.def("resolve_quant_config", [](const std::string& defaults, const std::string& overrides) {
        return quant_lstm::toCanonicalJson(quant_lstm::resolveQuantConfig(defaults, overrides));
    }, py::arg("defaults_json"), py::arg("override_json"));
    module.def("validate_resolved_quant_config", [](const std::string& config) {
        return quant_lstm::toCanonicalJson(quant_lstm::parseResolvedQuantConfig(config, false));
    }, py::arg("resolved_json"));
    module.def("audit_quant_params_bundle", &auditBundle, py::arg("bundle_json"),
               py::arg("require_exact_accumulation") = false);
    py::class_<CalibrationSessionBinding>(module, "CalibrationSession")
        .def(py::init<const std::string&, std::int64_t, std::int64_t, bool, const std::string&, float>(),
             py::arg("resolved_config_json"), py::arg("input_size"), py::arg("hidden_size"),
             py::arg("bias_enabled"), py::arg("method") = "minmax", py::arg("percentile") = 99.99F)
        .def("collect", &CalibrationSessionBinding::collect, py::arg("input"), py::arg("weight_ih"),
             py::arg("weight_hh"), py::arg("bias_ih") = std::nullopt,
             py::arg("bias_hh") = std::nullopt, py::arg("initial_hidden") = std::nullopt,
             py::arg("initial_cell") = std::nullopt, py::arg("batch_first") = false)
        .def("finalize", &CalibrationSessionBinding::finalize,
             py::arg("require_exact_accumulation") = false)
        .def("reset", &CalibrationSessionBinding::reset)
        .def_property_readonly("state", &CalibrationSessionBinding::state)
        .def_property_readonly("batch_count", &CalibrationSessionBinding::batchCount);
}
