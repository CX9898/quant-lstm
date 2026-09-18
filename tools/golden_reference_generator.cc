#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "lstm/forward_cpu.h"
#include "lstm/gate_layout.h"
#include "lstm/lstm_execution_params.h"
#include "lstm/quant_config_loader.h"
#include "lstm/quant_params.h"
#include "quantization/fixed_point_ops.h"

namespace {

using Json = nlohmann::json;
using quant_lstm::QuantOperator;

struct CaseDefinition {
    std::string id_prefix;
    std::string kind;
    quant_lstm::LstmShape shape;
    quant_lstm::quantization::ScaleMode scale_mode;
    std::vector<float> input;
};

struct PreparedCase {
    CaseDefinition definition;
    quant_lstm::LstmOperatorQuantConfig config;
    quant_lstm::LstmQuantParams quant_params;
    quant_lstm::LstmExecutionParams execution_params;
    std::vector<std::int32_t> input;
    std::vector<std::int32_t> initial_hidden;
    std::vector<std::int32_t> initial_cell;
    std::vector<std::int32_t> weight_ih;
    std::vector<std::int32_t> weight_hh;
    std::vector<std::int32_t> bias_ih;
    std::vector<std::int32_t> bias_hh;
};

template <typename T>
Json tensor(std::string_view dtype, const std::vector<std::size_t>& shape,
            const std::vector<T>& data) {
    return Json{{"dtype", dtype}, {"shape", shape}, {"data", data}};
}

template <typename T>
Json scalarTensor(std::string_view dtype, const T& value) {
    return tensor(dtype, {}, std::vector<T>{value});
}

std::vector<std::string> canonicalFloats(const std::vector<float>& values) {
    std::vector<std::string> result;
    result.reserve(values.size());
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Reference Golden 不允许非有限 float32");
        }
        bool formatted = false;
        for (int precision = 1; precision <= 9; ++precision) {
            std::array<char, 64> buffer{};
            const auto encoded = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value,
                                               std::chars_format::general, precision);
            if (encoded.ec != std::errc{}) {
                throw std::runtime_error("格式化 float32 失败");
            }
            float parsed = 0.0F;
            const auto decoded =
                std::from_chars(buffer.data(), encoded.ptr, parsed, std::chars_format::general);
            if (decoded.ec == std::errc{} && decoded.ptr == encoded.ptr && parsed == value &&
                std::signbit(parsed) == std::signbit(value)) {
                result.emplace_back(buffer.data(), encoded.ptr);
                formatted = true;
                break;
            }
        }
        if (!formatted) {
            throw std::runtime_error("无法生成可往返的 float32 字符串");
        }
    }
    return result;
}

Json floatTensor(const std::vector<std::size_t>& shape, const std::vector<float>& values) {
    return tensor("float32", shape, canonicalFloats(values));
}

quant_lstm::LstmOperatorQuantConfig makeConfig(quant_lstm::quantization::ScaleMode mode) {
    quant_lstm::LstmOperatorQuantConfig config;
    config.scale_mode = mode;
    for (auto& item : config.operators) {
        item.type = {8, false, true};
        item.granularity = quant_lstm::QuantGranularity::PerTensor;
    }
    for (QuantOperator id : {QuantOperator::WeightInputHidden, QuantOperator::WeightHiddenHidden,
                             QuantOperator::BiasInputHidden, QuantOperator::BiasHiddenHidden}) {
        config.at(id).granularity = quant_lstm::QuantGranularity::PerChannel;
    }
    for (QuantOperator id : {QuantOperator::InputGateOutput, QuantOperator::ForgetGateOutput,
                             QuantOperator::OutputGateOutput}) {
        config.at(id).type = {8, true, true};
    }
    config.validate();
    return config;
}

quant_lstm::CalibrationRange rangeFor(QuantOperator id) {
    switch (id) {
        case QuantOperator::Input:
        case QuantOperator::Output:
        case QuantOperator::CellGateOutput:
        case QuantOperator::CellTanhOutput:
            return {-1.0F, 1.0F};
        case QuantOperator::CellState:
            return {-2.0F, 2.0F};
        case QuantOperator::WeightInputHidden:
        case QuantOperator::WeightHiddenHidden:
        case QuantOperator::BiasInputHidden:
        case QuantOperator::BiasHiddenHidden:
            return {-0.5F, 0.5F};
        case QuantOperator::InputGateOutput:
        case QuantOperator::ForgetGateOutput:
        case QuantOperator::OutputGateOutput:
            return {0.0F, 1.0F};
        default:
            return {-4.0F, 4.0F};
    }
}

quant_lstm::LstmQuantParams makeParams(const quant_lstm::LstmOperatorQuantConfig& config,
                                       std::int64_t hidden) {
    quant_lstm::LstmQuantizationRanges ranges;
    const std::size_t channels = static_cast<std::size_t>(4 * hidden);
    for (std::size_t index = 0; index < quant_lstm::kQuantOperatorCount; ++index) {
        const auto id = static_cast<QuantOperator>(index);
        const std::size_t count = quant_lstm::isParameterOperator(id) ? channels : 1;
        ranges.at(id).assign(count, rangeFor(id));
    }
    return quant_lstm::finalizeQuantParams(config, ranges, hidden, true);
}

std::vector<std::int32_t> quantizeTensor(const std::vector<float>& source, QuantOperator id,
                                         const quant_lstm::LstmOperatorQuantConfig& config,
                                         const quant_lstm::LstmQuantParams& params,
                                         std::size_t row_width = 0) {
    std::vector<std::int32_t> result(source.size());
    const auto& point = params.at(id);
    for (std::size_t index = 0; index < source.size(); ++index) {
        const std::size_t parameter_index = row_width == 0 ? 0 : index / row_width;
        result[index] = quant_lstm::quantization::quantize(
            source[index], point.values.at(parameter_index), config.at(id).type);
    }
    return result;
}

PreparedCase prepare(const CaseDefinition& definition) {
    PreparedCase result;
    result.definition = definition;
    result.config = makeConfig(definition.scale_mode);
    result.quant_params = makeParams(result.config, definition.shape.hidden_size);
    result.execution_params = quant_lstm::deriveLstmExecutionParams(
        result.config, result.quant_params, definition.shape.input_size);

    const std::vector<float> weight_ih{0.25F,  -0.125F, -0.375F, 0.25F,  0.125F,  0.375F,
                                       -0.25F, 0.125F,  0.375F,  0.25F,  -0.125F, -0.25F,
                                       0.25F,  0.125F,  0.375F,  -0.375F};
    const std::vector<float> weight_hh{0.125F, -0.25F,  0.25F,  0.125F, -0.125F, 0.375F,
                                       0.25F,  -0.25F,  0.375F, 0.125F, -0.25F,  0.25F,
                                       0.125F, -0.375F, 0.25F,  0.125F};
    const std::vector<float> bias_ih{0.125F, -0.125F, 0.25F, 0.125F, -0.25F, 0.125F, 0.0F, 0.25F};
    const std::vector<float> bias_hh{0.0F, 0.125F, -0.125F, 0.0F, 0.125F, -0.25F, 0.125F, 0.0F};
    const std::vector<float> initial_hidden{0.125F, -0.25F};
    const std::vector<float> initial_cell{0.25F, -0.5F};

    result.input =
        quantizeTensor(definition.input, QuantOperator::Input, result.config, result.quant_params);
    result.initial_hidden =
        quantizeTensor(initial_hidden, QuantOperator::Output, result.config, result.quant_params);
    result.initial_cell =
        quantizeTensor(initial_cell, QuantOperator::CellState, result.config, result.quant_params);
    result.weight_ih = quantizeTensor(weight_ih, QuantOperator::WeightInputHidden, result.config,
                                      result.quant_params, 2);
    result.weight_hh = quantizeTensor(weight_hh, QuantOperator::WeightHiddenHidden, result.config,
                                      result.quant_params, 2);
    result.bias_ih = quantizeTensor(bias_ih, QuantOperator::BiasInputHidden, result.config,
                                    result.quant_params, 1);
    result.bias_hh = quantizeTensor(bias_hh, QuantOperator::BiasHiddenHidden, result.config,
                                    result.quant_params, 1);
    return result;
}

Json serializedInputs(const PreparedCase& source) {
    const auto& shape = source.definition.shape;
    const std::size_t hidden = static_cast<std::size_t>(shape.hidden_size);
    const std::size_t channels = 4 * hidden;
    Json inputs;
    inputs["shape"] = tensor("int64", {4},
                             std::vector<std::int64_t>{shape.sequence_length, shape.batch_size,
                                                       shape.input_size, shape.hidden_size});
    inputs["scale_mode"] =
        scalarTensor("string", std::string(quant_lstm::scaleModeName(source.config.scale_mode)));
    inputs["bias_enabled"] = scalarTensor("bool", source.quant_params.bias_enabled);

    std::vector<std::string> operator_names;
    std::vector<std::uint8_t> bitwidths;
    std::vector<bool> is_unsigned;
    std::vector<bool> is_symmetric;
    std::vector<std::string> granularities;
    std::vector<std::int32_t> value_offsets{0};
    std::vector<std::int32_t> group_counts;
    std::vector<float> scales;
    std::vector<std::int32_t> zero_points;
    for (std::size_t index = 0; index < quant_lstm::kQuantOperatorCount; ++index) {
        const auto id = static_cast<QuantOperator>(index);
        const auto& config = source.config.at(id);
        const auto& params = source.quant_params.at(id);
        operator_names.emplace_back(quant_lstm::quantOperatorName(id));
        bitwidths.push_back(config.type.bitwidth);
        is_unsigned.push_back(config.type.is_unsigned);
        is_symmetric.push_back(config.type.is_symmetric);
        granularities.emplace_back(quant_lstm::granularityName(params.source_granularity));
        group_counts.push_back(static_cast<std::int32_t>(params.group_diagnostics.size()));
        for (const auto& value : params.values) {
            scales.push_back(value.scale);
            zero_points.push_back(value.zero_point);
        }
        value_offsets.push_back(static_cast<std::int32_t>(scales.size()));
    }
    inputs["operator_names"] = tensor("string", {quant_lstm::kQuantOperatorCount}, operator_names);
    inputs["operator_bitwidth"] = tensor("uint8", {quant_lstm::kQuantOperatorCount}, bitwidths);
    inputs["operator_is_unsigned"] = tensor("bool", {quant_lstm::kQuantOperatorCount}, is_unsigned);
    inputs["operator_is_symmetric"] =
        tensor("bool", {quant_lstm::kQuantOperatorCount}, is_symmetric);
    inputs["operator_granularity"] =
        tensor("string", {quant_lstm::kQuantOperatorCount}, granularities);
    inputs["quant_param_value_offsets"] =
        tensor("int32", {quant_lstm::kQuantOperatorCount + 1}, value_offsets);
    inputs["quant_param_group_counts"] =
        tensor("int32", {quant_lstm::kQuantOperatorCount}, group_counts);
    inputs["quant_param_scales"] = floatTensor({scales.size()}, scales);
    inputs["quant_param_zero_points"] = tensor("int32", {zero_points.size()}, zero_points);

    inputs["input"] = tensor(
        "int32",
        {static_cast<std::size_t>(shape.sequence_length),
         static_cast<std::size_t>(shape.batch_size), static_cast<std::size_t>(shape.input_size)},
        source.input);
    inputs["initial_hidden"] = tensor("int32", {static_cast<std::size_t>(shape.batch_size), hidden},
                                      source.initial_hidden);
    inputs["initial_cell"] =
        tensor("int32", {static_cast<std::size_t>(shape.batch_size), hidden}, source.initial_cell);
    inputs["weight_ih"] =
        tensor("int32", {channels, static_cast<std::size_t>(shape.input_size)}, source.weight_ih);
    inputs["weight_hh"] = tensor("int32", {channels, hidden}, source.weight_hh);
    inputs["bias_ih"] = tensor("int32", {channels}, source.bias_ih);
    inputs["bias_hh"] = tensor("int32", {channels}, source.bias_hh);
    return inputs;
}

template <typename T>
std::vector<T> gateSlice(const std::vector<T>& source, std::size_t steps, std::size_t batch,
                         std::size_t hidden, quant_lstm::GateKind gate) {
    std::vector<T> result;
    result.reserve(steps * batch * hidden);
    const std::size_t channels = quant_lstm::kGateCount * hidden;
    for (std::size_t step = 0; step < steps; ++step) {
        for (std::size_t row = 0; row < batch; ++row) {
            const std::size_t base =
                (step * batch + row) * channels + quant_lstm::gateOffset(gate, hidden);
            result.insert(result.end(), source.begin() + base, source.begin() + base + hidden);
        }
    }
    return result;
}

std::int32_t checkedGridValue(float value) {
    if (!std::isfinite(value) || std::nearbyint(value) != value ||
        value < static_cast<float>(INT32_MIN) || value > static_cast<float>(INT32_MAX)) {
        throw std::runtime_error("CPU FP32 reference 产生了非 int32 q 网格值");
    }
    return static_cast<std::int32_t>(value);
}

std::vector<std::int32_t> checkedGridValues(const std::vector<float>& values) {
    std::vector<std::int32_t> result;
    result.reserve(values.size());
    for (float value : values) {
        result.push_back(checkedGridValue(value));
    }
    return result;
}

std::vector<std::int64_t> checkedWideIntegers(const std::vector<float>& values) {
    std::vector<std::int64_t> result;
    result.reserve(values.size());
    for (float value : values) {
        if (!std::isfinite(value) || std::nearbyint(value) != value ||
            value < static_cast<float>(INT64_MIN) || value > static_cast<float>(INT64_MAX)) {
            throw std::runtime_error("CPU FP32 reference 产生了非 int64 整数宽值");
        }
        result.push_back(static_cast<std::int64_t>(value));
    }
    return result;
}

template <typename T>
void addCheckpointSet(Json& checkpoints, const std::vector<T>& linear_ih,
                      const std::vector<T>& linear_hh, const std::vector<T>& gate_inputs,
                      const std::vector<T>& gate_outputs, const std::vector<T>& cell_states,
                      const std::vector<T>& cell_tanh_outputs, const std::vector<T>& hidden_outputs,
                      const std::vector<T>& final_hidden, const std::vector<T>& final_cell,
                      std::size_t steps, std::size_t batch, std::size_t hidden) {
    const auto qTensor = [&](const std::vector<T>& values, const std::vector<std::size_t>& shape) {
        std::vector<std::int32_t> converted;
        converted.reserve(values.size());
        for (const T value : values) {
            converted.push_back(static_cast<std::int32_t>(value));
        }
        return tensor("int32", shape, converted);
    };
    const std::vector<std::size_t> gate_shape{steps, batch, hidden};
    checkpoints["weight_ih_linear"] = qTensor(linear_ih, {steps, batch, 4, hidden});
    checkpoints["weight_hh_linear"] = qTensor(linear_hh, {steps, batch, 4, hidden});
    const std::array<std::pair<const char*, quant_lstm::GateKind>, 4> gates{{
        {"input", quant_lstm::GateKind::Input},
        {"forget", quant_lstm::GateKind::Forget},
        {"cell", quant_lstm::GateKind::Cell},
        {"output", quant_lstm::GateKind::Output},
    }};
    for (const auto& [name, gate] : gates) {
        checkpoints[std::string(name) + "_gate_input"] =
            qTensor(gateSlice(gate_inputs, steps, batch, hidden, gate), gate_shape);
        checkpoints[std::string(name) + "_gate_output"] =
            qTensor(gateSlice(gate_outputs, steps, batch, hidden, gate), gate_shape);
    }
    checkpoints["cell"] = qTensor(cell_states, {steps, batch, hidden});
    checkpoints["cell_tanh"] = qTensor(cell_tanh_outputs, {steps, batch, hidden});
    checkpoints["output"] = qTensor(hidden_outputs, {steps, batch, hidden});
    checkpoints["h_n"] = qTensor(final_hidden, {batch, hidden});
    checkpoints["c_n"] = qTensor(final_cell, {batch, hidden});
}

Json intExpected(const PreparedCase& source) {
    const auto& shape = source.definition.shape;
    const std::size_t steps = static_cast<std::size_t>(shape.sequence_length);
    const std::size_t batch = static_cast<std::size_t>(shape.batch_size);
    const std::size_t hidden = static_cast<std::size_t>(shape.hidden_size);
    std::vector<std::int32_t> output(steps * batch * hidden);
    std::vector<std::int32_t> final_hidden(batch * hidden);
    std::vector<std::int32_t> final_cell(batch * hidden);
    quant_lstm::LstmInt32ReferenceTrace trace;
    quant_lstm::lstmForwardInt32CpuReference(
        shape,
        {source.weight_ih.data(), source.weight_hh.data(), source.bias_ih.data(),
         source.bias_hh.data()},
        source.input.data(), source.initial_hidden.data(), source.initial_cell.data(),
        source.config, source.quant_params, source.execution_params, output.data(),
        final_hidden.data(), final_cell.data(), &trace);

    Json checkpoints;
    addCheckpointSet(checkpoints, trace.weight_ih_linear, trace.weight_hh_linear, trace.gate_inputs,
                     trace.gate_outputs, trace.cell_states, trace.cell_tanh_outputs,
                     trace.hidden_outputs, final_hidden, final_cell, steps, batch, hidden);
    std::vector<std::int64_t> cell_products;
    cell_products.reserve(2 * trace.p_forget.size());
    std::vector<std::string> scaled_contributions;
    scaled_contributions.reserve(2 * trace.scaled_forget_contributions.size());
    for (std::size_t index = 0; index < trace.p_forget.size(); ++index) {
        cell_products.push_back(trace.p_forget[index]);
        cell_products.push_back(trace.p_input[index]);
        scaled_contributions.push_back(trace.scaled_forget_contributions[index]);
        scaled_contributions.push_back(trace.scaled_input_contributions[index]);
    }
    Json diagnostics;
    diagnostics["cell_products"] = tensor("int64", {steps, batch, hidden, 2}, cell_products);
    diagnostics["scaled_cell_contributions"] =
        tensor("string", {steps, batch, hidden, 2}, scaled_contributions);
    diagnostics["cell_pre_round_sum"] =
        tensor("string", {steps, batch, hidden}, trace.cell_wide_sums);
    diagnostics["hidden_raw_product"] =
        tensor("int64", {steps, batch, hidden}, trace.hidden_products);
    return Json{{"checkpoints", checkpoints}, {"diagnostics", diagnostics}};
}

Json fpExpected(const PreparedCase& source) {
    const auto& shape = source.definition.shape;
    const std::size_t steps = static_cast<std::size_t>(shape.sequence_length);
    const std::size_t batch = static_cast<std::size_t>(shape.batch_size);
    const std::size_t hidden = static_cast<std::size_t>(shape.hidden_size);
    const auto toFloat = [](const std::vector<std::int32_t>& values) {
        return std::vector<float>(values.begin(), values.end());
    };
    const auto input = toFloat(source.input);
    const auto initial_hidden = toFloat(source.initial_hidden);
    const auto initial_cell = toFloat(source.initial_cell);
    const auto weight_ih = toFloat(source.weight_ih);
    const auto weight_hh = toFloat(source.weight_hh);
    const auto bias_ih = toFloat(source.bias_ih);
    const auto bias_hh = toFloat(source.bias_hh);
    std::vector<float> output(steps * batch * hidden);
    std::vector<float> final_hidden(batch * hidden);
    std::vector<float> final_cell(batch * hidden);
    quant_lstm::LstmFpReferenceTrace trace;
    quant_lstm::lstmForwardQuantizedFpCpuReference(
        shape, {weight_ih.data(), weight_hh.data(), bias_ih.data(), bias_hh.data()}, input.data(),
        initial_hidden.data(), initial_cell.data(), source.config, source.quant_params,
        source.execution_params, output.data(), final_hidden.data(), final_cell.data(), &trace);

    Json checkpoints;
    addCheckpointSet(
        checkpoints, checkedGridValues(trace.weight_ih_linear),
        checkedGridValues(trace.weight_hh_linear), checkedGridValues(trace.gate_inputs),
        checkedGridValues(trace.gate_outputs), checkedGridValues(trace.cell_states),
        checkedGridValues(trace.cell_tanh_outputs), checkedGridValues(trace.hidden_outputs),
        checkedGridValues(final_hidden), checkedGridValues(final_cell), steps, batch, hidden);
    std::vector<float> cell_products;
    std::vector<float> scaled_contributions;
    cell_products.reserve(2 * trace.p_forget.size());
    scaled_contributions.reserve(2 * trace.scaled_forget_contributions.size());
    for (std::size_t index = 0; index < trace.p_forget.size(); ++index) {
        cell_products.push_back(trace.p_forget[index]);
        cell_products.push_back(trace.p_input[index]);
        scaled_contributions.push_back(trace.scaled_forget_contributions[index]);
        scaled_contributions.push_back(trace.scaled_input_contributions[index]);
    }
    Json diagnostics;
    diagnostics["cell_products"] =
        tensor("int64", {steps, batch, hidden, 2}, checkedWideIntegers(cell_products));
    diagnostics["scaled_cell_contributions"] =
        floatTensor({steps, batch, hidden, 2}, scaled_contributions);
    diagnostics["cell_pre_round_sum"] = floatTensor({steps, batch, hidden}, trace.cell_wide_sums);
    diagnostics["hidden_raw_product"] =
        tensor("int64", {steps, batch, hidden}, checkedWideIntegers(trace.hidden_products));
    return Json{{"checkpoints", checkpoints}, {"diagnostics", diagnostics}};
}

Json generateDocument(const PreparedCase& source, bool fp_carrier) {
    return Json{
        {"schema_version", 1},
        {"case_id", source.definition.id_prefix + (fp_carrier ? "_cpu_fp32" : "_cpu_int32")},
        {"kind", source.definition.kind},
        {"execution_model", fp_carrier ? "cpu_fp32" : "cpu_int32"},
        {"attributes", Json::object()},
        {"inputs", serializedInputs(source)},
        {"expected", fp_carrier ? fpExpected(source) : intExpected(source)},
    };
}

void writeDocument(const std::filesystem::path& output_directory, const Json& document) {
    const std::filesystem::path kind_directory =
        output_directory / document.at("kind").get<std::string>();
    std::filesystem::create_directories(kind_directory);
    const std::filesystem::path output_path =
        kind_directory / (document.at("case_id").get<std::string>() + ".json");
    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error("无法写入 " + output_path.string());
    }
    output << document.dump(2) << '\n';
    if (!output) {
        throw std::runtime_error("写入失败 " + output_path.string());
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::invalid_argument("usage: golden_reference_generator <golden-spec-dir>");
        }
        const std::filesystem::path output_directory(argv[1]);
        const std::array<CaseDefinition, 2> cases{{
            {"reference_cell_t1_b1_i2_h2",
             "cell",
             {1, 1, 2, 2},
             quant_lstm::quantization::ScaleMode::Affine,
             {-0.75F, 0.25F}},
            {"reference_recurrent_t3_b1_i2_h2",
             "recurrent",
             {3, 1, 2, 2},
             quant_lstm::quantization::ScaleMode::Pot2,
             {-0.75F, 0.25F, 0.5F, -0.125F, 0.25F, 0.75F}},
        }};
        for (const auto& definition : cases) {
            const PreparedCase prepared = prepare(definition);
            writeDocument(output_directory, generateDocument(prepared, false));
            writeDocument(output_directory, generateDocument(prepared, true));
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
