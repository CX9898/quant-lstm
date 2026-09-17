#include "golden_fixtures.h"
#include "lstm/forward_cpu.h"
#include "lstm/gate_layout.h"
#include "lstm/lstm_execution_params.h"
#include "lstm/quant_config_loader.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace {

using Json = nlohmann::json;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

const Json& tensorData(const Json& tensor_map, std::string_view name) {
    return tensor_map.at(std::string(name)).at("data");
}

template <typename T>
std::vector<T> values(const Json& tensor_map, std::string_view name) {
    return tensorData(tensor_map, name).get<std::vector<T>>();
}

std::vector<float> floatValues(const Json& tensor_map,
                               std::string_view name) {
    std::vector<float> result;
    for (const Json& item : tensorData(tensor_map, name)) {
        result.push_back(quant_lstm::parseCanonicalFloat32Value(
            item.get<std::string>()));
    }
    return result;
}

std::size_t shapeElementCount(const Json& shape) {
    std::size_t count = 1;
    for (const Json& dimension : shape) {
        const std::size_t value = dimension.get<std::size_t>();
        if (value == 0 ||
            count > std::numeric_limits<std::size_t>::max() / value) {
            throw std::overflow_error("Reference Golden tensor shape 溢出");
        }
        count *= value;
    }
    return count;
}

void validateTensorMap(const Json& tensors) {
    for (const auto& [name, tensor] : tensors.items()) {
        require(shapeElementCount(tensor.at("shape")) ==
                    tensor.at("data").size(),
                "Reference Golden shape/data mismatch: " + name);
        if (tensor.at("dtype") == "float32") {
            for (const Json& item : tensor.at("data")) {
                const std::string text = item.get<std::string>();
                static_cast<void>(
                    quant_lstm::parseCanonicalFloat32Value(text));
            }
        }
    }
}

struct LoadedCase {
    quant_lstm::LstmShape shape{};
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

LoadedCase loadCase(const Json& inputs) {
    LoadedCase result;
    const auto shape = values<std::int64_t>(inputs, "shape");
    require(shape.size() == 4, "Reference Golden shape 必须为 T,B,I,H");
    result.shape = {shape[0], shape[1], shape[2], shape[3]};
    result.config.scale_mode = quant_lstm::parseScaleMode(
        tensorData(inputs, "scale_mode").at(0).get<std::string>());

    const auto operator_names =
        values<std::string>(inputs, "operator_names");
    const auto bitwidths =
        values<std::uint8_t>(inputs, "operator_bitwidth");
    const auto is_unsigned =
        values<bool>(inputs, "operator_is_unsigned");
    const auto is_symmetric =
        values<bool>(inputs, "operator_is_symmetric");
    const auto granularities =
        values<std::string>(inputs, "operator_granularity");
    const auto offsets =
        values<std::int32_t>(inputs, "quant_param_value_offsets");
    const auto group_counts =
        values<std::int32_t>(inputs, "quant_param_group_counts");
    const auto scales = floatValues(inputs, "quant_param_scales");
    const auto zero_points =
        values<std::int32_t>(inputs, "quant_param_zero_points");

    require(operator_names.size() == quant_lstm::kQuantOperatorCount &&
                bitwidths.size() == quant_lstm::kQuantOperatorCount &&
                is_unsigned.size() == quant_lstm::kQuantOperatorCount &&
                is_symmetric.size() == quant_lstm::kQuantOperatorCount &&
                granularities.size() == quant_lstm::kQuantOperatorCount &&
                group_counts.size() == quant_lstm::kQuantOperatorCount &&
                offsets.size() == quant_lstm::kQuantOperatorCount + 1 &&
                scales.size() == zero_points.size(),
            "Reference Golden 量化元数据长度非法");
    require(offsets.front() == 0 &&
                offsets.back() == static_cast<std::int32_t>(scales.size()),
            "Reference Golden quant param offsets 非法");

    result.quant_params.hidden_size = result.shape.hidden_size;
    result.quant_params.bias_enabled =
        tensorData(inputs, "bias_enabled").at(0).get<bool>();
    for (std::size_t index = 0; index < quant_lstm::kQuantOperatorCount;
         ++index) {
        const auto id = static_cast<quant_lstm::QuantOperator>(index);
        require(operator_names[index] == quant_lstm::quantOperatorName(id),
                "Reference Golden operator 顺序非法");
        require(offsets[index] >= 0 && offsets[index + 1] >= offsets[index] &&
                    offsets[index + 1] <=
                        static_cast<std::int32_t>(scales.size()) &&
                    group_counts[index] >= 0,
                "Reference Golden quant param span 非法");
        auto& config = result.config.at(id);
        config.type = {bitwidths[index], is_unsigned[index],
                       is_symmetric[index]};
        config.granularity =
            quant_lstm::parseGranularity(granularities[index]);
        auto& params = result.quant_params.operators[index];
        params.source_granularity = config.granularity;
        for (std::int32_t offset = offsets[index];
             offset < offsets[index + 1]; ++offset) {
            params.values.push_back(
                {scales[static_cast<std::size_t>(offset)],
                 zero_points[static_cast<std::size_t>(offset)]});
        }
        params.group_diagnostics.resize(
            static_cast<std::size_t>(group_counts[index]));
    }
    result.config.validate();
    result.quant_params.validate(result.config);
    result.execution_params = quant_lstm::deriveLstmExecutionParams(
        result.config, result.quant_params, result.shape.input_size);

    result.input = values<std::int32_t>(inputs, "input");
    result.initial_hidden =
        values<std::int32_t>(inputs, "initial_hidden");
    result.initial_cell = values<std::int32_t>(inputs, "initial_cell");
    result.weight_ih = values<std::int32_t>(inputs, "weight_ih");
    result.weight_hh = values<std::int32_t>(inputs, "weight_hh");
    result.bias_ih = values<std::int32_t>(inputs, "bias_ih");
    result.bias_hh = values<std::int32_t>(inputs, "bias_hh");
    return result;
}

template <typename T>
std::vector<T> gateSlice(const std::vector<T>& source, std::size_t steps,
                         std::size_t batch, std::size_t hidden,
                         quant_lstm::GateKind gate) {
    std::vector<T> result;
    result.reserve(steps * batch * hidden);
    const std::size_t channels = quant_lstm::kGateCount * hidden;
    for (std::size_t step = 0; step < steps; ++step) {
        for (std::size_t row = 0; row < batch; ++row) {
            const std::size_t base =
                (step * batch + row) * channels +
                quant_lstm::gateOffset(gate, hidden);
            result.insert(result.end(), source.begin() + base,
                          source.begin() + base + hidden);
        }
    }
    return result;
}

template <typename Actual, typename Expected>
void compareExact(const std::vector<Actual>& actual,
                  const std::vector<Expected>& expected,
                  const std::string& case_id, std::string_view field) {
    require(actual.size() == expected.size(),
            case_id + " " + std::string(field) + " length mismatch");
    for (std::size_t index = 0; index < actual.size(); ++index) {
        require(actual[index] == static_cast<Actual>(expected[index]),
                case_id + " " + std::string(field) + "[" +
                    std::to_string(index) + "] mismatch");
    }
}

template <typename T>
void checkCheckpoints(const Json& expected, const std::vector<T>& linear_ih,
                      const std::vector<T>& linear_hh,
                      const std::vector<T>& gate_inputs,
                      const std::vector<T>& gate_outputs,
                      const std::vector<T>& cell,
                      const std::vector<T>& cell_tanh,
                      const std::vector<T>& output,
                      const std::vector<T>& final_hidden,
                      const std::vector<T>& final_cell,
                      const quant_lstm::LstmShape& shape,
                      const std::string& case_id) {
    static const std::unordered_set<std::string> kCheckpointNames{
        "weight_ih_linear", "weight_hh_linear",
        "input_gate_input", "forget_gate_input", "cell_gate_input",
        "output_gate_input", "input_gate_output", "forget_gate_output",
        "cell_gate_output", "output_gate_output", "cell", "cell_tanh",
        "output", "h_n", "c_n"};
    require(expected.size() == kCheckpointNames.size(),
            case_id + " checkpoint 字段数非法");
    for (const auto& [name, unused] : expected.items()) {
        static_cast<void>(unused);
        require(kCheckpointNames.count(name) == 1,
                case_id + " 非法 checkpoint: " + name);
    }
    const std::size_t steps = static_cast<std::size_t>(shape.sequence_length);
    const std::size_t batch = static_cast<std::size_t>(shape.batch_size);
    const std::size_t hidden = static_cast<std::size_t>(shape.hidden_size);
    const auto check = [&](const std::vector<T>& actual,
                           std::string_view name) {
        compareExact(actual, values<std::int32_t>(expected, name), case_id,
                     name);
    };
    check(linear_ih, "weight_ih_linear");
    check(linear_hh, "weight_hh_linear");
    const std::array<std::pair<const char*, quant_lstm::GateKind>, 4> gates{{
        {"input", quant_lstm::GateKind::Input},
        {"forget", quant_lstm::GateKind::Forget},
        {"cell", quant_lstm::GateKind::Cell},
        {"output", quant_lstm::GateKind::Output},
    }};
    for (const auto& [name, gate] : gates) {
        check(gateSlice(gate_inputs, steps, batch, hidden, gate),
              std::string(name) + "_gate_input");
        check(gateSlice(gate_outputs, steps, batch, hidden, gate),
              std::string(name) + "_gate_output");
    }
    check(cell, "cell");
    check(cell_tanh, "cell_tanh");
    check(output, "output");
    check(final_hidden, "h_n");
    check(final_cell, "c_n");
}

void checkDiagnosticNames(const Json& diagnostics,
                          const std::string& case_id) {
    static const std::unordered_set<std::string> kDiagnosticNames{
        "cell_products", "scaled_cell_contributions",
        "cell_pre_round_sum", "hidden_raw_product"};
    require(diagnostics.size() == kDiagnosticNames.size(),
            case_id + " diagnostics 必须恰含四类中间值");
    for (const auto& [name, unused] : diagnostics.items()) {
        static_cast<void>(unused);
        require(kDiagnosticNames.count(name) == 1,
                case_id + " 非法 diagnostic: " + name);
    }
}

void checkInt32Case(const LoadedCase& loaded, const Json& expected,
                    const std::string& case_id) {
    const std::size_t steps =
        static_cast<std::size_t>(loaded.shape.sequence_length);
    const std::size_t batch =
        static_cast<std::size_t>(loaded.shape.batch_size);
    const std::size_t hidden =
        static_cast<std::size_t>(loaded.shape.hidden_size);
    std::vector<std::int32_t> output(steps * batch * hidden);
    std::vector<std::int32_t> final_hidden(batch * hidden);
    std::vector<std::int32_t> final_cell(batch * hidden);
    quant_lstm::LstmInt32ReferenceTrace trace;
    quant_lstm::lstmForwardInt32CpuReference(
        loaded.shape,
        {loaded.weight_ih.data(), loaded.weight_hh.data(),
         loaded.bias_ih.data(), loaded.bias_hh.data()},
        loaded.input.data(), loaded.initial_hidden.data(),
        loaded.initial_cell.data(), loaded.config, loaded.quant_params,
        loaded.execution_params, output.data(), final_hidden.data(),
        final_cell.data(), &trace);
    checkCheckpoints(expected.at("checkpoints"), trace.weight_ih_linear,
                     trace.weight_hh_linear, trace.gate_inputs,
                     trace.gate_outputs, trace.cell_states,
                     trace.cell_tanh_outputs, trace.hidden_outputs,
                     final_hidden, final_cell, loaded.shape, case_id);

    const Json& diagnostics = expected.at("diagnostics");
    checkDiagnosticNames(diagnostics, case_id);
    std::vector<std::int64_t> products;
    std::vector<std::string> scaled;
    for (std::size_t index = 0; index < trace.p_forget.size(); ++index) {
        products.push_back(trace.p_forget[index]);
        products.push_back(trace.p_input[index]);
        scaled.push_back(trace.scaled_forget_contributions[index]);
        scaled.push_back(trace.scaled_input_contributions[index]);
    }
    compareExact(products,
                 values<std::int64_t>(diagnostics, "cell_products"), case_id,
                 "cell_products");
    compareExact(scaled,
                 values<std::string>(diagnostics,
                                     "scaled_cell_contributions"),
                 case_id, "scaled_cell_contributions");
    compareExact(trace.cell_wide_sums,
                 values<std::string>(diagnostics, "cell_pre_round_sum"),
                 case_id, "cell_pre_round_sum");
    compareExact(trace.hidden_products,
                 values<std::int64_t>(diagnostics, "hidden_raw_product"),
                 case_id, "hidden_raw_product");
}

void checkFp32Case(const LoadedCase& loaded, const Json& expected,
                   const std::string& case_id) {
    const auto toFloat = [](const std::vector<std::int32_t>& source) {
        return std::vector<float>(source.begin(), source.end());
    };
    const auto input = toFloat(loaded.input);
    const auto initial_hidden = toFloat(loaded.initial_hidden);
    const auto initial_cell = toFloat(loaded.initial_cell);
    const auto weight_ih = toFloat(loaded.weight_ih);
    const auto weight_hh = toFloat(loaded.weight_hh);
    const auto bias_ih = toFloat(loaded.bias_ih);
    const auto bias_hh = toFloat(loaded.bias_hh);
    const std::size_t steps =
        static_cast<std::size_t>(loaded.shape.sequence_length);
    const std::size_t batch =
        static_cast<std::size_t>(loaded.shape.batch_size);
    const std::size_t hidden =
        static_cast<std::size_t>(loaded.shape.hidden_size);
    std::vector<float> output(steps * batch * hidden);
    std::vector<float> final_hidden(batch * hidden);
    std::vector<float> final_cell(batch * hidden);
    quant_lstm::LstmFpReferenceTrace trace;
    quant_lstm::lstmForwardQuantizedFpCpuReference(
        loaded.shape,
        {weight_ih.data(), weight_hh.data(), bias_ih.data(), bias_hh.data()},
        input.data(), initial_hidden.data(), initial_cell.data(), loaded.config,
        loaded.quant_params, loaded.execution_params, output.data(),
        final_hidden.data(), final_cell.data(), &trace);
    checkCheckpoints(expected.at("checkpoints"), trace.weight_ih_linear,
                     trace.weight_hh_linear, trace.gate_inputs,
                     trace.gate_outputs, trace.cell_states,
                     trace.cell_tanh_outputs, trace.hidden_outputs,
                     final_hidden, final_cell, loaded.shape, case_id);

    const Json& diagnostics = expected.at("diagnostics");
    checkDiagnosticNames(diagnostics, case_id);
    std::vector<float> products;
    std::vector<float> scaled;
    for (std::size_t index = 0; index < trace.p_forget.size(); ++index) {
        products.push_back(trace.p_forget[index]);
        products.push_back(trace.p_input[index]);
        scaled.push_back(trace.scaled_forget_contributions[index]);
        scaled.push_back(trace.scaled_input_contributions[index]);
    }
    compareExact(products,
                 values<std::int64_t>(diagnostics, "cell_products"), case_id,
                 "cell_products");
    compareExact(scaled,
                 floatValues(diagnostics, "scaled_cell_contributions"),
                 case_id, "scaled_cell_contributions");
    compareExact(trace.cell_wide_sums,
                 floatValues(diagnostics, "cell_pre_round_sum"), case_id,
                 "cell_pre_round_sum");
    compareExact(trace.hidden_products,
                 values<std::int64_t>(diagnostics, "hidden_raw_product"),
                 case_id, "hidden_raw_product");
}

}  // namespace

int main() {
    try {
        std::size_t checked = 0;
        for (const auto& fixture : quant_lstm::test::kGoldenDocuments) {
            if (fixture.kind != "cell" && fixture.kind != "recurrent") {
                continue;
            }
            const Json document = Json::parse(fixture.json);
            const std::string case_id =
                document.at("case_id").get<std::string>();
            require(case_id == fixture.case_id,
                    "Reference Golden fixture case_id mismatch");
            require(document.at("kind").get<std::string>() == fixture.kind,
                    case_id + " kind mismatch");
            require(document.at("execution_model").get<std::string>() ==
                        fixture.execution_model,
                    case_id + " execution_model mismatch");
            validateTensorMap(document.at("inputs"));
            validateTensorMap(document.at("expected").at("checkpoints"));
            validateTensorMap(document.at("expected").at("diagnostics"));
            const LoadedCase loaded = loadCase(document.at("inputs"));
            if (fixture.kind == "cell") {
                require(loaded.shape.sequence_length == 1 &&
                            loaded.shape.batch_size == 1,
                        case_id + " cell shape 必须为 T=1,B=1");
            } else {
                require(loaded.shape.sequence_length == 3 &&
                            loaded.shape.batch_size == 1,
                        case_id + " recurrent shape 必须为 T=3,B=1");
            }
            if (fixture.execution_model == "cpu_int32") {
                checkInt32Case(loaded, document.at("expected"), case_id);
            } else if (fixture.execution_model == "cpu_fp32") {
                checkFp32Case(loaded, document.at("expected"), case_id);
            } else {
                throw std::invalid_argument(case_id +
                                            " execution_model 非法");
            }
            ++checked;
        }
        require(checked == 4, "Reference Golden 必须恰有四个载体分离用例");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
