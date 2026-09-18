#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/numeric_safety_report.h"
#include "common/synthetic_precision_fixture.h"
#include "cuda/cuda_test_support.cuh"
#include "lstm/forward_cpu.h"
#include "lstm/forward_quantized_fp_cuda.h"

namespace {

using Json = nlohmann::json;
using quant_lstm::test::DeviceBuffer;

#ifndef QUANT_LSTM_SOURCE_DIR
#define QUANT_LSTM_SOURCE_DIR "."
#endif

struct Arguments {
    std::string selection;
    std::filesystem::path report;
};

Arguments parseArguments(int argc, char** argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if ((option == "--selection" || option == "--report") && index + 1 >= argc) {
            throw std::invalid_argument(option + " 缺少值");
        }
        if (option == "--selection") {
            result.selection = argv[++index];
        } else if (option == "--report") {
            result.report = argv[++index];
        } else {
            throw std::invalid_argument("未知参数: " + option);
        }
    }
    if (result.selection != "basic" && result.selection != "strict") {
        throw std::invalid_argument("--selection 必须为 basic 或 strict");
    }
    if (result.report.empty()) {
        throw std::invalid_argument("--report 不能为空");
    }
    return result;
}

Json readJson(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("无法读取 JSON: " + path.string());
    }
    Json result;
    stream >> result;
    return result;
}

void writeJson(const std::filesystem::path& path, const Json& document) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream stream(path);
    if (!stream) {
        throw std::runtime_error("无法写入报告: " + path.string());
    }
    stream << document.dump(2) << '\n';
}

std::vector<float> asFloat(const std::vector<std::int32_t>& source) {
    return std::vector<float>(source.begin(), source.end());
}

std::vector<float> sliceLastState(const std::vector<float>& source, std::size_t state_count) {
    if (source.size() < state_count) {
        throw std::invalid_argument("checkpoint state 长度非法");
    }
    return {source.end() - static_cast<std::ptrdiff_t>(state_count), source.end()};
}

void requireExact(const std::vector<float>& actual, const std::vector<float>& expected,
                  const std::string& field) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(field + " 长度不一致");
    }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (actual[index] != expected[index]) {
            throw std::runtime_error(
                field + "[" + std::to_string(index) + "] CUDA/CPU exact checkpoint 不一致: " +
                std::to_string(actual[index]) + " != " + std::to_string(expected[index]));
        }
    }
}

bool executionIsExact(const quant_lstm::LstmExecutionParams& execution) {
    const auto exact = [](const auto& report) {
        return report.fp32_accumulation ==
               quant_lstm::quantization::Fp32AccumulationClass::ExactIntegerRange;
    };
    return std::all_of(execution.diagnostics.input_hidden_linear.begin(),
                       execution.diagnostics.input_hidden_linear.end(), exact) &&
           std::all_of(execution.diagnostics.hidden_hidden_linear.begin(),
                       execution.diagnostics.hidden_hidden_linear.end(), exact) &&
           std::all_of(execution.diagnostics.gates.begin(), execution.diagnostics.gates.end(),
                       exact) &&
           exact(execution.diagnostics.cell) && exact(execution.diagnostics.hidden);
}

bool tf32OperandsAreExact(const quant_lstm::LstmOperatorQuantConfig& config) {
    const auto exactlyRepresentable = [&](quant_lstm::QuantOperator id) {
        const auto range = config.at(id).type.range();
        return range.minimum >= -2048 && range.maximum <= 2048;
    };
    return exactlyRepresentable(quant_lstm::QuantOperator::Input) &&
           exactlyRepresentable(quant_lstm::QuantOperator::Output) &&
           exactlyRepresentable(quant_lstm::QuantOperator::WeightInputHidden) &&
           exactlyRepresentable(quant_lstm::QuantOperator::WeightHiddenHidden);
}

Json metricJson(const quant_lstm::test::SyntheticNumericMetrics& metrics) {
    Json result{
        {"max_absolute_error", metrics.accuracy.maximum_absolute_error},
        {"mae", metrics.accuracy.mean_absolute_error},
        {"mse", metrics.accuracy.mean_squared_error},
        {"sqnr_db", metrics.sqnr_db},
        {"saturation_rate", metrics.saturation_rate},
    };
    if (metrics.accuracy.cosine_not_applicable) {
        result["cosine_similarity"] = nullptr;
        result["cosine_status"] = "both_near_zero";
    } else if (metrics.accuracy.one_sided_zero_norm) {
        result["cosine_similarity"] = nullptr;
        result["cosine_status"] = "one_near_zero";
    } else {
        result["cosine_similarity"] = metrics.accuracy.cosine_similarity;
        result["cosine_status"] = "valid";
    }
    return result;
}

bool passes(const quant_lstm::test::SyntheticNumericMetrics& metrics, const Json& threshold) {
    const auto& accuracy = metrics.accuracy;
    const bool cosine =
        accuracy.cosine_not_applicable
            ? accuracy.mean_absolute_error == 0.0 && accuracy.mean_squared_error == 0.0
            : !accuracy.one_sided_zero_norm &&
                  accuracy.cosine_similarity >= threshold.at("cosine_similarity").get<double>();
    return accuracy.mean_absolute_error < threshold.at("mae").get<double>() &&
           accuracy.mean_squared_error < threshold.at("mse").get<double>() && cosine;
}

std::vector<Json> selectProfiles(const Json& matrix, const std::string& selection) {
    std::vector<Json> candidates;
    for (const Json& item : matrix.at("cases")) {
        if (item.at("backend") == "cuda_fp32" && item.at("tier") == selection) {
            candidates.push_back(item);
        }
    }
    if (selection == "basic") {
        if (candidates.empty()) {
            throw std::runtime_error("v2 matrix 缺少 CUDA basic case");
        }
        const auto pedantic =
            std::find_if(candidates.begin(), candidates.end(),
                         [](const Json& item) { return item.at("math_mode") == "pedantic"; });
        return {pedantic == candidates.end() ? candidates.front() : *pedantic};
    }

    constexpr std::array<const char*, 4> kCaseIds{
        "stage4_strict_cuda_fp32_minimal_int8", "stage4_strict_cuda_fp32_short_int16",
        "stage4_strict_cuda_fp32_nonaligned_mixed", "stage4_strict_cuda_fp32_long_int8"};
    std::vector<Json> selected;
    for (const char* case_id : kCaseIds) {
        const auto match =
            std::find_if(candidates.begin(), candidates.end(),
                         [&](const Json& item) { return item.at("case_id") == case_id; });
        if (match == candidates.end()) {
            throw std::runtime_error(std::string("v2 matrix 缺少定向 CUDA strict case: ") +
                                     case_id);
        }
        selected.push_back(*match);
    }
    return selected;
}

Json runProfile(const Json& profile, const Json& thresholds, bool caller_workspace) {
    const auto fixture = quant_lstm::test::makeSyntheticPrecisionFixture(profile);
    const std::size_t steps = static_cast<std::size_t>(fixture.shape.sequence_length);
    const std::size_t batch = static_cast<std::size_t>(fixture.shape.batch_size);
    const std::size_t hidden = static_cast<std::size_t>(fixture.shape.hidden_size);
    const std::size_t channels = 4 * hidden;
    const std::size_t output_count = steps * batch * hidden;
    const std::size_t state_count = batch * hidden;
    const std::size_t gate_count = steps * batch * channels;

    DeviceBuffer<float> input(fixture.master.input.size());
    DeviceBuffer<float> weight_ih(fixture.master.weight_ih.size());
    DeviceBuffer<float> weight_hh(fixture.master.weight_hh.size());
    DeviceBuffer<float> bias_ih(fixture.master.bias_ih.size());
    DeviceBuffer<float> bias_hh(fixture.master.bias_hh.size());
    DeviceBuffer<float> initial_hidden(fixture.master.h0.size());
    DeviceBuffer<float> initial_cell(fixture.master.c0.size());
    DeviceBuffer<float> output(output_count);
    DeviceBuffer<float> final_hidden(state_count);
    DeviceBuffer<float> final_cell(state_count);
    input.copyFrom(fixture.master.input);
    weight_ih.copyFrom(fixture.master.weight_ih);
    weight_hh.copyFrom(fixture.master.weight_hh);
    bias_ih.copyFrom(fixture.master.bias_ih);
    bias_hh.copyFrom(fixture.master.bias_hh);
    initial_hidden.copyFrom(fixture.master.h0);
    initial_cell.copyFrom(fixture.master.c0);

    DeviceBuffer<float> linear_ih(gate_count);
    DeviceBuffer<float> linear_hh(gate_count);
    DeviceBuffer<float> gate_inputs(gate_count);
    DeviceBuffer<float> gate_outputs(gate_count);
    DeviceBuffer<float> cell_states(output_count);
    DeviceBuffer<float> cell_tanh(output_count);
    DeviceBuffer<float> hidden_outputs(output_count);
    quant_lstm::LstmQuantizedFpCudaCheckpoints checkpoints{
        linear_ih.get(),   linear_hh.get(), gate_inputs.get(),   gate_outputs.get(),
        cell_states.get(), cell_tanh.get(), hidden_outputs.get()};

    const std::size_t required_workspace =
        quant_lstm::lstmQuantizedFpCudaWorkspaceBytes(fixture.shape, fixture.master.bias_enabled);
    const auto workspace_breakdown = quant_lstm::lstmQuantizedFpCudaWorkspaceBreakdown(
        fixture.shape, fixture.master.bias_enabled);
    DeviceBuffer<std::byte> workspace(caller_workspace ? required_workspace : 0);
    quant_lstm::test::QuantizedCudaContextOwner context(workspace_breakdown.device_parameter_bytes);
    quant_lstm::LstmQuantizedFpCudaStats stats;
    const bool pedantic = profile.at("math_mode") == "pedantic";
    quant_lstm::lstmForwardQuantizedFpCuda(
        fixture.shape,
        {weight_ih.get(), weight_hh.get(), fixture.master.bias_enabled ? bias_ih.get() : nullptr,
         fixture.master.bias_enabled ? bias_hh.get() : nullptr},
        input.get(), fixture.master.has_explicit_state ? initial_hidden.get() : nullptr,
        fixture.master.has_explicit_state ? initial_cell.get() : nullptr, fixture.config,
        fixture.params, fixture.execution, output.get(), final_hidden.get(), final_cell.get(),
        context.get(),
        pedantic ? quant_lstm::LstmQuantizedFpCudaMathMode::Pedantic
                 : quant_lstm::LstmQuantizedFpCudaMathMode::Tf32,
        {workspace.get(), workspace.bytes()}, &checkpoints, &stats);
    context.synchronize();
    if (stats.input_gemm_calls != 1 || stats.recurrent_gemm_calls != steps ||
        stats.workspace_bytes != required_workspace ||
        stats.used_internal_workspace == caller_workspace) {
        throw std::runtime_error("CUDA forward stats/workspace 契约不满足");
    }

    quant_lstm::test::FloatLstmResult actual{output.copyToHost(), final_hidden.copyToHost(),
                                             final_cell.copyToHost()};
    const auto q_linear_ih = linear_ih.copyToHost();
    const auto q_linear_hh = linear_hh.copyToHost();
    const auto q_gate_inputs = gate_inputs.copyToHost();
    const auto q_gate_outputs = gate_outputs.copyToHost();
    const auto q_cell = cell_states.copyToHost();
    const auto q_cell_tanh = cell_tanh.copyToHost();
    const auto q_hidden = hidden_outputs.copyToHost();

    quant_lstm::LstmFpReferenceTrace cpu_trace;
    std::vector<float> cpu_q_output(output_count);
    std::vector<float> cpu_q_hidden(state_count);
    std::vector<float> cpu_q_cell(state_count);
    const auto q_input = asFloat(fixture.quantized.input);
    const auto q_weight_ih = asFloat(fixture.quantized.weight_ih);
    const auto q_weight_hh = asFloat(fixture.quantized.weight_hh);
    const auto q_bias_ih = asFloat(fixture.quantized.bias_ih);
    const auto q_bias_hh = asFloat(fixture.quantized.bias_hh);
    const auto q_h0 = asFloat(fixture.quantized.h0);
    const auto q_c0 = asFloat(fixture.quantized.c0);
    quant_lstm::lstmForwardQuantizedFpCpuReference(
        fixture.shape,
        {q_weight_ih.data(), q_weight_hh.data(),
         fixture.master.bias_enabled ? q_bias_ih.data() : nullptr,
         fixture.master.bias_enabled ? q_bias_hh.data() : nullptr},
        q_input.data(), fixture.master.has_explicit_state ? q_h0.data() : nullptr,
        fixture.master.has_explicit_state ? q_c0.data() : nullptr, fixture.config, fixture.params,
        fixture.execution, cpu_q_output.data(), cpu_q_hidden.data(), cpu_q_cell.data(), &cpu_trace);

    const bool derived_exact =
        executionIsExact(fixture.execution) && (pedantic || tf32OperandsAreExact(fixture.config));
    const std::string derived_class = derived_exact ? "exact_integer_range" : "precision_risk";
    if (profile.at("fp32_accumulation_class") != derived_class) {
        throw std::runtime_error(profile.at("case_id").get<std::string>() +
                                 " matrix NumericSafety 标签与派生报告不一致");
    }
    if (derived_exact) {
        requireExact(q_linear_ih, cpu_trace.weight_ih_linear, "weight_ih_linear");
        requireExact(q_linear_hh, cpu_trace.weight_hh_linear, "weight_hh_linear");
        requireExact(q_gate_inputs, cpu_trace.gate_inputs, "gate_inputs");
        requireExact(q_gate_outputs, cpu_trace.gate_outputs, "gate_outputs");
        requireExact(q_cell, cpu_trace.cell_states, "cell");
        requireExact(q_cell_tanh, cpu_trace.cell_tanh_outputs, "cell_tanh");
        requireExact(q_hidden, cpu_trace.hidden_outputs, "output");
        requireExact(sliceLastState(q_hidden, state_count), cpu_q_hidden, "h_n");
        requireExact(sliceLastState(q_cell, state_count), cpu_q_cell, "c_n");
    }

    const auto output_metrics = quant_lstm::test::computeSyntheticNumericMetrics(
        actual.output.data(), fixture.float_oracle.output.data(), output_count, q_hidden.data(),
        fixture.config.at(quant_lstm::QuantOperator::Output).type);
    const auto hidden_q = sliceLastState(q_hidden, state_count);
    const auto cell_q = sliceLastState(q_cell, state_count);
    const auto hidden_metrics = quant_lstm::test::computeSyntheticNumericMetrics(
        actual.final_hidden.data(), fixture.float_oracle.final_hidden.data(), state_count,
        hidden_q.data(), fixture.config.at(quant_lstm::QuantOperator::Output).type);
    const auto cell_metrics = quant_lstm::test::computeSyntheticNumericMetrics(
        actual.final_cell.data(), fixture.float_oracle.final_cell.data(), state_count,
        cell_q.data(), fixture.config.at(quant_lstm::QuantOperator::CellState).type);
    const Json& threshold = thresholds.at("profiles").at(profile.at("threshold_profile"));
    const bool passed = passes(output_metrics, threshold) && passes(hidden_metrics, threshold) &&
                        passes(cell_metrics, threshold);
    if (!passed) {
        throw std::runtime_error(
            profile.at("case_id").get<std::string>() + " 未通过既有 strict threshold: output=" +
            metricJson(output_metrics).dump() + ", h_n=" + metricJson(hidden_metrics).dump() +
            ", c_n=" + metricJson(cell_metrics).dump());
    }

    return {
        {"case_id", profile.at("case_id")},
        {"shape", profile.at("shape")},
        {"shape_profile", profile.at("shape_profile")},
        {"math_mode", profile.at("math_mode")},
        {"fp32_accumulation_class", derived_class},
        {"workspace",
         {{"kind", caller_workspace ? "caller" : "internal"},
          {"bytes", required_workspace},
          {"breakdown",
           {{"quantized_input_bytes", workspace_breakdown.quantized_input_bytes},
            {"device_parameter_bytes", workspace_breakdown.device_parameter_bytes}}}}},
        {"gemm_calls",
         {{"input", stats.input_gemm_calls}, {"recurrent", stats.recurrent_gemm_calls}}},
        {"metrics",
         {{"output", metricJson(output_metrics)},
          {"h_n", metricJson(hidden_metrics)},
          {"c_n", metricJson(cell_metrics)}}},
        {"numeric_safety", quant_lstm::test::executionSafetyJson(fixture.execution)},
        {"passed", true},
    };
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path report_path;
    Json report{
        {"schema_version", 1},
        {"stage", 4},
        {"validation_scope", "synthetic_numeric"},
        {"real_data_status", "not_configured"},
        {"cases", Json::array()},
    };
    try {
        const Arguments arguments = parseArguments(argc, argv);
        report_path = arguments.report;
        const std::filesystem::path source_dir = QUANT_LSTM_SOURCE_DIR;
        const Json matrix = readJson(source_dir / "tests/precision/config/strict_matrix_v2.json");
        const Json thresholds =
            readJson(source_dir / "tests/precision/config/strict_thresholds.json");
        report["matrix_version"] = matrix.at("matrix_version");
        report["selection"] = arguments.selection;
        const auto selected = selectProfiles(matrix, arguments.selection);
        bool caller_workspace = false;
        for (const Json& profile : selected) {
            report["cases"].push_back(runProfile(profile, thresholds, caller_workspace));
            caller_workspace = !caller_workspace;
        }
        report["summary"] = {
            {"selected_cases", selected.size()},
            {"passed", true},
        };
        writeJson(report_path, report);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        report["summary"] = {{"passed", false}};
        report["fatal_error"] = error.what();
        if (!report_path.empty()) {
            try {
                writeJson(report_path, report);
            } catch (const std::exception& write_error) {
                std::cerr << write_error.what() << '\n';
            }
        }
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
