#include "common/deterministic_rng.h"
#include "common/synthetic_precision_fixture.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace {

using Json = nlohmann::json;

#ifndef QUANT_LSTM_SOURCE_DIR
#define QUANT_LSTM_SOURCE_DIR "."
#endif

constexpr std::array<const char*, 8> kDirectedCaseIds{{
    "stage3_basic_int32_int8_bias_time",
    "stage3_basic_fp32_int16_bias_batch",
    "stage3_strict_int32_minimal_int8_g00",
    "stage3_strict_fp32_short_int16_g01",
    "stage3_strict_int32_nonaligned_mixed_g02",
    "stage3_strict_fp32_long_int8_g10",
    "stage3_strict_fp32_nonaligned_mixed_g12",
    "stage3_strict_int32_minimal_int16_no_bias",
}};

Json readJson(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("无法打开配置: " + path.string());
    }
    Json result;
    stream >> result;
    return result;
}

quant_lstm::test::MetricThresholds metricThresholds(const Json& threshold) {
    return {
        threshold.at("mae").get<double>(),
        threshold.at("mse").get<double>(),
        threshold.at("cosine_similarity").get<double>(),
    };
}

Json metricsJson(const quant_lstm::test::TensorMetrics& result) {
    const auto& metrics = result.metrics.accuracy;
    Json output{
        {"mae", metrics.mean_absolute_error},
        {"mse", metrics.mean_squared_error},
        {"max_absolute_error", metrics.maximum_absolute_error},
        {"passed", result.passed},
    };
    if (metrics.cosine_not_applicable) {
        output["cosine_similarity"] = nullptr;
        output["cosine_status"] = "both_near_zero";
    } else if (metrics.one_sided_zero_norm) {
        output["cosine_similarity"] = nullptr;
        output["cosine_status"] = "one_near_zero";
    } else {
        output["cosine_similarity"] = metrics.cosine_similarity;
        output["cosine_status"] = "valid";
    }
    return output;
}

Json carrierResult(
    const char* carrier, const quant_lstm::test::FloatLstmResult& actual,
    const quant_lstm::test::FloatLstmResult& expected,
    const quant_lstm::test::MetricThresholds& thresholds, bool* all_passed) {
    const auto output_metrics = quant_lstm::test::evaluateTensorMetrics(
        actual.output, expected.output, thresholds);
    const auto hidden_metrics = quant_lstm::test::evaluateTensorMetrics(
        actual.final_hidden, expected.final_hidden, thresholds);
    const auto cell_metrics = quant_lstm::test::evaluateTensorMetrics(
        actual.final_cell, expected.final_cell, thresholds);
    const bool passed =
        output_metrics.passed && hidden_metrics.passed && cell_metrics.passed;
    *all_passed = *all_passed && passed;
    return {
        {"carrier", carrier},
        {"passed", passed},
        {"tensors",
         {{"output", metricsJson(output_metrics)},
          {"h_n", metricsJson(hidden_metrics)},
          {"c_n", metricsJson(cell_metrics)}}},
    };
}

quant_lstm::test::FloatLstmResult dequantizeInt32Result(
    const quant_lstm::test::SyntheticPrecisionFixture& fixture) {
    return {
        quant_lstm::test::dequantizeTensor(
            fixture.int32_oracle.output, quant_lstm::QuantOperator::Output,
            fixture.config, fixture.params),
        quant_lstm::test::dequantizeTensor(
            fixture.int32_oracle.final_hidden,
            quant_lstm::QuantOperator::Output, fixture.config,
            fixture.params),
        quant_lstm::test::dequantizeTensor(
            fixture.int32_oracle.final_cell,
            quant_lstm::QuantOperator::CellState, fixture.config,
            fixture.params),
    };
}

quant_lstm::test::FloatLstmResult dequantizeFpResult(
    const quant_lstm::test::SyntheticPrecisionFixture& fixture) {
    return {
        quant_lstm::test::dequantizeTensor(
            fixture.fp_quantized_oracle.output,
            quant_lstm::QuantOperator::Output, fixture.config,
            fixture.params),
        quant_lstm::test::dequantizeTensor(
            fixture.fp_quantized_oracle.final_hidden,
            quant_lstm::QuantOperator::Output, fixture.config,
            fixture.params),
        quant_lstm::test::dequantizeTensor(
            fixture.fp_quantized_oracle.final_cell,
            quant_lstm::QuantOperator::CellState, fixture.config,
            fixture.params),
    };
}

Json runProfile(const Json& profile, const Json& thresholds,
                bool* all_passed) {
    const auto fixture =
        quant_lstm::test::makeSyntheticPrecisionFixture(profile);
    const std::uint64_t evaluation_seed =
        profile.at("data_seeds").at("evaluation").front().get<std::uint64_t>();
    const Json& threshold =
        thresholds.at("profiles").at(profile.at("threshold_profile"));
    const auto parsed_thresholds = metricThresholds(threshold);
    const auto int_dequantized = dequantizeInt32Result(fixture);
    const auto fp_dequantized = dequantizeFpResult(fixture);

    Json carriers = Json::array();
    carriers.push_back(carrierResult(
        "int32", int_dequantized, fixture.float_oracle, parsed_thresholds,
        all_passed));
    carriers.push_back(carrierResult(
        "float32_quantized_values", fp_dequantized, fixture.float_oracle,
        parsed_thresholds, all_passed));
    const bool profile_passed =
        carriers.at(0).at("passed").get<bool>() &&
        carriers.at(1).at("passed").get<bool>();
    const std::size_t hidden =
        static_cast<std::size_t>(fixture.shape.hidden_size);
    const std::size_t channels = 4 * hidden;
    return {
        {"case_id", profile.at("case_id")},
        {"source_backend", profile.at("backend")},
        {"shape_profile", profile.at("shape_profile")},
        {"shape", profile.at("shape")},
        {"state_profile", profile.at("state_profile")},
        {"bias_profile", profile.at("bias_profile")},
        {"layout_profile", profile.at("layout_profile")},
        {"scale_mode", profile.at("resolved_quant_config").at("scale_mode")},
        {"bitwidth_profile", profile.at("bitwidth_profile")},
        {"threshold_profile", profile.at("threshold_profile")},
        {"strict_thresholds", threshold},
        {"activation_profile", profile.at("activation_profile")},
        {"parameter_granularities", profile.at("parameter_granularities")},
        {"directed_profiles", profile.at("directed_profiles")},
        {"evaluation_seed", evaluation_seed},
        {"passed", profile_passed},
        {"carriers", std::move(carriers)},
        {"quantized_elements",
         {{"input", fixture.quantized.input.size()},
          {"weight_ih", fixture.quantized.weight_ih.size()},
          {"weight_hh", fixture.quantized.weight_hh.size()},
          {"bias", fixture.master.bias_enabled ? 2 * channels : 0}}},
    };
}

void writeReport(const std::filesystem::path& path, const Json& report) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("无法写入报告: " + path.string());
    }
    output << report.dump(2) << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    const std::filesystem::path source_dir = QUANT_LSTM_SOURCE_DIR;
    const std::filesystem::path report_path =
        argc > 1 ? std::filesystem::path(argv[1])
                 : std::filesystem::current_path() /
                       "synthetic_numeric_report.json";
    Json report{
        {"schema_version", 1},
        {"validation_scope", "synthetic_numeric"},
        {"real_data_status", "not_configured"},
        {"rng_version", quant_lstm::test::kRngVersion},
        {"rng_stream_registry", quant_lstm::test::kRngStreamRegistryVersion},
        {"distribution_profile", quant_lstm::test::kDistributionProfile},
        {"selection", "directed_strict_profile_set_v1"},
        {"cases", Json::array()},
    };
    try {
        const Json matrix = readJson(
            source_dir / "tests/precision/config/strict_matrix_v1.json");
        const Json thresholds = readJson(
            source_dir / "tests/precision/config/strict_thresholds.json");
        report["matrix_version"] = matrix.at("matrix_version");
        report["threshold_schema_version"] = thresholds.at("schema_version");
        std::unordered_set<std::string> required(kDirectedCaseIds.begin(),
                                                 kDirectedCaseIds.end());
        bool all_passed = true;
        for (const auto& profile : matrix.at("cases")) {
            const std::string id = profile.at("case_id");
            if (required.erase(id) == 0U) {
                continue;
            }
            report["cases"].push_back(
                runProfile(profile, thresholds, &all_passed));
            std::cout << id << ": "
                      << (report["cases"].back().at("passed").get<bool>()
                              ? "PASS"
                              : "FAIL")
                      << '\n';
        }
        if (!required.empty()) {
            throw std::runtime_error("strict matrix 缺少定向 case");
        }
        report["summary"] = {
            {"selected_cases", report["cases"].size()},
            {"executed_carriers", 2},
            {"passed", all_passed},
            {"coverage",
             {{"scale_modes", {"affine", "pot2"}},
              {"bitwidth_profiles",
               {"all_int8", "all_int16", "mixed_8_16"}},
              {"shape_profiles",
               {"minimal", "short_recurrent", "non_aligned",
                "long_sequence"}},
              {"state_behaviors",
               {"zero", "random", "saturation_boundary",
                "long_recurrence"}},
              {"bias", {true, false}},
              {"parameter_granularities",
               {"per_tensor", "per_gate", "per_channel"}},
              {"asymmetric_activation_types",
               {"signed_asymmetric", "unsigned_asymmetric"}}}},
        };
        writeReport(report_path, report);
        std::cout << "report: " << report_path << '\n';
        return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        report["fatal_error"] = error.what();
        report["summary"] = {{"passed", false}};
        try {
            writeReport(report_path, report);
        } catch (const std::exception& write_error) {
            std::cerr << write_error.what() << '\n';
        }
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
