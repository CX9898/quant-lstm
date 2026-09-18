#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/synthetic_precision_fixture.h"
#include "cuda/cuda_test_support.cuh"
#include "lstm/forward_quantized_fp_cuda.h"

namespace {

using Json = nlohmann::json;
using quant_lstm::test::DeviceBuffer;

#ifndef QUANT_LSTM_SOURCE_DIR
#define QUANT_LSTM_SOURCE_DIR "."
#endif

struct Arguments {
    int device = 0;
    int warmup = 10;
    int iterations = 100;
    std::filesystem::path report;
};

int parseInteger(const char* text, const char* name, bool allow_zero) {
    int value = 0;
    const std::string input = text;
    const auto parsed = std::from_chars(input.data(), input.data() + input.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != input.data() + input.size() ||
        value < (allow_zero ? 0 : 1)) {
        throw std::invalid_argument(std::string(name) + " 必须为" + (allow_zero ? "非负" : "正") +
                                    "整数");
    }
    return value;
}

Arguments parseArguments(int argc, char** argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (index + 1 >= argc) {
            throw std::invalid_argument(option + " 缺少值");
        }
        if (option == "--device") {
            result.device = parseInteger(argv[++index], "device", true);
        } else if (option == "--warmup-iterations") {
            result.warmup = parseInteger(argv[++index], "warmup", true);
        } else if (option == "--iterations") {
            result.iterations = parseInteger(argv[++index], "iterations", false);
        } else if (option == "--json-report") {
            result.report = argv[++index];
        } else {
            throw std::invalid_argument("未知参数: " + option);
        }
    }
    if (result.report.empty()) {
        throw std::invalid_argument("--json-report 不能为空");
    }
    return result;
}

Json readJson(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("无法读取 benchmark fixture: " + path.string());
    }
    Json result;
    stream >> result;
    return result;
}

std::vector<Json> benchmarkProfiles(const Json& matrix) {
    std::vector<Json> result;
    constexpr std::array<const char*, 2> kShapes{"cuda_gru_basic", "large_batch"};
    constexpr std::array<const char*, 2> kModes{"pedantic", "tf32"};
    for (const char* shape : kShapes) {
        for (const char* mode : kModes) {
            const auto match = std::find_if(
                matrix.at("cases").begin(), matrix.at("cases").end(), [&](const Json& item) {
                    return item.at("backend") == "cuda_fp32" && item.at("shape_profile") == shape &&
                           item.at("math_mode") == mode;
                });
            if (match == matrix.at("cases").end()) {
                throw std::runtime_error(std::string("strict_matrix_v2 缺少 benchmark profile: ") +
                                         shape + "/" + mode);
            }
            result.push_back(*match);
        }
    }
    return result;
}

struct EventSet {
    std::array<cudaEvent_t, 6> values{};

    EventSet() {
        try {
            for (cudaEvent_t& event : values) {
                quant_lstm::test::checkCuda(cudaEventCreate(&event), "cudaEventCreate timing");
            }
        } catch (...) {
            release();
            throw;
        }
    }

    ~EventSet() { release(); }
    EventSet(const EventSet&) = delete;
    EventSet& operator=(const EventSet&) = delete;

    quant_lstm::LstmQuantizedFpCudaTimingEvents view() const {
        return {values[0], values[1], values[2], values[3], values[4], values[5]};
    }

   private:
    void release() noexcept {
        for (cudaEvent_t& event : values) {
            if (event != nullptr) {
                cudaEventDestroy(event);
                event = nullptr;
            }
        }
    }
};

float elapsed(cudaEvent_t start, cudaEvent_t stop) {
    float value = 0.0F;
    quant_lstm::test::checkCuda(cudaEventElapsedTime(&value, start, stop), "cudaEventElapsedTime");
    if (!std::isfinite(value) || value < 0.0F) {
        throw std::runtime_error("CUDA event 计时不是有限非负数");
    }
    return value;
}

Json summary(std::vector<float> values) {
    if (values.empty()) {
        throw std::invalid_argument("benchmark samples 不能为空");
    }
    std::sort(values.begin(), values.end());
    const auto percentile = [&](double fraction) {
        return values[static_cast<std::size_t>(std::ceil(fraction * values.size())) - 1];
    };
    const double mean =
        std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    return {
        {"minimum", values.front()}, {"maximum", values.back()}, {"mean", mean},
        {"p50", percentile(0.50)},   {"p95", percentile(0.95)},
    };
}

Json metricsJson(const quant_lstm::test::SyntheticNumericMetrics& metrics) {
    const auto& accuracy = metrics.accuracy;
    Json result{
        {"max_absolute_error", accuracy.maximum_absolute_error},
        {"mae", accuracy.mean_absolute_error},
        {"mse", accuracy.mean_squared_error},
        {"sqnr_db", metrics.sqnr_db},
        {"saturation_rate", metrics.saturation_rate},
    };
    result["cosine_similarity"] = accuracy.cosine_not_applicable || accuracy.one_sided_zero_norm
                                      ? Json(nullptr)
                                      : Json(accuracy.cosine_similarity);
    return result;
}

Json workspaceJson(const quant_lstm::LstmQuantizedFpCudaWorkspaceBreakdown& value,
                   std::size_t persistent_parameter_cache_bytes) {
    return {
        {"quantized_input_bytes", value.quantized_input_bytes},
        {"quantized_weight_ih_bytes", value.quantized_weight_ih_bytes},
        {"quantized_weight_hh_bytes", value.quantized_weight_hh_bytes},
        {"quantized_bias_bytes", value.quantized_bias_bytes},
        {"quantized_state_bytes", value.quantized_state_bytes},
        {"input_linear_bytes", value.input_linear_bytes},
        {"recurrent_linear_bytes", value.recurrent_linear_bytes},
        {"weight_sum_bytes", value.weight_sum_bytes},
        {"device_parameter_bytes", value.device_parameter_bytes},
        {"persistent_parameter_cache_bytes", persistent_parameter_cache_bytes},
        {"alignment_padding_bytes", value.alignment_padding_bytes},
        {"total_bytes", value.total_bytes},
    };
}

Json runCase(const Json& profile, const Arguments& arguments) {
    const auto fixture = quant_lstm::test::makeSyntheticPrecisionFixture(profile);
    const std::size_t steps = static_cast<std::size_t>(fixture.shape.sequence_length);
    const std::size_t batch = static_cast<std::size_t>(fixture.shape.batch_size);
    const std::size_t hidden = static_cast<std::size_t>(fixture.shape.hidden_size);
    const std::size_t output_count = steps * batch * hidden;
    const std::size_t state_count = batch * hidden;

    DeviceBuffer<float> input(fixture.master.input.size());
    DeviceBuffer<float> weight_ih(fixture.master.weight_ih.size());
    DeviceBuffer<float> weight_hh(fixture.master.weight_hh.size());
    DeviceBuffer<float> bias_ih(fixture.master.bias_ih.size());
    DeviceBuffer<float> bias_hh(fixture.master.bias_hh.size());
    DeviceBuffer<float> h0(fixture.master.h0.size());
    DeviceBuffer<float> c0(fixture.master.c0.size());
    DeviceBuffer<float> output(output_count);
    DeviceBuffer<float> hn(state_count);
    DeviceBuffer<float> cn(state_count);
    DeviceBuffer<float> q_output(output_count);
    DeviceBuffer<float> q_cell(output_count);
    input.copyFrom(fixture.master.input);
    weight_ih.copyFrom(fixture.master.weight_ih);
    weight_hh.copyFrom(fixture.master.weight_hh);
    bias_ih.copyFrom(fixture.master.bias_ih);
    bias_hh.copyFrom(fixture.master.bias_hh);
    h0.copyFrom(fixture.master.h0);
    c0.copyFrom(fixture.master.c0);

    const auto breakdown = quant_lstm::lstmQuantizedFpCudaWorkspaceBreakdown(
        fixture.shape, fixture.master.bias_enabled);
    const std::size_t static_parameter_cache_bytes =
        quant_lstm::lstmQuantizedFpCudaStaticParameterBytes(fixture.shape,
                                                            fixture.master.bias_enabled);
    DeviceBuffer<std::byte> workspace(breakdown.total_bytes);
    quant_lstm::test::QuantizedCudaContextOwner context(breakdown.device_parameter_bytes,
                                                        static_parameter_cache_bytes);
    quant_lstm::LstmQuantizedFpCudaCheckpoints checkpoints{};
    checkpoints.hidden_outputs = q_output.get();
    checkpoints.cell_states = q_cell.get();
    const bool pedantic = profile.at("math_mode") == "pedantic";
    const auto math_mode = pedantic ? quant_lstm::LstmQuantizedFpCudaMathMode::Pedantic
                                    : quant_lstm::LstmQuantizedFpCudaMathMode::Tf32;
    constexpr std::uint64_t kStaticParameterCacheKey = 1;
    std::size_t static_cache_hits = 0;
    std::size_t static_cache_misses = 0;
    bool static_cache_primed = false;

    const auto forward = [&](const auto* timing) {
        quant_lstm::LstmQuantizedFpCudaStats stats;
        const std::string range_name = "quant_lstm_forward_" +
                                       profile.at("shape_profile").get<std::string>() + "_" +
                                       profile.at("math_mode").get<std::string>();
        nvtxRangePushA(range_name.c_str());
        quant_lstm::lstmForwardQuantizedFpCuda(
            fixture.shape,
            {weight_ih.get(), weight_hh.get(),
             fixture.master.bias_enabled ? bias_ih.get() : nullptr,
             fixture.master.bias_enabled ? bias_hh.get() : nullptr},
            input.get(), fixture.master.has_explicit_state ? h0.get() : nullptr,
            fixture.master.has_explicit_state ? c0.get() : nullptr, fixture.config, fixture.params,
            fixture.execution, output.get(), hn.get(), cn.get(), context.get(), math_mode,
            {workspace.get(), workspace.bytes()}, &checkpoints, &stats, timing,
            kStaticParameterCacheKey);
        nvtxRangePop();
        if (stats.input_gemm_calls != 1 || stats.recurrent_gemm_calls != steps) {
            throw std::runtime_error("benchmark GEMM 计数非法");
        }
        if (stats.static_parameter_cache_hit != static_cache_primed) {
            throw std::runtime_error("benchmark static parameter cache 状态非法");
        }
        static_cache_primed = true;
        stats.static_parameter_cache_hit ? ++static_cache_hits : ++static_cache_misses;
    };

    for (int index = 0; index < arguments.warmup; ++index) {
        forward(static_cast<const quant_lstm::LstmQuantizedFpCudaTimingEvents*>(nullptr));
    }
    context.synchronize();

    EventSet events;
    std::vector<float> end_to_end;
    std::vector<float> quantization;
    std::vector<float> core;
    end_to_end.reserve(arguments.iterations);
    quantization.reserve(arguments.iterations);
    core.reserve(arguments.iterations);
    for (int index = 0; index < arguments.iterations; ++index) {
        const auto timing = events.view();
        forward(&timing);
        quant_lstm::test::checkCuda(cudaEventSynchronize(timing.complete),
                                    "cudaEventSynchronize benchmark");
        end_to_end.push_back(elapsed(timing.start, timing.complete));
        quantization.push_back(elapsed(timing.start, timing.quantization_complete));
        core.push_back(elapsed(timing.quantization_complete, timing.core_complete));
    }

    const auto actual_output = output.copyToHost();
    const auto actual_hn = hn.copyToHost();
    const auto actual_cn = cn.copyToHost();
    const auto quantized_output = q_output.copyToHost();
    const auto quantized_cell = q_cell.copyToHost();
    const std::vector<float> quantized_hn(
        quantized_output.end() - static_cast<std::ptrdiff_t>(state_count), quantized_output.end());
    const std::vector<float> quantized_cn(
        quantized_cell.end() - static_cast<std::ptrdiff_t>(state_count), quantized_cell.end());
    const auto output_metrics = quant_lstm::test::computeSyntheticNumericMetrics(
        actual_output.data(), fixture.float_oracle.output.data(), output_count,
        quantized_output.data(), fixture.config.at(quant_lstm::QuantOperator::Output).type);
    const auto hidden_metrics = quant_lstm::test::computeSyntheticNumericMetrics(
        actual_hn.data(), fixture.float_oracle.final_hidden.data(), state_count,
        quantized_hn.data(), fixture.config.at(quant_lstm::QuantOperator::Output).type);
    const auto cell_metrics = quant_lstm::test::computeSyntheticNumericMetrics(
        actual_cn.data(), fixture.float_oracle.final_cell.data(), state_count, quantized_cn.data(),
        fixture.config.at(quant_lstm::QuantOperator::CellState).type);

    const Json end_to_end_summary = summary(end_to_end);
    const double median_ms = end_to_end_summary.at("p50");
    const double seconds = median_ms / 1000.0;
    if (!(seconds > 0.0) || !std::isfinite(seconds)) {
        throw std::runtime_error("benchmark median 必须为有限正数");
    }
    return {
        {"case_id", profile.at("case_id")},
        {"shape_profile", profile.at("shape_profile")},
        {"shape", profile.at("shape")},
        {"math_mode", profile.at("math_mode")},
        {"fp32_accumulation_class", profile.at("fp32_accumulation_class")},
        {"warmup_iterations", arguments.warmup},
        {"measured_iterations", arguments.iterations},
        {"timing_scope",
         "cuda_event_quantize_core_dequantize_"
         "cached_execution_and_static_params"},
        {"static_parameter_cache",
         {{"enabled", true},
          {"generation_key", kStaticParameterCacheKey},
          {"persistent_bytes", static_parameter_cache_bytes},
          {"total_hits", static_cache_hits},
          {"total_misses", static_cache_misses},
          {"measured_region_all_hits", arguments.warmup > 0}}},
        {"timing_ms",
         {{"end_to_end", end_to_end_summary},
          {"quantization_overhead", summary(quantization)},
          {"quantized_core", summary(core)}}},
        {"throughput",
         {{"sequences_per_second", static_cast<double>(batch) / seconds},
          {"time_steps_per_second", static_cast<double>(steps * batch) / seconds},
          {"output_elements_per_second", static_cast<double>(output_count) / seconds}}},
        {"workspace", workspaceJson(breakdown, static_parameter_cache_bytes)},
        {"gemm_calls",
         {{"input_per_forward", 1},
          {"recurrent_per_forward", steps},
          {"total_per_forward", 1 + steps}}},
        {"precision",
         {{"output", metricsJson(output_metrics)},
          {"h_n", metricsJson(hidden_metrics)},
          {"c_n", metricsJson(cell_metrics)}}},
    };
}

Json environmentJson(int device) {
    cudaDeviceProp properties{};
    quant_lstm::test::checkCuda(cudaGetDeviceProperties(&properties, device),
                                "cudaGetDeviceProperties");
    int driver = 0;
    int runtime = 0;
    int cublas = 0;
    quant_lstm::test::checkCuda(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
    quant_lstm::test::checkCuda(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");
    cublasHandle_t handle = nullptr;
    quant_lstm::test::checkCublas(cublasCreate(&handle), "cublasCreate");
    try {
        quant_lstm::test::checkCublas(cublasGetVersion(handle, &cublas), "cublasGetVersion");
    } catch (...) {
        cublasDestroy(handle);
        throw;
    }
    cublasDestroy(handle);
    return {
        {"gpu_name", properties.name},
        {"compute_capability",
         std::to_string(properties.major) + "." + std::to_string(properties.minor)},
        {"cuda_driver_version", driver},
        {"cuda_runtime_version", runtime},
        {"cublas_version", cublas},
    };
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments arguments = parseArguments(argc, argv);
        quant_lstm::test::checkCuda(cudaSetDevice(arguments.device), "cudaSetDevice");
        const Json matrix = readJson(std::filesystem::path(QUANT_LSTM_SOURCE_DIR) /
                                     "tests/precision/config/strict_matrix_v2.json");
        Json cases = Json::array();
        std::uint64_t total_profiled_gemm_calls = 0;
        for (const Json& profile : benchmarkProfiles(matrix)) {
            cases.push_back(runCase(profile, arguments));
            total_profiled_gemm_calls +=
                (1 + profile.at("shape").at(0).get<std::uint64_t>()) *
                static_cast<std::uint64_t>(arguments.iterations + arguments.warmup);
        }
        const Json report{
            {"schema_version", 3},
            {"benchmark_id", "quantized_fp_cuda_lstm_v1"},
            {"device", arguments.device},
            {"matrix_version", matrix.at("matrix_version")},
            {"environment", environmentJson(arguments.device)},
            {"total_profiled_cublas_sgemm_calls", total_profiled_gemm_calls},
            {"cases", std::move(cases)},
        };
        if (!arguments.report.parent_path().empty()) {
            std::filesystem::create_directories(arguments.report.parent_path());
        }
        std::ofstream stream(arguments.report);
        if (!stream) {
            throw std::runtime_error("无法写入 benchmark report");
        }
        stream << report.dump(2) << '\n';
        std::cout << report.dump() << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
