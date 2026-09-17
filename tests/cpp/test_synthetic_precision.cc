#include "common/deterministic_rng.h"
#include "common/numeric_metrics.h"
#include "lstm/forward_cpu.h"
#include "lstm/forward_float.h"
#include "lstm/lstm_execution_params.h"
#include "lstm/quant_config_loader.h"
#include "lstm/quant_params.h"
#include "quantization/fixed_point_ops.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::json;
using quant_lstm::QuantOperator;

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

struct ObservedRange {
    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();

    void add(float value) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("校准观测产生 Inf/NaN");
        }
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }

    quant_lstm::CalibrationRange expanded(float fraction = 0.15F) const {
        if (!std::isfinite(minimum) || !std::isfinite(maximum)) {
            throw std::runtime_error("校准范围没有观测值");
        }
        const float span = std::max(maximum - minimum, 1.0e-4F);
        const float padding = span * fraction;
        return {minimum - padding, maximum + padding};
    }
};

struct FloatMaster {
    std::vector<float> input;
    std::vector<float> weight_ih;
    std::vector<float> weight_hh;
    std::vector<float> bias_ih;
    std::vector<float> bias_hh;
    std::vector<float> h0;
    std::vector<float> c0;
    bool has_explicit_state = false;
    bool bias_enabled = true;
};

struct CalibrationObservations {
    ObservedRange input;
    ObservedRange output;
    ObservedRange cell;
    ObservedRange weight_ih_linear;
    ObservedRange weight_hh_linear;
    std::array<ObservedRange, 4> gate_inputs;
};

struct TensorMetrics {
    quant_lstm::test::NumericMetrics metrics;
    bool passed = false;
};

Json readJson(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("无法打开配置: " + path.string());
    }
    Json result;
    stream >> result;
    return result;
}

std::vector<float> canonicalInput(const quant_lstm::LstmShape& shape,
                                  std::uint64_t seed, bool batch_first,
                                  const std::string& activation_profile,
                                  const Json& directed_profiles) {
    const std::size_t steps = static_cast<std::size_t>(shape.sequence_length);
    const std::size_t batch = static_cast<std::size_t>(shape.batch_size);
    const std::size_t input_size = static_cast<std::size_t>(shape.input_size);
    std::vector<float> generated(steps * batch * input_size);
    quant_lstm::test::fillNormalLike(
        generated.data(), generated.size(), seed,
        quant_lstm::test::TensorStream::Input);

    const bool is_unsigned =
        activation_profile == "unsigned_symmetric" ||
        activation_profile == "unsigned_asymmetric";
    const bool is_asymmetric =
        activation_profile == "signed_asymmetric" ||
        activation_profile == "unsigned_asymmetric";
    for (float& value : generated) {
        value *= 0.35F;
        if (is_unsigned) {
            value = std::abs(value);
        } else if (is_asymmetric) {
            value += 0.20F;
        }
    }

    const std::unordered_set<std::string> directed(
        directed_profiles.begin(), directed_profiles.end());
    if (!generated.empty()) {
        if (directed.count("signed_asymmetric_minimum_boundary") != 0U) {
            generated.front() = -1.0F;
        } else if (directed.count("unsigned_symmetric_boundary") != 0U ||
                   directed.count("unsigned_asymmetric_boundary") != 0U) {
            generated.front() = 1.5F;
        }
    }

    if (!batch_first) {
        return generated;
    }
    std::vector<float> time_major(generated.size());
    for (std::size_t row = 0; row < batch; ++row) {
        for (std::size_t step = 0; step < steps; ++step) {
            const std::size_t source = (row * steps + step) * input_size;
            const std::size_t target = (step * batch + row) * input_size;
            std::copy_n(generated.data() + source, input_size,
                        time_major.data() + target);
        }
    }
    return time_major;
}

void makeInitialState(const quant_lstm::LstmShape& shape, std::uint64_t seed,
                      const std::string& profile, FloatMaster* master) {
    const std::size_t count =
        static_cast<std::size_t>(shape.batch_size * shape.hidden_size);
    if (profile == "omitted") {
        master->has_explicit_state = false;
        return;
    }
    master->has_explicit_state = true;
    master->h0.assign(count, 0.0F);
    master->c0.assign(count, 0.0F);
    if (profile == "explicit_zero") {
        return;
    }
    quant_lstm::test::fillNormalLike(
        master->h0.data(), count, seed,
        quant_lstm::test::TensorStream::InitialHidden);
    quant_lstm::test::fillNormalLike(
        master->c0.data(), count, seed,
        quant_lstm::test::TensorStream::InitialCell);
    for (std::size_t index = 0; index < count; ++index) {
        master->h0[index] *= 0.20F;
        master->c0[index] *= 0.20F;
    }
    if (profile == "near_quant_boundary") {
        for (std::size_t index = 0; index < count; ++index) {
            const float sign = index % 2 == 0 ? 1.0F : -1.0F;
            master->h0[index] = sign * 0.25F;
            master->c0[index] = sign * 0.45F;
        }
    } else if (profile == "mixed_h_random_c_boundary") {
        for (std::size_t index = 0; index < count; ++index) {
            master->c0[index] = (index % 2 == 0 ? 1.0F : -1.0F) * 0.45F;
        }
    }
}

FloatMaster makeMaster(const Json& profile, std::uint64_t data_seed) {
    const auto& dimensions = profile.at("shape");
    const quant_lstm::LstmShape shape{
        dimensions.at(0).get<std::int64_t>(),
        dimensions.at(1).get<std::int64_t>(),
        dimensions.at(2).get<std::int64_t>(),
        dimensions.at(3).get<std::int64_t>()};
    const std::size_t hidden = static_cast<std::size_t>(shape.hidden_size);
    const std::size_t channels = 4 * hidden;
    FloatMaster master;
    master.bias_enabled = profile.at("bias_profile") != "disabled";
    master.input = canonicalInput(
        shape, data_seed, profile.at("batch_first").get<bool>(),
        profile.at("activation_profile").get<std::string>(),
        profile.at("directed_profiles"));
    master.weight_ih.resize(channels *
                            static_cast<std::size_t>(shape.input_size));
    master.weight_hh.resize(channels * hidden);
    quant_lstm::test::fillLstmParameter(
        master.weight_ih.data(), master.weight_ih.size(), shape.hidden_size,
        profile.at("parameter_seed").get<std::uint64_t>(),
        quant_lstm::test::TensorStream::WeightInputHidden);
    quant_lstm::test::fillLstmParameter(
        master.weight_hh.data(), master.weight_hh.size(), shape.hidden_size,
        profile.at("parameter_seed").get<std::uint64_t>(),
        quant_lstm::test::TensorStream::WeightHiddenHidden);
    if (master.bias_enabled) {
        master.bias_ih.resize(channels);
        master.bias_hh.resize(channels);
        quant_lstm::test::fillLstmParameter(
            master.bias_ih.data(), channels, shape.hidden_size,
            profile.at("parameter_seed").get<std::uint64_t>(),
            quant_lstm::test::TensorStream::BiasInputHidden);
        quant_lstm::test::fillLstmParameter(
            master.bias_hh.data(), channels, shape.hidden_size,
            profile.at("parameter_seed").get<std::uint64_t>(),
            quant_lstm::test::TensorStream::BiasHiddenHidden);
    }
    if (shape.sequence_length >= 128) {
        for (float& value : master.weight_ih) {
            value *= 0.25F;
        }
        for (float& value : master.weight_hh) {
            value *= 0.25F;
        }
        if (master.bias_enabled) {
            const std::array<float, 4> gate_bias{0.25F, 0.75F, 0.25F,
                                                 0.50F};
            for (std::size_t gate = 0; gate < 4; ++gate) {
                for (std::size_t column = 0; column < hidden; ++column) {
                    const std::size_t index = gate * hidden + column;
                    master.bias_ih[index] =
                        gate_bias[gate] + 0.05F * master.bias_ih[index];
                    master.bias_hh[index] =
                        0.05F * master.bias_hh[index];
                }
            }
        }
    }
    if (profile.at("resolved_quant_config").at("scale_mode") == "pot2") {
        const auto snapToPot2Grid = [](std::vector<float>* values) {
            float maximum = 0.0F;
            for (float value : *values) {
                maximum = std::max(maximum, std::abs(value));
            }
            if (maximum == 0.0F) {
                return;
            }
            const float required_scale = maximum * 1.02F / 127.0F;
            const float scale = std::exp2(std::ceil(std::log2(required_scale)));
            for (float& value : *values) {
                value = std::nearbyint(value / scale) * scale;
            }
        };
        snapToPot2Grid(&master.weight_ih);
        snapToPot2Grid(&master.weight_hh);
        if (master.bias_enabled) {
            snapToPot2Grid(&master.bias_ih);
            snapToPot2Grid(&master.bias_hh);
        }
    }
    makeInitialState(shape, data_seed,
                     profile.at("state_profile").get<std::string>(), &master);
    return master;
}

float sigmoid(float value) {
    if (value >= 0.0F) {
        return 1.0F / (1.0F + std::exp(-value));
    }
    const float exp_value = std::exp(value);
    return exp_value / (1.0F + exp_value);
}

void observeCalibrationRun(const quant_lstm::LstmShape& shape,
                           const FloatMaster& master,
                           CalibrationObservations* observations) {
    const std::size_t steps = static_cast<std::size_t>(shape.sequence_length);
    const std::size_t batch = static_cast<std::size_t>(shape.batch_size);
    const std::size_t input_size = static_cast<std::size_t>(shape.input_size);
    const std::size_t hidden = static_cast<std::size_t>(shape.hidden_size);
    const std::size_t channels = 4 * hidden;
    for (float value : master.input) {
        observations->input.add(value);
    }
    std::vector<float> hidden_state(batch * hidden, 0.0F);
    std::vector<float> cell_state(batch * hidden, 0.0F);
    if (master.has_explicit_state) {
        hidden_state = master.h0;
        cell_state = master.c0;
    }
    std::vector<float> gates(channels);
    for (std::size_t step = 0; step < steps; ++step) {
        for (std::size_t row = 0; row < batch; ++row) {
            const float* input_row =
                master.input.data() + (step * batch + row) * input_size;
            float* hidden_row = hidden_state.data() + row * hidden;
            float* cell_row = cell_state.data() + row * hidden;
            for (std::size_t channel = 0; channel < channels; ++channel) {
                float input_linear =
                    master.bias_enabled ? master.bias_ih[channel] : 0.0F;
                float hidden_linear =
                    master.bias_enabled ? master.bias_hh[channel] : 0.0F;
                for (std::size_t column = 0; column < input_size; ++column) {
                    input_linear +=
                        master.weight_ih[channel * input_size + column] *
                        input_row[column];
                }
                for (std::size_t column = 0; column < hidden; ++column) {
                    hidden_linear +=
                        master.weight_hh[channel * hidden + column] *
                        hidden_row[column];
                }
                observations->weight_ih_linear.add(input_linear);
                observations->weight_hh_linear.add(hidden_linear);
                gates[channel] = input_linear + hidden_linear;
                observations->gate_inputs[channel / hidden].add(gates[channel]);
            }
            for (std::size_t column = 0; column < hidden; ++column) {
                const float input_gate = sigmoid(gates[column]);
                const float forget_gate = sigmoid(gates[hidden + column]);
                const float cell_gate = std::tanh(gates[2 * hidden + column]);
                const float output_gate = sigmoid(gates[3 * hidden + column]);
                cell_row[column] =
                    forget_gate * cell_row[column] + input_gate * cell_gate;
                hidden_row[column] =
                    output_gate * std::tanh(cell_row[column]);
                observations->cell.add(cell_row[column]);
                observations->output.add(hidden_row[column]);
            }
        }
    }
}

std::size_t groupCount(QuantOperator id,
                       quant_lstm::QuantGranularity granularity,
                       std::size_t channels) {
    if (!quant_lstm::isParameterOperator(id) ||
        granularity == quant_lstm::QuantGranularity::PerTensor) {
        return 1;
    }
    if (granularity == quant_lstm::QuantGranularity::PerGate) {
        return 4;
    }
    return channels;
}

void addParameterRanges(const std::vector<float>& values, std::size_t row_width,
                        std::size_t hidden, QuantOperator id,
                        const quant_lstm::LstmOperatorQuantConfig& config,
                        quant_lstm::LstmQuantizationRanges* ranges) {
    const std::size_t channels = 4 * hidden;
    const auto granularity = config.at(id).granularity;
    std::vector<ObservedRange> observed(
        groupCount(id, granularity, channels));
    for (std::size_t row = 0; row < channels; ++row) {
        std::size_t group = 0;
        if (granularity == quant_lstm::QuantGranularity::PerGate) {
            group = row / hidden;
        } else if (granularity ==
                   quant_lstm::QuantGranularity::PerChannel) {
            group = row;
        }
        for (std::size_t column = 0; column < row_width; ++column) {
            observed[group].add(values[row * row_width + column]);
        }
    }
    auto& destination = ranges->at(id);
    destination.reserve(observed.size());
    for (const auto& item : observed) {
        destination.push_back(item.expanded(0.02F));
    }
}

quant_lstm::LstmQuantizationRanges makeRanges(
    const Json& profile, const quant_lstm::LstmShape& shape,
    const quant_lstm::LstmOperatorQuantConfig& config,
    const FloatMaster& parameter_master) {
    CalibrationObservations observed;
    for (const auto& seed : profile.at("data_seeds").at("calibration")) {
        const auto calibration_master =
            makeMaster(profile, seed.get<std::uint64_t>());
        observeCalibrationRun(shape, calibration_master, &observed);
    }

    quant_lstm::LstmQuantizationRanges ranges;
    ranges.at(QuantOperator::Input).push_back(observed.input.expanded());
    ranges.at(QuantOperator::Output).push_back(observed.output.expanded(0.20F));
    ranges.at(QuantOperator::CellState).push_back(observed.cell.expanded());
    ranges.at(QuantOperator::WeightInputHiddenLinear)
        .push_back(observed.weight_ih_linear.expanded(0.20F));
    ranges.at(QuantOperator::WeightHiddenHiddenLinear)
        .push_back(observed.weight_hh_linear.expanded(0.20F));
    const std::array<QuantOperator, 4> gate_inputs{{
        QuantOperator::InputGateInput, QuantOperator::ForgetGateInput,
        QuantOperator::CellGateInput, QuantOperator::OutputGateInput}};
    for (std::size_t gate = 0; gate < gate_inputs.size(); ++gate) {
        ranges.at(gate_inputs[gate])
            .push_back(observed.gate_inputs[gate].expanded(0.20F));
    }
    const auto sigmoidMaximum = [](const ObservedRange& input) {
        return 1.0F / (1.0F + std::exp(-input.maximum));
    };
    ranges.at(QuantOperator::InputGateOutput)
        .push_back({0.0F, sigmoidMaximum(observed.gate_inputs[0])});
    ranges.at(QuantOperator::ForgetGateOutput)
        .push_back({0.0F, sigmoidMaximum(observed.gate_inputs[1])});
    ranges.at(QuantOperator::CellGateOutput)
        .push_back({std::tanh(observed.gate_inputs[2].minimum),
                    std::tanh(observed.gate_inputs[2].maximum)});
    ranges.at(QuantOperator::OutputGateOutput)
        .push_back({0.0F, sigmoidMaximum(observed.gate_inputs[3])});
    ranges.at(QuantOperator::CellTanhOutput)
        .push_back({std::tanh(observed.cell.minimum),
                    std::tanh(observed.cell.maximum)});

    const std::size_t hidden = static_cast<std::size_t>(shape.hidden_size);
    addParameterRanges(parameter_master.weight_ih,
                       static_cast<std::size_t>(shape.input_size), hidden,
                       QuantOperator::WeightInputHidden, config, &ranges);
    addParameterRanges(parameter_master.weight_hh, hidden, hidden,
                       QuantOperator::WeightHiddenHidden, config, &ranges);
    if (parameter_master.bias_enabled) {
        addParameterRanges(parameter_master.bias_ih, 1, hidden,
                           QuantOperator::BiasInputHidden, config, &ranges);
        addParameterRanges(parameter_master.bias_hh, 1, hidden,
                           QuantOperator::BiasHiddenHidden, config, &ranges);
    }
    return ranges;
}

std::vector<std::int32_t> quantizeTensor(
    const std::vector<float>& source, QuantOperator id,
    const quant_lstm::LstmOperatorQuantConfig& config,
    const quant_lstm::LstmQuantParams& params, std::size_t row_width = 0) {
    std::vector<std::int32_t> result(source.size());
    const auto& points = params.at(id).values;
    for (std::size_t index = 0; index < source.size(); ++index) {
        const std::size_t parameter_index =
            row_width == 0 ? 0 : index / row_width;
        result[index] = quant_lstm::quantization::quantize(
            source[index], points.at(parameter_index), config.at(id).type);
    }
    return result;
}

std::vector<float> asFloatCarrier(const std::vector<std::int32_t>& values) {
    return std::vector<float>(values.begin(), values.end());
}

template <typename Quantized>
std::vector<float> dequantizeTensor(
    const std::vector<Quantized>& source, QuantOperator id,
    const quant_lstm::LstmOperatorQuantConfig& config,
    const quant_lstm::LstmQuantParams& params) {
    std::vector<float> result(source.size());
    const auto& point = params.at(id).values.front();
    for (std::size_t index = 0; index < source.size(); ++index) {
        const auto q = static_cast<std::int32_t>(source[index]);
        result[index] = quant_lstm::quantization::dequantize(
            q, point, config.at(id).type);
    }
    return result;
}

TensorMetrics gateMetrics(const std::vector<float>& actual,
                          const std::vector<float>& expected,
                          const Json& threshold) {
    TensorMetrics result;
    result.metrics = quant_lstm::test::computeNumericMetrics(
        actual.data(), expected.data(), actual.size());
    const bool cosine_pass =
        result.metrics.cosine_not_applicable
            ? result.metrics.mean_absolute_error == 0.0 &&
                  result.metrics.mean_squared_error == 0.0
            : !result.metrics.one_sided_zero_norm &&
                  result.metrics.cosine_similarity >=
                      threshold.at("cosine_similarity").get<double>();
    result.passed =
        result.metrics.mean_absolute_error <
            threshold.at("mae").get<double>() &&
        result.metrics.mean_squared_error <
            threshold.at("mse").get<double>() &&
        cosine_pass;
    return result;
}

Json metricsJson(const TensorMetrics& result) {
    Json output{
        {"mae", result.metrics.mean_absolute_error},
        {"mse", result.metrics.mean_squared_error},
        {"max_absolute_error", result.metrics.maximum_absolute_error},
        {"passed", result.passed},
    };
    if (result.metrics.cosine_not_applicable) {
        output["cosine_similarity"] = nullptr;
        output["cosine_status"] = "both_near_zero";
    } else if (result.metrics.one_sided_zero_norm) {
        output["cosine_similarity"] = nullptr;
        output["cosine_status"] = "one_near_zero";
    } else {
        output["cosine_similarity"] = result.metrics.cosine_similarity;
        output["cosine_status"] = "valid";
    }
    return output;
}

Json carrierResult(
    const char* carrier, const std::vector<float>& output,
    const std::vector<float>& hidden, const std::vector<float>& cell,
    const std::vector<float>& expected_output,
    const std::vector<float>& expected_hidden,
    const std::vector<float>& expected_cell, const Json& threshold,
    bool* all_passed) {
    const auto output_metrics =
        gateMetrics(output, expected_output, threshold);
    const auto hidden_metrics =
        gateMetrics(hidden, expected_hidden, threshold);
    const auto cell_metrics = gateMetrics(cell, expected_cell, threshold);
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

Json runProfile(const Json& profile, const Json& thresholds,
                bool* all_passed) {
    const auto& dimensions = profile.at("shape");
    const quant_lstm::LstmShape shape{
        dimensions.at(0).get<std::int64_t>(),
        dimensions.at(1).get<std::int64_t>(),
        dimensions.at(2).get<std::int64_t>(),
        dimensions.at(3).get<std::int64_t>()};
    const auto config = quant_lstm::parseResolvedQuantConfig(
        profile.at("resolved_quant_config").dump(), false);
    const std::uint64_t evaluation_seed =
        profile.at("data_seeds").at("evaluation").front().get<std::uint64_t>();
    const FloatMaster master = makeMaster(profile, evaluation_seed);
    const auto ranges = makeRanges(profile, shape, config, master);
    const auto params = quant_lstm::finalizeQuantParams(
        config, ranges, shape.hidden_size, master.bias_enabled);
    const auto execution =
        quant_lstm::deriveLstmExecutionParams(config, params, shape.input_size);

    const std::size_t input_size = static_cast<std::size_t>(shape.input_size);
    const std::size_t hidden = static_cast<std::size_t>(shape.hidden_size);
    const std::size_t output_count = static_cast<std::size_t>(
        shape.sequence_length * shape.batch_size * shape.hidden_size);
    const std::size_t state_count =
        static_cast<std::size_t>(shape.batch_size * shape.hidden_size);
    const std::size_t channels = 4 * hidden;
    const auto q_input =
        quantizeTensor(master.input, QuantOperator::Input, config, params);
    const auto q_weight_ih = quantizeTensor(
        master.weight_ih, QuantOperator::WeightInputHidden, config, params,
        input_size);
    const auto q_weight_hh = quantizeTensor(
        master.weight_hh, QuantOperator::WeightHiddenHidden, config, params,
        hidden);
    const auto q_bias_ih =
        master.bias_enabled
            ? quantizeTensor(master.bias_ih, QuantOperator::BiasInputHidden,
                             config, params, 1)
            : std::vector<std::int32_t>{};
    const auto q_bias_hh =
        master.bias_enabled
            ? quantizeTensor(master.bias_hh, QuantOperator::BiasHiddenHidden,
                             config, params, 1)
            : std::vector<std::int32_t>{};
    const auto q_h0 =
        master.has_explicit_state
            ? quantizeTensor(master.h0, QuantOperator::Output, config, params)
            : std::vector<std::int32_t>{};
    const auto q_c0 =
        master.has_explicit_state
            ? quantizeTensor(master.c0, QuantOperator::CellState, config,
                             params)
            : std::vector<std::int32_t>{};

    std::vector<float> expected_output(output_count);
    std::vector<float> expected_hidden(state_count);
    std::vector<float> expected_cell(state_count);
    quant_lstm::lstmForwardFloatCpu(
        shape,
        {master.weight_ih.data(), master.weight_hh.data(),
         master.bias_enabled ? master.bias_ih.data() : nullptr,
         master.bias_enabled ? master.bias_hh.data() : nullptr},
        master.input.data(),
        master.has_explicit_state ? master.h0.data() : nullptr,
        master.has_explicit_state ? master.c0.data() : nullptr,
        expected_output.data(), expected_hidden.data(), expected_cell.data());

    std::vector<std::int32_t> int_output(output_count);
    std::vector<std::int32_t> int_hidden(state_count);
    std::vector<std::int32_t> int_cell(state_count);
    quant_lstm::lstmForwardInt32CpuReference(
        shape,
        {q_weight_ih.data(), q_weight_hh.data(),
         master.bias_enabled ? q_bias_ih.data() : nullptr,
         master.bias_enabled ? q_bias_hh.data() : nullptr},
        q_input.data(), master.has_explicit_state ? q_h0.data() : nullptr,
        master.has_explicit_state ? q_c0.data() : nullptr, config, params,
        execution, int_output.data(), int_hidden.data(), int_cell.data());

    const auto fp_input = asFloatCarrier(q_input);
    const auto fp_weight_ih = asFloatCarrier(q_weight_ih);
    const auto fp_weight_hh = asFloatCarrier(q_weight_hh);
    const auto fp_bias_ih = asFloatCarrier(q_bias_ih);
    const auto fp_bias_hh = asFloatCarrier(q_bias_hh);
    const auto fp_h0 = asFloatCarrier(q_h0);
    const auto fp_c0 = asFloatCarrier(q_c0);
    std::vector<float> fp_output(output_count);
    std::vector<float> fp_hidden(state_count);
    std::vector<float> fp_cell(state_count);
    quant_lstm::lstmForwardQuantizedFpCpuReference(
        shape,
        {fp_weight_ih.data(), fp_weight_hh.data(),
         master.bias_enabled ? fp_bias_ih.data() : nullptr,
         master.bias_enabled ? fp_bias_hh.data() : nullptr},
        fp_input.data(), master.has_explicit_state ? fp_h0.data() : nullptr,
        master.has_explicit_state ? fp_c0.data() : nullptr, config, params,
        execution, fp_output.data(), fp_hidden.data(), fp_cell.data());

    const auto int_dequant_output = dequantizeTensor(
        int_output, QuantOperator::Output, config, params);
    const auto int_dequant_hidden = dequantizeTensor(
        int_hidden, QuantOperator::Output, config, params);
    const auto int_dequant_cell = dequantizeTensor(
        int_cell, QuantOperator::CellState, config, params);
    const auto fp_dequant_output = dequantizeTensor(
        fp_output, QuantOperator::Output, config, params);
    const auto fp_dequant_hidden = dequantizeTensor(
        fp_hidden, QuantOperator::Output, config, params);
    const auto fp_dequant_cell = dequantizeTensor(
        fp_cell, QuantOperator::CellState, config, params);
    const Json& threshold =
        thresholds.at("profiles").at(profile.at("threshold_profile"));

    Json carriers = Json::array();
    carriers.push_back(carrierResult(
        "int32", int_dequant_output, int_dequant_hidden, int_dequant_cell,
        expected_output, expected_hidden, expected_cell, threshold,
        all_passed));
    carriers.push_back(carrierResult(
        "float32_quantized_values", fp_dequant_output, fp_dequant_hidden,
        fp_dequant_cell, expected_output, expected_hidden, expected_cell,
        threshold, all_passed));
    const bool profile_passed =
        carriers.at(0).at("passed").get<bool>() &&
        carriers.at(1).at("passed").get<bool>();
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
         {{"input", q_input.size()},
          {"weight_ih", q_weight_ih.size()},
          {"weight_hh", q_weight_hh.size()},
          {"bias", master.bias_enabled ? 2 * channels : 0}}},
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
