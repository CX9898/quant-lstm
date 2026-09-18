#include "common/synthetic_precision_fixture.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "common/deterministic_rng.h"
#include "lstm/forward_cpu.h"
#include "lstm/quant_config_loader.h"
#include "quantization/fixed_point_ops.h"

namespace quant_lstm::test {
namespace {

using Json = nlohmann::json;

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

    CalibrationRange expanded(float fraction = 0.15F) const {
        if (!std::isfinite(minimum) || !std::isfinite(maximum)) {
            throw std::runtime_error("校准范围没有观测值");
        }
        const float span = std::max(maximum - minimum, 1.0e-4F);
        const float padding = span * fraction;
        return {minimum - padding, maximum + padding};
    }
};

struct CalibrationObservations {
    ObservedRange input;
    ObservedRange output;
    ObservedRange cell;
    ObservedRange weight_ih_linear;
    ObservedRange weight_hh_linear;
    std::array<ObservedRange, 4> gate_inputs;
};

std::vector<float> canonicalInput(const LstmShape& shape, std::uint64_t seed, bool batch_first,
                                  const std::string& activation_profile,
                                  const Json& directed_profiles) {
    const std::size_t steps = static_cast<std::size_t>(shape.sequence_length);
    const std::size_t batch = static_cast<std::size_t>(shape.batch_size);
    const std::size_t input_size = static_cast<std::size_t>(shape.input_size);
    std::vector<float> generated(steps * batch * input_size);
    fillNormalLike(generated.data(), generated.size(), seed, TensorStream::Input);

    const bool is_unsigned =
        activation_profile == "unsigned_symmetric" || activation_profile == "unsigned_asymmetric";
    const bool is_asymmetric =
        activation_profile == "signed_asymmetric" || activation_profile == "unsigned_asymmetric";
    for (float& value : generated) {
        value *= 0.35F;
        if (is_unsigned) {
            value = std::abs(value);
        } else if (is_asymmetric) {
            value += 0.20F;
        }
    }

    const std::unordered_set<std::string> directed(directed_profiles.begin(),
                                                   directed_profiles.end());
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
            std::copy_n(generated.data() + source, input_size, time_major.data() + target);
        }
    }
    return time_major;
}

void makeInitialState(const LstmShape& shape, std::uint64_t seed, const std::string& profile,
                      FloatMaster* master) {
    const std::size_t count = static_cast<std::size_t>(shape.batch_size * shape.hidden_size);
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
    fillNormalLike(master->h0.data(), count, seed, TensorStream::InitialHidden);
    fillNormalLike(master->c0.data(), count, seed, TensorStream::InitialCell);
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

float sigmoid(float value) {
    if (value >= 0.0F) {
        return 1.0F / (1.0F + std::exp(-value));
    }
    const float exp_value = std::exp(value);
    return exp_value / (1.0F + exp_value);
}

void observeCalibrationRun(const LstmShape& shape, const FloatMaster& master,
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
            const float* input_row = master.input.data() + (step * batch + row) * input_size;
            float* hidden_row = hidden_state.data() + row * hidden;
            float* cell_row = cell_state.data() + row * hidden;
            for (std::size_t channel = 0; channel < channels; ++channel) {
                float input_linear = master.bias_enabled ? master.bias_ih[channel] : 0.0F;
                float hidden_linear = master.bias_enabled ? master.bias_hh[channel] : 0.0F;
                for (std::size_t column = 0; column < input_size; ++column) {
                    input_linear +=
                        master.weight_ih[channel * input_size + column] * input_row[column];
                }
                for (std::size_t column = 0; column < hidden; ++column) {
                    hidden_linear +=
                        master.weight_hh[channel * hidden + column] * hidden_row[column];
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
                cell_row[column] = forget_gate * cell_row[column] + input_gate * cell_gate;
                hidden_row[column] = output_gate * std::tanh(cell_row[column]);
                observations->cell.add(cell_row[column]);
                observations->output.add(hidden_row[column]);
            }
        }
    }
}

std::size_t groupCount(QuantOperator id, QuantGranularity granularity, std::size_t channels) {
    if (!isParameterOperator(id) || granularity == QuantGranularity::PerTensor) {
        return 1;
    }
    if (granularity == QuantGranularity::PerGate) {
        return 4;
    }
    return channels;
}

void addParameterRanges(const std::vector<float>& values, std::size_t row_width, std::size_t hidden,
                        QuantOperator id, const LstmOperatorQuantConfig& config,
                        LstmQuantizationRanges* ranges) {
    const std::size_t channels = 4 * hidden;
    const auto granularity = config.at(id).granularity;
    std::vector<ObservedRange> observed(groupCount(id, granularity, channels));
    for (std::size_t row = 0; row < channels; ++row) {
        std::size_t group = 0;
        if (granularity == QuantGranularity::PerGate) {
            group = row / hidden;
        } else if (granularity == QuantGranularity::PerChannel) {
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

template <typename Quantized>
std::vector<float> dequantizeTensorImpl(const std::vector<Quantized>& source, QuantOperator id,
                                        const LstmOperatorQuantConfig& config,
                                        const LstmQuantParams& params) {
    std::vector<float> result(source.size());
    const auto& point = params.at(id).values.front();
    for (std::size_t index = 0; index < source.size(); ++index) {
        const auto q = static_cast<std::int32_t>(source[index]);
        result[index] = quantization::dequantize(q, point, config.at(id).type);
    }
    return result;
}

template <typename Quantized>
double saturationRate(const Quantized* values, std::size_t count,
                      quantization::QuantizationType type) {
    if (values == nullptr && count != 0) {
        throw std::invalid_argument("饱和率输入指针不能为空");
    }
    if (count == 0) {
        return 0.0;
    }
    const auto range = type.range();
    std::size_t saturated = 0;
    for (std::size_t index = 0; index < count; ++index) {
        const double value = static_cast<double>(values[index]);
        if (value <= static_cast<double>(range.minimum) ||
            value >= static_cast<double>(range.maximum)) {
            ++saturated;
        }
    }
    return static_cast<double>(saturated) / static_cast<double>(count);
}

SyntheticNumericMetrics computeMetrics(const float* actual, const float* expected,
                                       std::size_t count) {
    SyntheticNumericMetrics result;
    result.accuracy = computeNumericMetrics(actual, expected, count);
    double signal_power = 0.0;
    double noise_power = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        const double signal = static_cast<double>(expected[index]);
        const double noise =
            static_cast<double>(actual[index]) - static_cast<double>(expected[index]);
        signal_power += signal * signal;
        noise_power += noise * noise;
    }
    if (noise_power == 0.0) {
        result.sqnr_db = 300.0;
    } else if (signal_power == 0.0) {
        result.sqnr_db = -300.0;
    } else {
        result.sqnr_db = 10.0 * std::log10(signal_power / noise_power);
    }
    return result;
}

}  // namespace

LstmShape shapeFromSyntheticProfile(const Json& profile) {
    const auto& dimensions = profile.at("shape");
    return {dimensions.at(0).get<std::int64_t>(), dimensions.at(1).get<std::int64_t>(),
            dimensions.at(2).get<std::int64_t>(), dimensions.at(3).get<std::int64_t>()};
}

FloatMaster makeFloatMaster(const Json& profile, std::uint64_t data_seed) {
    const LstmShape shape = shapeFromSyntheticProfile(profile);
    const std::size_t hidden = static_cast<std::size_t>(shape.hidden_size);
    const std::size_t channels = 4 * hidden;
    FloatMaster master;
    master.bias_enabled = profile.at("bias_profile") != "disabled";
    master.input = canonicalInput(shape, data_seed, profile.at("batch_first").get<bool>(),
                                  profile.at("activation_profile").get<std::string>(),
                                  profile.at("directed_profiles"));
    master.weight_ih.resize(channels * static_cast<std::size_t>(shape.input_size));
    master.weight_hh.resize(channels * hidden);
    fillLstmParameter(master.weight_ih.data(), master.weight_ih.size(), shape.hidden_size,
                      profile.at("parameter_seed").get<std::uint64_t>(),
                      TensorStream::WeightInputHidden);
    fillLstmParameter(master.weight_hh.data(), master.weight_hh.size(), shape.hidden_size,
                      profile.at("parameter_seed").get<std::uint64_t>(),
                      TensorStream::WeightHiddenHidden);
    if (master.bias_enabled) {
        master.bias_ih.resize(channels);
        master.bias_hh.resize(channels);
        fillLstmParameter(master.bias_ih.data(), channels, shape.hidden_size,
                          profile.at("parameter_seed").get<std::uint64_t>(),
                          TensorStream::BiasInputHidden);
        fillLstmParameter(master.bias_hh.data(), channels, shape.hidden_size,
                          profile.at("parameter_seed").get<std::uint64_t>(),
                          TensorStream::BiasHiddenHidden);
    }
    if (shape.sequence_length >= 128) {
        for (float& value : master.weight_ih) {
            value *= 0.25F;
        }
        for (float& value : master.weight_hh) {
            value *= 0.25F;
        }
        if (master.bias_enabled) {
            const std::array<float, 4> gate_bias{0.25F, 0.75F, 0.25F, 0.50F};
            for (std::size_t gate = 0; gate < 4; ++gate) {
                for (std::size_t column = 0; column < hidden; ++column) {
                    const std::size_t index = gate * hidden + column;
                    master.bias_ih[index] = gate_bias[gate] + 0.05F * master.bias_ih[index];
                    master.bias_hh[index] = 0.05F * master.bias_hh[index];
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
    makeInitialState(shape, data_seed, profile.at("state_profile").get<std::string>(), &master);
    return master;
}

LstmQuantizationRanges makeSyntheticRanges(const Json& profile, const LstmShape& shape,
                                           const LstmOperatorQuantConfig& config,
                                           const FloatMaster& parameter_master) {
    CalibrationObservations observed;
    for (const auto& seed : profile.at("data_seeds").at("calibration")) {
        const auto calibration_master = makeFloatMaster(profile, seed.get<std::uint64_t>());
        observeCalibrationRun(shape, calibration_master, &observed);
    }

    LstmQuantizationRanges ranges;
    ranges.at(QuantOperator::Input).push_back(observed.input.expanded());
    ranges.at(QuantOperator::Output).push_back(observed.output.expanded(0.20F));
    ranges.at(QuantOperator::CellState).push_back(observed.cell.expanded());
    ranges.at(QuantOperator::WeightInputHiddenLinear)
        .push_back(observed.weight_ih_linear.expanded(0.20F));
    ranges.at(QuantOperator::WeightHiddenHiddenLinear)
        .push_back(observed.weight_hh_linear.expanded(0.20F));
    const std::array<QuantOperator, 4> gate_inputs{
        {QuantOperator::InputGateInput, QuantOperator::ForgetGateInput,
         QuantOperator::CellGateInput, QuantOperator::OutputGateInput}};
    for (std::size_t gate = 0; gate < gate_inputs.size(); ++gate) {
        ranges.at(gate_inputs[gate]).push_back(observed.gate_inputs[gate].expanded(0.20F));
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
        .push_back({std::tanh(observed.cell.minimum), std::tanh(observed.cell.maximum)});

    const std::size_t hidden = static_cast<std::size_t>(shape.hidden_size);
    addParameterRanges(parameter_master.weight_ih, static_cast<std::size_t>(shape.input_size),
                       hidden, QuantOperator::WeightInputHidden, config, &ranges);
    addParameterRanges(parameter_master.weight_hh, hidden, hidden,
                       QuantOperator::WeightHiddenHidden, config, &ranges);
    if (parameter_master.bias_enabled) {
        addParameterRanges(parameter_master.bias_ih, 1, hidden, QuantOperator::BiasInputHidden,
                           config, &ranges);
        addParameterRanges(parameter_master.bias_hh, 1, hidden, QuantOperator::BiasHiddenHidden,
                           config, &ranges);
    }
    return ranges;
}

std::vector<std::int32_t> quantizeTensor(const std::vector<float>& source, QuantOperator id,
                                         const LstmOperatorQuantConfig& config,
                                         const LstmQuantParams& params, std::size_t row_width) {
    std::vector<std::int32_t> result(source.size());
    const auto& points = params.at(id).values;
    for (std::size_t index = 0; index < source.size(); ++index) {
        const std::size_t parameter_index = row_width == 0 ? 0 : index / row_width;
        result[index] =
            quantization::quantize(source[index], points.at(parameter_index), config.at(id).type);
    }
    return result;
}

std::vector<float> asFloatCarrier(const std::vector<std::int32_t>& values) {
    return std::vector<float>(values.begin(), values.end());
}

std::vector<float> dequantizeTensor(const std::vector<std::int32_t>& source, QuantOperator id,
                                    const LstmOperatorQuantConfig& config,
                                    const LstmQuantParams& params) {
    return dequantizeTensorImpl(source, id, config, params);
}

std::vector<float> dequantizeTensor(const std::vector<float>& source, QuantOperator id,
                                    const LstmOperatorQuantConfig& config,
                                    const LstmQuantParams& params) {
    return dequantizeTensorImpl(source, id, config, params);
}

QuantizedMaster quantizeMaster(const LstmShape& shape, const LstmOperatorQuantConfig& config,
                               const LstmQuantParams& params, const FloatMaster& master) {
    const std::size_t input_size = static_cast<std::size_t>(shape.input_size);
    const std::size_t hidden = static_cast<std::size_t>(shape.hidden_size);
    QuantizedMaster result;
    result.input = quantizeTensor(master.input, QuantOperator::Input, config, params);
    result.weight_ih = quantizeTensor(master.weight_ih, QuantOperator::WeightInputHidden, config,
                                      params, input_size);
    result.weight_hh =
        quantizeTensor(master.weight_hh, QuantOperator::WeightHiddenHidden, config, params, hidden);
    if (master.bias_enabled) {
        result.bias_ih =
            quantizeTensor(master.bias_ih, QuantOperator::BiasInputHidden, config, params, 1);
        result.bias_hh =
            quantizeTensor(master.bias_hh, QuantOperator::BiasHiddenHidden, config, params, 1);
    }
    if (master.has_explicit_state) {
        result.h0 = quantizeTensor(master.h0, QuantOperator::Output, config, params);
        result.c0 = quantizeTensor(master.c0, QuantOperator::CellState, config, params);
    }
    return result;
}

FloatLstmResult runCpuFloatOracle(const LstmShape& shape, const FloatMaster& master) {
    const std::size_t output_count =
        static_cast<std::size_t>(shape.sequence_length * shape.batch_size * shape.hidden_size);
    const std::size_t state_count = static_cast<std::size_t>(shape.batch_size * shape.hidden_size);
    FloatLstmResult result{{}, {}, {}};
    result.output.resize(output_count);
    result.final_hidden.resize(state_count);
    result.final_cell.resize(state_count);
    lstmForwardFloatCpu(shape,
                        {master.weight_ih.data(), master.weight_hh.data(),
                         master.bias_enabled ? master.bias_ih.data() : nullptr,
                         master.bias_enabled ? master.bias_hh.data() : nullptr},
                        master.input.data(), master.has_explicit_state ? master.h0.data() : nullptr,
                        master.has_explicit_state ? master.c0.data() : nullptr,
                        result.output.data(), result.final_hidden.data(), result.final_cell.data());
    return result;
}

Int32LstmResult runCpuInt32Oracle(const LstmShape& shape, const LstmOperatorQuantConfig& config,
                                  const LstmQuantParams& params,
                                  const LstmExecutionParams& execution, const FloatMaster& master,
                                  const QuantizedMaster& quantized) {
    const std::size_t output_count =
        static_cast<std::size_t>(shape.sequence_length * shape.batch_size * shape.hidden_size);
    const std::size_t state_count = static_cast<std::size_t>(shape.batch_size * shape.hidden_size);
    Int32LstmResult result{{}, {}, {}};
    result.output.resize(output_count);
    result.final_hidden.resize(state_count);
    result.final_cell.resize(state_count);
    lstmForwardInt32CpuReference(
        shape,
        {quantized.weight_ih.data(), quantized.weight_hh.data(),
         master.bias_enabled ? quantized.bias_ih.data() : nullptr,
         master.bias_enabled ? quantized.bias_hh.data() : nullptr},
        quantized.input.data(), master.has_explicit_state ? quantized.h0.data() : nullptr,
        master.has_explicit_state ? quantized.c0.data() : nullptr, config, params, execution,
        result.output.data(), result.final_hidden.data(), result.final_cell.data());
    return result;
}

FloatLstmResult runCpuFpQuantizedOracle(const LstmShape& shape,
                                        const LstmOperatorQuantConfig& config,
                                        const LstmQuantParams& params,
                                        const LstmExecutionParams& execution,
                                        const FloatMaster& master,
                                        const QuantizedMaster& quantized) {
    const auto input = asFloatCarrier(quantized.input);
    const auto weight_ih = asFloatCarrier(quantized.weight_ih);
    const auto weight_hh = asFloatCarrier(quantized.weight_hh);
    const auto bias_ih = asFloatCarrier(quantized.bias_ih);
    const auto bias_hh = asFloatCarrier(quantized.bias_hh);
    const auto h0 = asFloatCarrier(quantized.h0);
    const auto c0 = asFloatCarrier(quantized.c0);
    const std::size_t output_count =
        static_cast<std::size_t>(shape.sequence_length * shape.batch_size * shape.hidden_size);
    const std::size_t state_count = static_cast<std::size_t>(shape.batch_size * shape.hidden_size);
    FloatLstmResult result{{}, {}, {}};
    result.output.resize(output_count);
    result.final_hidden.resize(state_count);
    result.final_cell.resize(state_count);
    lstmForwardQuantizedFpCpuReference(
        shape,
        {weight_ih.data(), weight_hh.data(), master.bias_enabled ? bias_ih.data() : nullptr,
         master.bias_enabled ? bias_hh.data() : nullptr},
        input.data(), master.has_explicit_state ? h0.data() : nullptr,
        master.has_explicit_state ? c0.data() : nullptr, config, params, execution,
        result.output.data(), result.final_hidden.data(), result.final_cell.data());
    return result;
}

SyntheticPrecisionFixture makeSyntheticPrecisionFixture(const Json& profile) {
    SyntheticPrecisionFixture fixture;
    fixture.shape = shapeFromSyntheticProfile(profile);
    fixture.config = parseResolvedQuantConfig(profile.at("resolved_quant_config").dump(), false);
    const std::uint64_t evaluation_seed =
        profile.at("data_seeds").at("evaluation").front().get<std::uint64_t>();
    fixture.master = makeFloatMaster(profile, evaluation_seed);
    fixture.ranges = makeSyntheticRanges(profile, fixture.shape, fixture.config, fixture.master);
    fixture.params = finalizeQuantParams(fixture.config, fixture.ranges, fixture.shape.hidden_size,
                                         fixture.master.bias_enabled);
    fixture.execution =
        deriveLstmExecutionParams(fixture.config, fixture.params, fixture.shape.input_size);
    fixture.quantized =
        quantizeMaster(fixture.shape, fixture.config, fixture.params, fixture.master);
    fixture.float_oracle = runCpuFloatOracle(fixture.shape, fixture.master);
    fixture.int32_oracle = runCpuInt32Oracle(fixture.shape, fixture.config, fixture.params,
                                             fixture.execution, fixture.master, fixture.quantized);
    fixture.fp_quantized_oracle =
        runCpuFpQuantizedOracle(fixture.shape, fixture.config, fixture.params, fixture.execution,
                                fixture.master, fixture.quantized);
    return fixture;
}

SyntheticNumericMetrics computeSyntheticNumericMetrics(const float* actual, const float* expected,
                                                       std::size_t count) {
    return computeMetrics(actual, expected, count);
}

SyntheticNumericMetrics computeSyntheticNumericMetrics(
    const float* actual, const float* expected, std::size_t count,
    const std::int32_t* quantized_values, quantization::QuantizationType quantization_type) {
    auto result = computeMetrics(actual, expected, count);
    result.saturation_rate = saturationRate(quantized_values, count, quantization_type);
    return result;
}

SyntheticNumericMetrics computeSyntheticNumericMetrics(
    const float* actual, const float* expected, std::size_t count, const float* quantized_values,
    quantization::QuantizationType quantization_type) {
    auto result = computeMetrics(actual, expected, count);
    result.saturation_rate = saturationRate(quantized_values, count, quantization_type);
    return result;
}

TensorMetrics evaluateTensorMetrics(const std::vector<float>& actual,
                                    const std::vector<float>& expected,
                                    const MetricThresholds& thresholds) {
    if (actual.size() != expected.size()) {
        throw std::invalid_argument("指标张量元素数量不一致");
    }
    TensorMetrics result;
    result.metrics = computeSyntheticNumericMetrics(actual.data(), expected.data(), actual.size());
    const auto& accuracy = result.metrics.accuracy;
    const bool cosine_pass =
        accuracy.cosine_not_applicable
            ? accuracy.mean_absolute_error == 0.0 && accuracy.mean_squared_error == 0.0
            : !accuracy.one_sided_zero_norm &&
                  accuracy.cosine_similarity >= thresholds.minimum_cosine_similarity;
    result.passed = accuracy.mean_absolute_error < thresholds.maximum_mae &&
                    accuracy.mean_squared_error < thresholds.maximum_mse && cosine_pass;
    return result;
}

}  // namespace quant_lstm::test
