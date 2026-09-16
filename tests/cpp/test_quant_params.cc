#include "lstm/quant_config_loader.h"
#include "lstm/quant_params.h"
#include "quantization/scale_encoding.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void requireThrows(Function&& function, const char* message) {
    bool threw = false;
    try {
        function();
    } catch (const std::exception&) {
        threw = true;
    }
    require(threw, message);
}

std::size_t groupCount(quant_lstm::QuantOperator id,
                       quant_lstm::QuantGranularity granularity,
                       std::size_t channel_count) {
    if (!quant_lstm::isParameterOperator(id)) {
        return 1;
    }
    if (granularity == quant_lstm::QuantGranularity::PerTensor) {
        return 1;
    }
    if (granularity == quant_lstm::QuantGranularity::PerGate) {
        return 4;
    }
    return channel_count;
}

quant_lstm::LstmQuantizationRanges makeRanges(
    const quant_lstm::LstmOperatorQuantConfig& config, std::size_t channel_count,
    bool bias_enabled) {
    quant_lstm::LstmQuantizationRanges ranges;
    for (std::size_t index = 0; index < quant_lstm::kQuantOperatorCount; ++index) {
        const auto id = static_cast<quant_lstm::QuantOperator>(index);
        if (!bias_enabled && quant_lstm::isBiasOperator(id)) {
            continue;
        }
        const std::size_t count =
            groupCount(id, config.operators[index].granularity, channel_count);
        auto& groups = ranges.operators[index];
        for (std::size_t group = 0; group < count; ++group) {
            const float extent = static_cast<float>(group + 1);
            groups.push_back({-extent, extent * 0.75F});
        }
    }
    return ranges;
}

}  // namespace

int main() {
    try {
        constexpr std::int64_t hidden_size = 2;
        constexpr std::size_t channel_count = 8;
        const std::filesystem::path defaults_path =
            std::filesystem::path(QUANT_LSTM_SOURCE_DIR) /
            "config/defaults/lstm_quant_default_v1.json";
        const auto defaults = quant_lstm::resolveQuantConfigFiles(defaults_path);
        const std::string override_json = R"({
          "schema_version": 1,
          "operators": {
            "weight_ih": {"granularity": "per_tensor"},
            "weight_hh": {"granularity": "per_gate"},
            "bias_ih": {"granularity": "per_channel"},
            "bias_hh": {"granularity": "per_tensor"}
          }
        })";
        const auto config =
            quant_lstm::resolveQuantConfig(quant_lstm::toCanonicalJson(defaults), override_json);
        const auto ranges = makeRanges(config, channel_count, true);
        const auto params =
            quant_lstm::finalizeQuantParams(config, ranges, hidden_size, true);
        params.validate(config);

        for (quant_lstm::QuantOperator id :
             {quant_lstm::QuantOperator::WeightInputHidden,
              quant_lstm::QuantOperator::WeightHiddenHidden,
              quant_lstm::QuantOperator::BiasInputHidden,
              quant_lstm::QuantOperator::BiasHiddenHidden}) {
            const auto& values = params.at(id).values;
            require(values.size() == channel_count, "parameter values must be 4H");
            for (const auto& value : values) {
                require(value.zero_point == 0, "parameter zero point must be zero");
            }
        }

        const auto& per_tensor =
            params.at(quant_lstm::QuantOperator::WeightInputHidden).values;
        for (const auto& value : per_tensor) {
            require(value.scale == per_tensor.front().scale, "per-tensor expansion");
        }

        const auto& per_gate =
            params.at(quant_lstm::QuantOperator::WeightHiddenHidden).values;
        for (std::size_t gate = 0; gate < 4; ++gate) {
            require(per_gate[gate * hidden_size].scale ==
                        per_gate[gate * hidden_size + 1].scale,
                    "per-gate segment expansion");
            const auto expected = quant_lstm::quantization::calibrateMinMax(
                ranges.at(quant_lstm::QuantOperator::WeightHiddenHidden)[gate].minimum,
                ranges.at(quant_lstm::QuantOperator::WeightHiddenHidden)[gate].maximum,
                config.at(quant_lstm::QuantOperator::WeightHiddenHidden).type);
            require(per_gate[gate * hidden_size].scale == expected.param.scale,
                    "per-gate source mapping");
        }

        const auto& per_channel =
            params.at(quant_lstm::QuantOperator::BiasInputHidden).values;
        for (std::size_t channel = 0; channel < channel_count; ++channel) {
            const auto expected = quant_lstm::quantization::calibrateMinMax(
                ranges.at(quant_lstm::QuantOperator::BiasInputHidden)[channel].minimum,
                ranges.at(quant_lstm::QuantOperator::BiasInputHidden)[channel].maximum,
                config.at(quant_lstm::QuantOperator::BiasInputHidden).type);
            require(per_channel[channel].scale == expected.param.scale,
                    "per-channel source mapping");
        }

        const auto no_bias_ranges = makeRanges(config, channel_count, false);
        const auto no_bias_params =
            quant_lstm::finalizeQuantParams(config, no_bias_ranges, hidden_size, false);
        require(no_bias_params.at(quant_lstm::QuantOperator::BiasInputHidden).values.empty() &&
                    no_bias_params.at(quant_lstm::QuantOperator::BiasHiddenHidden).values.empty(),
                "bias=False parameters must be absent");
        no_bias_params.validate(config);

        for (quant_lstm::QuantOperator id :
             {quant_lstm::QuantOperator::WeightInputHidden,
              quant_lstm::QuantOperator::WeightHiddenHidden,
              quant_lstm::QuantOperator::BiasInputHidden,
              quant_lstm::QuantOperator::BiasHiddenHidden}) {
            for (quant_lstm::QuantGranularity granularity :
                 {quant_lstm::QuantGranularity::PerTensor,
                  quant_lstm::QuantGranularity::PerGate,
                  quant_lstm::QuantGranularity::PerChannel}) {
                auto independent_config = defaults;
                independent_config.at(id).granularity = granularity;
                const auto independent_ranges =
                    makeRanges(independent_config, channel_count, true);
                const auto independent_params = quant_lstm::finalizeQuantParams(
                    independent_config, independent_ranges, hidden_size, true);
                const auto& finalized = independent_params.at(id);
                require(finalized.values.size() == channel_count &&
                            finalized.source_granularity == granularity,
                        "independent parameter granularity");
                const std::size_t expected_groups =
                    groupCount(id, granularity, channel_count);
                require(finalized.group_diagnostics.size() == expected_groups,
                        "independent calibration group count");
            }
        }

        auto equal_gate_ranges = ranges;
        equal_gate_ranges.at(quant_lstm::QuantOperator::WeightHiddenHidden)
            .assign(4, quant_lstm::CalibrationRange{-1.0F, 1.0F});
        static_cast<void>(
            quant_lstm::finalizeQuantParams(config, equal_gate_ranges, hidden_size, true));

        for (std::size_t index = 0; index < quant_lstm::kQuantOperatorCount; ++index) {
            const auto id = static_cast<quant_lstm::QuantOperator>(index);
            if (quant_lstm::isParameterOperator(id)) {
                continue;
            }
            auto field_config = defaults;
            field_config.at(id).type = {16, true, false};
            const auto field_ranges = makeRanges(field_config, channel_count, true);
            const auto field_params =
                quant_lstm::finalizeQuantParams(field_config, field_ranges, hidden_size, true);
            const auto expected = quant_lstm::quantization::calibrateMinMax(
                field_ranges.at(id).front().minimum, field_ranges.at(id).front().maximum,
                field_config.at(id).type);
            require(field_params.at(id).values.front().scale == expected.param.scale &&
                        field_params.at(id).values.front().zero_point ==
                            expected.param.zero_point,
                    "non-parameter fields must reach finalize");
        }

        auto invalid_ranges = ranges;
        invalid_ranges.at(quant_lstm::QuantOperator::WeightHiddenHidden).pop_back();
        requireThrows(
            [&] {
                static_cast<void>(
                    quant_lstm::finalizeQuantParams(config, invalid_ranges, hidden_size, true));
            },
            "invalid group count must fail");
        auto invalid_per_tensor = params;
        invalid_per_tensor
            .operators[static_cast<std::size_t>(
                quant_lstm::QuantOperator::WeightInputHidden)]
            .values[1]
            .scale *= 2.0F;
        requireThrows([&] { invalid_per_tensor.validate(config); },
                      "per-tensor 4H repetition must be validated");
        auto invalid_diagnostics = params;
        invalid_diagnostics
            .operators[static_cast<std::size_t>(
                quant_lstm::QuantOperator::WeightHiddenHidden)]
            .group_diagnostics.pop_back();
        requireThrows([&] { invalid_diagnostics.validate(config); },
                      "calibration group count must be validated");

        auto pot2_config = config;
        pot2_config.scale_mode = quant_lstm::quantization::ScaleMode::Pot2;
        const auto pot2_params =
            quant_lstm::finalizeQuantParams(pot2_config, ranges, hidden_size, true);
        int exponent = 0;
        const float mantissa = std::frexp(
            pot2_params.at(quant_lstm::QuantOperator::Input).values.front().scale, &exponent);
        require(mantissa == 0.5F, "POT2 finalized scale");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
