#include "lstm/quant_config_loader.h"

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

}  // namespace

int main() {
    try {
        const std::filesystem::path defaults_path =
            std::filesystem::path(QUANT_LSTM_SOURCE_DIR) /
            "config/defaults/lstm_quant_default_v1.json";
        const auto defaults = quant_lstm::resolveQuantConfigFiles(defaults_path);
        require(defaults.schema_version == 1, "default schema version");
        require(defaults.scale_mode == quant_lstm::quantization::ScaleMode::Affine,
                "default scale mode");
        require(defaults.at(quant_lstm::QuantOperator::WeightInputHidden).granularity ==
                    quant_lstm::QuantGranularity::PerChannel,
                "default parameter granularity");
        require(defaults.at(quant_lstm::QuantOperator::InputGateOutput).type.is_unsigned,
                "sigmoid gate output default unsigned");

        const std::string override_json = R"({
          "schema_version": 1,
          "scale_mode": "pot2",
          "operators": {
            "weight_ih": {"bitwidth": 16, "granularity": "per_gate"},
            "input_gate_output": {"bitwidth": 16, "is_unsigned": true}
          }
        })";
        const auto resolved =
            quant_lstm::resolveQuantConfig(quant_lstm::toCanonicalJson(defaults), override_json);
        require(resolved.scale_mode == quant_lstm::quantization::ScaleMode::Pot2,
                "override scale mode");
        require(resolved.at(quant_lstm::QuantOperator::WeightInputHidden).type.bitwidth == 16 &&
                    resolved.at(quant_lstm::QuantOperator::WeightInputHidden).granularity ==
                        quant_lstm::QuantGranularity::PerGate,
                "parameter override");
        const std::string canonical = quant_lstm::toCanonicalJson(resolved);
        const auto reparsed = quant_lstm::parseResolvedQuantConfig(canonical);
        require(quant_lstm::toCanonicalJson(reparsed) == canonical,
                "resolved canonical round-trip");

        const std::string default_canonical = quant_lstm::toCanonicalJson(defaults);
        const auto parsed_default = quant_lstm::parseResolvedQuantConfig(default_canonical);
        require(quant_lstm::toCanonicalJson(parsed_default) == default_canonical,
                "default canonical round-trip");

        const std::string float_text = quant_lstm::formatCanonicalFloat32(0.1F);
        require(float_text == "0.1" &&
                    quant_lstm::parseCanonicalFloat32(float_text) == 0.1F,
                "canonical float32");
        requireThrows([] { static_cast<void>(quant_lstm::parseCanonicalFloat32("0.1000000")); },
                      "non-canonical float32");

        const std::string defaults_json = quant_lstm::toCanonicalJson(defaults);
        const auto reject_override = [&](const std::string& invalid) {
            requireThrows(
                [&] {
                    static_cast<void>(quant_lstm::resolveQuantConfig(defaults_json, invalid));
                },
                "invalid override was accepted");
        };
        reject_override(R"({"schema_version":1,"schema_version":1})");
        reject_override(R"({"schema_version":1,"unknown":1})");
        reject_override(R"({"schema_version":1,"scale_mode":null})");
        reject_override(
            R"({"schema_version":1,"operators":{"input":{"bitwidth":7}}})");
        reject_override(
            R"({"schema_version":1,"operators":{"input":{"bitwidth":8.0}}})");
        reject_override(
            R"({"schema_version":1,"operators":{"input":{"bitwidth":"8"}}})");
        reject_override(
            R"({"schema_version":1,"operators":{"weight_ih":{"is_symmetric":false}}})");
        reject_override(
            R"({"schema_version":1,"operators":{"weight_ih":{"is_unsigned":true}}})");
        reject_override(
            R"({"schema_version":1,"operators":{"input":{"granularity":"per_gate"}}})");
        reject_override(
            R"({"schema_version":1,"operators":{"mul_output_cell":{"bitwidth":8}}})");
        reject_override(R"({"schema_version":1,"pot_scale_method":"floor"})");
        reject_override(R"({"schema_version":1,"pot_scale_tolerance":0.02})");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
