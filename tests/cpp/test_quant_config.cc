#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "lstm/quant_config_loader.h"

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
        const std::filesystem::path defaults_path = std::filesystem::path(QUANT_LSTM_SOURCE_DIR) /
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
        for (const auto& value : defaults.operators) {
            require(value.type.bitwidth == 8, "pure INT8 default");
        }
        const auto empty_override = quant_lstm::resolveQuantConfig(
            quant_lstm::toCanonicalJson(defaults), R"({"schema_version":1})");
        require(
            quant_lstm::toCanonicalJson(empty_override) == quant_lstm::toCanonicalJson(defaults),
            "empty override");

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
        require(resolved.at(quant_lstm::QuantOperator::WeightInputHidden).type.is_symmetric &&
                    !resolved.at(quant_lstm::QuantOperator::WeightInputHidden).type.is_unsigned,
                "field-level shallow merge");
        require(resolved.at(quant_lstm::QuantOperator::Input).type.bitwidth == 8,
                "mixed INT8/INT16 config");
        const std::string canonical = quant_lstm::toCanonicalJson(resolved);
        const auto reparsed = quant_lstm::parseResolvedQuantConfig(canonical);
        require(quant_lstm::toCanonicalJson(reparsed) == canonical,
                "resolved canonical round-trip");
        requireThrows(
            [&] {
                static_cast<void>(quant_lstm::parseResolvedQuantConfig(
                    canonical.substr(0, canonical.size() - 1)));
            },
            "non-canonical resolved JSON");

        const std::string default_canonical = quant_lstm::toCanonicalJson(defaults);
        for (std::size_t index = 0; index < quant_lstm::kQuantOperatorCount; ++index) {
            const auto id = static_cast<quant_lstm::QuantOperator>(index);
            const std::string key = '"' + std::string(quant_lstm::quantOperatorName(id)) + "\": {";
            const auto start = default_canonical.find(key);
            require(start != std::string::npos, "canonical operator missing");
            const auto end = default_canonical.find('}', start);
            const bool exposes_granularity =
                default_canonical.substr(start, end - start).find("granularity") != std::string::npos;
            require(exposes_granularity == quant_lstm::isParameterOperator(id),
                    "only parameter config may expose granularity");
            const std::string operator_text = default_canonical.substr(start, end - start);
            if (quant_lstm::isParameterOperator(id)) {
                require(operator_text.find("is_unsigned") == std::string::npos &&
                            operator_text.find("is_symmetric") == std::string::npos,
                        "fixed parameter flags must not be exposed");
                require(!defaults.at(id).type.is_unsigned && defaults.at(id).type.is_symmetric,
                        "parameters must remain signed symmetric internally");
                for (const std::string field : {"\"is_unsigned\":false,", "\"is_symmetric\":true,"}) {
                    std::string invalid = default_canonical;
                    invalid.insert(start + key.size(), field);
                    requireThrows(
                        [&] { static_cast<void>(quant_lstm::parseResolvedQuantConfig(invalid, false)); },
                        "fixed parameter flags must not be accepted");
                }
            } else {
                require(defaults.at(id).granularity == quant_lstm::QuantGranularity::PerTensor,
                        "non-parameter granularity must remain per_tensor internally");
            }
        }
        std::string non_parameter_field = default_canonical;
        const std::string input_key = "\"input\": {";
        non_parameter_field.insert(non_parameter_field.find(input_key) + input_key.size(),
                                   "\n      \"granularity\": \"per_tensor\",");
        requireThrows(
            [&] {
                static_cast<void>(quant_lstm::parseResolvedQuantConfig(non_parameter_field, false));
            },
            "non-parameter granularity field must not be accepted in resolved config");
        std::string annotated = default_canonical;
        annotated.insert(annotated.find(input_key) + input_key.size(),
                         "\n      \"comment\": \"输入序列 x_t\",");
        require(quant_lstm::toCanonicalJson(quant_lstm::parseResolvedQuantConfig(annotated, false)) ==
                    default_canonical,
                "comments must not affect execution config");
        require(quant_lstm::toCanonicalJson(quant_lstm::resolveQuantConfig(
                    annotated, R"({"schema_version":1})")) == default_canonical,
                "annotated defaults must resolve normally");
        requireThrows(
            [&] { static_cast<void>(quant_lstm::parseResolvedQuantConfig(annotated)); },
            "canonical execution config must not contain comments");
        const auto commented_override = quant_lstm::resolveQuantConfig(
            default_canonical,
            R"({"schema_version":1,"operators":{"input":{"comment":"输入序列","bitwidth":16}}})");
        const auto plain_override = quant_lstm::resolveQuantConfig(
            default_canonical, R"({"schema_version":1,"operators":{"input":{"bitwidth":16}}})");
        require(quant_lstm::toCanonicalJson(commented_override) == quant_lstm::toCanonicalJson(plain_override),
                "override comments must not affect execution config");
        const auto parsed_default = quant_lstm::parseResolvedQuantConfig(default_canonical);
        require(quant_lstm::toCanonicalJson(parsed_default) == default_canonical,
                "default canonical round-trip");

        const std::string float_text = quant_lstm::formatCanonicalFloat32(0.1F);
        require(float_text == "0.1" && quant_lstm::parseCanonicalFloat32(float_text) == 0.1F,
                "canonical float32");
        require(quant_lstm::formatCanonicalFloat32Value(-0.5F) == "-0.5" &&
                    quant_lstm::parseCanonicalFloat32Value("-0.5") == -0.5F &&
                    quant_lstm::parseCanonicalFloat32Value("0") == 0.0F,
                "canonical general float32");
        requireThrows([] { static_cast<void>(quant_lstm::parseCanonicalFloat32("0.1000000")); },
                      "non-canonical float32");
        for (const std::string invalid : {"0", "-1", "nan", "inf", "1e-1"}) {
            requireThrows([&] { static_cast<void>(quant_lstm::parseCanonicalFloat32(invalid)); },
                          "invalid canonical float32");
        }

        const std::string defaults_json = quant_lstm::toCanonicalJson(defaults);
        std::string pure_int16_override =
            R"({"schema_version":1,"scale_mode":"affine","operators":{)";
        for (std::size_t index = 0; index < quant_lstm::kQuantOperatorCount; ++index) {
            const auto id = static_cast<quant_lstm::QuantOperator>(index);
            if (index != 0) {
                pure_int16_override += ',';
            }
            pure_int16_override +=
                '"' + std::string(quant_lstm::quantOperatorName(id)) + "\":{\"bitwidth\":16";
            if (quant_lstm::isParameterOperator(id)) {
                pure_int16_override +=
                    R"(,"granularity":"per_channel")";
            } else {
                pure_int16_override += R"(,"is_unsigned":false,"is_symmetric":true)";
            }
            pure_int16_override += '}';
        }
        pure_int16_override += "}}";
        const auto pure_int16 = quant_lstm::resolveQuantConfig(defaults_json, pure_int16_override);
        for (const auto& value : pure_int16.operators) {
            require(value.type.bitwidth == 16, "pure INT16 full override");
        }
        for (std::size_t index = 0; index < quant_lstm::kQuantOperatorCount; ++index) {
            const auto id = static_cast<quant_lstm::QuantOperator>(index);
            if (quant_lstm::isParameterOperator(id)) {
                continue;
            }
            const auto& original = defaults.at(id).type;
            const std::string per_operator_override =
                "{\"schema_version\":1,\"operators\":{\"" +
                std::string(quant_lstm::quantOperatorName(id)) +
                "\":{\"bitwidth\":16,\"is_unsigned\":" + (original.is_unsigned ? "false" : "true") +
                ",\"is_symmetric\":" + (original.is_symmetric ? "false" : "true") + "}}}";
            const auto per_operator =
                quant_lstm::resolveQuantConfig(defaults_json, per_operator_override);
            const auto& changed = per_operator.at(id).type;
            require(changed.bitwidth == 16 && changed.is_unsigned != original.is_unsigned &&
                        changed.is_symmetric != original.is_symmetric,
                    "each non-parameter operator fields must resolve independently");
        }

        const auto reject_override = [&](const std::string& invalid) {
            requireThrows(
                [&] { static_cast<void>(quant_lstm::resolveQuantConfig(defaults_json, invalid)); },
                "invalid override was accepted");
        };
        reject_override(R"({"schema_version":1,"operators":{"input":{"comment":12,"bitwidth":8}}})");
        reject_override(R"({"schema_version":1,"operators":{"input":{"comment":null,"bitwidth":8}}})");
        reject_override(R"({"schema_version":1,"operators":{"input":{"comment":{},"bitwidth":8}}})");
        reject_override(R"({"schema_version":1,"operators":{"input":{"comment":"说明"}}})");
        reject_override(R"({"schema_version":1,"operators":{"input":{"comment":"a","comment":"b","bitwidth":8}}})");
        reject_override(R"({"schema_version":1,"operators":{"weight_ih":{"is_unsigned":false}}})");
        reject_override(R"({"schema_version":1,"operators":{"weight_ih":{"is_symmetric":true}}})");
        reject_override("");
        reject_override(R"({"schema_version":1,"schema_version":1})");
        reject_override(
            R"({"schema_version":1,"operators":{"input":{"bitwidth":8,"bitwidth":16}}})");
        reject_override(R"({"schema_version":1,"unknown":1})");
        reject_override(R"({"schema_version":1,"scale_mode":null})");
        reject_override(R"({"schema_version":1,"operators":{"input":{"bitwidth":7}}})");
        reject_override(R"({"schema_version":1,"operators":{"input":{"bitwidth":8.0}}})");
        reject_override(R"({"schema_version":1,"operators":{"input":{"bitwidth":"8"}}})");
        reject_override(R"({"schema_version":1,"operators":{"input":null}})");
        reject_override(R"({"schema_version":1,"operators":{"input":{}}})");
        reject_override(R"({"schema_version":1,"operators":{"weight_ih":{"is_symmetric":false}}})");
        reject_override(R"({"schema_version":1,"operators":{"weight_ih":{"is_unsigned":true}}})");
        reject_override(R"({"schema_version":1,"operators":{"input":{"granularity":"per_gate"}}})");
        reject_override(R"({"schema_version":1,"operators":{"mul_output_cell":{"bitwidth":8}}})");
        reject_override(R"({"schema_version":1,"pot_scale_method":"floor"})");
        reject_override(R"({"schema_version":1,"pot_scale_tolerance":0.02})");
        requireThrows(
            [] {
                static_cast<void>(
                    quant_lstm::parseResolvedQuantConfig(R"({"schema_version":1})", false));
            },
            "incomplete resolved config");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
