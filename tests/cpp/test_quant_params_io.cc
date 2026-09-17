#include "lstm/quant_config_loader.h"
#include "lstm/quant_params_io.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <cstring>
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

quant_lstm::LstmOperatorQuantConfig defaultConfig() {
    return quant_lstm::resolveQuantConfigFiles(
        std::filesystem::path(QUANT_LSTM_SOURCE_DIR) /
        "config/defaults/lstm_quant_default_v1.json");
}

quant_lstm::LstmQuantParamsBundle makeBundle(bool bias_enabled) {
    using namespace quant_lstm;
    auto config = defaultConfig();
    config.at(QuantOperator::WeightInputHidden).granularity =
        QuantGranularity::PerTensor;
    config.at(QuantOperator::WeightHiddenHidden).granularity =
        QuantGranularity::PerGate;
    LstmQuantizationRanges ranges;
    ranges.reset(config, 2, bias_enabled);
    for (std::size_t index = 0; index < kQuantOperatorCount; ++index) {
        const auto id = static_cast<QuantOperator>(index);
        if (!bias_enabled && isBiasOperator(id)) {
            continue;
        }
        auto& groups = ranges.at(id);
        for (std::size_t group = 0; group < groups.size(); ++group) {
            const float extent = 0.25F + static_cast<float>(group) * 0.03125F;
            groups[group].observe(-extent);
            groups[group].observe(extent * 0.75F);
        }
    }
    return {1, 3, config, finalizeQuantParams(config, ranges, 2, bias_enabled)};
}

bool sameBits(float lhs, float rhs) {
    return std::memcmp(&lhs, &rhs, sizeof(float)) == 0;
}

}  // namespace

int main() {
    try {
        using namespace quant_lstm;
        const auto bundle = makeBundle(true);
        const std::string encoded = exportQuantParamsBundle(bundle);
        const auto decoded = importQuantParamsBundle(encoded, true);
        require(exportQuantParamsBundle(decoded) == encoded,
                "canonical export/import must be stable");
        for (std::size_t operator_index = 0;
             operator_index < kQuantOperatorCount; ++operator_index) {
            const auto& expected =
                bundle.quant_params.operators[operator_index].values;
            const auto& actual =
                decoded.quant_params.operators[operator_index].values;
            require(expected.size() == actual.size(), "round-trip value count");
            for (std::size_t index = 0; index < expected.size(); ++index) {
                require(sameBits(expected[index].scale, actual[index].scale) &&
                            expected[index].zero_point ==
                                actual[index].zero_point,
                        "FP32 scale and zero point round-trip");
            }
        }
        const auto execution = auditQuantParamsBundle(decoded);
        require(execution.input_size == 3 && execution.hidden_size == 2,
                "audit must derive execution parameters");

        nlohmann::json root = nlohmann::json::parse(encoded);
        require(root["operators"]["weight_ih"]["scales"].size() == 8,
                "parameter export must be full 4H");
        require(!root["operators"]["weight_ih"].contains("compact_values") &&
                    !root.contains("execution_params"),
                "external bundle must not contain execution encodings");

        auto numeric_scale = root;
        numeric_scale["operators"]["input"]["scales"][0] = 0.1;
        requireThrows(
            [&] {
                static_cast<void>(
                    importQuantParamsBundle(numeric_scale.dump()));
            },
            "numeric scale must be rejected");

        auto compact = root;
        compact["operators"]["weight_ih"]["scales"] =
            nlohmann::json::array({"0.1"});
        compact["operators"]["weight_ih"]["zero_points"] =
            nlohmann::json::array({0});
        requireThrows(
            [&] { static_cast<void>(importQuantParamsBundle(compact.dump())); },
            "compact parameter arrays must be rejected");

        auto raw_ratio = root;
        raw_ratio["operators"]["input"]["raw_ratio"] = "1";
        requireThrows(
            [&] {
                static_cast<void>(importQuantParamsBundle(raw_ratio.dump()));
            },
            "raw execution ratio must be rejected");

        auto broken_repetition = root;
        broken_repetition["operators"]["weight_ih"]["scales"][1] = "0.5";
        requireThrows(
            [&] {
                static_cast<void>(
                    importQuantParamsBundle(broken_repetition.dump()));
            },
            "per-tensor 4H repetition must be audited");

        const auto no_bias_bundle = makeBundle(false);
        const std::string no_bias_json =
            exportQuantParamsBundle(no_bias_bundle);
        const auto no_bias_root = nlohmann::json::parse(no_bias_json);
        require(!no_bias_root["operators"].contains("bias_ih") &&
                    !no_bias_root["operators"].contains("bias_hh"),
                "bias=False fields must be absent");
        auto forbidden_bias = no_bias_root;
        forbidden_bias["operators"]["bias_ih"] =
            root["operators"]["bias_ih"];
        requireThrows(
            [&] {
                static_cast<void>(
                    importQuantParamsBundle(forbidden_bias.dump()));
            },
            "bias=False must reject even populated bias fields");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
