#include <cstdlib>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

#include "golden_fixtures.h"
#include "lstm/quant_config_loader.h"
#include "quantization/fixed_point_ops.h"
#include "quantization/float_carrier_ops.h"
#include "quantization/real_activation.h"
#include "quantization/rounding.h"
#include "quantization/scale_encoding.h"

namespace {

using Json = nlohmann::json;
namespace q = quant_lstm::quantization;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

const Json& tensorData(const Json& tensor_map, const char* name) {
    return tensor_map.at(name).at("data");
}

std::size_t shapeElementCount(const Json& shape) {
    std::size_t count = 1;
    for (const Json& dimension : shape) {
        const std::size_t value = dimension.get<std::size_t>();
        if (value == 0 || count > std::numeric_limits<std::size_t>::max() / value) {
            throw std::overflow_error("Golden tensor shape 元素数溢出");
        }
        count *= value;
    }
    return count;
}

void validateTensorMap(const Json& tensor_map) {
    for (const auto& [name, tensor] : tensor_map.items()) {
        require(shapeElementCount(tensor.at("shape")) == tensor.at("data").size(),
                "Golden tensor shape/data mismatch: " + name);
        const std::string dtype = tensor.at("dtype").get<std::string>();
        if (dtype == "float32") {
            for (const Json& value : tensor.at("data")) {
                const std::string text = value.get<std::string>();
                static_cast<void>(quant_lstm::parseCanonicalFloat32Value(text));
            }
        }
    }
}

q::QuantizationType parseType(const Json& inputs, std::size_t index) {
    return {tensorData(inputs, "bitwidth").at(index).get<std::uint8_t>(),
            tensorData(inputs, "is_unsigned").at(index).get<bool>(),
            tensorData(inputs, "is_symmetric").at(index).get<bool>()};
}

void checkRound(const Json& document) {
    const Json& values = tensorData(document.at("inputs"), "values");
    const Json& expected = tensorData(document.at("expected").at("checkpoints"), "rounded");
    for (std::size_t index = 0; index < values.size(); ++index) {
        require(q::roundToNearestEven(values.at(index).get<double>()) ==
                    expected.at(index).get<double>(),
                "round Golden mismatch");
    }
}

void checkRange(const Json& document) {
    const Json& inputs = document.at("inputs");
    const Json& minimum = tensorData(document.at("expected").at("checkpoints"), "minimum");
    const Json& maximum = tensorData(document.at("expected").at("checkpoints"), "maximum");
    for (std::size_t index = 0; index < minimum.size(); ++index) {
        const auto actual = parseType(inputs, index).range();
        require(actual.minimum == minimum.at(index).get<std::int32_t>() &&
                    actual.maximum == maximum.at(index).get<std::int32_t>(),
                "qrange Golden mismatch");
    }
}

void checkMShift(const Json& document) {
    const Json& ratios = tensorData(document.at("inputs"), "ratio");
    const Json& expected = document.at("expected").at("checkpoints");
    const Json& multipliers = tensorData(expected, "multiplier");
    const Json& shifts = tensorData(expected, "shift");
    const Json& values = tensorData(document.at("inputs"), "value");
    for (std::size_t index = 0; index < ratios.size(); ++index) {
        const auto actual = q::encodeMShift(ratios.at(index).get<double>());
        require(actual.multiplier == multipliers.at(index).get<std::uint16_t>() &&
                    actual.shift == shifts.at(index).get<std::int8_t>() &&
                    q::applyRescale(values.at(index).get<std::int64_t>(), actual) ==
                        tensorData(expected, "int_apply").at(index).get<std::int64_t>() &&
                    q::applyRescale(values.at(index).get<float>(), actual) ==
                        tensorData(expected, "fp_apply").at(index).get<float>(),
                "M+shift Golden mismatch");
    }
}

void checkPot2(const Json& document) {
    const Json& inputs = document.at("inputs");
    const Json& checkpoints = document.at("expected").at("checkpoints");
    const Json& diagnostics = document.at("expected").at("diagnostics");
    const Json& range_minimum = tensorData(inputs, "range_minimum");
    for (std::size_t index = 0; index < range_minimum.size(); ++index) {
        const auto type = parseType(inputs, index);
        const auto calibrated = q::calibrateMinMax(
            quant_lstm::parseCanonicalFloat32Value(range_minimum.at(index).get<std::string>()),
            quant_lstm::parseCanonicalFloat32Value(
                tensorData(inputs, "range_maximum").at(index).get<std::string>()),
            type);
        const auto actual = q::convertScaleToPot2CoverRange(calibrated, type);
        require(
            quant_lstm::formatCanonicalFloat32(actual.param.scale) ==
                    tensorData(checkpoints, "scale").at(index).get<std::string>() &&
                actual.param.zero_point ==
                    tensorData(checkpoints, "zero_point").at(index).get<std::int32_t>() &&
                actual.exponent ==
                    tensorData(diagnostics, "exponent").at(index).get<std::int8_t>() &&
                actual.range_is_near_power_of_two ==
                    tensorData(diagnostics, "range_is_near_power_of_two").at(index).get<bool>() &&
                calibrated.diagnostics.fallback_used ==
                    tensorData(diagnostics, "fallback_used").at(index).get<bool>(),
            "POT2 Golden mismatch");
    }
}

void checkQuantDequant(const Json& document) {
    const Json& inputs = document.at("inputs");
    const Json& values = tensorData(inputs, "value");
    const Json& checkpoints = document.at("expected").at("checkpoints");
    const std::string operation = document.at("attributes").at("operation").get<std::string>();
    for (std::size_t index = 0; index < values.size(); ++index) {
        const auto type = parseType(inputs, index);
        const q::QuantParam param{quant_lstm::parseCanonicalFloat32(
                                      tensorData(inputs, "scale").at(index).get<std::string>()),
                                  tensorData(inputs, "zero_point").at(index).get<std::int32_t>()};
        if (operation == "quantize") {
            require(q::quantize(
                        quant_lstm::parseCanonicalFloat32Value(values.at(index).get<std::string>()),
                        param,
                        type) == tensorData(checkpoints, "quantized").at(index).get<std::int32_t>(),
                    "quantize Golden mismatch");
        } else if (operation == "dequantize") {
            require(q::dequantize(values.at(index).get<std::int32_t>(), param, type) ==
                        quant_lstm::parseCanonicalFloat32Value(
                            tensorData(checkpoints, "real").at(index).get<std::string>()),
                    "dequantize Golden mismatch");
        } else {
            throw std::invalid_argument("unknown quant_dequant operation");
        }
    }
}

void checkActivation(const Json& document) {
    const Json& inputs = document.at("inputs");
    const Json& values = tensorData(inputs, "quantized");
    const Json& expected = tensorData(document.at("expected").at("checkpoints"), "quantized");
    const bool fp32_carrier = document.at("execution_model").get<std::string>() == "cpu_fp32";
    for (std::size_t index = 0; index < values.size(); ++index) {
        const std::string activation = tensorData(inputs, "kind").at(index).get<std::string>();
        if (activation != "sigmoid" && activation != "tanh") {
            throw std::invalid_argument("unknown real activation kind");
        }
        const bool sigmoid = activation == "sigmoid";
        const q::QuantParam input_param{
            quant_lstm::parseCanonicalFloat32(
                tensorData(inputs, "input_scale").at(index).get<std::string>()),
            tensorData(inputs, "input_zero_point").at(index).get<std::int32_t>()};
        const q::QuantParam output_param{
            quant_lstm::parseCanonicalFloat32(
                tensorData(inputs, "output_scale").at(index).get<std::string>()),
            tensorData(inputs, "output_zero_point").at(index).get<std::int32_t>()};
        const auto kind = sigmoid ? q::RealActivationKind::Sigmoid : q::RealActivationKind::Tanh;
        if (fp32_carrier) {
            require(q::realActivation(values.at(index).get<float>(), input_param, {8, false, true},
                                      output_param, {8, sigmoid, true},
                                      kind) == expected.at(index).get<float>(),
                    "FP32 real activation Golden mismatch");
        } else {
            require(q::realActivation(values.at(index).get<std::int32_t>(), input_param,
                                      {8, false, true}, output_param, {8, sigmoid, true},
                                      kind) == expected.at(index).get<std::int32_t>(),
                    "int32 real activation Golden mismatch");
        }
    }
}

}  // namespace

int main() {
    try {
        for (const auto& fixture : quant_lstm::test::kGoldenDocuments) {
            if (fixture.kind != "primitive") {
                continue;
            }
            const Json document = Json::parse(fixture.json);
            require(document.at("case_id").get<std::string>() == fixture.case_id,
                    "generated fixture case_id mismatch");
            require(document.at("kind").get<std::string>() == fixture.kind,
                    "primitive Golden kind mismatch");
            require(document.at("execution_model").get<std::string>() == fixture.execution_model,
                    "generated fixture execution_model mismatch");
            validateTensorMap(document.at("inputs"));
            validateTensorMap(document.at("expected").at("checkpoints"));
            validateTensorMap(document.at("expected").at("diagnostics"));
            const std::string primitive = document.at("primitive").get<std::string>();
            if (primitive == "round_to_nearest_even") {
                checkRound(document);
            } else if (primitive == "quantized_range") {
                checkRange(document);
            } else if (primitive == "m_shift") {
                checkMShift(document);
            } else if (primitive == "pot2_cover_range") {
                checkPot2(document);
            } else if (primitive == "quant_dequant") {
                checkQuantDequant(document);
            } else if (primitive == "real_activation") {
                checkActivation(document);
            } else {
                throw std::invalid_argument("unknown primitive Golden");
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
