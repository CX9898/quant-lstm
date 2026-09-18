#include "lstm/quant_params_io.h"

#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace quant_lstm {
namespace {

using Json = nlohmann::json;
using OrderedJson = nlohmann::ordered_json;

void requireObject(const Json& value, std::string_view context) {
    if (!value.is_object()) {
        throw std::invalid_argument(std::string(context) + " 必须是 object");
    }
}

void rejectUnknownKeys(const Json& object, std::initializer_list<std::string_view> allowed,
                       std::string_view context) {
    requireObject(object, context);
    for (const auto& [key, unused] : object.items()) {
        static_cast<void>(unused);
        bool accepted = false;
        for (const std::string_view candidate : allowed) {
            accepted = accepted || key == candidate;
        }
        if (!accepted) {
            throw std::invalid_argument(std::string(context) + " unknown field: " + key);
        }
    }
}

template <typename T>
T requiredInteger(const Json& object, const char* field) {
    if (!object.contains(field) || !object[field].is_number_integer()) {
        throw std::invalid_argument(std::string(field) + " 必须是 JSON integer");
    }
    const auto value = object[field].get<std::int64_t>();
    if (value < static_cast<std::int64_t>(std::numeric_limits<T>::min()) ||
        value > static_cast<std::int64_t>(std::numeric_limits<T>::max())) {
        throw std::invalid_argument(std::string(field) + " 超出范围");
    }
    return static_cast<T>(value);
}

bool requiredBoolean(const Json& object, const char* field) {
    if (!object.contains(field) || !object[field].is_boolean()) {
        throw std::invalid_argument(std::string(field) + " 必须是 boolean");
    }
    return object[field].get<bool>();
}

std::string requiredString(const Json& object, const char* field) {
    if (!object.contains(field) || !object[field].is_string()) {
        throw std::invalid_argument(std::string(field) + " 必须是 string");
    }
    return object[field].get<std::string>();
}

std::string shortestFloat(float value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument("standard scale 必须为有限 FP32");
    }
    char buffer[64];
    const auto result =
        std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::general);
    if (result.ec != std::errc{}) {
        throw std::runtime_error("FP32 canonical decimal 序列化失败");
    }
    return {buffer, result.ptr};
}

float parseCanonicalFloat(const Json& value) {
    if (!value.is_string()) {
        throw std::invalid_argument("scale 必须是 canonical FP32 string");
    }
    const std::string text = value.get<std::string>();
    float parsed = 0.0F;
    const auto result =
        std::from_chars(text.data(), text.data() + text.size(), parsed, std::chars_format::general);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
        !std::isfinite(parsed) || shortestFloat(parsed) != text) {
        throw std::invalid_argument("scale 不是最短可往返 FP32 十进制字符串");
    }
    return parsed;
}

std::size_t groupCountForDiagnostics(QuantOperator id, QuantGranularity granularity,
                                     std::int64_t hidden_size) {
    return quantizationGroupCount(id, granularity, hidden_size);
}

OperatorQuantConfig parseOperatorConfig(QuantOperator id, const Json& value) {
    rejectUnknownKeys(
        value, {"bitwidth", "is_unsigned", "is_symmetric", "granularity", "scales", "zero_points"},
        quantOperatorName(id));
    OperatorQuantConfig result;
    const auto bitwidth = requiredInteger<std::int32_t>(value, "bitwidth");
    if (bitwidth != 8 && bitwidth != 16) {
        throw std::invalid_argument("bitwidth 只允许 8 或 16");
    }
    result.type.bitwidth = static_cast<std::uint8_t>(bitwidth);
    result.type.is_unsigned = requiredBoolean(value, "is_unsigned");
    result.type.is_symmetric = requiredBoolean(value, "is_symmetric");
    result.granularity = parseGranularity(requiredString(value, "granularity"));
    return result;
}

FinalizedOperatorQuantParams parseOperatorParams(QuantOperator id, const Json& value,
                                                 const OperatorQuantConfig& config,
                                                 std::int64_t hidden_size) {
    if (!value.contains("scales") || !value["scales"].is_array() ||
        !value.contains("zero_points") || !value["zero_points"].is_array()) {
        throw std::invalid_argument("scales/zero_points 必须是 array");
    }
    const Json& scales = value["scales"];
    const Json& zero_points = value["zero_points"];
    const std::size_t expected =
        isParameterOperator(id) ? static_cast<std::size_t>(4 * hidden_size) : 1;
    if (scales.size() != expected || zero_points.size() != expected) {
        throw std::invalid_argument("量化参数必须使用完整 4H 或单值表示");
    }

    FinalizedOperatorQuantParams result;
    result.source_granularity = config.granularity;
    result.values.reserve(expected);
    for (std::size_t index = 0; index < expected; ++index) {
        if (!zero_points[index].is_number_integer()) {
            throw std::invalid_argument("zero_point 必须是 JSON integer");
        }
        const auto zero_point = zero_points[index].get<std::int64_t>();
        if (zero_point < std::numeric_limits<std::int32_t>::min() ||
            zero_point > std::numeric_limits<std::int32_t>::max()) {
            throw std::invalid_argument("zero_point 超出 int32");
        }
        result.values.push_back(
            {parseCanonicalFloat(scales[index]), static_cast<std::int32_t>(zero_point)});
    }
    result.group_diagnostics.resize(groupCountForDiagnostics(id, config.granularity, hidden_size));
    return result;
}

OrderedJson operatorJson(const OperatorQuantConfig& config,
                         const FinalizedOperatorQuantParams& params) {
    OrderedJson scales = OrderedJson::array();
    OrderedJson zero_points = OrderedJson::array();
    for (const auto& value : params.values) {
        scales.push_back(shortestFloat(value.scale));
        zero_points.push_back(value.zero_point);
    }
    return OrderedJson{
        {"bitwidth", config.type.bitwidth},
        {"is_unsigned", config.type.is_unsigned},
        {"is_symmetric", config.type.is_symmetric},
        {"granularity", granularityName(config.granularity)},
        {"scales", std::move(scales)},
        {"zero_points", std::move(zero_points)},
    };
}

}  // namespace

void LstmQuantParamsBundle::validate() const {
    if (schema_version != 1) {
        throw std::invalid_argument("只支持 quant params bundle schema_version=1");
    }
    if (input_size <= 0) {
        throw std::invalid_argument("bundle input_size 必须大于 0");
    }
    config.validate();
    quant_params.validate(config);
}

std::string exportQuantParamsBundle(const LstmQuantParamsBundle& bundle) {
    bundle.validate();
    OrderedJson operators = OrderedJson::object();
    for (std::size_t index = 0; index < kQuantOperatorCount; ++index) {
        const auto id = static_cast<QuantOperator>(index);
        if (!bundle.quant_params.bias_enabled && isBiasOperator(id)) {
            continue;
        }
        operators[std::string(quantOperatorName(id))] =
            operatorJson(bundle.config.operators[index], bundle.quant_params.operators[index]);
    }
    const OrderedJson root{
        {"schema_version", bundle.schema_version},
        {"input_size", bundle.input_size},
        {"hidden_size", bundle.quant_params.hidden_size},
        {"bias_enabled", bundle.quant_params.bias_enabled},
        {"scale_mode", scaleModeName(bundle.config.scale_mode)},
        {"operators", std::move(operators)},
    };
    return root.dump();
}

LstmQuantParamsBundle importQuantParamsBundle(std::string_view json_text, bool require_canonical) {
    Json root;
    try {
        root = Json::parse(json_text.begin(), json_text.end());
    } catch (const Json::exception& error) {
        throw std::invalid_argument(std::string("quant params JSON 解析失败: ") + error.what());
    }
    rejectUnknownKeys(
        root,
        {"schema_version", "input_size", "hidden_size", "bias_enabled", "scale_mode", "operators"},
        "quant params root");

    LstmQuantParamsBundle result;
    result.schema_version = requiredInteger<std::int32_t>(root, "schema_version");
    result.input_size = requiredInteger<std::int64_t>(root, "input_size");
    result.quant_params.hidden_size = requiredInteger<std::int64_t>(root, "hidden_size");
    if (result.quant_params.hidden_size <= 0 ||
        result.quant_params.hidden_size > std::numeric_limits<std::int64_t>::max() / 4) {
        throw std::invalid_argument("hidden_size 非法或 4H 溢出");
    }
    result.quant_params.bias_enabled = requiredBoolean(root, "bias_enabled");
    result.config.schema_version = result.schema_version;
    result.config.scale_mode = parseScaleMode(requiredString(root, "scale_mode"));
    if (!root.contains("operators")) {
        throw std::invalid_argument("operators 缺失");
    }
    requireObject(root["operators"], "operators");

    std::unordered_set<std::string> expected_names;
    for (std::size_t index = 0; index < kQuantOperatorCount; ++index) {
        const auto id = static_cast<QuantOperator>(index);
        if (!result.quant_params.bias_enabled && isBiasOperator(id)) {
            continue;
        }
        const std::string name(quantOperatorName(id));
        expected_names.insert(name);
        if (!root["operators"].contains(name)) {
            throw std::invalid_argument("operators 缺少字段: " + name);
        }
        const Json& value = root["operators"][name];
        result.config.operators[index] = parseOperatorConfig(id, value);
        result.quant_params.operators[index] = parseOperatorParams(
            id, value, result.config.operators[index], result.quant_params.hidden_size);
    }
    for (const auto& [name, unused] : root["operators"].items()) {
        static_cast<void>(unused);
        if (expected_names.count(name) == 0U) {
            throw std::invalid_argument("operators 包含非法或禁用字段: " + name);
        }
    }
    result.validate();
    if (require_canonical && exportQuantParamsBundle(result) != json_text) {
        throw std::invalid_argument("quant params JSON 不是 canonical 表示");
    }
    return result;
}

LstmExecutionParams auditQuantParamsBundle(const LstmQuantParamsBundle& bundle,
                                           bool require_exact_accumulation) {
    bundle.validate();
    return deriveLstmExecutionParams(bundle.config, bundle.quant_params, bundle.input_size,
                                     require_exact_accumulation);
}

}  // namespace quant_lstm
