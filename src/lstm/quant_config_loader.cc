#include "lstm/quant_config_loader.h"

#include <nlohmann/json.hpp>

#include <array>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace quant_lstm {
namespace {

using Json = nlohmann::json;
using OrderedJson = nlohmann::ordered_json;

class StrictJsonScanner {
   public:
    explicit StrictJsonScanner(std::string_view text) : text_(text) {}

    void validate() {
        skipWhitespace();
        parseValue();
        skipWhitespace();
        if (position_ != text_.size()) {
            throw std::invalid_argument("JSON 包含尾随内容");
        }
    }

   private:
    void skipWhitespace() {
        while (position_ < text_.size() &&
               (text_[position_] == ' ' || text_[position_] == '\n' ||
                text_[position_] == '\r' || text_[position_] == '\t')) {
            ++position_;
        }
    }

    void expect(char character) {
        if (position_ >= text_.size() || text_[position_] != character) {
            throw std::invalid_argument("JSON 结构非法");
        }
        ++position_;
    }

    std::string parseString() {
        const std::size_t start = position_;
        expect('"');
        bool escaped = false;
        while (position_ < text_.size()) {
            const char character = text_[position_++];
            if (escaped) {
                escaped = false;
            } else if (character == '\\') {
                escaped = true;
            } else if (character == '"') {
                const std::string token(text_.substr(start, position_ - start));
                try {
                    return Json::parse(token).get<std::string>();
                } catch (const Json::exception& error) {
                    throw std::invalid_argument(std::string("JSON string 非法: ") + error.what());
                }
            }
        }
        throw std::invalid_argument("JSON string 未终止");
    }

    void parseObject() {
        expect('{');
        skipWhitespace();
        std::unordered_set<std::string> keys;
        if (position_ < text_.size() && text_[position_] == '}') {
            ++position_;
            return;
        }
        while (true) {
            skipWhitespace();
            const std::string key = parseString();
            if (!keys.insert(key).second) {
                throw std::invalid_argument("JSON 包含重复 key: " + key);
            }
            skipWhitespace();
            expect(':');
            skipWhitespace();
            parseValue();
            skipWhitespace();
            if (position_ < text_.size() && text_[position_] == '}') {
                ++position_;
                return;
            }
            expect(',');
        }
    }

    void parseArray() {
        expect('[');
        skipWhitespace();
        if (position_ < text_.size() && text_[position_] == ']') {
            ++position_;
            return;
        }
        while (true) {
            parseValue();
            skipWhitespace();
            if (position_ < text_.size() && text_[position_] == ']') {
                ++position_;
                return;
            }
            expect(',');
            skipWhitespace();
        }
    }

    void parsePrimitive() {
        const std::size_t start = position_;
        while (position_ < text_.size()) {
            const char character = text_[position_];
            if (character == ',' || character == ']' || character == '}' || character == ' ' ||
                character == '\n' || character == '\r' || character == '\t') {
                break;
            }
            ++position_;
        }
        if (position_ == start) {
            throw std::invalid_argument("JSON primitive 非法");
        }
    }

    void parseValue() {
        skipWhitespace();
        if (position_ >= text_.size()) {
            throw std::invalid_argument("JSON value 缺失");
        }
        switch (text_[position_]) {
            case '{':
                parseObject();
                return;
            case '[':
                parseArray();
                return;
            case '"':
                static_cast<void>(parseString());
                return;
            default:
                parsePrimitive();
                return;
        }
    }

    std::string_view text_;
    std::size_t position_ = 0;
};

Json parseStrictJson(std::string_view text) {
    StrictJsonScanner(text).validate();
    try {
        return Json::parse(text.begin(), text.end());
    } catch (const Json::exception& error) {
        throw std::invalid_argument(std::string("JSON 解析失败: ") + error.what());
    }
}

void requireObject(const Json& value, std::string_view context) {
    if (!value.is_object()) {
        throw std::invalid_argument(std::string(context) + " 必须是 object");
    }
}

void rejectUnknownKeys(const Json& object,
                       std::initializer_list<std::string_view> allowed,
                       std::string_view context) {
    requireObject(object, context);
    for (const auto& [key, unused] : object.items()) {
        static_cast<void>(unused);
        bool found = false;
        for (const std::string_view candidate : allowed) {
            if (key == candidate) {
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::invalid_argument(std::string(context) + " unknown field: " + key);
        }
    }
}

std::int32_t readSchemaVersion(const Json& root) {
    if (!root.contains("schema_version") || !root["schema_version"].is_number_integer()) {
        throw std::invalid_argument("schema_version 必须是 JSON integer");
    }
    const auto version = root["schema_version"].get<std::int64_t>();
    if (version != 1) {
        throw std::invalid_argument("只支持 schema_version=1");
    }
    return 1;
}

std::uint8_t readBitwidth(const Json& value) {
    if (!value.is_number_integer()) {
        throw std::invalid_argument("bitwidth 必须是 JSON integer");
    }
    const auto bitwidth = value.get<std::int64_t>();
    if (bitwidth != 8 && bitwidth != 16) {
        throw std::invalid_argument("bitwidth 只允许 8 或 16");
    }
    return static_cast<std::uint8_t>(bitwidth);
}

bool readBoolean(const Json& value, std::string_view field) {
    if (!value.is_boolean()) {
        throw std::invalid_argument(std::string(field) + " 必须是 boolean");
    }
    return value.get<bool>();
}

std::string readString(const Json& value, std::string_view field) {
    if (!value.is_string()) {
        throw std::invalid_argument(std::string(field) + " 必须是 string");
    }
    return value.get<std::string>();
}

OperatorQuantConfig parseResolvedOperator(QuantOperator id, const Json& value) {
    rejectUnknownKeys(value, {"bitwidth", "is_unsigned", "is_symmetric", "granularity"},
                      quantOperatorName(id));
    for (const char* field : {"bitwidth", "is_unsigned", "is_symmetric", "granularity"}) {
        if (!value.contains(field) || value[field].is_null()) {
            throw std::invalid_argument(std::string(quantOperatorName(id)) +
                                        " resolved field 缺失: " + field);
        }
    }
    OperatorQuantConfig result;
    result.type.bitwidth = readBitwidth(value["bitwidth"]);
    result.type.is_unsigned = readBoolean(value["is_unsigned"], "is_unsigned");
    result.type.is_symmetric = readBoolean(value["is_symmetric"], "is_symmetric");
    result.granularity = parseGranularity(readString(value["granularity"], "granularity"));
    return result;
}

void applyOperatorOverride(QuantOperator id, const Json& value, OperatorQuantConfig* target) {
    const bool parameter = isParameterOperator(id);
    if (parameter) {
        rejectUnknownKeys(value, {"bitwidth", "is_unsigned", "is_symmetric", "granularity"},
                          quantOperatorName(id));
    } else {
        rejectUnknownKeys(value, {"bitwidth", "is_unsigned", "is_symmetric"},
                          quantOperatorName(id));
    }
    if (value.empty()) {
        throw std::invalid_argument(std::string(quantOperatorName(id)) +
                                    " override 不能为空 object");
    }
    for (const auto& [field, field_value] : value.items()) {
        if (field_value.is_null()) {
            throw std::invalid_argument("override 字段不能为 null: " + field);
        }
        if (field == "bitwidth") {
            target->type.bitwidth = readBitwidth(field_value);
        } else if (field == "is_unsigned") {
            const bool requested = readBoolean(field_value, field);
            if (parameter && requested) {
                throw std::invalid_argument("weight/bias is_unsigned 只能为 false");
            }
            target->type.is_unsigned = requested;
        } else if (field == "is_symmetric") {
            const bool requested = readBoolean(field_value, field);
            if (parameter && !requested) {
                throw std::invalid_argument("weight/bias is_symmetric 只能为 true");
            }
            target->type.is_symmetric = requested;
        } else if (field == "granularity") {
            target->granularity = parseGranularity(readString(field_value, field));
        }
    }
}

std::string readTextFile(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("无法打开 JSON 文件: " + path.string());
    }
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

}  // namespace

LstmOperatorQuantConfig parseResolvedQuantConfig(std::string_view json_text,
                                                 bool require_canonical) {
    const Json root = parseStrictJson(json_text);
    rejectUnknownKeys(root, {"schema_version", "scale_mode", "operators"}, "resolved root");
    LstmOperatorQuantConfig result;
    result.schema_version = readSchemaVersion(root);
    if (!root.contains("scale_mode") || root["scale_mode"].is_null()) {
        throw std::invalid_argument("resolved scale_mode 缺失");
    }
    result.scale_mode = parseScaleMode(readString(root["scale_mode"], "scale_mode"));
    if (!root.contains("operators")) {
        throw std::invalid_argument("resolved operators 缺失");
    }
    const Json& operators = root["operators"];
    requireObject(operators, "resolved operators");
    if (operators.size() != kQuantOperatorCount) {
        throw std::invalid_argument("resolved config 必须包含全部 18 个 operator");
    }
    for (std::size_t index = 0; index < kQuantOperatorCount; ++index) {
        const auto id = static_cast<QuantOperator>(index);
        const std::string name(quantOperatorName(id));
        if (!operators.contains(name)) {
            throw std::invalid_argument("resolved operator 缺失: " + name);
        }
        result.operators[index] = parseResolvedOperator(id, operators.at(name));
    }
    for (const auto& [name, unused] : operators.items()) {
        static_cast<void>(unused);
        static_cast<void>(parseQuantOperatorName(name));
    }
    result.validate();
    if (require_canonical && toCanonicalJson(result) != json_text) {
        throw std::invalid_argument("resolved JSON 不是 canonical 字节表示");
    }
    return result;
}

LstmOperatorQuantConfig resolveQuantConfig(std::string_view default_json,
                                           std::string_view override_json) {
    LstmOperatorQuantConfig result = parseResolvedQuantConfig(default_json, true);
    const Json root = parseStrictJson(override_json);
    rejectUnknownKeys(root, {"schema_version", "scale_mode", "operators"}, "override root");
    static_cast<void>(readSchemaVersion(root));
    if (root.contains("scale_mode")) {
        if (root["scale_mode"].is_null()) {
            throw std::invalid_argument("override scale_mode 不能为 null");
        }
        result.scale_mode = parseScaleMode(readString(root["scale_mode"], "scale_mode"));
    }
    if (root.contains("operators")) {
        const Json& operators = root["operators"];
        requireObject(operators, "override operators");
        for (const auto& [name, value] : operators.items()) {
            const QuantOperator id = parseQuantOperatorName(name);
            requireObject(value, name);
            applyOperatorOverride(id, value, &result.at(id));
        }
    }
    result.validate();
    return result;
}

LstmOperatorQuantConfig resolveQuantConfigFiles(
    const std::filesystem::path& default_path,
    const std::optional<std::filesystem::path>& override_path) {
    const std::string defaults = readTextFile(default_path);
    if (!override_path.has_value()) {
        return parseResolvedQuantConfig(defaults, true);
    }
    return resolveQuantConfig(defaults, readTextFile(*override_path));
}

std::string toCanonicalJson(const LstmOperatorQuantConfig& config) {
    config.validate();
    OrderedJson root = OrderedJson::object();
    root["schema_version"] = config.schema_version;
    root["scale_mode"] = scaleModeName(config.scale_mode);
    OrderedJson operators = OrderedJson::object();
    for (std::size_t index = 0; index < kQuantOperatorCount; ++index) {
        const auto id = static_cast<QuantOperator>(index);
        const OperatorQuantConfig& value = config.operators[index];
        OrderedJson operator_json = OrderedJson::object();
        operator_json["bitwidth"] = value.type.bitwidth;
        operator_json["is_unsigned"] = value.type.is_unsigned;
        operator_json["is_symmetric"] = value.type.is_symmetric;
        operator_json["granularity"] = granularityName(value.granularity);
        operators[std::string(quantOperatorName(id))] = std::move(operator_json);
    }
    root["operators"] = std::move(operators);
    return root.dump(2) + '\n';
}

std::string formatCanonicalFloat32Value(float value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument("canonical float32 必须是有限数");
    }
    std::array<char, 64> buffer{};
    const auto [end, error] =
        std::to_chars(buffer.data(), buffer.data() + buffer.size(), value,
                      std::chars_format::general);
    if (error != std::errc()) {
        throw std::runtime_error("float32 canonical 格式化失败");
    }
    return std::string(buffer.data(), end);
}

float parseCanonicalFloat32Value(std::string_view text) {
    float value = 0.0F;
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), value,
                        std::chars_format::general);
    if (error != std::errc() || end != text.data() + text.size() ||
        !std::isfinite(value)) {
        throw std::invalid_argument("canonical float32 字符串非法");
    }
    if (formatCanonicalFloat32Value(value) != text) {
        throw std::invalid_argument("float32 字符串不是最短 canonical 表示");
    }
    return value;
}

std::string formatCanonicalFloat32(float value) {
    if (value <= 0.0F) {
        throw std::invalid_argument("canonical float32 scale 必须是正数");
    }
    return formatCanonicalFloat32Value(value);
}

float parseCanonicalFloat32(std::string_view text) {
    const float value = parseCanonicalFloat32Value(text);
    if (value <= 0.0F) {
        throw std::invalid_argument("canonical float32 scale 必须是正数");
    }
    return value;
}

}  // namespace quant_lstm
