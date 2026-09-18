#include "lstm/quant_config.h"

#include <stdexcept>
#include <string>

namespace quant_lstm {
namespace {

constexpr std::array<std::string_view, kQuantOperatorCount> kOperatorNames{
    "input",
    "output",
    "cell_state",
    "weight_ih",
    "weight_hh",
    "bias_ih",
    "bias_hh",
    "weight_ih_linear",
    "weight_hh_linear",
    "input_gate_input",
    "forget_gate_input",
    "cell_gate_input",
    "output_gate_input",
    "input_gate_output",
    "forget_gate_output",
    "cell_gate_output",
    "output_gate_output",
    "cell_tanh_output",
};

std::size_t operatorIndex(QuantOperator id) {
    const auto index = static_cast<std::size_t>(id);
    if (index >= kQuantOperatorCount) {
        throw std::out_of_range("QuantOperator 枚举值非法");
    }
    return index;
}

}  // namespace

const OperatorQuantConfig& LstmOperatorQuantConfig::at(QuantOperator id) const {
    return operators.at(operatorIndex(id));
}

OperatorQuantConfig& LstmOperatorQuantConfig::at(QuantOperator id) {
    return operators.at(operatorIndex(id));
}

void LstmOperatorQuantConfig::validate() const {
    if (schema_version != 1) {
        throw std::invalid_argument("只支持 quant config schema_version=1");
    }
    if (scale_mode != quantization::ScaleMode::Affine &&
        scale_mode != quantization::ScaleMode::Pot2) {
        throw std::invalid_argument("ScaleMode 枚举值非法");
    }
    for (std::size_t index = 0; index < operators.size(); ++index) {
        const auto id = static_cast<QuantOperator>(index);
        const OperatorQuantConfig& config = operators[index];
        config.type.validate();
        if (config.granularity != QuantGranularity::PerTensor &&
            config.granularity != QuantGranularity::PerGate &&
            config.granularity != QuantGranularity::PerChannel) {
            throw std::invalid_argument("QuantGranularity 枚举值非法");
        }
        if (isParameterOperator(id)) {
            if (config.type.is_unsigned || !config.type.is_symmetric) {
                throw std::invalid_argument("weight/bias 必须为 signed symmetric");
            }
        } else if (config.granularity != QuantGranularity::PerTensor) {
            throw std::invalid_argument("非参数量化点 granularity 必须为 per_tensor");
        }
    }
}

std::string_view quantOperatorName(QuantOperator id) {
    return kOperatorNames.at(operatorIndex(id));
}

QuantOperator parseQuantOperatorName(std::string_view name) {
    for (std::size_t index = 0; index < kOperatorNames.size(); ++index) {
        if (kOperatorNames[index] == name) {
            return static_cast<QuantOperator>(index);
        }
    }
    throw std::invalid_argument("unknown quant operator: " + std::string(name));
}

bool isParameterOperator(QuantOperator id) noexcept {
    return id == QuantOperator::WeightInputHidden || id == QuantOperator::WeightHiddenHidden ||
           id == QuantOperator::BiasInputHidden || id == QuantOperator::BiasHiddenHidden;
}

bool isBiasOperator(QuantOperator id) noexcept {
    return id == QuantOperator::BiasInputHidden || id == QuantOperator::BiasHiddenHidden;
}

bool isSigmoidGateOutput(QuantOperator id) noexcept {
    return id == QuantOperator::InputGateOutput || id == QuantOperator::ForgetGateOutput ||
           id == QuantOperator::OutputGateOutput;
}

std::string_view granularityName(QuantGranularity granularity) {
    switch (granularity) {
        case QuantGranularity::PerTensor:
            return "per_tensor";
        case QuantGranularity::PerGate:
            return "per_gate";
        case QuantGranularity::PerChannel:
            return "per_channel";
    }
    throw std::invalid_argument("QuantGranularity 枚举值非法");
}

QuantGranularity parseGranularity(std::string_view name) {
    if (name == "per_tensor") {
        return QuantGranularity::PerTensor;
    }
    if (name == "per_gate") {
        return QuantGranularity::PerGate;
    }
    if (name == "per_channel") {
        return QuantGranularity::PerChannel;
    }
    throw std::invalid_argument("unknown granularity: " + std::string(name));
}

std::string_view scaleModeName(quantization::ScaleMode mode) {
    switch (mode) {
        case quantization::ScaleMode::Affine:
            return "affine";
        case quantization::ScaleMode::Pot2:
            return "pot2";
    }
    throw std::invalid_argument("ScaleMode 枚举值非法");
}

quantization::ScaleMode parseScaleMode(std::string_view name) {
    if (name == "affine") {
        return quantization::ScaleMode::Affine;
    }
    if (name == "pot2") {
        return quantization::ScaleMode::Pot2;
    }
    throw std::invalid_argument("unknown scale_mode: " + std::string(name));
}

}  // namespace quant_lstm
