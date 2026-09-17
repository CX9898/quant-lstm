#include "lstm/quant_params.h"

#include "lstm/gate_layout.h"
#include "quantization/scale_encoding.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace quant_lstm {
namespace {

std::size_t operatorIndex(QuantOperator id) {
    const auto index = static_cast<std::size_t>(id);
    if (index >= kQuantOperatorCount) {
        throw std::out_of_range("QuantOperator 枚举值非法");
    }
    return index;
}

std::size_t checkedChannelCount(std::int64_t hidden_size) {
    if (hidden_size <= 0 ||
        hidden_size > std::numeric_limits<std::int64_t>::max() /
                          static_cast<std::int64_t>(kGateCount)) {
        throw std::invalid_argument("hidden_size 非法或 4H 溢出");
    }
    return static_cast<std::size_t>(hidden_size) * kGateCount;
}

quantization::CalibrationResult calibrateGroup(
    const CalibrationRange& range, const quantization::QuantizationType& type,
    quantization::ScaleMode scale_mode) {
    quantization::CalibrationResult result =
        quantization::calibrateMinMax(range.minimum, range.maximum, type);
    if (scale_mode == quantization::ScaleMode::Pot2) {
        result.param = quantization::convertScaleToPot2CoverRange(result, type).param;
    }
    return result;
}

}  // namespace

bool CalibrationRange::empty() const noexcept {
    return !std::isfinite(minimum) || !std::isfinite(maximum) || minimum > maximum;
}

void CalibrationRange::observe(float value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument("校准值必须为有限 FP32");
    }
    minimum = std::min(minimum, value);
    maximum = std::max(maximum, value);
    ++sample_count;
}

void CalibrationRange::observe(const float* values, std::size_t count) {
    if (values == nullptr && count != 0) {
        throw std::invalid_argument("校准张量指针不能为空");
    }
    for (std::size_t index = 0; index < count; ++index) {
        observe(values[index]);
    }
}

void CalibrationRange::merge(const CalibrationRange& other) {
    if (other.empty()) {
        return;
    }
    minimum = std::min(minimum, other.minimum);
    maximum = std::max(maximum, other.maximum);
    sample_count += other.sample_count;
}

std::size_t quantizationGroupCount(QuantOperator id, QuantGranularity granularity,
                                   std::int64_t hidden_size) {
    const std::size_t channel_count = checkedChannelCount(hidden_size);
    if (!isParameterOperator(id)) {
        if (granularity != QuantGranularity::PerTensor) {
            throw std::invalid_argument("非参数量化点 granularity 必须为 per_tensor");
        }
        return 1;
    }
    switch (granularity) {
        case QuantGranularity::PerTensor:
            return 1;
        case QuantGranularity::PerGate:
            return kGateCount;
        case QuantGranularity::PerChannel:
            return channel_count;
    }
    throw std::invalid_argument("QuantGranularity 枚举值非法");
}

std::size_t quantizationGroupIndex(QuantOperator id, QuantGranularity granularity,
                                   std::size_t channel, std::int64_t hidden_size) {
    const std::size_t channel_count = checkedChannelCount(hidden_size);
    if (channel >= channel_count) {
        throw std::out_of_range("参数 channel 超出 4H");
    }
    if (!isParameterOperator(id)) {
        throw std::invalid_argument("非参数量化点没有 channel group");
    }
    switch (granularity) {
        case QuantGranularity::PerTensor:
            return 0;
        case QuantGranularity::PerGate:
            return channel / static_cast<std::size_t>(hidden_size);
        case QuantGranularity::PerChannel:
            return channel;
    }
    throw std::invalid_argument("QuantGranularity 枚举值非法");
}

void LstmQuantizationRanges::reset(const LstmOperatorQuantConfig& config,
                                   std::int64_t hidden_size, bool bias_enabled) {
    config.validate();
    for (std::size_t index = 0; index < operators.size(); ++index) {
        const auto id = static_cast<QuantOperator>(index);
        auto& groups = operators[index];
        if (!bias_enabled && isBiasOperator(id)) {
            groups.clear();
            continue;
        }
        groups.assign(
            quantizationGroupCount(id, config.operators[index].granularity, hidden_size), {});
    }
}

void LstmQuantizationRanges::validateComplete(
    const LstmOperatorQuantConfig& config, std::int64_t hidden_size,
    bool bias_enabled) const {
    config.validate();
    for (std::size_t index = 0; index < operators.size(); ++index) {
        const auto id = static_cast<QuantOperator>(index);
        const auto& groups = operators[index];
        if (!bias_enabled && isBiasOperator(id)) {
            if (!groups.empty()) {
                throw std::invalid_argument("bias=False 时 bias 校准范围必须缺失");
            }
            continue;
        }
        const std::size_t expected = quantizationGroupCount(
            id, config.operators[index].granularity, hidden_size);
        if (groups.size() != expected) {
            throw std::invalid_argument("校准 range 数量与 granularity 不匹配");
        }
        for (const CalibrationRange& group : groups) {
            if (group.empty()) {
                throw std::invalid_argument("校准 range 缺少有限观测值");
            }
        }
    }
}

const std::vector<CalibrationRange>& LstmQuantizationRanges::at(QuantOperator id) const {
    return operators.at(operatorIndex(id));
}

std::vector<CalibrationRange>& LstmQuantizationRanges::at(QuantOperator id) {
    return operators.at(operatorIndex(id));
}

const FinalizedOperatorQuantParams& LstmQuantParams::at(QuantOperator id) const {
    return operators.at(operatorIndex(id));
}

void LstmQuantParams::validate(const LstmOperatorQuantConfig& config) const {
    config.validate();
    const std::size_t channel_count = checkedChannelCount(hidden_size);
    for (std::size_t index = 0; index < operators.size(); ++index) {
        const auto id = static_cast<QuantOperator>(index);
        const auto& finalized = operators[index];
        if (!bias_enabled && isBiasOperator(id)) {
            if (!finalized.values.empty() || !finalized.group_diagnostics.empty()) {
                throw std::invalid_argument("bias=False 时 bias 量化参数必须缺失");
            }
            continue;
        }
        const std::size_t expected_values = isParameterOperator(id) ? channel_count : 1;
        const std::size_t expected_groups =
            quantizationGroupCount(id, finalized.source_granularity, hidden_size);
        if (finalized.values.size() != expected_values ||
            finalized.group_diagnostics.size() != expected_groups ||
            finalized.source_granularity != config.at(id).granularity) {
            throw std::invalid_argument("finalized 参数长度或 granularity 不匹配");
        }
        for (const auto& value : finalized.values) {
            value.validate(config.at(id).type);
            if (isParameterOperator(id) && value.zero_point != 0) {
                throw std::invalid_argument("weight/bias 4H zero point 必须全为 0");
            }
        }
        if (isParameterOperator(id) &&
            finalized.source_granularity == QuantGranularity::PerTensor) {
            for (const auto& value : finalized.values) {
                if (value.scale != finalized.values.front().scale ||
                    value.zero_point != finalized.values.front().zero_point) {
                    throw std::invalid_argument("per_tensor 4H 参数没有位级重复");
                }
            }
        }
        if (isParameterOperator(id) &&
            finalized.source_granularity == QuantGranularity::PerGate) {
            for (std::size_t gate = 0; gate < kGateCount; ++gate) {
                const std::size_t start = gate * static_cast<std::size_t>(hidden_size);
                for (std::size_t channel = start + 1;
                     channel < start + static_cast<std::size_t>(hidden_size); ++channel) {
                    if (finalized.values[channel].scale != finalized.values[start].scale ||
                        finalized.values[channel].zero_point !=
                            finalized.values[start].zero_point) {
                        throw std::invalid_argument("per_gate 4H 参数门段没有位级重复");
                    }
                }
            }
        }
    }
}

LstmQuantParams finalizeQuantParams(const LstmOperatorQuantConfig& config,
                                    const LstmQuantizationRanges& ranges,
                                    std::int64_t hidden_size, bool bias_enabled) {
    config.validate();
    const std::size_t channel_count = checkedChannelCount(hidden_size);
    LstmQuantParams result;
    result.hidden_size = hidden_size;
    result.bias_enabled = bias_enabled;
    ranges.validateComplete(config, hidden_size, bias_enabled);

    for (std::size_t index = 0; index < kQuantOperatorCount; ++index) {
        const auto id = static_cast<QuantOperator>(index);
        const OperatorQuantConfig& operator_config = config.operators[index];
        auto& destination = result.operators[index];
        destination.source_granularity = operator_config.granularity;
        const auto& source_ranges = ranges.operators[index];

        if (!bias_enabled && isBiasOperator(id)) {
            if (!source_ranges.empty()) {
                throw std::invalid_argument("bias=False 时 bias 校准范围必须缺失");
            }
            continue;
        }

        const std::size_t group_count =
            quantizationGroupCount(id, operator_config.granularity, hidden_size);
        if (source_ranges.size() != group_count) {
            throw std::invalid_argument("校准 range 数量与 granularity 不匹配");
        }
        std::vector<quantization::QuantParam> compact_values;
        compact_values.reserve(group_count);
        destination.group_diagnostics.reserve(group_count);
        for (const CalibrationRange& source_range : source_ranges) {
            const auto calibrated =
                calibrateGroup(source_range, operator_config.type, config.scale_mode);
            compact_values.push_back(calibrated.param);
            destination.group_diagnostics.push_back(calibrated.diagnostics);
        }

        if (!isParameterOperator(id)) {
            destination.values = std::move(compact_values);
        } else if (operator_config.granularity == QuantGranularity::PerTensor) {
            destination.values.assign(channel_count, compact_values.front());
        } else if (operator_config.granularity == QuantGranularity::PerGate) {
            destination.values.reserve(channel_count);
            for (const auto& value : compact_values) {
                destination.values.insert(destination.values.end(),
                                          static_cast<std::size_t>(hidden_size), value);
            }
        } else {
            destination.values = std::move(compact_values);
        }
    }

    result.validate(config);
    return result;
}

}  // namespace quant_lstm
