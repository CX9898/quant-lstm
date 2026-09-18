#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "lstm/quant_config.h"
#include "quantization/quant_param.h"

// 本模块将各粒度校准组一次性物化为执行期 4H 参数；kernel 不再广播。
namespace quant_lstm {

struct CalibrationRange {
    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();
    std::uint64_t sample_count = 0;

    bool empty() const noexcept;
    void observe(float value);
    void observe(const float* values, std::size_t count);
    void merge(const CalibrationRange& other);
};

struct LstmQuantizationRanges {
    std::array<std::vector<CalibrationRange>, kQuantOperatorCount> operators;

    void reset(const LstmOperatorQuantConfig& config, std::int64_t hidden_size, bool bias_enabled);
    void validateComplete(const LstmOperatorQuantConfig& config, std::int64_t hidden_size,
                          bool bias_enabled) const;
    const std::vector<CalibrationRange>& at(QuantOperator id) const;
    std::vector<CalibrationRange>& at(QuantOperator id);
};

std::size_t quantizationGroupCount(QuantOperator id, QuantGranularity granularity,
                                   std::int64_t hidden_size);
std::size_t quantizationGroupIndex(QuantOperator id, QuantGranularity granularity,
                                   std::size_t channel, std::int64_t hidden_size);

struct FinalizedOperatorQuantParams {
    QuantGranularity source_granularity = QuantGranularity::PerTensor;
    std::vector<quantization::QuantParam> values;
    std::vector<quantization::CalibrationDiagnostics> group_diagnostics;
};

struct LstmQuantParams {
    std::int64_t hidden_size = 0;
    bool bias_enabled = true;
    std::array<FinalizedOperatorQuantParams, kQuantOperatorCount> operators;

    const FinalizedOperatorQuantParams& at(QuantOperator id) const;
    void validate(const LstmOperatorQuantConfig& config) const;
};

LstmQuantParams finalizeQuantParams(const LstmOperatorQuantConfig& config,
                                    const LstmQuantizationRanges& ranges, std::int64_t hidden_size,
                                    bool bias_enabled);

}  // namespace quant_lstm
