#pragma once

#include "lstm/quant_config.h"
#include "quantization/quant_param.h"

#include <array>
#include <cstdint>
#include <vector>

// 本模块将各粒度校准组一次性物化为执行期 4H 参数；kernel 不再广播。
namespace quant_lstm {

struct CalibrationRange {
    float minimum = 0.0F;
    float maximum = 0.0F;
};

struct LstmQuantizationRanges {
    std::array<std::vector<CalibrationRange>, kQuantOperatorCount> operators;

    const std::vector<CalibrationRange>& at(QuantOperator id) const;
    std::vector<CalibrationRange>& at(QuantOperator id);
};

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
                                    const LstmQuantizationRanges& ranges,
                                    std::int64_t hidden_size, bool bias_enabled);

}  // namespace quant_lstm
