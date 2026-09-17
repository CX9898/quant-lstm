#pragma once

#include "lstm/forward_float.h"
#include "lstm/lstm_execution_params.h"
#include "lstm/quant_config.h"
#include "lstm/quant_params.h"
#include "quantization/histogram.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace quant_lstm {

enum class CalibrationMethod : std::uint8_t {
    MinMax,
    Sqnr,
    Percentile,
};

enum class CalibrationState : std::uint8_t {
    Empty,
    Dirty,
    Locked,
};

struct LstmContributionRanges {
    CalibrationRange forget_times_old_cell;
    CalibrationRange input_times_cell;
    CalibrationRange output_times_cell_tanh;
};

struct CalibrationGroupReport {
    CalibrationRange observed;
    std::int64_t quantized_steps = 0;
    quantization::CalibrationDiagnostics diagnostics;
    quantization::QuantParam finalized;
};

struct CalibrationOperatorReport {
    QuantOperator id = QuantOperator::Input;
    QuantGranularity granularity = QuantGranularity::PerTensor;
    std::vector<CalibrationGroupReport> groups;
};

struct LstmCalibrationReport {
    CalibrationMethod method = CalibrationMethod::MinMax;
    std::uint64_t batch_count = 0;
    std::array<CalibrationOperatorReport, kQuantOperatorCount> operators;
    LstmContributionRanges contributions;
    ExecutionSafetyDiagnostics execution_safety;
};

struct FinalizedLstmCalibration {
    LstmQuantParams quant_params;
    LstmExecutionParams execution_params;
    LstmCalibrationReport report;
};

// 负责把正式 FP32 reference checkpoint 累积到 18 个真实量化点。
class LstmCalibrationCollector {
   public:
    LstmCalibrationCollector(LstmOperatorQuantConfig config, std::int64_t input_size,
                             std::int64_t hidden_size, bool bias_enabled,
                             bool collect_histograms = false,
                             std::size_t histogram_bin_count = 2048);

    void reset();
    void collect(const LstmShape& shape, const LstmFloatWeights& weights,
                 const float* input, const float* initial_hidden,
                 const float* initial_cell);

    const LstmQuantizationRanges& ranges() const noexcept;
    const LstmContributionRanges& contributions() const noexcept;
    std::uint64_t batchCount() const noexcept;
    std::int64_t inputSize() const noexcept;
    std::int64_t hiddenSize() const noexcept;
    bool biasEnabled() const noexcept;
    const LstmOperatorQuantConfig& config() const noexcept;
    const std::array<std::vector<quantization::HistogramCollector>,
                     kQuantOperatorCount>&
    histograms() const noexcept;

   private:
    void observeOperator(QuantOperator id, const float* values,
                         std::size_t count);
    void observeParameter(QuantOperator id, const float* values,
                          std::size_t row_width);

    LstmOperatorQuantConfig config_;
    std::int64_t input_size_;
    std::int64_t hidden_size_;
    bool bias_enabled_;
    std::uint64_t batch_count_ = 0;
    LstmQuantizationRanges ranges_;
    LstmContributionRanges contributions_;
    bool collect_histograms_;
    std::size_t histogram_bin_count_;
    std::array<std::vector<quantization::HistogramCollector>,
               kQuantOperatorCount>
        histograms_;
};

// 会话封装 Empty -> Dirty -> Locked 生命周期。finalize 幂等；Locked 后
// collect 被拒绝，reset 是开始新一轮校准的唯一入口。
class LstmCalibrationSession {
   public:
    LstmCalibrationSession(LstmOperatorQuantConfig config, std::int64_t input_size,
                           std::int64_t hidden_size, bool bias_enabled,
                           CalibrationMethod method = CalibrationMethod::MinMax,
                           quantization::HistogramCalibrationOptions
                               histogram_options = {},
                           std::size_t histogram_bin_count = 2048);

    void collect(const LstmShape& shape, const LstmFloatWeights& weights,
                 const float* input, const float* initial_hidden,
                 const float* initial_cell);
    const FinalizedLstmCalibration& finalize(
        bool require_exact_accumulation = false);
    void reset();

    CalibrationState state() const noexcept;
    CalibrationMethod method() const noexcept;
    const LstmCalibrationCollector& collector() const noexcept;
    const FinalizedLstmCalibration* finalized() const noexcept;

   private:
    CalibrationMethod method_;
    quantization::HistogramCalibrationOptions histogram_options_;
    CalibrationState state_ = CalibrationState::Empty;
    LstmCalibrationCollector collector_;
    FinalizedLstmCalibration finalized_;
    bool has_finalized_ = false;
};

std::string calibrationMethodName(CalibrationMethod method);
std::string calibrationStateName(CalibrationState state);

}  // namespace quant_lstm
