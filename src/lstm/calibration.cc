#include "lstm/calibration.h"

#include "lstm/gate_layout.h"

#include <array>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace quant_lstm {
namespace {

constexpr std::array<QuantOperator, kGateCount> kGateInputOperators{
    QuantOperator::InputGateInput,
    QuantOperator::ForgetGateInput,
    QuantOperator::CellGateInput,
    QuantOperator::OutputGateInput,
};

constexpr std::array<QuantOperator, kGateCount> kGateOutputOperators{
    QuantOperator::InputGateOutput,
    QuantOperator::ForgetGateOutput,
    QuantOperator::CellGateOutput,
    QuantOperator::OutputGateOutput,
};

std::size_t checkedSize(std::int64_t value, const char* name) {
    if (value <= 0) {
        throw std::invalid_argument(std::string(name) + " 必须大于 0");
    }
    return static_cast<std::size_t>(value);
}

const float* stateOrZeros(const float* state, std::vector<float>* zeros,
                          std::size_t count) {
    if (state != nullptr) {
        return state;
    }
    zeros->assign(count, 0.0F);
    return zeros->data();
}

std::size_t finalizedGroupValueIndex(QuantGranularity granularity,
                                     std::size_t group,
                                     std::size_t hidden_size) {
    if (granularity == QuantGranularity::PerGate) {
        return group * hidden_size;
    }
    return granularity == QuantGranularity::PerChannel ? group : 0;
}

}  // namespace

LstmCalibrationCollector::LstmCalibrationCollector(
    LstmOperatorQuantConfig config, std::int64_t input_size,
    std::int64_t hidden_size, bool bias_enabled, bool collect_histograms,
    std::size_t histogram_bin_count)
    : config_(std::move(config)),
      input_size_(input_size),
      hidden_size_(hidden_size),
      bias_enabled_(bias_enabled),
      collect_histograms_(collect_histograms),
      histogram_bin_count_(histogram_bin_count) {
    config_.validate();
    checkedSize(input_size_, "input_size");
    checkedSize(hidden_size_, "hidden_size");
    if (histogram_bin_count_ < 2) {
        throw std::invalid_argument("histogram_bin_count 必须至少为 2");
    }
    reset();
}

void LstmCalibrationCollector::reset() {
    batch_count_ = 0;
    ranges_.reset(config_, hidden_size_, bias_enabled_);
    contributions_ = {};
    for (std::size_t index = 0; index < kQuantOperatorCount; ++index) {
        const auto id = static_cast<QuantOperator>(index);
        auto& collectors = histograms_[index];
        if (!collect_histograms_ ||
            (!bias_enabled_ && isBiasOperator(id))) {
            collectors.clear();
            continue;
        }
        collectors.assign(
            quantizationGroupCount(id, config_.operators[index].granularity,
                                   hidden_size_),
            quantization::HistogramCollector(histogram_bin_count_));
    }
}

void LstmCalibrationCollector::observeOperator(QuantOperator id,
                                               const float* values,
                                               std::size_t count) {
    ranges_.at(id).front().observe(values, count);
    if (collect_histograms_) {
        histograms_[static_cast<std::size_t>(id)].front().collect(values,
                                                                  count);
    }
}

void LstmCalibrationCollector::observeParameter(QuantOperator id,
                                                const float* values,
                                                std::size_t row_width) {
    if (values == nullptr) {
        throw std::invalid_argument("参数校准指针不能为空");
    }
    const std::size_t hidden = static_cast<std::size_t>(hidden_size_);
    const std::size_t channels = kGateCount * hidden;
    const QuantGranularity granularity = config_.at(id).granularity;
    auto& groups = ranges_.at(id);
    auto& histograms = histograms_[static_cast<std::size_t>(id)];
    if (granularity == QuantGranularity::PerTensor) {
        groups.front().observe(values, channels * row_width);
        if (collect_histograms_) {
            histograms.front().collect(values, channels * row_width);
        }
        return;
    }
    if (granularity == QuantGranularity::PerGate) {
        for (std::size_t gate = 0; gate < kGateCount; ++gate) {
            const float* gate_values =
                values + gate * hidden * row_width;
            groups[gate].observe(gate_values, hidden * row_width);
            if (collect_histograms_) {
                histograms[gate].collect(gate_values, hidden * row_width);
            }
        }
        return;
    }
    for (std::size_t channel = 0; channel < channels; ++channel) {
        const std::size_t group =
            quantizationGroupIndex(id, granularity, channel, hidden_size_);
        groups[group].observe(values + channel * row_width, row_width);
        if (collect_histograms_) {
            histograms[group].collect(values + channel * row_width,
                                      row_width);
        }
    }
}

void LstmCalibrationCollector::collect(
    const LstmShape& shape, const LstmFloatWeights& weights, const float* input,
    const float* initial_hidden, const float* initial_cell) {
    if (shape.input_size != input_size_ || shape.hidden_size != hidden_size_) {
        throw std::invalid_argument("校准 shape 与 collector 契约不匹配");
    }
    if ((weights.bias_ih != nullptr) != bias_enabled_) {
        throw std::invalid_argument("校准 bias 状态与 collector 契约不匹配");
    }

    const std::size_t steps = checkedSize(shape.sequence_length, "sequence_length");
    const std::size_t batch = checkedSize(shape.batch_size, "batch_size");
    const std::size_t input_size = static_cast<std::size_t>(input_size_);
    const std::size_t hidden = static_cast<std::size_t>(hidden_size_);
    const std::size_t state_count = batch * hidden;

    std::vector<float> output(steps * state_count);
    std::vector<float> final_hidden(state_count);
    std::vector<float> final_cell(state_count);
    LstmFloatReferenceTrace trace;
    lstmForwardFloatCpu(shape, weights, input, initial_hidden, initial_cell,
                        output.data(), final_hidden.data(), final_cell.data(), &trace);

    observeOperator(QuantOperator::Input, input,
                    steps * batch * input_size);

    std::vector<float> zero_hidden;
    std::vector<float> zero_cell;
    const float* initial_hidden_values =
        stateOrZeros(initial_hidden, &zero_hidden, state_count);
    const float* initial_cell_values =
        stateOrZeros(initial_cell, &zero_cell, state_count);
    observeOperator(QuantOperator::Output, initial_hidden_values,
                    state_count);
    observeOperator(QuantOperator::CellState, initial_cell_values,
                    state_count);
    observeOperator(QuantOperator::Output, trace.hidden_outputs.data(),
                    trace.hidden_outputs.size());
    observeOperator(QuantOperator::CellState, trace.cell_states.data(),
                    trace.cell_states.size());

    observeParameter(QuantOperator::WeightInputHidden, weights.weight_ih,
                     input_size);
    observeParameter(QuantOperator::WeightHiddenHidden, weights.weight_hh,
                     hidden);
    if (bias_enabled_) {
        observeParameter(QuantOperator::BiasInputHidden, weights.bias_ih, 1);
        observeParameter(QuantOperator::BiasHiddenHidden, weights.bias_hh, 1);
    }

    observeOperator(QuantOperator::WeightInputHiddenLinear,
        trace.weight_input_hidden_linear.data(),
        trace.weight_input_hidden_linear.size());
    observeOperator(QuantOperator::WeightHiddenHiddenLinear,
        trace.weight_hidden_hidden_linear.data(),
        trace.weight_hidden_hidden_linear.size());
    observeOperator(QuantOperator::CellTanhOutput,
        trace.cell_tanh_outputs.data(), trace.cell_tanh_outputs.size());

    const std::size_t gate_stride = kGateCount * hidden;
    for (std::size_t row = 0; row < steps * batch; ++row) {
        for (std::size_t gate = 0; gate < kGateCount; ++gate) {
            const std::size_t offset = row * gate_stride + gate * hidden;
            observeOperator(kGateInputOperators[gate],
                trace.gate_inputs.data() + offset, hidden);
            observeOperator(kGateOutputOperators[gate],
                trace.gate_outputs.data() + offset, hidden);
        }
    }

    for (std::size_t step = 0; step < steps; ++step) {
        for (std::size_t row = 0; row < batch; ++row) {
            const std::size_t state_offset = (step * batch + row) * hidden;
            const float* previous_cell =
                step == 0 ? initial_cell_values + row * hidden
                          : trace.cell_states.data() +
                                ((step - 1) * batch + row) * hidden;
            for (std::size_t channel = 0; channel < hidden; ++channel) {
                const std::size_t gate_base =
                    (step * batch + row) * gate_stride + channel;
                const float input_gate = trace.gate_outputs[
                    gate_base + gateOffset(GateKind::Input, hidden)];
                const float forget_gate = trace.gate_outputs[
                    gate_base + gateOffset(GateKind::Forget, hidden)];
                const float cell_gate = trace.gate_outputs[
                    gate_base + gateOffset(GateKind::Cell, hidden)];
                const float output_gate = trace.gate_outputs[
                    gate_base + gateOffset(GateKind::Output, hidden)];
                const float cell_tanh =
                    trace.cell_tanh_outputs[state_offset + channel];
                contributions_.forget_times_old_cell.observe(
                    forget_gate * previous_cell[channel]);
                contributions_.input_times_cell.observe(input_gate * cell_gate);
                contributions_.output_times_cell_tanh.observe(
                    output_gate * cell_tanh);
            }
        }
    }
    ++batch_count_;
}

const LstmQuantizationRanges& LstmCalibrationCollector::ranges() const noexcept {
    return ranges_;
}

const LstmContributionRanges&
LstmCalibrationCollector::contributions() const noexcept {
    return contributions_;
}

std::uint64_t LstmCalibrationCollector::batchCount() const noexcept {
    return batch_count_;
}

std::int64_t LstmCalibrationCollector::inputSize() const noexcept {
    return input_size_;
}

std::int64_t LstmCalibrationCollector::hiddenSize() const noexcept {
    return hidden_size_;
}

bool LstmCalibrationCollector::biasEnabled() const noexcept {
    return bias_enabled_;
}

const LstmOperatorQuantConfig&
LstmCalibrationCollector::config() const noexcept {
    return config_;
}

const std::array<std::vector<quantization::HistogramCollector>,
                 kQuantOperatorCount>&
LstmCalibrationCollector::histograms() const noexcept {
    return histograms_;
}

LstmCalibrationSession::LstmCalibrationSession(
    LstmOperatorQuantConfig config, std::int64_t input_size,
    std::int64_t hidden_size, bool bias_enabled, CalibrationMethod method,
    quantization::HistogramCalibrationOptions histogram_options,
    std::size_t histogram_bin_count)
    : method_(method),
      histogram_options_(histogram_options),
      collector_(std::move(config), input_size, hidden_size, bias_enabled,
                 method != CalibrationMethod::MinMax,
                 histogram_bin_count) {
    if (method_ != CalibrationMethod::MinMax &&
        method_ != CalibrationMethod::Sqnr &&
        method_ != CalibrationMethod::Percentile) {
        throw std::invalid_argument("CalibrationMethod 枚举值非法");
    }
}

void LstmCalibrationSession::collect(
    const LstmShape& shape, const LstmFloatWeights& weights, const float* input,
    const float* initial_hidden, const float* initial_cell) {
    if (state_ == CalibrationState::Locked) {
        throw std::logic_error("Locked 校准会话拒绝继续采集");
    }
    collector_.collect(shape, weights, input, initial_hidden, initial_cell);
    state_ = CalibrationState::Dirty;
    has_finalized_ = false;
}

const FinalizedLstmCalibration& LstmCalibrationSession::finalize(
    bool require_exact_accumulation) {
    if (state_ == CalibrationState::Locked) {
        return finalized_;
    }
    if (state_ == CalibrationState::Empty) {
        throw std::logic_error("Empty 校准会话不能 finalize");
    }

    LstmQuantizationRanges selected_ranges = collector_.ranges();
    if (method_ != CalibrationMethod::MinMax) {
        const auto histogram_method =
            method_ == CalibrationMethod::Sqnr
                ? quantization::HistogramCalibrationMethod::Sqnr
                : quantization::HistogramCalibrationMethod::Percentile;
        for (std::size_t index = 0; index < kQuantOperatorCount; ++index) {
            const auto id = static_cast<QuantOperator>(index);
            if (!collector_.biasEnabled() && isBiasOperator(id)) {
                continue;
            }
            auto& ranges = selected_ranges.operators[index];
            const auto& histograms = collector_.histograms()[index];
            for (std::size_t group = 0; group < ranges.size(); ++group) {
                const auto candidate = quantization::calibrateHistogramRange(
                    histograms[group].histogram(),
                    collector_.config().operators[index].type,
                    histogram_method, histogram_options_);
                ranges[group].minimum = candidate.first;
                ranges[group].maximum = candidate.second;
            }
        }
    }
    finalized_.quant_params = finalizeQuantParams(
        collector_.config(), selected_ranges, collector_.hiddenSize(),
        collector_.biasEnabled());
    finalized_.execution_params = deriveLstmExecutionParams(
        collector_.config(), finalized_.quant_params, collector_.inputSize(),
        require_exact_accumulation);

    auto& report = finalized_.report;
    report = {};
    report.method = method_;
    report.batch_count = collector_.batchCount();
    report.contributions = collector_.contributions();
    report.execution_safety = finalized_.execution_params.diagnostics;
    const std::size_t hidden =
        static_cast<std::size_t>(collector_.hiddenSize());
    for (std::size_t index = 0; index < kQuantOperatorCount; ++index) {
        const auto id = static_cast<QuantOperator>(index);
        auto& operator_report = report.operators[index];
        operator_report.id = id;
        operator_report.granularity =
            finalized_.quant_params.operators[index].source_granularity;
        if (!collector_.biasEnabled() && isBiasOperator(id)) {
            continue;
        }
        const auto& ranges = collector_.ranges().operators[index];
        const auto& params = finalized_.quant_params.operators[index];
        operator_report.groups.reserve(ranges.size());
        for (std::size_t group = 0; group < ranges.size(); ++group) {
            const std::size_t value_index = finalizedGroupValueIndex(
                params.source_granularity, group, hidden);
            const auto quantized_range =
                collector_.config().operators[index].type.range();
            const std::int64_t quantized_steps =
                static_cast<std::int64_t>(quantized_range.maximum) -
                quantized_range.minimum;
            operator_report.groups.push_back(
                {ranges[group], quantized_steps,
                 params.group_diagnostics[group],
                 params.values[value_index]});
        }
    }

    state_ = CalibrationState::Locked;
    has_finalized_ = true;
    return finalized_;
}

void LstmCalibrationSession::reset() {
    collector_.reset();
    finalized_ = {};
    has_finalized_ = false;
    state_ = CalibrationState::Empty;
}

CalibrationState LstmCalibrationSession::state() const noexcept {
    return state_;
}

CalibrationMethod LstmCalibrationSession::method() const noexcept {
    return method_;
}

const LstmCalibrationCollector&
LstmCalibrationSession::collector() const noexcept {
    return collector_;
}

const FinalizedLstmCalibration*
LstmCalibrationSession::finalized() const noexcept {
    return has_finalized_ ? &finalized_ : nullptr;
}

std::string calibrationMethodName(CalibrationMethod method) {
    switch (method) {
        case CalibrationMethod::MinMax:
            return "minmax";
        case CalibrationMethod::Sqnr:
            return "sqnr";
        case CalibrationMethod::Percentile:
            return "percentile";
    }
    throw std::invalid_argument("CalibrationMethod 枚举值非法");
}

std::string calibrationStateName(CalibrationState state) {
    switch (state) {
        case CalibrationState::Empty:
            return "empty";
        case CalibrationState::Dirty:
            return "dirty";
        case CalibrationState::Locked:
            return "locked";
    }
    throw std::invalid_argument("CalibrationState 枚举值非法");
}

}  // namespace quant_lstm
