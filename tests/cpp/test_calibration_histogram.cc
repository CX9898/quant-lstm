#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "lstm/calibration.h"
#include "lstm/quant_config_loader.h"
#include "quantization/histogram.h"

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

quant_lstm::LstmOperatorQuantConfig defaultConfig() {
    return quant_lstm::resolveQuantConfigFiles(std::filesystem::path(QUANT_LSTM_SOURCE_DIR) /
                                               "config/defaults/lstm_quant_default_v1.json");
}

void testBatchHistogramParity(quant_lstm::CalibrationMethod method) {
    using namespace quant_lstm;
    using quantization::HistogramCollector;
    constexpr std::size_t bins = 64;
    const LstmShape shape{7, 2, 3, 4};
    auto config = defaultConfig();
    config.at(QuantOperator::WeightInputHidden).granularity = QuantGranularity::PerChannel;
    config.at(QuantOperator::WeightHiddenHidden).granularity = QuantGranularity::PerGate;
    config.at(QuantOperator::BiasInputHidden).granularity = QuantGranularity::PerChannel;
    LstmCalibrationSession actual(config, 3, 4, true, method, {}, bins);
    std::array<std::vector<HistogramCollector>, kQuantOperatorCount> expected;
    for (std::size_t index = 0; index < kQuantOperatorCount; ++index) {
        expected[index].assign(quantizationGroupCount(static_cast<QuantOperator>(index),
            config.operators[index].granularity, 4), HistogramCollector(bins));
    }
    std::vector<float> weight_ih(16 * 3), weight_hh(16 * 4), bias_ih(16), bias_hh(16);
    const auto fill = [](std::vector<float>* values, float factor) {
        for (std::size_t index = 0; index < values->size(); ++index)
            (*values)[index] = factor * std::sin(static_cast<float>(index + 1) * 0.71F);
    };
    fill(&weight_ih, 0.6F);
    fill(&weight_hh, 0.3F);
    fill(&bias_ih, 0.1F);
    fill(&bias_hh, -0.2F);
    const LstmFloatWeights weights{weight_ih.data(), weight_hh.data(), bias_ih.data(), bias_hh.data()};
    constexpr std::array<QuantOperator, 4> gate_inputs{QuantOperator::InputGateInput,
        QuantOperator::ForgetGateInput, QuantOperator::CellGateInput, QuantOperator::OutputGateInput};
    constexpr std::array<QuantOperator, 4> gate_outputs{QuantOperator::InputGateOutput,
        QuantOperator::ForgetGateOutput, QuantOperator::CellGateOutput, QuantOperator::OutputGateOutput};
    for (std::size_t batch = 0; batch < 5; ++batch) {
        std::vector<float> input(7 * 2 * 3), hidden(8), cell(8), output(7 * 2 * 4);
        std::vector<float> final_hidden(8), final_cell(8);
        fill(&input, 0.35F + static_cast<float>(batch) * 0.27F);
        fill(&hidden, 0.2F + static_cast<float>(batch) * 0.03F);
        fill(&cell, -0.4F - static_cast<float>(batch) * 0.09F);
        LstmFloatReferenceTrace trace;
        lstmForwardFloatCpu(shape, weights, input.data(), hidden.data(), cell.data(),
                            output.data(), final_hidden.data(), final_cell.data(), &trace);
        std::array<std::vector<HistogramCollector>, kQuantOperatorCount> statistics;
        for (std::size_t index = 0; index < kQuantOperatorCount; ++index)
            statistics[index].assign(expected[index].size(), HistogramCollector(bins));
        const auto observe = [&](QuantOperator id, const std::vector<float>& values) {
            statistics[static_cast<std::size_t>(id)].front().collect(values.data(), values.size());
        };
        const auto parameter = [&](QuantOperator id, const std::vector<float>& values,
                                    std::size_t width) {
            std::vector<std::vector<float>> groups(expected[static_cast<std::size_t>(id)].size());
            for (std::size_t channel = 0; channel < 16; ++channel) {
                auto& group = groups[quantizationGroupIndex(id, config.at(id).granularity, channel, 4)];
                group.insert(group.end(), values.begin() + channel * width,
                             values.begin() + (channel + 1) * width);
            }
            for (std::size_t group = 0; group < groups.size(); ++group)
                statistics[static_cast<std::size_t>(id)][group].collect(
                    groups[group].data(), groups[group].size());
        };
        observe(QuantOperator::Input, input);
        observe(QuantOperator::Output, hidden);
        observe(QuantOperator::Output, output);
        observe(QuantOperator::CellState, cell);
        observe(QuantOperator::CellState, trace.cell_states);
        observe(QuantOperator::WeightInputHiddenLinear, trace.weight_input_hidden_linear);
        observe(QuantOperator::WeightHiddenHiddenLinear, trace.weight_hidden_hidden_linear);
        observe(QuantOperator::CellTanhOutput, trace.cell_tanh_outputs);
        parameter(QuantOperator::WeightInputHidden, weight_ih, 3);
        parameter(QuantOperator::WeightHiddenHidden, weight_hh, 4);
        parameter(QuantOperator::BiasInputHidden, bias_ih, 1);
        parameter(QuantOperator::BiasHiddenHidden, bias_hh, 1);
        for (std::size_t gate = 0; gate < 4; ++gate) {
            std::vector<float> inputs, outputs;
            for (std::size_t row = 0; row < 14; ++row) {
                const auto offset = row * 16 + gate * 4;
                inputs.insert(inputs.end(), trace.gate_inputs.begin() + offset,
                              trace.gate_inputs.begin() + offset + 4);
                outputs.insert(outputs.end(), trace.gate_outputs.begin() + offset,
                               trace.gate_outputs.begin() + offset + 4);
            }
            observe(gate_inputs[gate], inputs);
            observe(gate_outputs[gate], outputs);
        }
        actual.collect(shape, weights, input.data(), hidden.data(), cell.data());
        for (std::size_t index = 0; index < kQuantOperatorCount; ++index) {
            for (std::size_t group = 0; group < expected[index].size(); ++group) {
                expected[index][group].merge(statistics[index][group].histogram());
                const auto& reference = expected[index][group].histogram();
                const auto& observed = actual.collector().histograms()[index][group].histogram();
                if (reference.minimum != observed.minimum || reference.maximum != observed.maximum ||
                    reference.counts != observed.counts || reference.total_count != observed.total_count)
                    throw std::runtime_error("same-trace batch histogram mismatch: " +
                        std::string(quantOperatorName(static_cast<QuantOperator>(index))));
            }
        }
    }
    actual.finalize();
}

}  // namespace

int main() {
    try {
        using namespace quant_lstm;
        using namespace quant_lstm::quantization;

        testBatchHistogramParity(CalibrationMethod::Percentile);
        testBatchHistogramParity(CalibrationMethod::Sqnr);

        HistogramCollector first(32);
        const float first_values[]{
            0.0F,
            0.1F,
            0.2F,
            0.3F,
            0.4F,
            0.5F,
            0.6F,
            0.7F,
            0.8F,
            100.0F,
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::infinity(),
        };
        first.collect(first_values, sizeof(first_values) / sizeof(first_values[0]));
        require(first.histogram().total_count == 10, "histogram must ignore NaN and infinity");

        HistogramCollector second(32);
        const float second_values[]{-2.0F, -1.0F};
        second.collect(second_values, 2);
        first.merge(second.histogram());
        require(first.histogram().total_count == 12 && first.histogram().minimum <= -2.0F &&
                    first.histogram().maximum >= 100.0F,
                "histogram merge must preserve range and count");

        const auto percentile = calibrateHistogramRange(
            first.histogram(), {8, false, false}, HistogramCalibrationMethod::Percentile,
            HistogramCalibrationOptions{80.0F, 101, 17, 21});
        require(
            percentile.first < percentile.second && percentile.second < first.histogram().maximum,
            "percentile must independently clip sparse outlier bins");

        const auto sqnr = calibrateHistogramRange(first.histogram(), {8, false, true},
                                                  HistogramCalibrationMethod::Sqnr);
        require(std::isfinite(sqnr.first) && std::isfinite(sqnr.second) &&
                    sqnr.first < sqnr.second && std::abs(sqnr.first) == std::abs(sqnr.second),
                "SQNR must return a symmetric finite candidate");

        const LstmShape shape{2, 1, 1, 1};
        const float input[]{0.25F, -0.5F};
        const float weight_ih[4]{};
        const float weight_hh[4]{};
        const LstmFloatWeights weights{weight_ih, weight_hh, nullptr, nullptr};
        for (CalibrationMethod method : {CalibrationMethod::Sqnr, CalibrationMethod::Percentile}) {
            LstmCalibrationSession session(defaultConfig(), 1, 1, false, method, {}, 64);
            session.collect(shape, weights, input, nullptr, nullptr);
            const auto& finalized = session.finalize();
            require(
                session.state() == CalibrationState::Locked && finalized.report.method == method,
                "histogram session state and method");
            for (std::size_t index = 0; index < kQuantOperatorCount; ++index) {
                const auto id = static_cast<QuantOperator>(index);
                if (isBiasOperator(id)) {
                    continue;
                }
                const auto& collectors = session.collector().histograms()[index];
                require(!collectors.empty() && !collectors.front().histogram().empty(),
                        "all enabled histogram quant points must collect");
                for (const auto& value : finalized.quant_params.operators[index].values) {
                    require(std::isfinite(value.scale) && value.scale > 0.0F,
                            "histogram finalized scales must be finite positive");
                }
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
