#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>

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

}  // namespace

int main() {
    try {
        using namespace quant_lstm;
        using namespace quant_lstm::quantization;

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
