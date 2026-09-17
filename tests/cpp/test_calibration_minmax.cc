#include "lstm/calibration.h"
#include "lstm/quant_config_loader.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool near(float actual, float expected, float tolerance = 1.0e-6F) {
    return std::abs(actual - expected) <= tolerance;
}

template <typename Function>
void requireThrows(Function&& function, const char* message) {
    bool threw = false;
    try {
        function();
    } catch (const std::exception&) {
        threw = true;
    }
    require(threw, message);
}

quant_lstm::LstmOperatorQuantConfig defaultConfig() {
    const std::filesystem::path path =
        std::filesystem::path(QUANT_LSTM_SOURCE_DIR) /
        "config/defaults/lstm_quant_default_v1.json";
    return quant_lstm::resolveQuantConfigFiles(path);
}

}  // namespace

int main() {
    try {
        using namespace quant_lstm;
        auto config = defaultConfig();
        config.at(QuantOperator::WeightInputHidden).granularity =
            QuantGranularity::PerTensor;
        config.at(QuantOperator::WeightHiddenHidden).granularity =
            QuantGranularity::PerGate;

        const LstmShape shape{2, 1, 1, 1};
        const float input[2]{0.25F, -0.5F};
        const float weight_ih[4]{};
        const float weight_hh[4]{};
        const LstmFloatWeights weights{weight_ih, weight_hh, nullptr, nullptr};
        const float h0[1]{0.75F};
        const float c0[1]{1.0F};

        LstmFloatReferenceTrace trace;
        float output[2]{};
        float h_n[1]{};
        float c_n[1]{};
        lstmForwardFloatCpu(shape, weights, input, h0, c0, output, h_n, c_n,
                            &trace);
        require(trace.gate_inputs.size() == 8 &&
                    trace.gate_outputs.size() == 8 &&
                    trace.cell_states.size() == 2,
                "float checkpoint shape");
        require(near(trace.gate_outputs[0], 0.5F) &&
                    near(trace.gate_outputs[1], 0.5F) &&
                    near(trace.gate_outputs[2], 0.0F) &&
                    near(trace.gate_outputs[3], 0.5F),
                "float checkpoint gate values");
        require(near(trace.cell_states[0], 0.5F) &&
                    near(trace.cell_states[1], 0.25F),
                "float checkpoint recurrent values");

        LstmCalibrationSession session(config, 1, 1, false);
        require(session.state() == CalibrationState::Empty,
                "initial state must be Empty");
        requireThrows([&] { static_cast<void>(session.finalize()); },
                      "Empty finalize must fail");
        session.collect(shape, weights, input, h0, c0);
        require(session.state() == CalibrationState::Dirty,
                "collect must make session Dirty");

        const auto& ranges = session.collector().ranges();
        require(near(ranges.at(QuantOperator::Input)[0].minimum, -0.5F) &&
                    near(ranges.at(QuantOperator::Input)[0].maximum, 0.25F),
                "input range");
        require(near(ranges.at(QuantOperator::Output)[0].maximum, 0.75F),
                "h0 must share output range");
        require(near(ranges.at(QuantOperator::CellState)[0].maximum, 1.0F),
                "c0 must share cell range");
        require(ranges.at(QuantOperator::WeightInputHidden).size() == 1 &&
                    ranges.at(QuantOperator::WeightHiddenHidden).size() == 4,
                "parameter group allocation");
        require(ranges.at(QuantOperator::BiasInputHidden).empty() &&
                    ranges.at(QuantOperator::BiasHiddenHidden).empty(),
                "bias-disabled ranges must be absent");
        for (std::size_t index = 0; index < kQuantOperatorCount; ++index) {
            const auto id = static_cast<QuantOperator>(index);
            if (!isBiasOperator(id)) {
                require(!ranges.at(id).front().empty(),
                        "all enabled real quant points must be observed");
            }
        }

        const auto& finalized = session.finalize();
        require(session.state() == CalibrationState::Locked,
                "finalize must lock session");
        require(finalized.report.batch_count == 1 &&
                    finalized.report.method == CalibrationMethod::MinMax,
                "calibration report identity");
        require(finalized.report.operators
                        [static_cast<std::size_t>(QuantOperator::WeightHiddenHidden)]
                            .groups.size() == 4,
                "report must retain compact calibration groups");
        require(finalized.report.operators
                        [static_cast<std::size_t>(QuantOperator::Input)]
                            .groups.front()
                            .quantized_steps == 254,
                "report must include quantized step count N");
        require(!finalized.report.contributions.forget_times_old_cell.empty() &&
                    finalized.report.contributions.input_times_cell.minimum == 0.0F,
                "contributions are diagnostics");
        require(&session.finalize() == &finalized,
                "Locked finalize must be idempotent");
        requireThrows([&] { session.collect(shape, weights, input, h0, c0); },
                      "Locked collect must fail");

        session.reset();
        require(session.state() == CalibrationState::Empty &&
                    session.collector().batchCount() == 0 &&
                    session.finalized() == nullptr,
                "reset must clear session");
        const float second_input[2]{-2.0F, 1.0F};
        session.collect(shape, weights, input, h0, c0);
        session.reset();
        session.collect(shape, weights, input, h0, c0);
        session.collect(shape, weights, second_input, h0, c0);
        require(session.collector().batchCount() == 2 &&
                    near(session.collector()
                             .ranges()
                             .at(QuantOperator::Input)[0]
                             .minimum,
                         -2.0F) &&
                    near(session.collector()
                             .ranges()
                             .at(QuantOperator::Input)[0]
                             .maximum,
                         1.0F),
                "multiple collect calls must form a range union");

        CalibrationRange range;
        range.observe(-1.0F);
        range.observe(2.0F);
        CalibrationRange other;
        other.observe(-3.0F);
        range.merge(other);
        require(range.sample_count == 3 && range.minimum == -3.0F &&
                    range.maximum == 2.0F,
                "range observe and merge");
        requireThrows(
            [&] { range.observe(std::numeric_limits<float>::infinity()); },
            "non-finite calibration values must fail");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
