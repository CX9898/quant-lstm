#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "lstm/calibration.h"
#include "lstm/forward_cpu.h"
#include "lstm/lstm_execution_params.h"
#include "quantization/fixed_point_ops.h"

namespace {

quant_lstm::LstmOperatorQuantConfig makeDefaultConfig() {
    using quant_lstm::QuantGranularity;
    using quant_lstm::QuantOperator;
    quant_lstm::LstmOperatorQuantConfig config;
    for (auto& item : config.operators) {
        item.type = {8, false, true};
        item.granularity = QuantGranularity::PerTensor;
    }
    for (QuantOperator id : {QuantOperator::WeightInputHidden, QuantOperator::WeightHiddenHidden,
                             QuantOperator::BiasInputHidden, QuantOperator::BiasHiddenHidden}) {
        config.at(id).granularity = QuantGranularity::PerChannel;
    }
    for (QuantOperator id : {QuantOperator::InputGateOutput, QuantOperator::ForgetGateOutput,
                             QuantOperator::OutputGateOutput}) {
        config.at(id).type = {8, true, true};
    }
    config.validate();
    return config;
}

std::vector<std::int32_t> quantizeTensor(const std::vector<float>& source,
                                         quant_lstm::QuantOperator id,
                                         const quant_lstm::LstmOperatorQuantConfig& config,
                                         const quant_lstm::LstmQuantParams& params,
                                         std::size_t row_width = 0) {
    std::vector<std::int32_t> result(source.size());
    const auto& operator_params = params.at(id);
    for (std::size_t index = 0; index < source.size(); ++index) {
        const std::size_t parameter_index = row_width == 0 ? 0 : index / row_width;
        result[index] = quant_lstm::quantization::quantize(
            source[index], operator_params.values.at(parameter_index), config.at(id).type);
    }
    return result;
}

}  // namespace

int main() {
    try {
        using quant_lstm::QuantOperator;
        const quant_lstm::LstmShape shape{3, 1, 2, 2};
        const std::vector<float> input{0.20F, -0.10F, -0.40F, 0.30F, 0.15F, 0.25F};
        const std::vector<float> weight_ih{0.10F, -0.20F, 0.05F, 0.15F, -0.12F, 0.08F,
                                           0.20F, -0.05F, 0.07F, 0.11F, -0.09F, 0.14F,
                                           0.03F, -0.06F, 0.13F, 0.04F};
        const std::vector<float> weight_hh{0.04F, -0.03F, 0.02F, 0.05F, -0.06F, 0.01F,
                                           0.03F, -0.02F, 0.05F, 0.02F, -0.04F, 0.06F,
                                           0.01F, -0.05F, 0.02F, 0.03F};
        const std::vector<float> bias_ih{0.01F, -0.02F, 0.03F, 0.04F, -0.01F, 0.02F, 0.01F, -0.03F};
        const std::vector<float> bias_hh{-0.01F, 0.01F, 0.02F, -0.02F, 0.01F, 0.00F, -0.01F, 0.02F};
        const quant_lstm::LstmFloatWeights weights{weight_ih.data(), weight_hh.data(),
                                                   bias_ih.data(), bias_hh.data()};

        const auto config = makeDefaultConfig();
        quant_lstm::LstmCalibrationSession calibration(config, shape.input_size, shape.hidden_size,
                                                       true);
        calibration.collect(shape, weights, input.data(), nullptr, nullptr);
        const auto& finalized = calibration.finalize();

        const auto q_input =
            quantizeTensor(input, QuantOperator::Input, config, finalized.quant_params);
        const auto q_weight_ih =
            quantizeTensor(weight_ih, QuantOperator::WeightInputHidden, config,
                           finalized.quant_params, static_cast<std::size_t>(shape.input_size));
        const auto q_weight_hh =
            quantizeTensor(weight_hh, QuantOperator::WeightHiddenHidden, config,
                           finalized.quant_params, static_cast<std::size_t>(shape.hidden_size));
        const auto q_bias_ih = quantizeTensor(bias_ih, QuantOperator::BiasInputHidden, config,
                                              finalized.quant_params, 1);
        const auto q_bias_hh = quantizeTensor(bias_hh, QuantOperator::BiasHiddenHidden, config,
                                              finalized.quant_params, 1);

        const std::size_t output_count =
            static_cast<std::size_t>(shape.sequence_length * shape.batch_size * shape.hidden_size);
        const std::size_t state_count =
            static_cast<std::size_t>(shape.batch_size * shape.hidden_size);
        std::vector<std::int32_t> q_output(output_count);
        std::vector<std::int32_t> q_hidden(state_count);
        std::vector<std::int32_t> q_cell(state_count);
        quant_lstm::lstmForwardInt32CpuReference(
            shape, {q_weight_ih.data(), q_weight_hh.data(), q_bias_ih.data(), q_bias_hh.data()},
            q_input.data(), nullptr, nullptr, config, finalized.quant_params,
            finalized.execution_params, q_output.data(), q_hidden.data(), q_cell.data());

        const auto& output_params = finalized.quant_params.at(QuantOperator::Output);
        std::cout << std::fixed << std::setprecision(6);
        for (std::int64_t time = 0; time < shape.sequence_length; ++time) {
            std::cout << "t=" << time << ':';
            for (std::int64_t hidden = 0; hidden < shape.hidden_size; ++hidden) {
                const std::size_t index =
                    static_cast<std::size_t>(time * shape.hidden_size + hidden);
                const float value = quant_lstm::quantization::dequantize(
                    q_output[index], output_params.values.front(),
                    config.at(QuantOperator::Output).type);
                std::cout << ' ' << value;
            }
            std::cout << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
