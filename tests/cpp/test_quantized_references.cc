#include "lstm/forward_cpu.h"
#include "lstm/forward_float.h"
#include "lstm/lstm_execution_params.h"
#include "common/numeric_metrics.h"
#include "quantization/fixed_point_ops.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using quant_lstm::QuantOperator;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

quant_lstm::LstmOperatorQuantConfig makeConfig(
    quant_lstm::quantization::ScaleMode mode) {
    quant_lstm::LstmOperatorQuantConfig config;
    config.scale_mode = mode;
    for (auto& item : config.operators) {
        item.type = {8, false, true};
        item.granularity = quant_lstm::QuantGranularity::PerTensor;
    }
    for (QuantOperator id :
         {QuantOperator::WeightInputHidden,
          QuantOperator::WeightHiddenHidden, QuantOperator::BiasInputHidden,
          QuantOperator::BiasHiddenHidden}) {
        config.at(id).granularity =
            quant_lstm::QuantGranularity::PerChannel;
    }
    for (QuantOperator id :
         {QuantOperator::InputGateOutput, QuantOperator::ForgetGateOutput,
          QuantOperator::OutputGateOutput}) {
        config.at(id).type = {8, true, true};
    }
    config.validate();
    return config;
}

quant_lstm::CalibrationRange rangeFor(QuantOperator id) {
    switch (id) {
        case QuantOperator::Input:
        case QuantOperator::Output:
        case QuantOperator::CellGateOutput:
        case QuantOperator::CellTanhOutput:
            return {-1.0F, 1.0F};
        case QuantOperator::CellState:
            return {-2.0F, 2.0F};
        case QuantOperator::WeightInputHidden:
        case QuantOperator::WeightHiddenHidden:
        case QuantOperator::BiasInputHidden:
        case QuantOperator::BiasHiddenHidden:
            return {-0.5F, 0.5F};
        case QuantOperator::InputGateOutput:
        case QuantOperator::ForgetGateOutput:
        case QuantOperator::OutputGateOutput:
            return {0.0F, 1.0F};
        default:
            return {-4.0F, 4.0F};
    }
}

quant_lstm::LstmQuantParams makeParams(
    const quant_lstm::LstmOperatorQuantConfig& config, std::int64_t hidden) {
    quant_lstm::LstmQuantizationRanges ranges;
    const std::size_t channels = static_cast<std::size_t>(4 * hidden);
    for (std::size_t index = 0; index < quant_lstm::kQuantOperatorCount;
         ++index) {
        const auto id = static_cast<QuantOperator>(index);
        std::size_t count = 1;
        if (quant_lstm::isParameterOperator(id)) {
            switch (config.at(id).granularity) {
                case quant_lstm::QuantGranularity::PerTensor:
                    count = 1;
                    break;
                case quant_lstm::QuantGranularity::PerGate:
                    count = 4;
                    break;
                case quant_lstm::QuantGranularity::PerChannel:
                    count = channels;
                    break;
            }
        }
        ranges.at(id).assign(count, rangeFor(id));
    }
    return quant_lstm::finalizeQuantParams(config, ranges, hidden, true);
}

std::vector<std::int32_t> quantizeTensor(
    const std::vector<float>& source, QuantOperator id,
    const quant_lstm::LstmOperatorQuantConfig& config,
    const quant_lstm::LstmQuantParams& params, std::size_t row_width = 0) {
    std::vector<std::int32_t> result(source.size());
    const auto& point = params.at(id);
    for (std::size_t index = 0; index < source.size(); ++index) {
        const std::size_t parameter_index =
            row_width == 0 ? 0 : index / row_width;
        result[index] = quant_lstm::quantization::quantize(
            source[index], point.values.at(parameter_index),
            config.at(id).type);
    }
    return result;
}

void runMode(quant_lstm::quantization::ScaleMode mode) {
    const quant_lstm::LstmShape shape{3, 1, 2, 2};
    const auto config = makeConfig(mode);
    const auto params = makeParams(config, shape.hidden_size);
    const auto execution =
        quant_lstm::deriveLstmExecutionParams(config, params, shape.input_size);

    const std::vector<float> input{
        -0.75F, 0.25F, 0.5F, -0.125F, 0.25F, 0.75F};
    const std::vector<float> weight_ih{
        0.25F, -0.125F, -0.375F, 0.25F, 0.125F, 0.375F, -0.25F, 0.125F,
        0.375F, 0.25F, -0.125F, -0.25F, 0.25F, 0.125F, 0.375F, -0.375F};
    const std::vector<float> weight_hh{
        0.125F, -0.25F, 0.25F, 0.125F, -0.125F, 0.375F, 0.25F, -0.25F,
        0.375F, 0.125F, -0.25F, 0.25F, 0.125F, -0.375F, 0.25F, 0.125F};
    const std::vector<float> bias_ih{
        0.125F, -0.125F, 0.25F, 0.125F, -0.25F, 0.125F, 0.0F, 0.25F};
    const std::vector<float> bias_hh{
        0.0F, 0.125F, -0.125F, 0.0F, 0.125F, -0.25F, 0.125F, 0.0F};
    const std::vector<float> initial_hidden{0.0F, 0.0F};
    const std::vector<float> initial_cell{0.0F, 0.0F};
    const std::size_t channels = 8;

    const auto q_input =
        quantizeTensor(input, QuantOperator::Input, config, params);
    const auto q_weight_ih = quantizeTensor(
        weight_ih, QuantOperator::WeightInputHidden, config, params, 2);
    const auto q_weight_hh = quantizeTensor(
        weight_hh, QuantOperator::WeightHiddenHidden, config, params, 2);
    const auto q_bias_ih = quantizeTensor(
        bias_ih, QuantOperator::BiasInputHidden, config, params, 1);
    const auto q_bias_hh = quantizeTensor(
        bias_hh, QuantOperator::BiasHiddenHidden, config, params, 1);
    const auto q_initial_hidden =
        quantizeTensor(initial_hidden, QuantOperator::Output, config, params);
    const auto q_initial_cell =
        quantizeTensor(initial_cell, QuantOperator::CellState, config, params);

    std::vector<std::int32_t> int_output(6);
    std::vector<std::int32_t> int_hidden(2);
    std::vector<std::int32_t> int_cell(2);
    quant_lstm::LstmInt32ReferenceTrace int_trace;
    quant_lstm::lstmForwardInt32CpuReference(
        shape,
        {q_weight_ih.data(), q_weight_hh.data(), q_bias_ih.data(),
         q_bias_hh.data()},
        q_input.data(), q_initial_hidden.data(), q_initial_cell.data(), config,
        params, execution, int_output.data(), int_hidden.data(),
        int_cell.data(), &int_trace);

    const auto toFloat = [](const std::vector<std::int32_t>& values) {
        return std::vector<float>(values.begin(), values.end());
    };
    const auto fp_input = toFloat(q_input);
    const auto fp_weight_ih = toFloat(q_weight_ih);
    const auto fp_weight_hh = toFloat(q_weight_hh);
    const auto fp_bias_ih = toFloat(q_bias_ih);
    const auto fp_bias_hh = toFloat(q_bias_hh);
    const auto fp_initial_hidden = toFloat(q_initial_hidden);
    const auto fp_initial_cell = toFloat(q_initial_cell);
    std::vector<float> fp_output(6);
    std::vector<float> fp_hidden(2);
    std::vector<float> fp_cell(2);
    quant_lstm::LstmFpReferenceTrace fp_trace;
    quant_lstm::lstmForwardQuantizedFpCpuReference(
        shape,
        {fp_weight_ih.data(), fp_weight_hh.data(), fp_bias_ih.data(),
         fp_bias_hh.data()},
        fp_input.data(), fp_initial_hidden.data(), fp_initial_cell.data(),
        config, params, execution, fp_output.data(), fp_hidden.data(),
        fp_cell.data(), &fp_trace);

    require(int_trace.weight_ih_linear.size() == 3 * channels,
            "linear trace shape");
    require(int_trace.p_forget.size() == 6, "cell diagnostics shape");
    for (std::size_t index = 0; index < int_output.size(); ++index) {
        require(static_cast<float>(int_output[index]) == fp_output[index],
                "cross-carrier output q mismatch");
    }
    for (std::size_t index = 0; index < int_trace.gate_outputs.size();
         ++index) {
        require(static_cast<float>(int_trace.gate_outputs[index]) ==
                    fp_trace.gate_outputs[index],
                "cross-carrier gate q mismatch");
    }

    std::vector<float> dequantized(int_output.size());
    for (std::size_t index = 0; index < int_output.size(); ++index) {
        dequantized[index] = quant_lstm::quantization::dequantize(
            int_output[index], params.at(QuantOperator::Output).values.front(),
            config.at(QuantOperator::Output).type);
    }
    for (float value : dequantized) {
        require(std::isfinite(value) && std::abs(value) <= 1.01F,
                "dequantized output range");
    }
    std::vector<float> float_output(6);
    std::vector<float> float_hidden(2);
    std::vector<float> float_cell(2);
    quant_lstm::lstmForwardFloatCpu(
        shape,
        {weight_ih.data(), weight_hh.data(), bias_ih.data(), bias_hh.data()},
        input.data(), initial_hidden.data(), initial_cell.data(),
        float_output.data(), float_hidden.data(), float_cell.data());
    std::vector<float> dequantized_cell(int_cell.size());
    for (std::size_t index = 0; index < int_cell.size(); ++index) {
        dequantized_cell[index] = quant_lstm::quantization::dequantize(
            int_cell[index],
            params.at(QuantOperator::CellState).values.front(),
            config.at(QuantOperator::CellState).type);
    }
    const auto output_metrics = quant_lstm::test::computeNumericMetrics(
        dequantized.data(), float_output.data(), dequantized.size());
    const auto cell_metrics = quant_lstm::test::computeNumericMetrics(
        dequantized_cell.data(), float_cell.data(), dequantized_cell.size());
    require(output_metrics.mean_absolute_error < 0.05 &&
                output_metrics.mean_squared_error < 0.001,
            "synthetic int8 output precision gate");
    require(cell_metrics.mean_absolute_error < 0.05 &&
                cell_metrics.mean_squared_error < 0.001,
            "synthetic int8 cell precision gate");
    if (!output_metrics.cosine_not_applicable) {
        require(!output_metrics.one_sided_zero_norm &&
                    output_metrics.cosine_similarity >= 0.999,
                "synthetic int8 output cosine gate");
    }
    auto invalid_input = q_input;
    invalid_input.front() = 256;
    bool rejected = false;
    try {
        quant_lstm::lstmForwardInt32CpuReference(
            shape,
            {q_weight_ih.data(), q_weight_hh.data(), q_bias_ih.data(),
             q_bias_hh.data()},
            invalid_input.data(), q_initial_hidden.data(),
            q_initial_cell.data(), config, params, execution,
            int_output.data(), int_hidden.data(), int_cell.data());
    } catch (const std::out_of_range&) {
        rejected = true;
    }
    require(rejected, "int32 reference must reject q outside configured range");
}

}  // namespace

int main() {
    try {
        runMode(quant_lstm::quantization::ScaleMode::Affine);
        runMode(quant_lstm::quantization::ScaleMode::Pot2);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
