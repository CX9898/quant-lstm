#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "lstm/lstm_execution_params.h"
#include "lstm/quantized_cell_fp.h"
#include "lstm/quantized_cell_int.h"

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
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

quant_lstm::QuantizedPoint signedPoint(float scale, std::int32_t zero_point = 0) {
    return {{scale, zero_point}, {8, false, false}};
}

quant_lstm::QuantizedPoint unsignedPoint(float scale, std::int32_t zero_point = 0) {
    return {{scale, zero_point}, {8, true, false}};
}

quant_lstm::CellExecutionParams makeCellParams() {
    quant_lstm::CellExecutionParams params;
    params.forget_gate = unsignedPoint(0.5F);
    params.old_cell = signedPoint(0.25F, -3);
    params.input_gate = unsignedPoint(0.25F);
    params.cell_gate = signedPoint(0.5F);
    params.new_cell = params.old_cell;
    params.forget_scale.multiplier = std::int64_t{1} << 30;
    params.input_scale.multiplier = std::int64_t{1} << 30;
    return params;
}

quant_lstm::LstmOperatorQuantConfig makeConfig(quant_lstm::quantization::ScaleMode mode) {
    quant_lstm::LstmOperatorQuantConfig config;
    config.scale_mode = mode;
    for (std::size_t index = 0; index < quant_lstm::kQuantOperatorCount; ++index) {
        const auto id = static_cast<quant_lstm::QuantOperator>(index);
        config.operators[index].type =
            quant_lstm::isParameterOperator(id)
                ? quant_lstm::quantization::QuantizationType{8, false, true}
                : quant_lstm::quantization::QuantizationType{8, false, false};
        config.operators[index].granularity = quant_lstm::QuantGranularity::PerTensor;
    }
    for (quant_lstm::QuantOperator id :
         {quant_lstm::QuantOperator::InputGateOutput, quant_lstm::QuantOperator::ForgetGateOutput,
          quant_lstm::QuantOperator::OutputGateOutput}) {
        config.at(id).type = {8, true, false};
    }
    return config;
}

quant_lstm::LstmQuantParams makeQuantParams(const quant_lstm::LstmOperatorQuantConfig& config) {
    quant_lstm::LstmQuantParams params;
    params.hidden_size = 1;
    params.bias_enabled = true;
    constexpr float scale = 1.0F / 128.0F;
    for (std::size_t index = 0; index < quant_lstm::kQuantOperatorCount; ++index) {
        const auto id = static_cast<quant_lstm::QuantOperator>(index);
        auto& output = params.operators[index];
        output.source_granularity = quant_lstm::QuantGranularity::PerTensor;
        output.group_diagnostics.resize(1);
        const std::size_t count = quant_lstm::isParameterOperator(id) ? 4 : 1;
        output.values.assign(count, {scale, 0});
    }
    params.validate(config);
    return params;
}

}  // namespace

int main() {
    try {
        const auto cell_params = makeCellParams();
        const auto integer = quant_lstm::computeQuantizedCellInt(4, 5, 8, -1, cell_params);
        require(integer.value == 9, "Cell int 双比例融合");
        require(integer.diagnostics.p_forget == 32 && integer.diagnostics.p_input == -8 &&
                    integer.diagnostics.pre_round_sum ==
                        static_cast<__int128>(24) * (std::int64_t{1} << 30),
                "Cell int diagnostics");

        const auto fp = quant_lstm::computeQuantizedCellFp(4.0F, 5.0F, 8.0F, -1.0F, cell_params);
        require(
            fp.value == static_cast<float>(integer.value) && fp.diagnostics.pre_round_sum == 12.0F,
            "Cell FP 模拟相同 Q31 编码");

        auto delayed_round = makeCellParams();
        delayed_round.forget_gate = signedPoint(1.0F);
        delayed_round.old_cell = signedPoint(1.0F);
        delayed_round.input_gate = signedPoint(1.0F);
        delayed_round.cell_gate = signedPoint(1.0F);
        delayed_round.new_cell = signedPoint(1.0F);
        const auto tie = quant_lstm::computeQuantizedCellInt(1, 1, 1, 1, delayed_round);
        require(tie.value == 1, "Cell 必须合并后只做一次 RoundShift31");

        auto saturated = delayed_round;
        saturated.forget_scale.multiplier = std::int64_t{1} << 40;
        saturated.input_scale.multiplier = std::int64_t{1} << 40;
        require(quant_lstm::computeQuantizedCellInt(127, 127, 127, 127, saturated).value == 127,
                "Cell 最终边界饱和");

        quant_lstm::HiddenExecutionParams hidden;
        hidden.output_gate = signedPoint(0.5F);
        hidden.cell_tanh = signedPoint(0.5F);
        hidden.output = signedPoint(0.5F, -2);
        hidden.product_to_output.kind = quant_lstm::ExecutionRescaleKind::MShift;
        hidden.product_to_output.m_shift = {32768, 16};
        const auto hidden_int = quant_lstm::computeQuantizedHiddenInt(3, 5, hidden);
        const auto hidden_fp = quant_lstm::computeQuantizedHiddenFp(3.0F, 5.0F, hidden);
        require(hidden_int.value == 6 && hidden_fp.value == 6.0F &&
                    hidden_int.diagnostics.raw_product == 15,
                "Hidden M+shift 编码融合");
        hidden.product_to_output.kind = quant_lstm::ExecutionRescaleKind::Pot2;
        hidden.product_to_output.pot2 = {1};
        require(quant_lstm::computeQuantizedHiddenInt(3, 5, hidden).value == 6,
                "Hidden POT2 编码融合");

        requireThrows(
            [&] {
                static_cast<void>(quant_lstm::computeQuantizedCellInt(-1, 0, 0, 0, cell_params));
            },
            "unsigned 门输入越界必须失败");
        requireThrows(
            [&] {
                static_cast<void>(
                    quant_lstm::computeQuantizedCellFp(1.5F, 0.0F, 0.0F, 0.0F, cell_params));
            },
            "FP carrier 非整数 q 必须失败");

        for (const auto mode : {quant_lstm::quantization::ScaleMode::Affine,
                                quant_lstm::quantization::ScaleMode::Pot2}) {
            const auto config = makeConfig(mode);
            const auto params = makeQuantParams(config);
            const auto execution = quant_lstm::deriveLstmExecutionParams(config, params, 2, true);
            const auto expected_kind = mode == quant_lstm::quantization::ScaleMode::Affine
                                           ? quant_lstm::ExecutionRescaleKind::MShift
                                           : quant_lstm::ExecutionRescaleKind::Pot2;
            require(execution.input_hidden_linear.weights.size() == 4 &&
                        execution.gates.size() == 4 &&
                        execution.input_hidden_linear.accumulator_to_output.front().kind ==
                            expected_kind &&
                        execution.hidden.product_to_output.kind == expected_kind,
                    "集中派生 Linear/四门/Hidden 编码");
            require(execution.cell.forget_scale.multiplier == (std::int64_t{1} << 24) &&
                        execution.cell.input_scale.multiplier == (std::int64_t{1} << 24) &&
                        execution.diagnostics.cell.safe() && execution.diagnostics.hidden.safe(),
                    "集中派生 Q31 与 NumericSafety diagnostics");
        }

        const auto config = makeConfig(quant_lstm::quantization::ScaleMode::Affine);
        const auto params = makeQuantParams(config);
        requireThrows(
            [&] { static_cast<void>(quant_lstm::deriveLstmExecutionParams(config, params, 0)); },
            "非法 input_size 必须失败");
        requireThrows(
            [&] {
                static_cast<void>(
                    quant_lstm::deriveLstmExecutionParams(config, params, 2000, true));
            },
            "exact 模式必须拒绝 FP32 precision risk");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
