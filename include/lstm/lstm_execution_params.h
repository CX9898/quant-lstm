#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "lstm/quant_config.h"
#include "lstm/quant_params.h"
#include "quantization/numeric_safety.h"

// 本模块集中把 standard scale 派生为执行编码；执行原语不得重新计算比例。
namespace quant_lstm {

enum class ExecutionRescaleKind : std::uint8_t {
    MShift,
    Pot2,
};

struct ExecutionRescale {
    ExecutionRescaleKind kind = ExecutionRescaleKind::MShift;
    quantization::FixedPointScale m_shift;
    quantization::Pot2Rescale pot2;
};

struct QuantizedPoint {
    quantization::QuantParam param;
    quantization::QuantizationType type;
};

struct LinearExecutionParams {
    QuantizedPoint input;
    QuantizedPoint output;
    std::vector<QuantizedPoint> weights;
    std::vector<QuantizedPoint> biases;
    std::vector<ExecutionRescale> bias_to_accumulator;
    std::vector<ExecutionRescale> accumulator_to_output;
};

struct GateExecutionParams {
    QuantizedPoint input_hidden_linear;
    QuantizedPoint hidden_hidden_linear;
    QuantizedPoint gate_input;
    QuantizedPoint gate_output;
    ExecutionRescale input_hidden_to_gate;
    ExecutionRescale hidden_hidden_to_gate;
};

struct Q31Scale {
    static constexpr std::uint8_t kFractionalBits = 31;

    std::int64_t multiplier = 0;
};

struct CellExecutionParams {
    QuantizedPoint forget_gate;
    QuantizedPoint old_cell;
    QuantizedPoint input_gate;
    QuantizedPoint cell_gate;
    QuantizedPoint new_cell;
    Q31Scale forget_scale;
    Q31Scale input_scale;
};

struct HiddenExecutionParams {
    QuantizedPoint output_gate;
    QuantizedPoint cell_tanh;
    QuantizedPoint output;
    ExecutionRescale product_to_output;
};

struct ExecutionSafetyDiagnostics {
    std::vector<quantization::NumericSafetyReport> input_hidden_linear;
    std::vector<quantization::NumericSafetyReport> hidden_hidden_linear;
    std::array<quantization::NumericSafetyReport, 4> gates;
    quantization::NumericSafetyReport cell;
    quantization::NumericSafetyReport hidden;
};

struct LstmExecutionParams {
    std::int64_t input_size = 0;
    std::int64_t hidden_size = 0;
    LinearExecutionParams input_hidden_linear;
    LinearExecutionParams hidden_hidden_linear;
    std::array<GateExecutionParams, 4> gates;
    CellExecutionParams cell;
    HiddenExecutionParams hidden;
    ExecutionSafetyDiagnostics diagnostics;
};

// 从已完成校准的参数一次性派生全部 Linear、四门、Cell 与 Hidden 执行编码。
// 无法证明整数边界安全时立即抛出；exact 模式还拒绝 FP32 精度风险。
LstmExecutionParams deriveLstmExecutionParams(const LstmOperatorQuantConfig& config,
                                              const LstmQuantParams& quant_params,
                                              std::int64_t input_size,
                                              bool require_exact_accumulation = false);

// 以下执行函数只消费编码，不接受或暴露 raw ratio。
std::int64_t applyExecutionRescale(std::int64_t value, const ExecutionRescale& encoded);
float applyExecutionRescale(float value, const ExecutionRescale& encoded);

}  // namespace quant_lstm
