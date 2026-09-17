#pragma once

#include <cstdint>

// 仅供 forward_quantized_fp_cuda.cu 使用的扁平 device 参数。
// 每通道数组始终为 4H；这里没有 raw scale ratio 或运行时 granularity 分支。
namespace quant_lstm::cuda_detail {

enum class DeviceRescaleKind : std::uint8_t {
    MShift = 0,
    Pot2 = 1,
};

struct DeviceQuantPoint {
    float scale;
    float zero_point;
    float minimum;
    float maximum;
};

struct DeviceRescale {
    DeviceRescaleKind kind;
    std::uint16_t multiplier;
    std::int8_t shift;
};

struct DeviceLinearChannelParams {
    DeviceQuantPoint weight_ih;
    DeviceQuantPoint weight_hh;
    DeviceQuantPoint bias_ih;
    DeviceQuantPoint bias_hh;
    DeviceQuantPoint input_linear;
    DeviceQuantPoint recurrent_linear;
    DeviceRescale bias_ih_to_accumulator;
    DeviceRescale input_accumulator_to_linear;
    DeviceRescale bias_hh_to_accumulator;
    DeviceRescale recurrent_accumulator_to_linear;
};

struct DeviceGateParams {
    DeviceQuantPoint input;
    DeviceQuantPoint output;
    DeviceRescale input_linear_to_gate;
    DeviceRescale recurrent_linear_to_gate;
};

struct DeviceScalarParams {
    DeviceQuantPoint input;
    DeviceQuantPoint hidden;
    DeviceQuantPoint cell;
    DeviceQuantPoint cell_tanh;
    DeviceGateParams gates[4];
    DeviceRescale hidden_product_to_output;
    std::int64_t cell_forget_q31;
    std::int64_t cell_input_q31;
    std::int32_t bias_enabled;
};

}  // namespace quant_lstm::cuda_detail
