#pragma once

#include "quantization/quant_param.h"

#include <cstdint>

// 两种 reference carrier 共用的真实 sigmoid/tanh 量化边界；首版不包含 LUT。
namespace quant_lstm::quantization {

enum class RealActivationKind : std::uint8_t {
    Sigmoid,
    Tanh,
};

std::int32_t realActivation(std::int32_t quantized_input, const QuantParam& input_param,
                            const QuantizationType& input_type,
                            const QuantParam& output_param,
                            const QuantizationType& output_type, RealActivationKind kind);

float realActivation(float quantized_input, const QuantParam& input_param,
                     const QuantizationType& input_type, const QuantParam& output_param,
                     const QuantizationType& output_type, RealActivationKind kind);

}  // namespace quant_lstm::quantization
