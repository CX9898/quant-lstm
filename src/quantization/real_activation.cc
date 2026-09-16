#include "quantization/real_activation.h"

#include "quantization/fixed_point_ops.h"
#include "quantization/rounding.h"

#include <cmath>
#include <stdexcept>

namespace quant_lstm::quantization {
namespace {

float activate(float value, RealActivationKind kind) {
    if (kind == RealActivationKind::Tanh) {
        return std::tanh(value);
    }
    if (value >= 0.0F) {
        const float exponential = std::exp(-value);
        return 1.0F / (1.0F + exponential);
    }
    const float exponential = std::exp(value);
    return exponential / (1.0F + exponential);
}

}  // namespace

std::int32_t realActivation(std::int32_t quantized_input, const QuantParam& input_param,
                            const QuantizationType& input_type,
                            const QuantParam& output_param,
                            const QuantizationType& output_type, RealActivationKind kind) {
    const float real_input = dequantize(quantized_input, input_param, input_type);
    return quantize(activate(real_input, kind), output_param, output_type);
}

float realActivation(float quantized_input, const QuantParam& input_param,
                     const QuantizationType& input_type, const QuantParam& output_param,
                     const QuantizationType& output_type, RealActivationKind kind) {
    if (!std::isfinite(quantized_input) ||
        roundToNearestEven(quantized_input) != quantized_input ||
        quantized_input < input_type.range().minimum ||
        quantized_input > input_type.range().maximum) {
        throw std::invalid_argument("FP carrier 激活输入必须位于合法整数网格");
    }
    return static_cast<float>(
        realActivation(static_cast<std::int32_t>(quantized_input), input_param, input_type,
                       output_param, output_type, kind));
}

}  // namespace quant_lstm::quantization
