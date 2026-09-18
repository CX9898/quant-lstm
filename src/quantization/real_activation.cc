#include "quantization/real_activation.h"

#include <cmath>
#include <stdexcept>

#include "../lstm/cuda/quantized_fp_math.cuh"
#include "quantization/rounding.h"

namespace quant_lstm::quantization {

std::int32_t realActivation(std::int32_t quantized_input, const QuantParam& input_param,
                            const QuantizationType& input_type, const QuantParam& output_param,
                            const QuantizationType& output_type, RealActivationKind kind) {
    input_param.validate(input_type);
    input_type.validateValue(quantized_input);
    output_param.validate(output_type);
    const auto output_range = output_type.range();
    return static_cast<std::int32_t>(cuda_detail::realActivationCore(
        static_cast<float>(quantized_input), input_param.scale, input_param.zero_point,
        output_param.scale, output_param.zero_point, output_range.minimum, output_range.maximum,
        kind));
}

float realActivation(float quantized_input, const QuantParam& input_param,
                     const QuantizationType& input_type, const QuantParam& output_param,
                     const QuantizationType& output_type, RealActivationKind kind) {
    if (!std::isfinite(quantized_input) || roundToNearestEven(quantized_input) != quantized_input ||
        quantized_input < input_type.range().minimum ||
        quantized_input > input_type.range().maximum) {
        throw std::invalid_argument("FP carrier 激活输入必须位于合法整数网格");
    }
    return static_cast<float>(realActivation(static_cast<std::int32_t>(quantized_input),
                                             input_param, input_type, output_param, output_type,
                                             kind));
}

}  // namespace quant_lstm::quantization
