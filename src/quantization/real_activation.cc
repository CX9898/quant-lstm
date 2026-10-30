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
    return static_cast<std::int32_t>(cuda_detail::realActivationQuantizedCore(
        static_cast<double>(quantized_input), input_param.scale, input_param.zero_point,
        output_param.scale, output_param.zero_point, output_range.minimum, output_range.maximum,
        kind));
}

float realActivation(float quantized_input, const QuantParam& input_param,
                     const QuantizationType& input_type, const QuantParam& output_param,
                     const QuantizationType& output_type, RealActivationKind kind) {
    input_param.validate(input_type);
    validateFloatCarrierValue(quantized_input, input_type);
    output_param.validate(output_type);
    const auto output_range = output_type.range();
    return cuda_detail::realActivationCore(
        quantized_input, input_param.scale, input_param.zero_point, output_param.scale,
        output_param.zero_point, output_range.minimum, output_range.maximum, kind);
}

}  // namespace quant_lstm::quantization
