#include "quantization/quant_param.h"

#include <cmath>
#include <stdexcept>

namespace quant_lstm::quantization {

void QuantParam::validate(const QuantizationType& type) const {
    if (!std::isfinite(scale) || scale <= 0.0F) {
        throw std::invalid_argument("standard scale 必须是有限正数");
    }
    type.validateValue(zero_point);
    if (type.is_symmetric && zero_point != 0) {
        throw std::invalid_argument("symmetric 量化的 zero point 必须为 0");
    }
}

}  // namespace quant_lstm::quantization
