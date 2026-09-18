#include "quantization/scale_encoding.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "quantization/rounding.h"

namespace quant_lstm::quantization {
namespace {

std::int32_t computeZeroPoint(float lower, float scale, const QuantizationType& type) {
    const QuantizedRange range = type.range();
    const double candidate =
        static_cast<double>(range.minimum) - static_cast<double>(lower) / scale;
    if (candidate <= static_cast<double>(range.minimum)) {
        return range.minimum;
    }
    if (candidate >= static_cast<double>(range.maximum)) {
        return range.maximum;
    }
    const auto rounded = roundToInteger<std::int64_t>(candidate);
    return static_cast<std::int32_t>(
        std::clamp<std::int64_t>(rounded, range.minimum, range.maximum));
}

float checkedScale(double value) {
    const float result = static_cast<float>(value);
    if (!std::isfinite(result) || result <= 0.0F) {
        throw std::invalid_argument("计算得到的 standard scale 不是有限正 float32");
    }
    return result;
}

}  // namespace

float minimumScale(const QuantizationType& type) {
    const QuantizedRange range = type.range();
    const double steps = static_cast<double>(range.maximum) - range.minimum;
    if (steps <= 0.0) {
        throw std::invalid_argument("量化整数范围为空");
    }
    return static_cast<float>(std::min(std::ldexp(1.0, -23), 0.01 / steps));
}

CalibrationResult calibrateMinMax(float range_minimum, float range_maximum,
                                  const QuantizationType& type) {
    type.validate();
    if (!std::isfinite(range_minimum) || !std::isfinite(range_maximum) ||
        range_minimum > range_maximum) {
        throw std::invalid_argument("校准范围必须有限且 minimum <= maximum");
    }

    const QuantizedRange quantized_range = type.range();
    const double steps = static_cast<double>(quantized_range.maximum) - quantized_range.minimum;
    const float minimum_scale = minimumScale(type);
    float adjusted_minimum = range_minimum;
    float adjusted_maximum = range_maximum;
    double candidate_scale = 0.0;

    if (type.is_symmetric && !type.is_unsigned) {
        const double absolute_maximum = std::max(std::abs(static_cast<double>(range_minimum)),
                                                 std::abs(static_cast<double>(range_maximum)));
        candidate_scale = absolute_maximum / quantized_range.maximum;
    } else if (type.is_symmetric) {
        adjusted_minimum = 0.0F;
        adjusted_maximum = std::max(range_maximum, 0.0F);
        candidate_scale = static_cast<double>(adjusted_maximum) / quantized_range.maximum;
    } else {
        adjusted_minimum = std::min(range_minimum, 0.0F);
        adjusted_maximum = std::max(range_maximum, 0.0F);
        candidate_scale = (static_cast<double>(adjusted_maximum) - adjusted_minimum) / steps;
    }

    const bool fallback_used = candidate_scale < static_cast<double>(minimum_scale);
    float scale = 0.0F;
    if (fallback_used) {
        scale = minimum_scale;
        if (type.is_symmetric && !type.is_unsigned) {
            adjusted_minimum = -static_cast<float>(quantized_range.maximum) * minimum_scale;
            adjusted_maximum = static_cast<float>(quantized_range.maximum) * minimum_scale;
        } else if (type.is_symmetric) {
            adjusted_minimum = 0.0F;
            adjusted_maximum = static_cast<float>(quantized_range.maximum) * minimum_scale;
        } else {
            adjusted_minimum = std::min(range_minimum, 0.0F);
            adjusted_maximum = adjusted_minimum + static_cast<float>(steps) * minimum_scale;
        }
    } else {
        scale = checkedScale(candidate_scale);
    }

    const std::int32_t zero_point =
        type.is_symmetric ? 0 : computeZeroPoint(adjusted_minimum, scale, type);
    QuantParam param{scale, zero_point};
    param.validate(type);
    return {param,
            {range_minimum, range_maximum, adjusted_minimum, adjusted_maximum, minimum_scale,
             fallback_used}};
}

FixedPointScale encodeMShift(double ratio) {
    if (!std::isfinite(ratio) || ratio <= 0.0) {
        throw std::invalid_argument("M+shift ratio 必须是有限正数");
    }
    int exponent = 0;
    const double mantissa = std::frexp(ratio, &exponent);
    std::int64_t multiplier =
        roundToInteger<std::int64_t>(mantissa * static_cast<double>(std::uint64_t{1} << 16U));
    if (multiplier == 65536) {
        multiplier = 32768;
        ++exponent;
    }
    if (multiplier < 32768 || multiplier > 65535) {
        throw std::runtime_error("M+shift multiplier 未规范化");
    }
    const int shift = 16 - exponent;
    if (shift < std::numeric_limits<std::int8_t>::min() ||
        shift > std::numeric_limits<std::int8_t>::max()) {
        throw std::overflow_error("M+shift shift 超出 int8_t");
    }
    return {static_cast<std::uint16_t>(multiplier), static_cast<std::int8_t>(shift)};
}

double decodeMShift(const FixedPointScale& encoded) noexcept {
    return std::ldexp(static_cast<double>(encoded.multiplier), -encoded.shift);
}

Pot2Rescale encodePot2Rescale(float source_scale, float destination_scale) {
    const auto exponentOf = [](float scale) {
        if (!std::isfinite(scale) || scale <= 0.0F) {
            throw std::invalid_argument("POT2 rescale 的 standard scale 必须是有限正数");
        }
        int exponent = 0;
        const float mantissa = std::frexp(scale, &exponent);
        if (mantissa != 0.5F) {
            throw std::invalid_argument("POT2 rescale 只接受精确幂次 standard scale");
        }
        return exponent;
    };
    const int source_exponent = exponentOf(source_scale);
    const int destination_exponent = exponentOf(destination_scale);
    const int shift = destination_exponent - source_exponent;
    if (shift < std::numeric_limits<std::int8_t>::min() ||
        shift > std::numeric_limits<std::int8_t>::max()) {
        throw std::overflow_error("POT2 rescale shift 超出 int8_t");
    }
    return {static_cast<std::int8_t>(shift)};
}

Pot2ScaleResult convertScaleToPot2CoverRange(const CalibrationResult& calibrated,
                                             const QuantizationType& type) {
    calibrated.param.validate(type);
    const double range_minimum = calibrated.diagnostics.adjusted_min;
    const double range_maximum = calibrated.diagnostics.adjusted_max;
    const double real_range = std::abs(range_maximum - range_minimum);
    if (!std::isfinite(real_range) || real_range <= 0.0) {
        throw std::invalid_argument("POT2 CoverRange 需要有限正 range");
    }

    const double range_log2 = std::log2(real_range);
    const int nearest_range_exponent = roundToInteger<int>(range_log2);
    const double nearest_range_power = std::ldexp(1.0, nearest_range_exponent);
    const double relative_distance =
        std::abs(real_range - nearest_range_power) / nearest_range_power;
    const bool near_power = relative_distance < kPot2RelativeTolerance;

    const double exponent_value = -std::log2(static_cast<double>(calibrated.param.scale));
    const double selected_exponent =
        near_power ? roundToNearestEven(exponent_value) : std::floor(exponent_value);
    if (!std::isfinite(selected_exponent) ||
        selected_exponent < std::numeric_limits<std::int8_t>::min() ||
        selected_exponent > std::numeric_limits<std::int8_t>::max()) {
        throw std::overflow_error("POT2 exponent 超出 int8_t");
    }
    const auto exponent = static_cast<std::int8_t>(selected_exponent);
    const float standard_scale = checkedScale(std::ldexp(1.0, -exponent));
    const std::int32_t zero_point =
        type.is_symmetric
            ? 0
            : computeZeroPoint(calibrated.diagnostics.adjusted_min, standard_scale, type);
    QuantParam param{standard_scale, zero_point};
    param.validate(type);
    return {param, exponent, near_power};
}

}  // namespace quant_lstm::quantization
