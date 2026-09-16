#pragma once

#include "quantization/quant_param.h"
#include "quantization/rounding.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

// CPU int32 carrier 使用的边界量化与受检执行编码。
namespace quant_lstm::quantization {

inline std::int32_t clampToRange(std::int64_t value, const QuantizationType& type) {
    const QuantizedRange range = type.range();
    return static_cast<std::int32_t>(
        std::clamp<std::int64_t>(value, range.minimum, range.maximum));
}

inline std::int32_t quantize(float value, const QuantParam& param,
                             const QuantizationType& type) {
    param.validate(type);
    if (!std::isfinite(value)) {
        throw std::invalid_argument("量化输入必须有限");
    }
    const double translated =
        static_cast<double>(value) / param.scale + static_cast<double>(param.zero_point);
    const QuantizedRange range = type.range();
    if (translated <= static_cast<double>(range.minimum)) {
        return range.minimum;
    }
    if (translated >= static_cast<double>(range.maximum)) {
        return range.maximum;
    }
    return clampToRange(roundToInteger<std::int64_t>(translated), type);
}

inline float dequantize(std::int32_t value, const QuantParam& param,
                        const QuantizationType& type) {
    param.validate(type);
    type.validateValue(value);
    const double real_value =
        static_cast<double>(static_cast<std::int64_t>(value) - param.zero_point) *
        static_cast<double>(param.scale);
    const float result = static_cast<float>(real_value);
    if (!std::isfinite(result)) {
        throw std::overflow_error("反量化结果超出有限 float32");
    }
    return result;
}

#if defined(__SIZEOF_INT128__)
inline std::int64_t checkedInt128ToInt64(__int128 value) {
    if (value < std::numeric_limits<std::int64_t>::min() ||
        value > std::numeric_limits<std::int64_t>::max()) {
        throw std::overflow_error("定点执行结果溢出 int64");
    }
    return static_cast<std::int64_t>(value);
}

inline __int128 checkedScaleInt128ByPowerOfTwo(__int128 value, int left_shift) {
    if (left_shift < 0 || left_shift >= 127) {
        throw std::invalid_argument("int128 受检左移位数非法");
    }
    using Unsigned128 = unsigned __int128;
    const Unsigned128 maximum = (Unsigned128{1} << 127) - 1U;
    const __int128 signed_maximum = static_cast<__int128>(maximum);
    const __int128 signed_minimum = -signed_maximum - 1;
    const __int128 factor = static_cast<__int128>(Unsigned128{1} << left_shift);
    if ((value > 0 && value > signed_maximum / factor) ||
        (value < 0 && value < signed_minimum / factor)) {
        throw std::overflow_error("定点负 shift 导致 int128 溢出");
    }
    return value * factor;
}
#endif

inline std::int64_t applyRescale(std::int64_t value, const FixedPointScale& encoded) {
    if (encoded.multiplier < 32768U) {
        throw std::invalid_argument("M+shift multiplier 未规范化");
    }
#if defined(__SIZEOF_INT128__)
    const __int128 product =
        static_cast<__int128>(value) * static_cast<__int128>(encoded.multiplier);
    const __int128 rounded =
        encoded.shift >= 0
            ? roundShiftRight128(product, encoded.shift)
            : checkedScaleInt128ByPowerOfTwo(product, -static_cast<int>(encoded.shift));
    return checkedInt128ToInt64(rounded);
#else
    throw std::runtime_error("当前编译器不支持阶段 2 所需的 int128");
#endif
}

inline std::int64_t applyRescale(std::int64_t value, const Pot2Rescale& encoded) {
    return encoded.shift >= 0
               ? roundShiftRight(value, encoded.shift)
               : checkedScaleByPowerOfTwo(value, -static_cast<int>(encoded.shift));
}

}  // namespace quant_lstm::quantization
