#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "quantization/fixed_point_ops.h"

// Cell 双比例使用的固定 Q31 编码；本模块只做宽域合并，不执行 zero-point 或 Clamp。
namespace quant_lstm::quantization {

inline constexpr int kQ31FractionalBits = 31;

struct Q31Scale {
    std::int64_t multiplier = 0;
};

inline Q31Scale encodeQ31Scale(double ratio) {
    if (!std::isfinite(ratio) || ratio < 0.0) {
        throw std::invalid_argument("Q31 ratio 必须是有限非负数");
    }
    const long double scaled = static_cast<long double>(ratio) *
                               static_cast<long double>(std::uint64_t{1} << kQ31FractionalBits);
    if (scaled > static_cast<long double>(std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error("Q31 multiplier 超出 int64");
    }
    return {roundToInteger<std::int64_t>(static_cast<double>(scaled))};
}

#if defined(__SIZEOF_INT128__)
struct Q31IntDiagnostics {
    __int128 scaled_forget = 0;
    __int128 scaled_input = 0;
    __int128 wide_sum = 0;
};

inline std::int64_t applyDualQ31(std::int64_t p_forget, Q31Scale forget_scale, std::int64_t p_input,
                                 Q31Scale input_scale, Q31IntDiagnostics* diagnostics = nullptr) {
    const __int128 scaled_forget = static_cast<__int128>(p_forget) * forget_scale.multiplier;
    const __int128 scaled_input = static_cast<__int128>(p_input) * input_scale.multiplier;
    using Unsigned128 = unsigned __int128;
    const __int128 maximum = static_cast<__int128>((Unsigned128{1} << 127) - 1U);
    const __int128 minimum = -maximum - 1;
    if ((scaled_input > 0 && scaled_forget > maximum - scaled_input) ||
        (scaled_input < 0 && scaled_forget < minimum - scaled_input)) {
        throw std::overflow_error("Q31 Cell contribution 合并溢出 int128");
    }
    const __int128 wide_sum = scaled_forget + scaled_input;
    if (diagnostics != nullptr) {
        *diagnostics = {scaled_forget, scaled_input, wide_sum};
    }
    return checkedInt128ToInt64(roundShiftRight128(wide_sum, kQ31FractionalBits));
}
#endif

struct Q31FpDiagnostics {
    float scaled_forget = 0.0F;
    float scaled_input = 0.0F;
    float wide_sum = 0.0F;
};

inline float applyDualQ31(float p_forget, Q31Scale forget_scale, float p_input,
                          Q31Scale input_scale, Q31FpDiagnostics* diagnostics = nullptr) {
    if (!std::isfinite(p_forget) || !std::isfinite(p_input)) {
        throw std::invalid_argument("FP Q31 contribution 必须有限");
    }
    const float forget_ratio =
        std::ldexp(static_cast<float>(forget_scale.multiplier), -kQ31FractionalBits);
    const float input_ratio =
        std::ldexp(static_cast<float>(input_scale.multiplier), -kQ31FractionalBits);
    const float scaled_forget = p_forget * forget_ratio;
    const float scaled_input = p_input * input_ratio;
    const float wide_sum = scaled_forget + scaled_input;
    if (!std::isfinite(wide_sum)) {
        throw std::overflow_error("FP Q31 Cell contribution 产生 Inf/NaN");
    }
    if (diagnostics != nullptr) {
        *diagnostics = {scaled_forget, scaled_input, wide_sum};
    }
    return roundToNearestEven(wide_sum);
}

}  // namespace quant_lstm::quantization
