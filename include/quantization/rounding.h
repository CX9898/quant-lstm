#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#if defined(__CUDACC__)
#define QUANT_LSTM_HOST_DEVICE __host__ __device__
#else
#define QUANT_LSTM_HOST_DEVICE
#endif

// 所有量化舍入均收敛到本模块；ties-to-even 不依赖当前浮点舍入模式。
namespace quant_lstm::quantization {

template <typename Float>
QUANT_LSTM_HOST_DEVICE inline Float roundToNearestEven(Float value) noexcept {
    static_assert(std::is_floating_point_v<Float>);
    using std::floor;
    using std::isfinite;
    if (!isfinite(value)) {
        return value;
    }
    const Float lower = floor(value);
    const Float fraction = value - lower;
    if (fraction < static_cast<Float>(0.5)) {
        return lower;
    }
    if (fraction > static_cast<Float>(0.5)) {
        return lower + static_cast<Float>(1);
    }
    const Float half = lower * static_cast<Float>(0.5);
    const bool lower_is_even = floor(half) == half;
    return lower_is_even ? lower : lower + static_cast<Float>(1);
}

template <typename Result, typename Float>
inline Result roundToInteger(Float value) {
    static_assert(std::is_integral_v<Result>);
    static_assert(std::is_floating_point_v<Float>);
    if (!std::isfinite(value)) {
        throw std::invalid_argument("不能将 NaN/Inf 舍入为整数");
    }
    const Float rounded = roundToNearestEven(value);
    const long double wide = static_cast<long double>(rounded);
    if (wide < static_cast<long double>(std::numeric_limits<Result>::lowest()) ||
        wide > static_cast<long double>(std::numeric_limits<Result>::max())) {
        throw std::overflow_error("舍入结果超出目标整数类型");
    }
    return static_cast<Result>(rounded);
}

inline std::int64_t roundShiftRight(std::int64_t value, int fractional_bits) {
    if (fractional_bits < 0) {
        throw std::invalid_argument("roundShiftRight 的 fractional_bits 不能为负");
    }
    if (fractional_bits == 0) {
        return value;
    }
    constexpr int width = std::numeric_limits<std::uint64_t>::digits;
    if (fractional_bits >= width) {
        return 0;
    }

    const bool negative = value < 0;
    const std::uint64_t raw = static_cast<std::uint64_t>(value);
    const std::uint64_t magnitude = negative ? std::uint64_t{0} - raw : raw;
    std::uint64_t quotient = magnitude >> fractional_bits;
    const std::uint64_t mask = (std::uint64_t{1} << fractional_bits) - 1U;
    const std::uint64_t remainder = magnitude & mask;
    const std::uint64_t half = std::uint64_t{1} << (fractional_bits - 1);
    if (remainder > half || (remainder == half && (quotient & 1U) != 0U)) {
        ++quotient;
    }

    if (!negative) {
        if (quotient > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
            throw std::overflow_error("右移舍入结果溢出 int64");
        }
        return static_cast<std::int64_t>(quotient);
    }
    const std::uint64_t minimum_magnitude =
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) + 1U;
    if (quotient == minimum_magnitude) {
        return std::numeric_limits<std::int64_t>::min();
    }
    return -static_cast<std::int64_t>(quotient);
}

#if defined(__SIZEOF_INT128__)
inline __int128 roundShiftRight128(__int128 value, int fractional_bits) {
    if (fractional_bits < 0) {
        throw std::invalid_argument("roundShiftRight128 的 fractional_bits 不能为负");
    }
    if (fractional_bits == 0) {
        return value;
    }
    if (fractional_bits >= 128) {
        return 0;
    }

    using Unsigned128 = unsigned __int128;
    const bool negative = value < 0;
    const Unsigned128 raw = static_cast<Unsigned128>(value);
    const Unsigned128 magnitude = negative ? Unsigned128{0} - raw : raw;
    Unsigned128 quotient = magnitude >> fractional_bits;
    const Unsigned128 mask = (Unsigned128{1} << fractional_bits) - 1U;
    const Unsigned128 remainder = magnitude & mask;
    const Unsigned128 half = Unsigned128{1} << (fractional_bits - 1);
    if (remainder > half || (remainder == half && (quotient & 1U) != 0U)) {
        ++quotient;
    }
    const Unsigned128 maximum = (Unsigned128{1} << 127) - 1U;
    if (!negative) {
        if (quotient > maximum) {
            throw std::overflow_error("右移舍入结果溢出 int128");
        }
        return static_cast<__int128>(quotient);
    }
    if (quotient == maximum + 1U) {
        return -static_cast<__int128>(maximum) - 1;
    }
    return -static_cast<__int128>(quotient);
}
#endif

inline std::int64_t checkedScaleByPowerOfTwo(std::int64_t value, int left_shift) {
    if (left_shift < 0 || left_shift >= 127) {
        throw std::invalid_argument("受检左移位数非法");
    }
    if (value == 0 || left_shift == 0) {
        return value;
    }
    if (left_shift >= 63) {
        if (left_shift == 63 && value == -1) {
            return std::numeric_limits<std::int64_t>::min();
        }
        throw std::overflow_error("受检左移溢出 int64");
    }
    const std::int64_t factor = std::int64_t{1} << left_shift;
    if (value > std::numeric_limits<std::int64_t>::max() / factor ||
        value < std::numeric_limits<std::int64_t>::min() / factor) {
        throw std::overflow_error("受检左移溢出 int64");
    }
    return value * factor;
}

}  // namespace quant_lstm::quantization

#undef QUANT_LSTM_HOST_DEVICE
