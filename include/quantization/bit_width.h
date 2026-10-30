#pragma once

#include <cstdint>
#include <stdexcept>

// 本模块定义真实量化点的整数域；内部累加器和编码宽度不使用本类型。
namespace quant_lstm::quantization {

struct QuantizedRange {
    std::int32_t minimum;
    std::int32_t maximum;

    constexpr bool contains(std::int64_t value) const noexcept {
        return value >= minimum && value <= maximum;
    }
};

struct QuantizationType {
    std::uint8_t bitwidth = 8;
    bool is_unsigned = false;
    bool is_symmetric = true;

    static std::uint8_t checkedBitwidth(std::int64_t value) {
        if (value < 1 || value > 32) {
            throw std::invalid_argument("operator bitwidth must be in [1, 32]");
        }
        return static_cast<std::uint8_t>(value);
    }

    void validate() const {
        static_cast<void>(checkedBitwidth(bitwidth));
        if (is_unsigned && bitwidth == 32) {
            throw std::invalid_argument("UINT32 exceeds the int32 quantization metadata range");
        }
        if (!is_unsigned && is_symmetric && bitwidth == 1) {
            throw std::invalid_argument("signed symmetric INT1 has qmax=0 and is unsupported");
        }
    }

    QuantizedRange range() const {
        validate();
        if (is_unsigned) {
            return {0, static_cast<std::int32_t>((std::uint64_t{1} << bitwidth) - 1U)};
        }
        const auto maximum = static_cast<std::int32_t>((std::uint64_t{1} << (bitwidth - 1U)) - 1U);
        // Symmetric calibration fixes zero_point=0 and divides max_abs by qmax.
        // Saturation still uses the complete signed storage range, as in GRU.
        return QuantizedRange{-maximum - 1, maximum};
    }

    void validateValue(std::int64_t value) const {
        const QuantizedRange valid_range = range();
        if (!valid_range.contains(value)) {
            throw std::out_of_range("量化值超出当前 signedness/symmetry 的有效范围");
        }
    }
};

}  // namespace quant_lstm::quantization
