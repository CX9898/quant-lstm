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

    void validate() const {
        if (bitwidth != 8 && bitwidth != 16) {
            throw std::invalid_argument("operator bitwidth 只允许 8 或 16");
        }
    }

    QuantizedRange range() const {
        validate();
        if (is_unsigned) {
            return {0, static_cast<std::int32_t>((std::uint32_t{1} << bitwidth) - 1U)};
        }
        const auto maximum =
            static_cast<std::int32_t>((std::uint32_t{1} << (bitwidth - 1U)) - 1U);
        return is_symmetric ? QuantizedRange{-maximum, maximum}
                            : QuantizedRange{-maximum - 1, maximum};
    }

    void validateValue(std::int64_t value) const {
        const QuantizedRange valid_range = range();
        if (!valid_range.contains(value)) {
            throw std::out_of_range("量化值超出当前 signedness/symmetry 的有效范围");
        }
    }
};

}  // namespace quant_lstm::quantization
