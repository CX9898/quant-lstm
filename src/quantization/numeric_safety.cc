#include "quantization/numeric_safety.h"

#include <limits>
#include <stdexcept>

namespace quant_lstm::quantization {
namespace {

#if defined(__SIZEOF_INT128__)
using Unsigned128 = unsigned __int128;

constexpr Unsigned128 kInt64Limit =
    static_cast<Unsigned128>(std::numeric_limits<std::int64_t>::max());
constexpr Unsigned128 kInt128Limit = (Unsigned128{1} << 127) - 1U;

std::string toDecimal(Unsigned128 value) {
    if (value == 0) {
        return "0";
    }
    std::string result;
    while (value != 0) {
        result.push_back(static_cast<char>('0' + value % 10));
        value /= 10;
    }
    return std::string(result.rbegin(), result.rend());
}

bool checkedMultiply(Unsigned128 lhs, Unsigned128 rhs, Unsigned128 limit, Unsigned128* result) {
    if (lhs != 0 && rhs > limit / lhs) {
        *result = limit;
        return false;
    }
    *result = lhs * rhs;
    return true;
}

struct BoundAnalysis {
    bool safe = false;
    Unsigned128 bound = 0;
};

BoundAnalysis analyzeInt64Product(std::uint64_t lhs, std::uint64_t rhs) {
    if (lhs > kInt64Limit || rhs > kInt64Limit) {
        return {};
    }
    BoundAnalysis analysis;
    analysis.safe = checkedMultiply(lhs, rhs, kInt64Limit, &analysis.bound);
    return analysis;
}

BoundAnalysis analyzeInt64Sum(std::uint64_t lhs, std::uint64_t rhs) {
    if (lhs > kInt64Limit || rhs > kInt64Limit || lhs > kInt64Limit - rhs) {
        return {};
    }
    return {true, static_cast<Unsigned128>(lhs) + rhs};
}

BoundAnalysis analyzeRoundedRightShift(Unsigned128 value, std::uint8_t fractional_bits) {
    if (fractional_bits > 127) {
        return {};
    }
    Unsigned128 rounded = value;
    if (fractional_bits != 0) {
        const Unsigned128 quotient = value >> fractional_bits;
        const Unsigned128 mask = (Unsigned128{1} << fractional_bits) - 1U;
        rounded = quotient + ((value & mask) != 0 ? 1U : 0U);
    }
    return {rounded <= kInt64Limit, rounded};
}

struct RescaleAnalysis {
    bool encoding_valid = false;
    bool safe = false;
    Unsigned128 result_bound = 0;
};

RescaleAnalysis analyzeRescale(const EncodedRescaleSafetyInput& input) {
    RescaleAnalysis analysis;
    if (input.encoding == SafetyRescaleEncoding::None) {
        analysis.encoding_valid =
            input.source_maximum == 0 && input.multiplier_maximum == 0 && input.shift == 0;
        analysis.safe = analysis.encoding_valid;
        return analysis;
    }
    if (input.shift < std::numeric_limits<std::int8_t>::min() ||
        input.shift > std::numeric_limits<std::int8_t>::max() || input.shift <= -127 ||
        input.source_maximum > kInt64Limit) {
        return analysis;
    }

    Unsigned128 intermediate = input.source_maximum;
    if (input.encoding == SafetyRescaleEncoding::MShift) {
        if (input.multiplier_maximum < 32768U) {
            return analysis;
        }
        analysis.encoding_valid = true;
        if (!checkedMultiply(intermediate, input.multiplier_maximum, kInt128Limit, &intermediate)) {
            return analysis;
        }
    } else if (input.encoding == SafetyRescaleEncoding::Pot2) {
        if (input.multiplier_maximum != 0) {
            return analysis;
        }
        analysis.encoding_valid = true;
    } else {
        return analysis;
    }

    if (input.shift >= 0) {
        if (input.shift == 0) {
            analysis.result_bound = intermediate;
        } else {
            const int shift = input.shift;
            const Unsigned128 quotient = intermediate >> shift;
            const Unsigned128 mask = (Unsigned128{1} << shift) - 1U;
            analysis.result_bound = quotient + ((intermediate & mask) != 0 ? 1U : 0U);
        }
    } else {
        const int left_shift = -input.shift;
        const Unsigned128 factor = Unsigned128{1} << left_shift;
        if (!checkedMultiply(intermediate, factor, kInt128Limit, &analysis.result_bound)) {
            return analysis;
        }
    }
    analysis.safe = analysis.result_bound <= kInt64Limit;
    return analysis;
}

RescaleAnalysis analyzeChainedRescale(const EncodedRescaleSafetyInput& input,
                                      Unsigned128 source_bound) {
    if (source_bound > kInt64Limit ||
        (input.source_maximum != 0 && input.source_maximum != source_bound)) {
        return {};
    }
    EncodedRescaleSafetyInput chained = input;
    chained.source_maximum = static_cast<std::uint64_t>(source_bound);
    return analyzeRescale(chained);
}

std::string formatBound(const BoundAnalysis& analysis, const char* limit) {
    return analysis.safe ? toDecimal(analysis.bound) : std::string(">") + limit;
}

std::string formatRescaleBound(const RescaleAnalysis& analysis) {
    if (!analysis.encoding_valid) {
        return "invalid_encoding";
    }
    return analysis.safe ? toDecimal(analysis.result_bound) : ">INT64_MAX";
}
#endif

}  // namespace

NumericSafetyReport buildNumericSafetyReport(const NumericSafetyInput& input) {
    NumericSafetyReport report;
#if defined(__SIZEOF_INT128__)
    Unsigned128 gemm_bound = input.gemm_reduction;
    const bool gemm_operands_safe =
        input.gemm_lhs_maximum <= kInt64Limit && input.gemm_rhs_maximum <= kInt64Limit;
    const bool gemm_representable =
        gemm_operands_safe &&
        checkedMultiply(gemm_bound, input.gemm_lhs_maximum, kInt64Limit, &gemm_bound) &&
        checkedMultiply(gemm_bound, input.gemm_rhs_maximum, kInt64Limit, &gemm_bound);
    report.int64_gemm_safe = gemm_representable && gemm_bound <= kInt64Limit;
    report.gemm_bound = report.int64_gemm_safe ? toDecimal(gemm_bound) : ">INT64_MAX";

    const RescaleAnalysis bias = analyzeRescale(input.bias_rescale);
    report.bias_rescale_safe = bias.safe;
    report.bias_rescale_bound = formatRescaleBound(bias);
    if (report.int64_gemm_safe && bias.safe && gemm_bound <= kInt64Limit - bias.result_bound) {
        report.int64_linear_accumulation_safe = true;
        report.linear_accumulation_bound = toDecimal(gemm_bound + bias.result_bound);
    } else {
        report.linear_accumulation_bound = ">INT64_MAX";
    }
    const RescaleAnalysis linear_output = analyzeChainedRescale(
        input.linear_output_rescale,
        report.int64_linear_accumulation_safe ? gemm_bound + bias.result_bound : kInt64Limit + 1U);
    report.linear_output_rescale_safe = linear_output.safe;
    report.linear_output_rescale_bound = formatRescaleBound(linear_output);
    const BoundAnalysis linear_boundary =
        linear_output.safe ? analyzeInt64Sum(static_cast<std::uint64_t>(linear_output.result_bound),
                                             input.linear_target_zero_point_maximum)
                           : BoundAnalysis{};
    report.int64_linear_boundary_safe = linear_boundary.safe;
    report.linear_boundary_bound = formatBound(linear_boundary, "INT64_MAX");

    const BoundAnalysis cell_product =
        analyzeInt64Product(input.cell_lhs_maximum, input.cell_rhs_maximum);
    report.int64_cell_product_safe = cell_product.safe;
    report.cell_product_bound = formatBound(cell_product, "INT64_MAX");

    Unsigned128 cell_q31_bound = cell_product.bound;
    const bool q31_multiplier_safe = input.q31_multiplier_maximum <= kInt64Limit;
    const bool cell_q31_representable =
        cell_product.safe && q31_multiplier_safe &&
        checkedMultiply(cell_q31_bound, input.q31_multiplier_maximum, kInt128Limit,
                        &cell_q31_bound) &&
        checkedMultiply(cell_q31_bound, 2U, kInt128Limit, &cell_q31_bound);
    report.int128_cell_safe = cell_q31_representable && cell_q31_bound <= kInt128Limit;
    report.cell_q31_bound = report.int128_cell_safe ? toDecimal(cell_q31_bound) : ">INT128_MAX";
    const BoundAnalysis cell_requantize =
        report.int128_cell_safe
            ? analyzeRoundedRightShift(cell_q31_bound, input.cell_fractional_bits)
            : BoundAnalysis{};
    report.int64_cell_requantize_safe = cell_requantize.safe;
    report.cell_requantize_bound = formatBound(cell_requantize, "INT64_MAX");
    const BoundAnalysis cell_boundary =
        cell_requantize.safe ? analyzeInt64Sum(static_cast<std::uint64_t>(cell_requantize.bound),
                                               input.cell_target_zero_point_maximum)
                             : BoundAnalysis{};
    report.int64_cell_boundary_safe = cell_boundary.safe;
    report.cell_boundary_bound = formatBound(cell_boundary, "INT64_MAX");

    const BoundAnalysis hidden_product =
        analyzeInt64Product(input.hidden_lhs_maximum, input.hidden_rhs_maximum);
    report.int64_hidden_product_safe = hidden_product.safe;
    report.hidden_product_bound = formatBound(hidden_product, "INT64_MAX");

    const RescaleAnalysis hidden = analyzeChainedRescale(
        input.hidden_rescale, hidden_product.safe ? hidden_product.bound : kInt64Limit + 1U);
    report.hidden_rescale_safe = hidden.safe;
    report.hidden_rescale_bound = formatRescaleBound(hidden);
    const BoundAnalysis hidden_boundary =
        hidden.safe ? analyzeInt64Sum(static_cast<std::uint64_t>(hidden.result_bound),
                                      input.hidden_target_zero_point_maximum)
                    : BoundAnalysis{};
    report.int64_hidden_boundary_safe = hidden_boundary.safe;
    report.hidden_boundary_bound = formatBound(hidden_boundary, "INT64_MAX");
    const RescaleAnalysis gate_lhs = analyzeRescale(input.gate_lhs_rescale);
    const RescaleAnalysis gate_rhs = analyzeRescale(input.gate_rhs_rescale);
    report.gate_lhs_rescale_safe = gate_lhs.safe;
    report.gate_rhs_rescale_safe = gate_rhs.safe;
    report.gate_lhs_rescale_bound = formatRescaleBound(gate_lhs);
    report.gate_rhs_rescale_bound = formatRescaleBound(gate_rhs);
    const BoundAnalysis gate_merge =
        gate_lhs.safe && gate_rhs.safe
            ? analyzeInt64Sum(static_cast<std::uint64_t>(gate_lhs.result_bound),
                              static_cast<std::uint64_t>(gate_rhs.result_bound))
            : BoundAnalysis{};
    report.int64_gate_merge_safe = gate_merge.safe;
    report.gate_merge_bound = formatBound(gate_merge, "INT64_MAX");
    const BoundAnalysis gate_boundary =
        gate_merge.safe ? analyzeInt64Sum(static_cast<std::uint64_t>(gate_merge.bound),
                                          input.gate_target_zero_point_maximum)
                        : BoundAnalysis{};
    report.int64_gate_boundary_safe = gate_boundary.safe;
    report.gate_boundary_bound = formatBound(gate_boundary, "INT64_MAX");
    report.encodings_valid = bias.encoding_valid && linear_output.encoding_valid &&
                             hidden.encoding_valid && gate_lhs.encoding_valid &&
                             gate_rhs.encoding_valid;
#else
    report.gemm_bound = "unsupported";
    report.bias_rescale_bound = "unsupported";
    report.linear_accumulation_bound = "unsupported";
    report.linear_output_rescale_bound = "unsupported";
    report.linear_boundary_bound = "unsupported";
    report.cell_product_bound = "unsupported";
    report.cell_q31_bound = "unsupported";
    report.cell_requantize_bound = "unsupported";
    report.cell_boundary_bound = "unsupported";
    report.hidden_product_bound = "unsupported";
    report.hidden_rescale_bound = "unsupported";
    report.hidden_boundary_bound = "unsupported";
    report.gate_lhs_rescale_bound = "unsupported";
    report.gate_rhs_rescale_bound = "unsupported";
    report.gate_merge_bound = "unsupported";
    report.gate_boundary_bound = "unsupported";
#endif

    if (input.may_produce_non_finite) {
        report.fp32_accumulation = Fp32AccumulationClass::UnsafeNonFinite;
    } else if (input.fp32_accumulator_maximum < (std::uint64_t{1} << 24U)) {
        report.fp32_accumulation = Fp32AccumulationClass::ExactIntegerRange;
    } else {
        report.fp32_accumulation = Fp32AccumulationClass::PrecisionRisk;
    }
    return report;
}

void enforceNumericSafety(const NumericSafetyReport& report, bool require_exact_accumulation) {
    if (!report.safe()) {
        throw std::overflow_error("NumericSafetyReport 无法证明执行安全");
    }
    if (require_exact_accumulation &&
        report.fp32_accumulation != Fp32AccumulationClass::ExactIntegerRange) {
        throw std::overflow_error("require_exact_accumulation 拒绝 FP32 precision_risk");
    }
}

}  // namespace quant_lstm::quantization
