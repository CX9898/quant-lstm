#pragma once

#include <cstdint>
#include <string>

// Setup 阶段集中证明载体上界；后端只消费报告，不重复边界公式。
namespace quant_lstm::quantization {

enum class Fp32AccumulationClass : std::uint8_t {
    ExactIntegerRange,
    PrecisionRisk,
    UnsafeNonFinite,
};

enum class SafetyRescaleEncoding : std::uint8_t {
    None,
    MShift,
    Pot2,
};

struct EncodedRescaleSafetyInput {
    SafetyRescaleEncoding encoding = SafetyRescaleEncoding::None;
    std::uint64_t source_maximum = 0;
    std::uint16_t multiplier_maximum = 0;
    std::int16_t shift = 0;
};

struct NumericSafetyInput {
    std::uint64_t gemm_reduction = 0;
    std::uint64_t gemm_lhs_maximum = 0;
    std::uint64_t gemm_rhs_maximum = 0;
    EncodedRescaleSafetyInput bias_rescale;
    EncodedRescaleSafetyInput linear_output_rescale;
    std::uint64_t linear_target_zero_point_maximum = 0;
    std::uint64_t cell_lhs_maximum = 0;
    std::uint64_t cell_rhs_maximum = 0;
    std::uint64_t q31_multiplier_maximum = 0;
    std::uint8_t cell_fractional_bits = 31;
    std::uint64_t cell_target_zero_point_maximum = 0;
    std::uint64_t hidden_lhs_maximum = 0;
    std::uint64_t hidden_rhs_maximum = 0;
    EncodedRescaleSafetyInput hidden_rescale;
    std::uint64_t hidden_target_zero_point_maximum = 0;
    EncodedRescaleSafetyInput gate_lhs_rescale;
    EncodedRescaleSafetyInput gate_rhs_rescale;
    std::uint64_t gate_target_zero_point_maximum = 0;
    std::uint64_t fp32_accumulator_maximum = 0;
    bool may_produce_non_finite = false;
};

struct NumericSafetyReport {
    bool int64_gemm_safe = false;
    bool bias_rescale_safe = false;
    bool int64_linear_accumulation_safe = false;
    bool linear_output_rescale_safe = false;
    bool int64_linear_boundary_safe = false;
    bool int64_cell_product_safe = false;
    bool int128_cell_safe = false;
    bool int64_cell_requantize_safe = false;
    bool int64_cell_boundary_safe = false;
    bool int64_hidden_product_safe = false;
    bool hidden_rescale_safe = false;
    bool int64_hidden_boundary_safe = false;
    bool gate_lhs_rescale_safe = false;
    bool gate_rhs_rescale_safe = false;
    bool int64_gate_merge_safe = false;
    bool int64_gate_boundary_safe = false;
    bool encodings_valid = false;
    Fp32AccumulationClass fp32_accumulation = Fp32AccumulationClass::UnsafeNonFinite;
    std::string gemm_bound;
    std::string bias_rescale_bound;
    std::string linear_accumulation_bound;
    std::string linear_output_rescale_bound;
    std::string linear_boundary_bound;
    std::string cell_product_bound;
    std::string cell_q31_bound;
    std::string cell_requantize_bound;
    std::string cell_boundary_bound;
    std::string hidden_product_bound;
    std::string hidden_rescale_bound;
    std::string hidden_boundary_bound;
    std::string gate_lhs_rescale_bound;
    std::string gate_rhs_rescale_bound;
    std::string gate_merge_bound;
    std::string gate_boundary_bound;

    bool safe() const noexcept {
        return int64_gemm_safe && bias_rescale_safe &&
               int64_linear_accumulation_safe && linear_output_rescale_safe &&
               int64_linear_boundary_safe && int64_cell_product_safe &&
               int128_cell_safe && int64_cell_requantize_safe &&
               int64_cell_boundary_safe && int64_hidden_product_safe &&
               hidden_rescale_safe && int64_hidden_boundary_safe &&
               gate_lhs_rescale_safe &&
               gate_rhs_rescale_safe && int64_gate_merge_safe &&
               int64_gate_boundary_safe && encodings_valid &&
               fp32_accumulation != Fp32AccumulationClass::UnsafeNonFinite;
    }
};

NumericSafetyReport buildNumericSafetyReport(const NumericSafetyInput& input);

void enforceNumericSafety(const NumericSafetyReport& report,
                          bool require_exact_accumulation);

}  // namespace quant_lstm::quantization
