#include "quantization/bit_width.h"
#include "quantization/fixed_point_ops.h"
#include "quantization/float_carrier_ops.h"
#include "quantization/numeric_safety.h"
#include "quantization/real_activation.h"
#include "quantization/rounding.h"
#include "quantization/scale_encoding.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void requireThrows(Function&& function, const char* message) {
    bool threw = false;
    try {
        function();
    } catch (const std::exception&) {
        threw = true;
    }
    require(threw, message);
}

}  // namespace

int main() {
    namespace q = quant_lstm::quantization;
    try {
        require(q::roundToNearestEven(0.5) == 0.0, "0.5 ties-to-even");
        require(q::roundToNearestEven(1.5) == 2.0, "1.5 ties-to-even");
        require(q::roundToNearestEven(2.5) == 2.0, "2.5 ties-to-even");
        require(q::roundToNearestEven(-1.5) == -2.0, "-1.5 ties-to-even");
        require(q::roundShiftRight(7, 1) == 4, "7/2 ties-to-even");
        require(q::roundShiftRight(5, 1) == 2, "5/2 ties-to-even");
        require(q::roundShiftRight(-7, 1) == -4, "-7/2 ties-to-even");
        require(q::roundShiftRight(std::numeric_limits<std::int64_t>::min(), 1) ==
                    std::numeric_limits<std::int64_t>::min() / 2,
                "INT64_MIN right shift");
        require(q::roundShiftRight(std::numeric_limits<std::int64_t>::max(), 64) == 0,
                "wide right shift");
        requireThrows([] { static_cast<void>(q::roundShiftRight(1, -1)); },
                      "negative right shift");
        requireThrows(
            [] {
                static_cast<void>(q::checkedScaleByPowerOfTwo(
                    std::numeric_limits<std::int64_t>::max(), 1));
            },
            "checked left shift overflow");
        require(q::checkedScaleByPowerOfTwo(-1, 63) ==
                    std::numeric_limits<std::int64_t>::min() &&
                    q::checkedScaleByPowerOfTwo(0, 126) == 0,
                "checked left shift signed boundaries");
        requireThrows([] { static_cast<void>(q::checkedScaleByPowerOfTwo(1, 63)); },
                      "checked left shift wide positive overflow");
        requireThrows([] { static_cast<void>(q::checkedScaleByPowerOfTwo(0, 127)); },
                      "checked left shift invalid width");

        const q::QuantizationType signed_symmetric_8{8, false, true};
        const q::QuantizationType signed_asymmetric_8{8, false, false};
        const q::QuantizationType unsigned_symmetric_8{8, true, true};
        require(signed_symmetric_8.range().minimum == -127 &&
                    signed_symmetric_8.range().maximum == 127,
                "signed symmetric INT8 range");
        require(signed_asymmetric_8.range().minimum == -128 &&
                    signed_asymmetric_8.range().maximum == 127,
                "signed asymmetric INT8 range");
        require(unsigned_symmetric_8.range().minimum == 0 &&
                    unsigned_symmetric_8.range().maximum == 255,
                "unsigned INT8 range");
        requireThrows([&] { signed_symmetric_8.validateValue(-128); },
                      "symmetric -128 must be rejected");
        signed_asymmetric_8.validateValue(-128);

        const auto signed_calibration = q::calibrateMinMax(-12.7F, 12.7F, signed_symmetric_8);
        require(std::abs(signed_calibration.param.scale - 0.1F) < 1.0e-7F,
                "signed symmetric scale");
        require(q::quantize(12.7F, signed_calibration.param, signed_symmetric_8) == 127,
                "signed quantize boundary");

        const auto unsigned_calibration =
            q::calibrateMinMax(-2.0F, 1.0F, unsigned_symmetric_8);
        require(q::quantize(-1.0F, unsigned_calibration.param, unsigned_symmetric_8) == 0,
                "unsigned negative clamp");

        const auto fallback = q::calibrateMinMax(0.0F, 0.0F, signed_symmetric_8);
        require(fallback.diagnostics.fallback_used &&
                    fallback.param.scale == q::minimumScale(signed_symmetric_8),
                "degenerate fallback");
        for (const q::QuantizationType type :
             {q::QuantizationType{8, false, true}, q::QuantizationType{8, true, true},
              q::QuantizationType{8, false, false}, q::QuantizationType{8, true, false}}) {
            const auto mode_fallback = q::calibrateMinMax(0.0F, 0.0F, type);
            require(mode_fallback.diagnostics.fallback_used &&
                        mode_fallback.param.scale == q::minimumScale(type),
                    "all four fallback modes");
        }
        require(q::calibrateMinMax(1.0F, 2.0F, signed_asymmetric_8).param.zero_point == -128,
                "positive-only signed asymmetric zero point");
        require(q::calibrateMinMax(-2.0F, -1.0F, signed_asymmetric_8).param.zero_point == 127,
                "negative-only signed asymmetric zero point");
        const q::QuantizationType unsigned_asymmetric_8{8, true, false};
        require(q::calibrateMinMax(1.0F, 2.0F, unsigned_asymmetric_8).param.zero_point == 0,
                "positive-only unsigned asymmetric zero point");
        require(q::calibrateMinMax(-2.0F, -1.0F, unsigned_asymmetric_8).param.zero_point ==
                    255,
                "negative-only unsigned asymmetric zero point");
        require(q::quantize(std::numeric_limits<float>::max(),
                            signed_calibration.param, signed_symmetric_8) == 127,
                "large finite quantize clamp");
        requireThrows(
            [&] {
                static_cast<void>(q::quantize(std::numeric_limits<float>::infinity(),
                                              signed_calibration.param,
                                              signed_symmetric_8));
            },
            "non-finite quantize input");
        requireThrows(
            [] {
                static_cast<void>(q::dequantize(
                    65535, q::QuantParam{std::numeric_limits<float>::max(), 0},
                    q::QuantizationType{16, true, false}));
            },
            "non-finite dequantize result");

        const q::FixedPointScale ratio_one = q::encodeMShift(1.0);
        require(ratio_one.multiplier == 32768 && ratio_one.shift == 15,
                "ratio one M+shift");
        const q::FixedPointScale ratio_half = q::encodeMShift(0.5);
        require(ratio_half.multiplier == 32768 && ratio_half.shift == 16,
                "ratio half M+shift");
        const q::FixedPointScale even_tie = q::encodeMShift(32768.5 / 65536.0);
        require(even_tie.multiplier == 32768, "M+shift even half tie");
        const q::FixedPointScale normalize_tie = q::encodeMShift(65535.5 / 65536.0);
        require(normalize_tie.multiplier == 32768 && normalize_tie.shift == 15,
                "M+shift 65536 normalization");
        require(q::applyRescale(std::int64_t{7}, ratio_half) == 4,
                "integer M+shift apply");
        require(q::applyRescale(7.0F, ratio_half) == 4.0F, "FP M+shift apply");
        require(q::applyRescale(std::int64_t{5}, ratio_half) == 2 &&
                    q::applyRescale(std::int64_t{-5}, ratio_half) == -2 &&
                    q::applyRescale(-5.0F, ratio_half) == -2.0F,
                "M+shift signed even half ties");
        requireThrows([] { static_cast<void>(q::encodeMShift(0.0)); },
                      "zero M+shift ratio");
        requireThrows(
            [] { static_cast<void>(q::encodeMShift(std::numeric_limits<double>::infinity())); },
            "infinite M+shift ratio");
        requireThrows(
            [] { static_cast<void>(q::encodeMShift(std::ldexp(1.0, -200))); },
            "M+shift positive shift overflow");
        requireThrows(
            [] { static_cast<void>(q::encodeMShift(std::ldexp(1.0, 200))); },
            "M+shift negative shift overflow");
        requireThrows(
            [] {
                static_cast<void>(
                    q::applyRescale(std::int64_t{1}, q::FixedPointScale{32767, 1}));
            },
            "unnormalized M+shift apply");
        requireThrows(
            [] {
                static_cast<void>(q::applyRescale(
                    std::numeric_limits<std::int64_t>::max(), q::Pot2Rescale{-1}));
            },
            "POT2 left shift overflow");
        const auto pot2_up = q::encodePot2Rescale(0.25F, 0.125F);
        const auto pot2_down = q::encodePot2Rescale(0.125F, 0.25F);
        require(pot2_up.shift == -1 && q::applyRescale(std::int64_t{3}, pot2_up) == 6,
                "POT2 rescale up");
        require(pot2_down.shift == 1 && q::applyRescale(std::int64_t{3}, pot2_down) == 2,
                "POT2 rescale down ties-to-even");
        requireThrows(
            [] { static_cast<void>(q::encodePot2Rescale(0.1F, 0.25F)); },
            "POT2 rescale non-power scale");
        requireThrows(
            [] {
                static_cast<void>(q::encodePot2Rescale(std::ldexp(1.0F, -127),
                                                       std::ldexp(1.0F, 127)));
            },
            "POT2 rescale shift overflow");

        const auto pot_calibration = q::calibrateMinMax(-1.0F, 1.0F, signed_symmetric_8);
        const auto pot = q::convertScaleToPot2CoverRange(pot_calibration, signed_symmetric_8);
        require(pot.param.scale > 0.0F && pot.param.zero_point == 0,
                "POT2 symmetric result");

        const auto positive_asymmetric =
            q::calibrateMinMax(1.0F, 2.0F, signed_asymmetric_8);
        const auto positive_pot =
            q::convertScaleToPot2CoverRange(positive_asymmetric, signed_asymmetric_8);
        require(positive_pot.param.zero_point == signed_asymmetric_8.range().minimum,
                "POT2 asymmetric must use zero-including r_lo");
        const float tolerance_boundary = 1.02F;
        const auto inside_tolerance = q::convertScaleToPot2CoverRange(
            q::calibrateMinMax(0.0F,
                               std::nextafter(tolerance_boundary, 0.0F),
                               signed_symmetric_8),
            signed_symmetric_8);
        const auto outside_tolerance = q::convertScaleToPot2CoverRange(
            q::calibrateMinMax(
                0.0F, std::nextafter(tolerance_boundary,
                                     std::numeric_limits<float>::infinity()),
                signed_symmetric_8),
            signed_symmetric_8);
        require(inside_tolerance.range_is_near_power_of_two &&
                    !outside_tolerance.range_is_near_power_of_two,
                "POT2 strict 2% boundary neighborhood");
        auto exponent_overflow = fallback;
        exponent_overflow.param.scale = std::numeric_limits<float>::denorm_min();
        exponent_overflow.diagnostics.adjusted_min = 0.0F;
        exponent_overflow.diagnostics.adjusted_max = 1.0F;
        requireThrows(
            [&] {
                static_cast<void>(
                    q::convertScaleToPot2CoverRange(exponent_overflow,
                                                   signed_symmetric_8));
            },
            "POT2 exponent overflow");

        const q::QuantParam activation_input{0.1F, 0};
        const q::QuantParam sigmoid_output{1.0F / 256.0F, 0};
        const auto sigmoid_q =
            q::realActivation(0, activation_input, signed_symmetric_8, sigmoid_output,
                              unsigned_symmetric_8, q::RealActivationKind::Sigmoid);
        require(sigmoid_q == 128, "real sigmoid activation");
        require(q::realActivation(0.0F, activation_input, signed_symmetric_8, sigmoid_output,
                                  unsigned_symmetric_8, q::RealActivationKind::Sigmoid) ==
                    static_cast<float>(sigmoid_q),
                "FP/int activation boundary");

        q::NumericSafetyInput safety_input;
        safety_input.gemm_reduction = 16;
        safety_input.gemm_lhs_maximum = 127;
        safety_input.gemm_rhs_maximum = 127;
        safety_input.bias_rescale = {
            q::SafetyRescaleEncoding::MShift, 127, 32768, 15};
        safety_input.linear_output_rescale = {
            q::SafetyRescaleEncoding::Pot2, 0, 0, 0};
        safety_input.cell_lhs_maximum = 32767;
        safety_input.cell_rhs_maximum = 32767;
        safety_input.q31_multiplier_maximum = std::uint64_t{1} << 31U;
        safety_input.hidden_lhs_maximum = 255;
        safety_input.hidden_rhs_maximum = 32767;
        safety_input.hidden_rescale = {
            q::SafetyRescaleEncoding::Pot2, 255U * 32767U, 0, 1};
        safety_input.gate_lhs_rescale = {
            q::SafetyRescaleEncoding::Pot2, 127, 0, 0};
        safety_input.gate_rhs_rescale = {
            q::SafetyRescaleEncoding::Pot2, 127, 0, 0};
        safety_input.fp32_accumulator_maximum = (std::uint64_t{1} << 24U) - 1U;
        const auto safe_report = q::buildNumericSafetyReport(safety_input);
        require(safe_report.safe() &&
                    safe_report.fp32_accumulation == q::Fp32AccumulationClass::ExactIntegerRange,
                "numeric safety exact range");
        q::enforceNumericSafety(safe_report, true);
        auto linear_boundary_input = safety_input;
        linear_boundary_input.gemm_reduction =
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
        linear_boundary_input.gemm_lhs_maximum = 1;
        linear_boundary_input.gemm_rhs_maximum = 1;
        linear_boundary_input.bias_rescale = {};
        linear_boundary_input.linear_target_zero_point_maximum = 1;
        const auto linear_boundary_report =
            q::buildNumericSafetyReport(linear_boundary_input);
        require(linear_boundary_report.linear_output_rescale_safe &&
                    !linear_boundary_report.int64_linear_boundary_safe,
                "unsafe Linear zero-point addition");
        auto hidden_boundary_input = safety_input;
        hidden_boundary_input.hidden_lhs_maximum =
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
        hidden_boundary_input.hidden_rhs_maximum = 1;
        hidden_boundary_input.hidden_rescale = {
            q::SafetyRescaleEncoding::Pot2, 0, 0, 0};
        hidden_boundary_input.hidden_target_zero_point_maximum = 1;
        const auto hidden_boundary_report =
            q::buildNumericSafetyReport(hidden_boundary_input);
        require(hidden_boundary_report.hidden_rescale_safe &&
                    !hidden_boundary_report.int64_hidden_boundary_safe,
                "unsafe Hidden zero-point addition");
        auto cell_boundary_input = safety_input;
        cell_boundary_input.cell_lhs_maximum =
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
        cell_boundary_input.cell_rhs_maximum = 1;
        cell_boundary_input.q31_multiplier_maximum = 1;
        cell_boundary_input.cell_fractional_bits = 1;
        cell_boundary_input.cell_target_zero_point_maximum = 1;
        const auto cell_boundary_report =
            q::buildNumericSafetyReport(cell_boundary_input);
        require(cell_boundary_report.int64_cell_requantize_safe &&
                    !cell_boundary_report.int64_cell_boundary_safe,
                "unsafe Cell zero-point addition");
        safety_input.fp32_accumulator_maximum = std::uint64_t{1} << 24U;
        const auto risk_report = q::buildNumericSafetyReport(safety_input);
        require(risk_report.fp32_accumulation == q::Fp32AccumulationClass::PrecisionRisk,
                "numeric safety precision risk");
        requireThrows([&] { q::enforceNumericSafety(risk_report, true); },
                      "exact mode must reject precision risk");
        safety_input.gemm_reduction = std::numeric_limits<std::uint64_t>::max();
        const auto unsafe_integer_report = q::buildNumericSafetyReport(safety_input);
        require(!unsafe_integer_report.int64_gemm_safe, "unsafe int64 GEMM bound");
        safety_input.gemm_reduction = 1;
        safety_input.cell_lhs_maximum = std::numeric_limits<std::uint64_t>::max();
        safety_input.cell_rhs_maximum = 2;
        safety_input.q31_multiplier_maximum = std::numeric_limits<std::uint64_t>::max();
        const auto unsafe_cell_report = q::buildNumericSafetyReport(safety_input);
        require(!unsafe_cell_report.int64_cell_product_safe &&
                    !unsafe_cell_report.int128_cell_safe,
                "unsafe Cell intermediate bounds");
        safety_input.cell_lhs_maximum = 1;
        safety_input.cell_rhs_maximum = 1;
        safety_input.q31_multiplier_maximum = 1;
        safety_input.hidden_rescale = {
            q::SafetyRescaleEncoding::MShift, 1, 1, 0};
        const auto invalid_encoding_report = q::buildNumericSafetyReport(safety_input);
        require(!invalid_encoding_report.encodings_valid &&
                    !invalid_encoding_report.hidden_rescale_safe,
                "invalid M+shift safety encoding");
        safety_input.hidden_rescale = {
            q::SafetyRescaleEncoding::Pot2, 1, 0, 0};
        safety_input.gemm_reduction =
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
        safety_input.gemm_lhs_maximum = 1;
        safety_input.gemm_rhs_maximum = 1;
        safety_input.bias_rescale = {
            q::SafetyRescaleEncoding::Pot2, 1, 0, 0};
        const auto unsafe_linear_sum_report =
            q::buildNumericSafetyReport(safety_input);
        require(unsafe_linear_sum_report.int64_gemm_safe &&
                    unsafe_linear_sum_report.bias_rescale_safe &&
                    !unsafe_linear_sum_report.int64_linear_accumulation_safe,
                "unsafe GEMM plus bias accumulation");
        safety_input.gate_lhs_rescale = {
            q::SafetyRescaleEncoding::Pot2,
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()),
            0,
            0};
        safety_input.gate_rhs_rescale = {
            q::SafetyRescaleEncoding::Pot2, 1, 0, 0};
        const auto unsafe_gate_merge_report =
            q::buildNumericSafetyReport(safety_input);
        require(!unsafe_gate_merge_report.int64_gate_merge_safe,
                "unsafe gate merge accumulation");
        safety_input.gate_rhs_rescale = {};
        safety_input.gate_target_zero_point_maximum = 1;
        const auto unsafe_gate_boundary_report =
            q::buildNumericSafetyReport(safety_input);
        require(unsafe_gate_boundary_report.int64_gate_merge_safe &&
                    !unsafe_gate_boundary_report.int64_gate_boundary_safe,
                "unsafe gate zero-point addition");
        safety_input.gate_lhs_rescale = {
            q::SafetyRescaleEncoding::Pot2, 1, 0, 0};
        safety_input.gate_rhs_rescale = {
            q::SafetyRescaleEncoding::Pot2, 1, 0, 0};
        safety_input.gate_target_zero_point_maximum = 0;
        safety_input.bias_rescale = {
            q::SafetyRescaleEncoding::MShift,
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()),
            65535,
            -126};
        const auto unsafe_rescale_report = q::buildNumericSafetyReport(safety_input);
        require(!unsafe_rescale_report.bias_rescale_safe,
                "unsafe bias rescale bound");
        safety_input.bias_rescale = {
            q::SafetyRescaleEncoding::MShift, 1, 32768, -127};
        require(!q::buildNumericSafetyReport(safety_input).encodings_valid,
                "left shift 127 must match execution rejection");
        safety_input.bias_rescale = {
            q::SafetyRescaleEncoding::MShift, 1, 32768, -128};
        require(!q::buildNumericSafetyReport(safety_input).encodings_valid,
                "left shift 128 must fail without undefined behavior");
        safety_input.bias_rescale = {};
        safety_input.may_produce_non_finite = true;
        const auto non_finite_report = q::buildNumericSafetyReport(safety_input);
        require(!non_finite_report.safe(), "unsafe FP32 non-finite report");
        requireThrows([&] { q::enforceNumericSafety(non_finite_report, false); },
                      "unsafe report must fail fast");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
