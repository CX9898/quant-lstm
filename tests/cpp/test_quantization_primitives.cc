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
        requireThrows(
            [] {
                static_cast<void>(q::checkedScaleByPowerOfTwo(
                    std::numeric_limits<std::int64_t>::max(), 1));
            },
            "checked left shift overflow");

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
        requireThrows([] { static_cast<void>(q::encodeMShift(0.0)); },
                      "zero M+shift ratio");
        requireThrows(
            [] { static_cast<void>(q::encodeMShift(std::numeric_limits<double>::infinity())); },
            "infinite M+shift ratio");

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
        safety_input.cell_product_maximum = 32767U * 32767U;
        safety_input.q31_multiplier_maximum = std::uint64_t{1} << 31U;
        safety_input.fp32_accumulator_maximum = (std::uint64_t{1} << 24U) - 1U;
        const auto safe_report = q::buildNumericSafetyReport(safety_input);
        require(safe_report.safe() &&
                    safe_report.fp32_accumulation == q::Fp32AccumulationClass::ExactIntegerRange,
                "numeric safety exact range");
        q::enforceNumericSafety(safe_report, true);
        safety_input.fp32_accumulator_maximum = std::uint64_t{1} << 24U;
        const auto risk_report = q::buildNumericSafetyReport(safety_input);
        require(risk_report.fp32_accumulation == q::Fp32AccumulationClass::PrecisionRisk,
                "numeric safety precision risk");
        requireThrows([&] { q::enforceNumericSafety(risk_report, true); },
                      "exact mode must reject precision risk");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
