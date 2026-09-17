#include "common/deterministic_rng.h"
#include "common/numeric_metrics.h"
#include "lstm/quantized_cell_fp.h"
#include "lstm/quantized_cell_int.h"
#include "quantization/q31_ops.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace q = quant_lstm::quantization;

static_assert(q::kQ31FractionalBits == 31, "底层 Q31 小数位契约漂移");
static_assert(quant_lstm::Q31Scale::kFractionalBits == 31,
              "Cell Q31 小数位契约漂移");
static_assert(
    std::is_same_v<decltype(q::Q31Scale::multiplier), std::int64_t> &&
        std::is_same_v<decltype(quant_lstm::Q31Scale::multiplier),
                       std::int64_t>,
    "Q31 multiplier 必须保持有符号 64-bit");
static_assert(std::numeric_limits<float>::radix == 2 &&
                  std::numeric_limits<float>::digits == 24,
              "FP carrier 验证要求 IEEE 风格 binary32 精度");

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Exception, typename Function>
void requireThrowsAs(Function&& function, const char* message) {
    try {
        function();
    } catch (const Exception&) {
        return;
    } catch (const std::exception&) {
        throw std::runtime_error(std::string(message) + "（异常类型错误）");
    }
    throw std::runtime_error(std::string(message) + "（未抛异常）");
}

#if defined(__SIZEOF_INT128__)

// 独立 oracle：只使用整数位运算实现 Q31 ties-to-even，不调用生产舍入。
__int128 oracleRoundShift31(__int128 value) {
    using U128 = unsigned __int128;
    constexpr int fractional_bits = 31;
    const bool negative = value < 0;
    const U128 raw = static_cast<U128>(value);
    const U128 magnitude = negative ? U128{0} - raw : raw;
    U128 quotient = magnitude >> fractional_bits;
    const U128 remainder =
        magnitude & ((U128{1} << fractional_bits) - U128{1});
    const U128 half = U128{1} << (fractional_bits - 1);
    if (remainder > half ||
        (remainder == half && (quotient & U128{1}) != 0)) {
        ++quotient;
    }
    return negative ? -static_cast<__int128>(quotient)
                    : static_cast<__int128>(quotient);
}

std::int64_t oracleDualQ31(std::int64_t p_forget, std::int64_t m_forget,
                           std::int64_t p_input, std::int64_t m_input) {
    const __int128 numerator =
        static_cast<__int128>(p_forget) * m_forget +
        static_cast<__int128>(p_input) * m_input;
    const __int128 rounded = oracleRoundShift31(numerator);
    if (rounded < std::numeric_limits<std::int64_t>::min() ||
        rounded > std::numeric_limits<std::int64_t>::max()) {
        throw std::overflow_error("oracle Q31 结果超出 int64");
    }
    return static_cast<std::int64_t>(rounded);
}

std::pair<std::int64_t, std::int64_t> oracleRange(
    const q::QuantizationType& type) {
    require(type.bitwidth == 8 || type.bitwidth == 16,
            "oracle 仅支持 8/16 bit");
    if (type.is_unsigned) {
        return {0, (std::int64_t{1} << type.bitwidth) - 1};
    }
    const std::int64_t maximum =
        (std::int64_t{1} << (type.bitwidth - 1U)) - 1;
    return {type.is_symmetric ? -maximum : -maximum - 1, maximum};
}

std::int32_t oracleCell(std::int32_t forget_gate, std::int32_t old_cell,
                        std::int32_t input_gate, std::int32_t cell_gate,
                        const quant_lstm::CellExecutionParams& params) {
    const __int128 p_forget =
        static_cast<__int128>(
            static_cast<std::int64_t>(forget_gate) -
            params.forget_gate.param.zero_point) *
        (static_cast<std::int64_t>(old_cell) -
         params.old_cell.param.zero_point);
    const __int128 p_input =
        static_cast<__int128>(
            static_cast<std::int64_t>(input_gate) -
            params.input_gate.param.zero_point) *
        (static_cast<std::int64_t>(cell_gate) -
         params.cell_gate.param.zero_point);
    const __int128 centered = oracleRoundShift31(
        p_forget * params.forget_scale.multiplier +
        p_input * params.input_scale.multiplier);
    const __int128 translated = centered + params.new_cell.param.zero_point;
    const auto [minimum, maximum] = oracleRange(params.new_cell.type);
    if (translated <= minimum) {
        return static_cast<std::int32_t>(minimum);
    }
    if (translated >= maximum) {
        return static_cast<std::int32_t>(maximum);
    }
    return static_cast<std::int32_t>(translated);
}

quant_lstm::QuantizedPoint makePoint(q::QuantizationType type, float scale,
                                     std::int32_t zero_point) {
    return {{scale, zero_point}, type};
}

quant_lstm::CellExecutionParams makeCellParams(std::uint8_t bitwidth,
                                                q::ScaleMode mode) {
    const bool affine = mode == q::ScaleMode::Affine;
    const q::QuantizationType gate_type{bitwidth, true, !affine};
    const q::QuantizationType signed_type{bitwidth, false, !affine};
    const std::int32_t signed_zero_point =
        affine ? (bitwidth == 8 ? -3 : -257) : 0;
    const std::int32_t output_zero_point =
        affine ? (bitwidth == 8 ? 5 : 1021) : 0;
    const float gate_scale =
        affine ? (bitwidth == 8 ? 0.00391F : 0.0000153F)
               : std::ldexp(1.0F, bitwidth == 8 ? -8 : -16);
    const float state_scale =
        affine ? (bitwidth == 8 ? 0.017F : 0.000071F)
               : std::ldexp(1.0F, bitwidth == 8 ? -6 : -12);
    quant_lstm::CellExecutionParams params;
    params.forget_gate = makePoint(gate_type, gate_scale, 0);
    params.input_gate = makePoint(gate_type, gate_scale, 0);
    params.old_cell =
        makePoint(signed_type, state_scale, signed_zero_point);
    params.cell_gate =
        makePoint(signed_type, state_scale, signed_zero_point);
    params.new_cell =
        makePoint(signed_type, state_scale, output_zero_point);
    return params;
}

std::int32_t qFromCentered(std::int64_t centered,
                           const quant_lstm::QuantizedPoint& point) {
    const std::int64_t value = centered + point.param.zero_point;
    const auto [minimum, maximum] = oracleRange(point.type);
    require(value >= minimum && value <= maximum,
            "定向样本超出量化范围");
    return static_cast<std::int32_t>(value);
}

std::int32_t sampleQuantized(quant_lstm::test::Pcg32& rng,
                             const quant_lstm::QuantizedPoint& point) {
    const auto [minimum, maximum] = oracleRange(point.type);
    const std::uint64_t width =
        static_cast<std::uint64_t>(maximum - minimum + 1);
    return static_cast<std::int32_t>(
        minimum + static_cast<std::int64_t>(rng.nextUint32() % width));
}

struct CellInputs {
    std::int32_t forget_gate;
    std::int32_t old_cell;
    std::int32_t input_gate;
    std::int32_t cell_gate;
};

struct ScenarioResult {
    std::uint8_t bitwidth = 0;
    q::ScaleMode mode = q::ScaleMode::Affine;
    std::size_t samples = 0;
    std::uint64_t int_mismatches = 0;
    std::uint64_t int_saturations = 0;
    std::uint64_t fp_saturations = 0;
    quant_lstm::test::NumericMetrics int_metrics;
    quant_lstm::test::NumericMetrics fp_metrics;
};

CellInputs directedInputs(std::size_t index,
                          const quant_lstm::CellExecutionParams& params) {
    const std::int64_t magnitude =
        params.new_cell.type.bitwidth == 8 ? 96 : 30000;
    const auto gate_maximum = oracleRange(params.forget_gate.type).second;
    const std::int64_t gate =
        index == 0 ? gate_maximum / 2 : gate_maximum;
    switch (index) {
        case 0:  // 精确抵消
        case 4:  // 极端比例下的异号贡献
            return {qFromCentered(gate, params.forget_gate),
                    qFromCentered(magnitude, params.old_cell),
                    qFromCentered(gate, params.input_gate),
                    qFromCentered(-magnitude, params.cell_gate)};
        case 1:  // 同号累加
        case 2:  // 正饱和
            return {qFromCentered(gate, params.forget_gate),
                    qFromCentered(magnitude, params.old_cell),
                    qFromCentered(gate, params.input_gate),
                    qFromCentered(magnitude, params.cell_gate)};
        default:  // 负饱和
            return {qFromCentered(gate, params.forget_gate),
                    qFromCentered(-magnitude, params.old_cell),
                    qFromCentered(gate, params.input_gate),
                    qFromCentered(-magnitude, params.cell_gate)};
    }
}

std::pair<std::int64_t, std::int64_t> multipliersFor(
    std::size_t index, std::uint8_t bitwidth, q::ScaleMode mode,
    quant_lstm::test::Pcg32& rng) {
    constexpr std::int64_t one = std::int64_t{1} << 31;
    const std::int64_t natural = one >> bitwidth;
    if (index == 0) {
        return {natural, natural};
    }
    if (index == 1) {
        return {natural, natural};
    }
    if (index == 2 || index == 3) {
        return {one << 3, one << 3};
    }
    if (index == 4) {
        return {1, one << 3};
    }
    const std::array<std::int64_t, 9> affine{
        0,
        1,
        natural / 8 + 3,
        natural / 3,
        natural / 2 + 1,
        natural - natural / 7,
        natural,
        natural + natural / 3,
        natural * 4 + 1};
    const std::array<std::int64_t, 9> pot2{
        0,           1,           natural / 8,
        natural / 4, natural / 2, natural,
        natural * 2, natural * 4, natural * 8};
    const auto& pool = mode == q::ScaleMode::Affine ? affine : pot2;
    return {pool[rng.nextUint32() % pool.size()],
            pool[rng.nextUint32() % pool.size()]};
}

bool cosinePasses(const quant_lstm::test::NumericMetrics& metrics,
                  double threshold) {
    return metrics.cosine_not_applicable ||
           (!metrics.one_sided_zero_norm &&
            metrics.cosine_similarity >= threshold);
}

ScenarioResult runScenario(std::uint8_t bitwidth, q::ScaleMode mode) {
    constexpr std::size_t sample_count = 8192;
    auto params = makeCellParams(bitwidth, mode);
    quant_lstm::test::Pcg32 rng(
        0x51315EEDULL + bitwidth +
            (mode == q::ScaleMode::Pot2 ? 0x10000ULL : 0ULL),
        0x31U + bitwidth);
    std::vector<std::int32_t> expected(sample_count);
    std::vector<std::int32_t> int_actual(sample_count);
    std::vector<float> fp_actual(sample_count);
    std::int32_t oracle_state = params.new_cell.param.zero_point;
    std::int32_t int_state = oracle_state;
    float fp_state = static_cast<float>(oracle_state);
    const auto [output_minimum, output_maximum] =
        oracleRange(params.new_cell.type);

    ScenarioResult result;
    result.bitwidth = bitwidth;
    result.mode = mode;
    result.samples = sample_count;
    for (std::size_t index = 0; index < sample_count; ++index) {
        const CellInputs inputs =
            index < 5
                ? directedInputs(index, params)
                : CellInputs{sampleQuantized(rng, params.forget_gate),
                             oracle_state,
                             sampleQuantized(rng, params.input_gate),
                             sampleQuantized(rng, params.cell_gate)};
        const auto [forget_multiplier, input_multiplier] =
            multipliersFor(index, bitwidth, mode, rng);
        params.forget_scale = {forget_multiplier};
        params.input_scale = {input_multiplier};
        const std::int32_t oracle_old_cell =
            index < 5 ? inputs.old_cell : oracle_state;
        const std::int32_t int_old_cell =
            index < 5 ? inputs.old_cell : int_state;
        const float fp_old_cell =
            index < 5 ? static_cast<float>(inputs.old_cell) : fp_state;

        expected[index] =
            oracleCell(inputs.forget_gate, oracle_old_cell, inputs.input_gate,
                       inputs.cell_gate, params);
        int_actual[index] =
            quant_lstm::computeQuantizedCellInt(
                inputs.forget_gate, int_old_cell, inputs.input_gate,
                inputs.cell_gate, params)
                .value;
        fp_actual[index] =
            quant_lstm::computeQuantizedCellFp(
                static_cast<float>(inputs.forget_gate), fp_old_cell,
                static_cast<float>(inputs.input_gate),
                static_cast<float>(inputs.cell_gate), params)
                .value;
        result.int_mismatches += int_actual[index] != expected[index] ? 1U : 0U;
        result.int_saturations +=
            expected[index] == output_minimum ||
                    expected[index] == output_maximum
                ? 1U
                : 0U;
        result.fp_saturations +=
            fp_actual[index] == static_cast<float>(output_minimum) ||
                    fp_actual[index] == static_cast<float>(output_maximum)
                ? 1U
                : 0U;
        oracle_state = expected[index];
        int_state = int_actual[index];
        fp_state = fp_actual[index];

        if (index == 0) {
            require(expected[index] == params.new_cell.param.zero_point,
                    "定向抵消样本未归零");
        } else if (index == 1) {
            require(expected[index] > params.new_cell.param.zero_point,
                    "定向同号样本未累加");
        } else if (index == 2) {
            require(expected[index] == output_maximum,
                    "定向正饱和样本未饱和");
        } else if (index == 3) {
            require(expected[index] == output_minimum,
                    "定向负饱和样本未饱和");
        } else if (index == 4) {
            require(expected[index] < params.new_cell.param.zero_point,
                    "定向极端比例未保留主导项符号");
        }
    }
    result.int_metrics = quant_lstm::test::computeNumericMetrics(
        int_actual.data(), expected.data(), sample_count);
    result.fp_metrics = quant_lstm::test::computeNumericMetrics(
        fp_actual.data(), expected.data(), sample_count);
    return result;
}

void runStaticBoundariesAndFailFast() {
    require(q::encodeQ31Scale(std::ldexp(0.5, -31)).multiplier == 0,
            "Q31 even half tie");
    require(q::encodeQ31Scale(std::ldexp(1.5, -31)).multiplier == 2,
            "Q31 odd half tie");
    require(q::applyDualQ31(std::int64_t{3},
                            q::Q31Scale{std::int64_t{1} << 30},
                            std::int64_t{1},
                            q::Q31Scale{std::int64_t{1} << 30}) == 2,
            "Q31 known vector");
    require(q::applyDualQ31(std::int64_t{-3},
                            q::Q31Scale{std::int64_t{1} << 30},
                            std::int64_t{-1},
                            q::Q31Scale{std::int64_t{1} << 30}) == -2,
            "Q31 signed known vector");

    q::Q31IntDiagnostics diagnostics;
    require(q::applyDualQ31(std::int64_t{7},
                            q::Q31Scale{std::int64_t{1} << 30},
                            std::int64_t{-3},
                            q::Q31Scale{std::int64_t{1} << 30},
                            &diagnostics) == 2 &&
                diagnostics.wide_sum ==
                    static_cast<__int128>(4) *
                        (std::int64_t{1} << 30),
            "Q31 diagnostics boundary");

    requireThrowsAs<std::invalid_argument>(
        [] { static_cast<void>(q::encodeQ31Scale(-1.0)); },
        "negative Q31 ratio fail-fast");
    requireThrowsAs<std::invalid_argument>(
        [] {
            static_cast<void>(
                q::encodeQ31Scale(std::numeric_limits<double>::infinity()));
        },
        "non-finite Q31 ratio fail-fast");
    requireThrowsAs<std::overflow_error>(
        [] {
            static_cast<void>(q::encodeQ31Scale(
                std::nextafter(std::ldexp(1.0, 32),
                               std::numeric_limits<double>::infinity())));
        },
        "Q31 encode overflow fail-fast");
    requireThrowsAs<std::overflow_error>(
        [] {
            const auto maximum = std::numeric_limits<std::int64_t>::max();
            static_cast<void>(
                q::applyDualQ31(maximum, q::Q31Scale{maximum}, maximum,
                                q::Q31Scale{maximum}));
        },
        "Q31 int128 merge overflow fail-fast");
    requireThrowsAs<std::overflow_error>(
        [] {
            const auto maximum = std::numeric_limits<std::int64_t>::max();
            static_cast<void>(q::applyDualQ31(
                maximum, q::Q31Scale{std::int64_t{1} << 31}, maximum,
                q::Q31Scale{std::int64_t{1} << 31}));
        },
        "Q31 int64 result overflow fail-fast");
    requireThrowsAs<std::invalid_argument>(
        [] {
            static_cast<void>(q::applyDualQ31(
                std::numeric_limits<float>::quiet_NaN(), q::Q31Scale{1},
                1.0F, q::Q31Scale{1}));
        },
        "Q31 FP NaN fail-fast");

    auto params = makeCellParams(8, q::ScaleMode::Affine);
    params.forget_scale = {std::int64_t{1} << 31};
    params.input_scale = {std::int64_t{1} << 31};
    requireThrowsAs<std::out_of_range>(
        [&] {
            static_cast<void>(quant_lstm::computeQuantizedCellInt(
                256, 0, 0, 0, params));
        },
        "Cell int range fail-fast");
    requireThrowsAs<std::invalid_argument>(
        [&] {
            auto invalid = params;
            invalid.old_cell.param.scale = 0.0F;
            static_cast<void>(
                quant_lstm::computeQuantizedCellInt(0, 0, 0, 0, invalid));
        },
        "Cell invalid QuantParam fail-fast");
    requireThrowsAs<std::invalid_argument>(
        [&] {
            auto invalid = params;
            invalid.input_scale.multiplier = -1;
            static_cast<void>(
                quant_lstm::computeQuantizedCellInt(0, 0, 0, 0, invalid));
        },
        "Cell negative multiplier fail-fast");
    requireThrowsAs<std::invalid_argument>(
        [&] {
            static_cast<void>(quant_lstm::computeQuantizedCellFp(
                0.5F, 0.0F, 0.0F, 0.0F, params));
        },
        "Cell FP fractional q fail-fast");
    requireThrowsAs<std::invalid_argument>(
        [&] {
            static_cast<void>(quant_lstm::computeQuantizedCellFp(
                std::numeric_limits<float>::infinity(), 0.0F, 0.0F, 0.0F,
                params));
        },
        "Cell FP non-finite q fail-fast");
}

std::uint64_t runReducedDomainExhaustive() {
    std::uint64_t mismatches = 0;
    for (std::int64_t p_forget = -8; p_forget <= 8; ++p_forget) {
        for (std::int64_t p_input = -8; p_input <= 8; ++p_input) {
            for (std::int64_t m_forget = 0; m_forget <= 8; ++m_forget) {
                for (std::int64_t m_input = 0; m_input <= 8; ++m_input) {
                    const std::int64_t actual =
                        q::applyDualQ31(p_forget, q::Q31Scale{m_forget},
                                       p_input, q::Q31Scale{m_input});
                    const std::int64_t expected =
                        oracleDualQ31(p_forget, m_forget, p_input, m_input);
                    mismatches += actual != expected ? 1U : 0U;
                }
            }
        }
    }
    return mismatches;
}

void writeMetrics(std::ofstream& report,
                  const quant_lstm::test::NumericMetrics& metrics) {
    report << "{\"max_abs\":" << metrics.maximum_absolute_error
           << ",\"mae\":" << metrics.mean_absolute_error
           << ",\"mse\":" << metrics.mean_squared_error << ",\"cosine\":";
    if (metrics.cosine_not_applicable) {
        report << "null";
    } else {
        report << metrics.cosine_similarity;
    }
    report << ",\"cosine_not_applicable\":"
           << (metrics.cosine_not_applicable ? "true" : "false")
           << ",\"one_sided_zero_norm\":"
           << (metrics.one_sided_zero_norm ? "true" : "false") << '}';
}

void writeReport(std::uint64_t exhaustive_mismatches,
                 const std::vector<ScenarioResult>& scenarios) {
    std::ofstream report("q31_validation_report.json",
                         std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(report), "Q31 report open failed");
    report << "{\"oracle\":\"independent_integer_ties_to_even\","
              "\"seed\":\"0x51315EED\",\"reduced_domain_mismatches\":"
           << exhaustive_mismatches << ",\"scenarios\":[";
    for (std::size_t index = 0; index < scenarios.size(); ++index) {
        if (index != 0) {
            report << ',';
        }
        const auto& scenario = scenarios[index];
        report << "{\"bitwidth\":" << static_cast<int>(scenario.bitwidth)
               << ",\"mode\":\""
               << (scenario.mode == q::ScaleMode::Affine ? "Affine" : "POT2")
               << "\",\"samples\":" << scenario.samples
               << ",\"int_mismatches\":" << scenario.int_mismatches
               << ",\"int_saturations\":" << scenario.int_saturations
               << ",\"fp_saturations\":" << scenario.fp_saturations
               << ",\"int_metrics\":";
        writeMetrics(report, scenario.int_metrics);
        report << ",\"fp_metrics\":";
        writeMetrics(report, scenario.fp_metrics);
        report << '}';
    }
    report << "]}\n";
    require(static_cast<bool>(report), "Q31 report write failed");
}

#endif

}  // namespace

int main() {
    try {
#if !defined(__SIZEOF_INT128__)
        throw std::runtime_error("Q31 reference 要求编译器支持 __int128");
#else
        runStaticBoundariesAndFailFast();
        const std::uint64_t exhaustive_mismatches =
            runReducedDomainExhaustive();
        require(exhaustive_mismatches == 0,
                "Q31 reduced-domain exhaustive mismatch");

        std::vector<ScenarioResult> scenarios;
        for (const std::uint8_t bitwidth :
             {std::uint8_t{8}, std::uint8_t{16}}) {
            scenarios.push_back(runScenario(bitwidth, q::ScaleMode::Affine));
            scenarios.push_back(runScenario(bitwidth, q::ScaleMode::Pot2));
        }
        writeReport(exhaustive_mismatches, scenarios);

        for (const auto& scenario : scenarios) {
            require(scenario.int_mismatches == 0 &&
                        scenario.int_metrics.maximum_absolute_error == 0.0 &&
                        cosinePasses(scenario.int_metrics, 1.0 - 1.0e-12),
                    "integer carrier oracle metric gate");
            require(scenario.fp_metrics.maximum_absolute_error <= 1.0 &&
                        scenario.fp_metrics.mean_absolute_error <= 0.002 &&
                        cosinePasses(scenario.fp_metrics, 0.999999),
                    "FP carrier oracle metric gate");
            require(scenario.int_saturations != 0 &&
                        scenario.fp_saturations != 0,
                    "adversarial saturation coverage gate");
        }
#endif
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
