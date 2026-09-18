#pragma once

#include <cstddef>
#include <cstdint>
#include <nlohmann/json_fwd.hpp>
#include <vector>

#include "common/numeric_metrics.h"
#include "lstm/forward_float.h"
#include "lstm/lstm_execution_params.h"
#include "lstm/quant_config.h"
#include "lstm/quant_params.h"

namespace quant_lstm::test {

struct FloatMaster {
    std::vector<float> input;
    std::vector<float> weight_ih;
    std::vector<float> weight_hh;
    std::vector<float> bias_ih;
    std::vector<float> bias_hh;
    std::vector<float> h0;
    std::vector<float> c0;
    bool has_explicit_state = false;
    bool bias_enabled = true;
};

struct QuantizedMaster {
    std::vector<std::int32_t> input;
    std::vector<std::int32_t> weight_ih;
    std::vector<std::int32_t> weight_hh;
    std::vector<std::int32_t> bias_ih;
    std::vector<std::int32_t> bias_hh;
    std::vector<std::int32_t> h0;
    std::vector<std::int32_t> c0;
};

struct FloatLstmResult {
    std::vector<float> output;
    std::vector<float> final_hidden;
    std::vector<float> final_cell;
};

struct Int32LstmResult {
    std::vector<std::int32_t> output;
    std::vector<std::int32_t> final_hidden;
    std::vector<std::int32_t> final_cell;
};

struct SyntheticPrecisionFixture {
    LstmShape shape;
    LstmOperatorQuantConfig config;
    FloatMaster master;
    LstmQuantizationRanges ranges;
    LstmQuantParams params;
    LstmExecutionParams execution;
    QuantizedMaster quantized;
    FloatLstmResult float_oracle;
    Int32LstmResult int32_oracle;
    FloatLstmResult fp_quantized_oracle;
};

struct SyntheticNumericMetrics {
    NumericMetrics accuracy;
    // JSON-safe SQNR；零噪声/零信号分别封顶为 +300/-300 dB。
    double sqnr_db = 0.0;
    double saturation_rate = 0.0;
};

struct MetricThresholds {
    double maximum_mae = 0.0;
    double maximum_mse = 0.0;
    double minimum_cosine_similarity = 0.0;
};

struct TensorMetrics {
    SyntheticNumericMetrics metrics;
    bool passed = false;
};

LstmShape shapeFromSyntheticProfile(const nlohmann::json& profile);

FloatMaster makeFloatMaster(const nlohmann::json& profile, std::uint64_t data_seed);

LstmQuantizationRanges makeSyntheticRanges(const nlohmann::json& profile, const LstmShape& shape,
                                           const LstmOperatorQuantConfig& config,
                                           const FloatMaster& parameter_master);

QuantizedMaster quantizeMaster(const LstmShape& shape, const LstmOperatorQuantConfig& config,
                               const LstmQuantParams& params, const FloatMaster& master);

SyntheticPrecisionFixture makeSyntheticPrecisionFixture(const nlohmann::json& profile);

std::vector<std::int32_t> quantizeTensor(const std::vector<float>& source, QuantOperator id,
                                         const LstmOperatorQuantConfig& config,
                                         const LstmQuantParams& params, std::size_t row_width = 0);

std::vector<float> asFloatCarrier(const std::vector<std::int32_t>& values);

std::vector<float> dequantizeTensor(const std::vector<std::int32_t>& source, QuantOperator id,
                                    const LstmOperatorQuantConfig& config,
                                    const LstmQuantParams& params);

std::vector<float> dequantizeTensor(const std::vector<float>& source, QuantOperator id,
                                    const LstmOperatorQuantConfig& config,
                                    const LstmQuantParams& params);

FloatLstmResult runCpuFloatOracle(const LstmShape& shape, const FloatMaster& master);

Int32LstmResult runCpuInt32Oracle(const LstmShape& shape, const LstmOperatorQuantConfig& config,
                                  const LstmQuantParams& params,
                                  const LstmExecutionParams& execution, const FloatMaster& master,
                                  const QuantizedMaster& quantized);

FloatLstmResult runCpuFpQuantizedOracle(const LstmShape& shape,
                                        const LstmOperatorQuantConfig& config,
                                        const LstmQuantParams& params,
                                        const LstmExecutionParams& execution,
                                        const FloatMaster& master,
                                        const QuantizedMaster& quantized);

SyntheticNumericMetrics computeSyntheticNumericMetrics(const float* actual, const float* expected,
                                                       std::size_t count);

SyntheticNumericMetrics computeSyntheticNumericMetrics(
    const float* actual, const float* expected, std::size_t count,
    const std::int32_t* quantized_values, quantization::QuantizationType quantization_type);

SyntheticNumericMetrics computeSyntheticNumericMetrics(
    const float* actual, const float* expected, std::size_t count, const float* quantized_values,
    quantization::QuantizationType quantization_type);

TensorMetrics evaluateTensorMetrics(const std::vector<float>& actual,
                                    const std::vector<float>& expected,
                                    const MetricThresholds& thresholds);

}  // namespace quant_lstm::test
