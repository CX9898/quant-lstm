#pragma once

#include "quantization/quant_param.h"

#include <cstdint>

// 本模块从 standard scale/range 派生内部执行编码；不保存第二套外部 scale。
namespace quant_lstm::quantization {

inline constexpr double kPot2RelativeTolerance = 0.02;

struct Pot2ScaleResult {
    QuantParam param;
    std::int8_t exponent = 0;  // standard_scale = 2^(-exponent)
    bool range_is_near_power_of_two = false;
};

float minimumScale(const QuantizationType& type);

CalibrationResult calibrateMinMax(float range_minimum, float range_maximum,
                                  const QuantizationType& type);

FixedPointScale encodeMShift(double ratio);
double decodeMShift(const FixedPointScale& encoded) noexcept;

Pot2Rescale encodePot2Rescale(float source_scale, float destination_scale);

Pot2ScaleResult convertScaleToPot2CoverRange(const CalibrationResult& calibrated,
                                             const QuantizationType& type);

}  // namespace quant_lstm::quantization
