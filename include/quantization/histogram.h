#pragma once

#include "quantization/quant_param.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace quant_lstm::quantization {

struct Histogram {
    float minimum = 0.0F;
    float maximum = 0.0F;
    std::vector<double> counts;
    std::uint64_t total_count = 0;

    bool empty() const noexcept;
    float binWidth() const;
    float binCenter(std::size_t index) const;
    std::pair<float, float> percentileRange(float percentile) const;
};

class HistogramCollector {
   public:
    explicit HistogramCollector(std::size_t bin_count = 2048);

    void reset();
    void collect(const float* values, std::size_t count);
    void merge(const Histogram& other);

    std::size_t binCount() const noexcept;
    const Histogram& histogram() const noexcept;

   private:
    std::size_t bin_count_;
    Histogram histogram_;
};

enum class HistogramCalibrationMethod : std::uint8_t {
    Sqnr,
    Percentile,
};

struct HistogramCalibrationOptions {
    float percentile = 99.99F;
    std::size_t symmetric_candidates = 101;
    std::size_t asymmetric_candidates = 17;
    std::size_t offset_candidates = 21;
};

// 返回候选连续范围；调用方仍须通过 calibrateMinMax 及 POT2
// CoverRange 统一生成最终 standard scale/zp。
std::pair<float, float> calibrateHistogramRange(
    const Histogram& histogram, const QuantizationType& type,
    HistogramCalibrationMethod method,
    const HistogramCalibrationOptions& options = {});

}  // namespace quant_lstm::quantization
