#include "quantization/histogram.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "quantization/fixed_point_ops.h"
#include "quantization/scale_encoding.h"

namespace quant_lstm::quantization {
namespace {

std::pair<float, float> nonDegenerateRange(float minimum, float maximum) {
    if (minimum < maximum) {
        return {minimum, maximum};
    }
    const float padding =
        std::max(std::abs(minimum) * std::numeric_limits<float>::epsilon(), 1.0e-6F);
    return {minimum - padding, maximum + padding};
}

std::size_t binIndex(float value, float minimum, float maximum, std::size_t bin_count) {
    if (value <= minimum) {
        return 0;
    }
    if (value >= maximum) {
        return bin_count - 1;
    }
    const double normalized =
        (static_cast<double>(value) - minimum) / (static_cast<double>(maximum) - minimum);
    return std::min(static_cast<std::size_t>(normalized * static_cast<double>(bin_count)),
                    bin_count - 1);
}

void addRebinned(const Histogram& source, Histogram* destination) {
    for (std::size_t index = 0; index < source.counts.size(); ++index) {
        const double count = source.counts[index];
        if (count == 0.0) {
            continue;
        }
        const std::size_t target = binIndex(source.binCenter(index), destination->minimum,
                                            destination->maximum, destination->counts.size());
        destination->counts[target] += count;
    }
    destination->total_count += source.total_count;
}

double quantizationNoise(const Histogram& histogram, float minimum, float maximum,
                         const QuantizationType& type) {
    const auto calibrated = calibrateMinMax(minimum, maximum, type);
    double noise = 0.0;
    for (std::size_t index = 0; index < histogram.counts.size(); ++index) {
        const double count = histogram.counts[index];
        if (count == 0.0) {
            continue;
        }
        const float value = histogram.binCenter(index);
        const std::int32_t quantized = quantize(value, calibrated.param, type);
        const float reconstructed = dequantize(quantized, calibrated.param, type);
        const double difference = static_cast<double>(reconstructed) - value;
        noise += difference * difference * count;
    }
    return noise;
}

}  // namespace

bool Histogram::empty() const noexcept { return total_count == 0 || counts.empty(); }

float Histogram::binWidth() const {
    if (empty() || !(minimum < maximum)) {
        throw std::logic_error("空或退化直方图没有 bin width");
    }
    return (maximum - minimum) / static_cast<float>(counts.size());
}

float Histogram::binCenter(std::size_t index) const {
    if (index >= counts.size()) {
        throw std::out_of_range("histogram bin index 越界");
    }
    return minimum + (static_cast<float>(index) + 0.5F) * binWidth();
}

std::pair<float, float> Histogram::percentileRange(float percentile) const {
    if (empty()) {
        throw std::logic_error("空直方图不能计算 percentile");
    }
    if (!std::isfinite(percentile) || percentile <= 0.0F || percentile > 100.0F) {
        throw std::invalid_argument("percentile 必须在 (0,100] 内");
    }
    if (percentile == 100.0F) {
        return {minimum, maximum};
    }
    const double clipped_fraction = (100.0 - static_cast<double>(percentile)) / 200.0;
    const double lower_target = static_cast<double>(total_count) * clipped_fraction;
    const double upper_target = static_cast<double>(total_count) * (1.0 - clipped_fraction);
    double cumulative = 0.0;
    std::size_t lower = 0;
    std::size_t upper = counts.size() - 1;
    bool lower_found = false;
    for (std::size_t index = 0; index < counts.size(); ++index) {
        cumulative += counts[index];
        if (!lower_found && cumulative >= lower_target) {
            lower = index;
            lower_found = true;
        }
        if (cumulative >= upper_target) {
            upper = index;
            break;
        }
    }
    const float width = binWidth();
    const float result_minimum = minimum + static_cast<float>(lower) * width;
    const float result_maximum = minimum + static_cast<float>(upper + 1) * width;
    return result_minimum < result_maximum ? std::pair<float, float>{result_minimum, result_maximum}
                                           : std::pair<float, float>{minimum, maximum};
}

HistogramCollector::HistogramCollector(std::size_t bin_count) : bin_count_(bin_count) {
    if (bin_count_ < 2) {
        throw std::invalid_argument("histogram bin_count 必须至少为 2");
    }
    reset();
}

void HistogramCollector::reset() {
    histogram_ = {};
    histogram_.counts.assign(bin_count_, 0.0);
}

void HistogramCollector::collect(const float* values, std::size_t count) {
    if (values == nullptr && count != 0) {
        throw std::invalid_argument("histogram 输入指针不能为空");
    }
    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();
    std::uint64_t finite_count = 0;
    for (std::size_t index = 0; index < count; ++index) {
        if (std::isfinite(values[index])) {
            minimum = std::min(minimum, values[index]);
            maximum = std::max(maximum, values[index]);
            ++finite_count;
        }
    }
    if (finite_count == 0) {
        return;
    }
    const auto normalized = nonDegenerateRange(minimum, maximum);
    Histogram batch;
    batch.minimum = normalized.first;
    batch.maximum = normalized.second;
    batch.counts.assign(bin_count_, 0.0);
    batch.total_count = finite_count;
    for (std::size_t index = 0; index < count; ++index) {
        if (!std::isfinite(values[index])) {
            continue;
        }
        ++batch.counts[binIndex(values[index], batch.minimum, batch.maximum, bin_count_)];
    }
    merge(batch);
}

void HistogramCollector::merge(const Histogram& other) {
    if (other.empty()) {
        return;
    }
    if (other.counts.size() != bin_count_ || !(other.minimum < other.maximum)) {
        throw std::invalid_argument("待合并 histogram 的布局非法");
    }
    if (histogram_.empty()) {
        histogram_ = other;
        return;
    }
    Histogram merged;
    merged.minimum = std::min(histogram_.minimum, other.minimum);
    merged.maximum = std::max(histogram_.maximum, other.maximum);
    merged.counts.assign(bin_count_, 0.0);
    addRebinned(histogram_, &merged);
    addRebinned(other, &merged);
    histogram_ = std::move(merged);
}

std::size_t HistogramCollector::binCount() const noexcept { return bin_count_; }

const Histogram& HistogramCollector::histogram() const noexcept { return histogram_; }

std::pair<float, float> calibrateHistogramRange(const Histogram& histogram,
                                                const QuantizationType& type,
                                                HistogramCalibrationMethod method,
                                                const HistogramCalibrationOptions& options) {
    if (histogram.empty()) {
        throw std::invalid_argument("空直方图不能校准");
    }
    type.validate();
    if (method == HistogramCalibrationMethod::Percentile) {
        return histogram.percentileRange(options.percentile);
    }
    if (method != HistogramCalibrationMethod::Sqnr) {
        throw std::invalid_argument("HistogramCalibrationMethod 枚举值非法");
    }

    std::pair<float, float> best{histogram.minimum, histogram.maximum};
    double best_noise = quantizationNoise(histogram, best.first, best.second, type);
    if (type.is_symmetric) {
        if (options.symmetric_candidates == 0) {
            throw std::invalid_argument("SQNR symmetric_candidates 不能为 0");
        }
        const float full_extent =
            type.is_unsigned ? std::max(0.0F, histogram.maximum)
                             : std::max(std::abs(histogram.minimum), std::abs(histogram.maximum));
        for (std::size_t candidate = 1; candidate <= options.symmetric_candidates; ++candidate) {
            const float extent = full_extent * static_cast<float>(candidate) /
                                 static_cast<float>(options.symmetric_candidates);
            const std::pair<float, float> range = type.is_unsigned
                                                      ? std::pair<float, float>{0.0F, extent}
                                                      : std::pair<float, float>{-extent, extent};
            const double noise = quantizationNoise(histogram, range.first, range.second, type);
            if (noise < best_noise) {
                best_noise = noise;
                best = range;
            }
        }
        return best;
    }

    if (options.asymmetric_candidates == 0 || options.offset_candidates == 0) {
        throw std::invalid_argument("SQNR candidate 数量不能为 0");
    }
    const float full_span = histogram.maximum - histogram.minimum;
    for (std::size_t width_index = 1; width_index <= options.asymmetric_candidates; ++width_index) {
        const float span = full_span * static_cast<float>(width_index) /
                           static_cast<float>(options.asymmetric_candidates);
        const float offset_span = full_span - span;
        for (std::size_t offset_index = 0; offset_index < options.offset_candidates;
             ++offset_index) {
            const float fraction = options.offset_candidates == 1
                                       ? 0.0F
                                       : static_cast<float>(offset_index) /
                                             static_cast<float>(options.offset_candidates - 1);
            const float minimum = histogram.minimum + offset_span * fraction;
            const float maximum = minimum + span;
            const double noise = quantizationNoise(histogram, minimum, maximum, type);
            if (noise < best_noise) {
                best_noise = noise;
                best = {minimum, maximum};
            }
        }
    }
    return best;
}

}  // namespace quant_lstm::quantization
