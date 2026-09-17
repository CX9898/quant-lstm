#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

// 测试指标统一使用 FP64 累加；生产代码不得依赖本模块。
namespace quant_lstm::test {

struct NumericMetrics {
    double maximum_absolute_error = 0.0;
    double mean_absolute_error = 0.0;
    double mean_squared_error = 0.0;
    double cosine_similarity = std::numeric_limits<double>::quiet_NaN();
    bool cosine_not_applicable = false;
    bool one_sided_zero_norm = false;
};

template <typename Actual, typename Expected>
NumericMetrics computeNumericMetrics(const Actual* actual, const Expected* expected,
                                     std::size_t count, double epsilon = 1.0e-12) {
    if ((actual == nullptr || expected == nullptr) && count != 0) {
        throw std::invalid_argument("指标输入指针不能为空");
    }
    NumericMetrics result;
    double absolute_sum = 0.0;
    double squared_sum = 0.0;
    double dot = 0.0;
    double actual_norm = 0.0;
    double expected_norm = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        const double lhs = static_cast<double>(actual[index]);
        const double rhs = static_cast<double>(expected[index]);
        const double difference = lhs - rhs;
        const double absolute = std::abs(difference);
        result.maximum_absolute_error =
            std::max(result.maximum_absolute_error, absolute);
        absolute_sum += absolute;
        squared_sum += difference * difference;
        dot += lhs * rhs;
        actual_norm += lhs * lhs;
        expected_norm += rhs * rhs;
    }
    if (count != 0) {
        result.mean_absolute_error = absolute_sum / static_cast<double>(count);
        result.mean_squared_error = squared_sum / static_cast<double>(count);
    }
    const double actual_l2 = std::sqrt(actual_norm);
    const double expected_l2 = std::sqrt(expected_norm);
    const bool actual_zero = actual_l2 <= epsilon;
    const bool expected_zero = expected_l2 <= epsilon;
    result.cosine_not_applicable = actual_zero && expected_zero;
    result.one_sided_zero_norm = actual_zero != expected_zero;
    if (!result.cosine_not_applicable && !result.one_sided_zero_norm) {
        result.cosine_similarity = dot / (actual_l2 * expected_l2);
    }
    return result;
}

}  // namespace quant_lstm::test
