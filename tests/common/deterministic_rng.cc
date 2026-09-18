#include "common/deterministic_rng.h"

#include <cmath>
#include <stdexcept>

namespace quant_lstm::test {
namespace {

constexpr std::uint64_t kMultiplier = 6364136223846793005ULL;

std::uint32_t rotateRight(std::uint32_t value, std::uint32_t rotation) noexcept {
    return (value >> rotation) | (value << ((0U - rotation) & 31U));
}

}  // namespace

Pcg32::Pcg32(std::uint64_t seed, std::uint64_t stream_id) : increment_((stream_id << 1U) | 1U) {
    nextUint32();
    state_ += seed;
    nextUint32();
}

std::uint32_t Pcg32::nextUint32() noexcept {
    const std::uint64_t old_state = state_;
    state_ = old_state * kMultiplier + increment_;
    const std::uint32_t xor_shifted =
        static_cast<std::uint32_t>(((old_state >> 18U) ^ old_state) >> 27U);
    const std::uint32_t rotation = static_cast<std::uint32_t>(old_state >> 59U);
    return rotateRight(xor_shifted, rotation);
}

float uint32ToUniformFloat(std::uint32_t value) noexcept {
    return static_cast<float>(value >> 8U) * 0x1p-24F;
}

float Pcg32::uniformFloat32() noexcept { return uint32ToUniformFloat(nextUint32()); }

float Pcg32::normalLikeFloat32() noexcept {
    float value = 0.0F;
    for (int index = 0; index < 12; ++index) {
        value += uniformFloat32();
    }
    return value - 6.0F;
}

void fillNormalLike(float* data, std::size_t count, std::uint64_t seed, TensorStream stream) {
    if (data == nullptr && count != 0) {
        throw std::invalid_argument("随机张量指针不能为空");
    }
    Pcg32 rng(seed, static_cast<std::uint64_t>(stream));
    for (std::size_t index = 0; index < count; ++index) {
        data[index] = rng.normalLikeFloat32();
    }
}

void fillLstmParameter(float* data, std::size_t count, std::int64_t hidden_size, std::uint64_t seed,
                       TensorStream stream) {
    if ((data == nullptr && count != 0) || hidden_size <= 0) {
        throw std::invalid_argument("参数张量指针或 hidden_size 非法");
    }
    const float bound = 1.0F / std::sqrt(static_cast<float>(hidden_size));
    Pcg32 rng(seed, static_cast<std::uint64_t>(stream));
    for (std::size_t index = 0; index < count; ++index) {
        const float centered = 2.0F * rng.uniformFloat32() - 1.0F;
        data[index] = centered * bound;
    }
}

}  // namespace quant_lstm::test
