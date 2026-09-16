#include "common/deterministic_rng.h"

#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <utility>

namespace {

std::uint32_t floatBits(float value) {
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

}  // namespace

int main() {
    using quant_lstm::test::Pcg32;
    using quant_lstm::test::TensorStream;

    Pcg32 official_rng(42, 54);
    constexpr std::array<std::uint32_t, 5> official_sequence{
        0xa15c02b7U, 0x7b47f409U, 0xba1d3330U, 0x83d2f293U, 0xbfa4784bU};
    for (const std::uint32_t expected : official_sequence) {
        if (official_rng.nextUint32() != expected) {
            std::cerr << "PCG32 官方已知向量不匹配\n";
            return EXIT_FAILURE;
        }
    }

    if (quant_lstm::test::uint32ToUniformFloat(0U) != 0.0F ||
        quant_lstm::test::uint32ToUniformFloat(0xffffffffU) != 1.0F - 0x1p-24F) {
        std::cerr << "float32 高 24 bit 映射端点错误\n";
        return EXIT_FAILURE;
    }

    Pcg32 normal_rng(1001, static_cast<std::uint64_t>(TensorStream::Input));
    Pcg32 manual_rng(1001, static_cast<std::uint64_t>(TensorStream::Input));
    const float normal_value = normal_rng.normalLikeFloat32();
    float manual_value = 0.0F;
    for (int index = 0; index < 12; ++index) {
        manual_value += manual_rng.uniformFloat32();
    }
    manual_value -= 6.0F;
    if (normal_value != manual_value || normal_rng.nextUint32() != manual_rng.nextUint32()) {
        std::cerr << "类正态映射未固定消费 12 个随机字\n";
        return EXIT_FAILURE;
    }

    Pcg32 input_first(3001, static_cast<std::uint64_t>(TensorStream::Input));
    const std::uint32_t expected_input = input_first.nextUint32();
    Pcg32 unused_bias(3001, static_cast<std::uint64_t>(TensorStream::BiasInputHidden));
    static_cast<void>(unused_bias.nextUint32());
    Pcg32 input_after_bias(3001, static_cast<std::uint64_t>(TensorStream::Input));
    if (input_after_bias.nextUint32() != expected_input) {
        std::cerr << "独立 stream 被其他张量的消费顺序污染\n";
        return EXIT_FAILURE;
    }

    using StreamVector =
        std::pair<TensorStream, std::array<std::uint32_t, 3>>;
    constexpr std::array<StreamVector, 7> registry_vectors{{
        {TensorStream::Input, {0xcbf35ab7U, 0x8db3c0abU, 0x459d9da1U}},
        {TensorStream::InitialHidden, {0x52897be8U, 0x7d41c103U, 0xb28c55d4U}},
        {TensorStream::InitialCell, {0xa49fd35eU, 0xf2f90b57U, 0x8859342cU}},
        {TensorStream::WeightInputHidden, {0x5b10bb82U, 0x4439a62fU, 0x96c0daf2U}},
        {TensorStream::WeightHiddenHidden, {0xdb014451U, 0x5f7f86a2U, 0x540208b7U}},
        {TensorStream::BiasInputHidden, {0x2e00574cU, 0x70a7ec54U, 0x37d62e78U}},
        {TensorStream::BiasHiddenHidden, {0x0d9ad15cU, 0x78af582dU, 0xc8fe15d2U}},
    }};
    for (const auto& [stream, expected_values] : registry_vectors) {
        Pcg32 stream_rng(3001, static_cast<std::uint64_t>(stream));
        for (const std::uint32_t expected : expected_values) {
            if (stream_rng.nextUint32() != expected) {
                std::cerr << "registry stream 已知向量不匹配\n";
                return EXIT_FAILURE;
            }
        }
    }

    std::array<float, 4> normal_values{};
    quant_lstm::test::fillNormalLike(normal_values.data(), normal_values.size(), 3001,
                                     TensorStream::Input);
    constexpr std::array<std::uint32_t, 4> normal_bits{
        0x3fece4c0U, 0x3f723158U, 0xbfaeb960U, 0x3f1aef38U};
    for (std::size_t index = 0; index < normal_values.size(); ++index) {
        if (floatBits(normal_values[index]) != normal_bits[index]) {
            std::cerr << "pytorch_typical_v1 类正态位模式不匹配\n";
            return EXIT_FAILURE;
        }
    }

    std::array<float, 4> parameter_values{};
    quant_lstm::test::fillLstmParameter(parameter_values.data(), parameter_values.size(), 32, 3001,
                                         TensorStream::WeightInputHidden);
    constexpr std::array<std::uint32_t, 4> parameter_bits{
        0xbd50ef15U, 0xbda91196U, 0x3d00b656U, 0xbdf92eb6U};
    for (std::size_t index = 0; index < parameter_values.size(); ++index) {
        if (floatBits(parameter_values[index]) != parameter_bits[index]) {
            std::cerr << "pytorch_typical_v1 参数位模式不匹配\n";
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
