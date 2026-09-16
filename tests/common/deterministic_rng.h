#pragma once

#include <cstddef>
#include <cstdint>

// 测试随机数契约 pcg32-xsh-rr-v1；生产代码不得依赖本模块。
namespace quant_lstm::test {

inline constexpr const char* kRngVersion = "pcg32-xsh-rr-v1";
inline constexpr const char* kRngStreamRegistryVersion = "v1";
inline constexpr const char* kDistributionProfile = "pytorch_typical_v1";

enum class TensorStream : std::uint64_t {
    Input = 1,
    InitialHidden = 2,
    InitialCell = 3,
    WeightInputHidden = 10,
    WeightHiddenHidden = 11,
    BiasInputHidden = 12,
    BiasHiddenHidden = 13,
};

class Pcg32 {
   public:
    Pcg32(std::uint64_t seed, std::uint64_t stream_id);

    std::uint32_t nextUint32() noexcept;
    float uniformFloat32() noexcept;
    float normalLikeFloat32() noexcept;

   private:
    std::uint64_t state_ = 0;
    std::uint64_t increment_ = 0;
};

/// 将 uint32 的高 24 bit 确定性映射到 [0,1)。
float uint32ToUniformFloat(std::uint32_t value) noexcept;

/// 生成 row-major float32 张量；每个元素固定消费 12 个随机字。
void fillNormalLike(float* data, std::size_t count, std::uint64_t seed, TensorStream stream);

/// 按 PyTorch LSTM 默认范围 [-1/sqrt(H),1/sqrt(H)) 生成参数。
void fillLstmParameter(float* data, std::size_t count, std::int64_t hidden_size,
                       std::uint64_t seed, TensorStream stream);

}  // namespace quant_lstm::test
