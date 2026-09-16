#pragma once

#include <cstddef>

// 本模块冻结 LSTM 内部的 PyTorch 门顺序；所有后端均以连续 (i,f,g,o) 门段寻址。
namespace quant_lstm {

enum class GateKind : std::size_t {
    Input = 0,
    Forget = 1,
    Cell = 2,
    Output = 3,
};

inline constexpr std::size_t kGateCount = 4;

/// 返回指定门在长度为 4H 的连续向量中的起始偏移。
constexpr std::size_t gateOffset(GateKind gate, std::size_t hidden_size) noexcept {
    return static_cast<std::size_t>(gate) * hidden_size;
}

}  // namespace quant_lstm
