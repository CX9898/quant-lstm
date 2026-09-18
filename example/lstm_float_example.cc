#include <array>
#include <iostream>

#include "lstm/forward_float.h"

int main() {
    constexpr quant_lstm::LstmShape shape{2, 1, 1, 1};
    const std::array<float, 2> input{0.25F, -0.5F};
    const std::array<float, 4> weight_ih{};
    const std::array<float, 4> weight_hh{};
    const quant_lstm::LstmFloatWeights weights{weight_ih.data(), weight_hh.data(), nullptr,
                                               nullptr};
    std::array<float, 2> output{};
    std::array<float, 1> final_hidden{};
    std::array<float, 1> final_cell{};

    quant_lstm::lstmForwardFloatCpu(shape, weights, input.data(), nullptr, nullptr, output.data(),
                                    final_hidden.data(), final_cell.data());
    std::cout << "output=[" << output[0] << ", " << output[1] << "]"
              << ", h_n=" << final_hidden[0] << ", c_n=" << final_cell[0] << '\n';
    return 0;
}
