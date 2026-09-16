#include "lstm/gate_layout.h"

#include <iostream>

int main() {
    constexpr std::size_t hidden_size = 32;
    std::cout << "output gate offset: "
              << quant_lstm::gateOffset(quant_lstm::GateKind::Output, hidden_size) << '\n';
    return 0;
}
