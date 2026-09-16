#include "lstm/gate_layout.h"

#include <cstdlib>
#include <iostream>

int main() {
    constexpr std::size_t hidden_size = 7;
    using quant_lstm::GateKind;

    if (quant_lstm::kGateCount != 4 ||
        quant_lstm::gateOffset(GateKind::Input, hidden_size) != 0 ||
        quant_lstm::gateOffset(GateKind::Forget, hidden_size) != 7 ||
        quant_lstm::gateOffset(GateKind::Cell, hidden_size) != 14 ||
        quant_lstm::gateOffset(GateKind::Output, hidden_size) != 21) {
        std::cerr << "门布局不是连续的 (i,f,g,o)\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
