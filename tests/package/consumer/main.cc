#include <cstdlib>

#include "lstm/forward_cpu.h"
#include "quantization/bit_width.h"

#ifdef QUANT_LSTM_WITH_CUDA
#error "CPU-only installed target 不得传播 QUANT_LSTM_WITH_CUDA"
#endif

int main() {
    const quant_lstm::LstmShape shape{1, 1, 1, 1};
    const quant_lstm::quantization::QuantizationType type{8, false, true};
    const auto range = type.range();
    return shape.hidden_size == 1 && range.minimum == -127 && range.maximum == 127 ? EXIT_SUCCESS
                                                                                   : EXIT_FAILURE;
}
