#pragma once

// 对外聚合头文件；公共 C++ 调用方只需包含本文件。
#include "lstm/forward_float.h"

#if defined(QUANT_LSTM_WITH_CUDA)
#include "lstm/forward_float_cuda.h"
#endif
