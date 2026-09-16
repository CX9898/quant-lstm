#pragma once

// 对外聚合头文件；公共 C++ 调用方只需包含本文件。
#include "lstm/forward_float.h"
#include "lstm/quant_config.h"
#include "lstm/quant_config_loader.h"
#include "lstm/quant_params.h"

#if defined(QUANT_LSTM_WITH_CUDA)
#include "lstm/forward_float_cuda.h"
#endif
