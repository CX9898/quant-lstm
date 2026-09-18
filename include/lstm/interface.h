#pragma once

// 对外聚合头文件；公共 C++ 调用方只需包含本文件。
#include "lstm/calibration.h"
#include "lstm/forward_cpu.h"
#include "lstm/forward_float.h"
#include "lstm/lstm_execution_params.h"
#include "lstm/quant_config.h"
#include "lstm/quant_config_loader.h"
#include "lstm/quant_params.h"
#include "lstm/quant_params_io.h"

#if defined(QUANT_LSTM_WITH_CUDA)
#include "lstm/forward_float_cuda.h"
#include "lstm/forward_quantized_fp_cuda.h"
#endif
