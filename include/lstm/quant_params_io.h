#pragma once

#include "lstm/lstm_execution_params.h"
#include "lstm/quant_config.h"
#include "lstm/quant_params.h"

#include <cstdint>
#include <string>
#include <string_view>

namespace quant_lstm {

struct LstmQuantParamsBundle {
    std::int32_t schema_version = 1;
    std::int64_t input_size = 0;
    LstmOperatorQuantConfig config;
    LstmQuantParams quant_params;

    void validate() const;
};

// 输出无多余空白、key 顺序固定的 canonical JSON。scale 使用最短可往返
// FP32 十进制字符串；zero point 保持 JSON integer。
std::string exportQuantParamsBundle(const LstmQuantParamsBundle& bundle);
LstmQuantParamsBundle importQuantParamsBundle(std::string_view json_text,
                                              bool require_canonical = false);

// 审计外部 standard scale/zp 并重新派生执行编码；外部 bundle 不接受
// raw ratio、M+shift 或 POT2 shift 字段。
LstmExecutionParams auditQuantParamsBundle(
    const LstmQuantParamsBundle& bundle,
    bool require_exact_accumulation = false);

}  // namespace quant_lstm
