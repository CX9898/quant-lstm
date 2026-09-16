#pragma once

#include "quantization/quant_param.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

// 本模块冻结 18 个真实量化点及 resolved config 的强类型表示。
namespace quant_lstm {

enum class QuantOperator : std::uint8_t {
    Input,
    Output,
    CellState,
    WeightInputHidden,
    WeightHiddenHidden,
    BiasInputHidden,
    BiasHiddenHidden,
    WeightInputHiddenLinear,
    WeightHiddenHiddenLinear,
    InputGateInput,
    ForgetGateInput,
    CellGateInput,
    OutputGateInput,
    InputGateOutput,
    ForgetGateOutput,
    CellGateOutput,
    OutputGateOutput,
    CellTanhOutput,
};

inline constexpr std::size_t kQuantOperatorCount = 18;

enum class QuantGranularity : std::uint8_t {
    PerTensor,
    PerGate,
    PerChannel,
};

struct OperatorQuantConfig {
    quantization::QuantizationType type;
    QuantGranularity granularity = QuantGranularity::PerTensor;
};

struct LstmOperatorQuantConfig {
    std::int32_t schema_version = 1;
    quantization::ScaleMode scale_mode = quantization::ScaleMode::Affine;
    std::array<OperatorQuantConfig, kQuantOperatorCount> operators{};

    const OperatorQuantConfig& at(QuantOperator id) const;
    OperatorQuantConfig& at(QuantOperator id);
    void validate() const;
};

std::string_view quantOperatorName(QuantOperator id);
QuantOperator parseQuantOperatorName(std::string_view name);
bool isParameterOperator(QuantOperator id) noexcept;
bool isBiasOperator(QuantOperator id) noexcept;
bool isSigmoidGateOutput(QuantOperator id) noexcept;

std::string_view granularityName(QuantGranularity granularity);
QuantGranularity parseGranularity(std::string_view name);

std::string_view scaleModeName(quantization::ScaleMode mode);
quantization::ScaleMode parseScaleMode(std::string_view name);

}  // namespace quant_lstm
