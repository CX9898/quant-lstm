#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cuda/cuda_common.cuh"
#include "cuda/quantized_fp_math.cuh"
#include "cuda/quantized_fp_params.cuh"
#include "lstm/forward_quantized_fp_cuda.h"
#include "lstm/gate_layout.h"
#include "quantization/rounding.h"

namespace quant_lstm {
namespace {

using cuda_detail::checkCublas;
using cuda_detail::checkCuda;
using cuda_detail::DeviceGateParams;
using cuda_detail::DeviceLinearChannelParams;
using cuda_detail::DeviceQuantPoint;
using cuda_detail::DeviceRescale;
using cuda_detail::DeviceRescaleKind;
using cuda_detail::DeviceScalarParams;

constexpr std::size_t kWorkspaceAlignment = 256;
constexpr int kThreads = 256;
constexpr unsigned int kMaximumOneDimensionalBlocks = 65535;

std::size_t checkedAdd(std::size_t lhs, std::size_t rhs, const char* description) {
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
        throw std::invalid_argument(std::string(description) + " 字节数溢出");
    }
    return lhs + rhs;
}

std::size_t checkedMul(std::size_t lhs, std::size_t rhs, const char* description) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::invalid_argument(std::string(description) + " 元素数溢出");
    }
    return lhs * rhs;
}

std::size_t positiveSize(std::int64_t value, const char* name) {
    if (value <= 0 || static_cast<std::uint64_t>(value) >
                          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument(std::string(name) + " 必须是可表示的正数");
    }
    return static_cast<std::size_t>(value);
}

std::size_t floatBytes(std::size_t elements, const char* description) {
    return checkedMul(elements, sizeof(float), description);
}

std::size_t alignUp(std::size_t value) {
    const std::size_t extra = kWorkspaceAlignment - 1;
    if (value > std::numeric_limits<std::size_t>::max() - extra) {
        throw std::invalid_argument("workspace 对齐字节数溢出");
    }
    return (value + extra) & ~extra;
}

struct WorkspaceLayout {
    LstmQuantizedFpCudaWorkspaceBreakdown breakdown;
    std::size_t quantized_input = 0;
    std::size_t quantized_weight_ih = 0;
    std::size_t quantized_weight_hh = 0;
    std::size_t quantized_bias = 0;
    std::size_t quantized_state = 0;
    std::size_t input_linear = 0;
    std::size_t recurrent_linear = 0;
    std::size_t weight_sum = 0;
    std::size_t device_parameters = 0;
};

struct StaticParameterLayout {
    std::size_t quantized_weight_ih = 0;
    std::size_t quantized_weight_hh = 0;
    std::size_t quantized_bias = 0;
    std::size_t weight_sum = 0;
    std::size_t total_bytes = 0;
};

void appendRegion(std::size_t bytes, std::size_t* cursor, std::size_t* offset) {
    const std::size_t aligned = alignUp(*cursor);
    *offset = aligned;
    *cursor = checkedAdd(aligned, bytes, "workspace");
}

WorkspaceLayout makeWorkspaceLayout(const LstmShape& shape, bool bias_enabled) {
    const std::size_t steps = positiveSize(shape.sequence_length, "sequence_length");
    const std::size_t batch = positiveSize(shape.batch_size, "batch_size");
    const std::size_t input = positiveSize(shape.input_size, "input_size");
    const std::size_t hidden = positiveSize(shape.hidden_size, "hidden_size");
    const std::size_t channels = checkedMul(kGateCount, hidden, "4H");
    const std::size_t sequence_batch = checkedMul(steps, batch, "T*B");
    const std::size_t states = checkedMul(batch, hidden, "B*H");

    WorkspaceLayout layout;
    auto& result = layout.breakdown;
    result.quantized_input_bytes =
        floatBytes(checkedMul(sequence_batch, input, "T*B*I"), "quantized input");
    result.quantized_weight_ih_bytes =
        floatBytes(checkedMul(channels, input, "4H*I"), "quantized weight_ih");
    result.quantized_weight_hh_bytes =
        floatBytes(checkedMul(channels, hidden, "4H*H"), "quantized weight_hh");
    result.quantized_bias_bytes =
        bias_enabled ? floatBytes(checkedMul(2, channels, "2*4H"), "quantized bias") : 0;
    result.quantized_state_bytes = floatBytes(checkedMul(2, states, "2*B*H"), "quantized states");
    result.input_linear_bytes =
        floatBytes(checkedMul(sequence_batch, channels, "T*B*4H"), "input linear");
    result.recurrent_linear_bytes =
        floatBytes(checkedMul(batch, channels, "B*4H"), "recurrent linear");
    result.weight_sum_bytes = floatBytes(checkedMul(2, channels, "2*4H"), "weight sums");
    result.device_parameter_bytes =
        checkedMul(channels, sizeof(DeviceLinearChannelParams), "device linear parameters");

    std::size_t cursor = 0;
    appendRegion(result.quantized_input_bytes, &cursor, &layout.quantized_input);
    appendRegion(result.quantized_weight_ih_bytes, &cursor, &layout.quantized_weight_ih);
    appendRegion(result.quantized_weight_hh_bytes, &cursor, &layout.quantized_weight_hh);
    appendRegion(result.quantized_bias_bytes, &cursor, &layout.quantized_bias);
    appendRegion(result.quantized_state_bytes, &cursor, &layout.quantized_state);
    appendRegion(result.input_linear_bytes, &cursor, &layout.input_linear);
    appendRegion(result.recurrent_linear_bytes, &cursor, &layout.recurrent_linear);
    appendRegion(result.weight_sum_bytes, &cursor, &layout.weight_sum);
    appendRegion(result.device_parameter_bytes, &cursor, &layout.device_parameters);
    result.total_bytes = alignUp(cursor);

    std::size_t payload = 0;
    payload = checkedAdd(payload, result.quantized_input_bytes, "workspace");
    payload = checkedAdd(payload, result.quantized_weight_ih_bytes, "workspace");
    payload = checkedAdd(payload, result.quantized_weight_hh_bytes, "workspace");
    payload = checkedAdd(payload, result.quantized_bias_bytes, "workspace");
    payload = checkedAdd(payload, result.quantized_state_bytes, "workspace");
    payload = checkedAdd(payload, result.input_linear_bytes, "workspace");
    payload = checkedAdd(payload, result.recurrent_linear_bytes, "workspace");
    payload = checkedAdd(payload, result.weight_sum_bytes, "workspace");
    payload = checkedAdd(payload, result.device_parameter_bytes, "workspace");
    result.alignment_padding_bytes = result.total_bytes - payload;
    return layout;
}

StaticParameterLayout makeStaticParameterLayout(const LstmShape& shape, bool bias_enabled) {
    const auto workspace = makeWorkspaceLayout(shape, bias_enabled);
    const auto& breakdown = workspace.breakdown;
    StaticParameterLayout layout;
    std::size_t cursor = 0;
    appendRegion(breakdown.quantized_weight_ih_bytes, &cursor, &layout.quantized_weight_ih);
    appendRegion(breakdown.quantized_weight_hh_bytes, &cursor, &layout.quantized_weight_hh);
    appendRegion(breakdown.quantized_bias_bytes, &cursor, &layout.quantized_bias);
    appendRegion(breakdown.weight_sum_bytes, &cursor, &layout.weight_sum);
    layout.total_bytes = alignUp(cursor);
    return layout;
}

bool sameType(const quantization::QuantizationType& lhs,
              const quantization::QuantizationType& rhs) {
    return lhs.bitwidth == rhs.bitwidth && lhs.is_unsigned == rhs.is_unsigned &&
           lhs.is_symmetric == rhs.is_symmetric;
}

bool samePoint(const QuantizedPoint& lhs, const QuantizedPoint& rhs) {
    return lhs.param.scale == rhs.param.scale && lhs.param.zero_point == rhs.param.zero_point &&
           sameType(lhs.type, rhs.type);
}

bool sameRescale(const ExecutionRescale& lhs, const ExecutionRescale& rhs) {
    if (lhs.kind != rhs.kind) {
        return false;
    }
    if (lhs.kind == ExecutionRescaleKind::MShift) {
        return lhs.m_shift.multiplier == rhs.m_shift.multiplier &&
               lhs.m_shift.shift == rhs.m_shift.shift;
    }
    return lhs.pot2.shift == rhs.pot2.shift;
}

bool sameLinear(const LinearExecutionParams& lhs, const LinearExecutionParams& rhs) {
    if (!samePoint(lhs.input, rhs.input) || !samePoint(lhs.output, rhs.output) ||
        lhs.weights.size() != rhs.weights.size() || lhs.biases.size() != rhs.biases.size() ||
        lhs.bias_to_accumulator.size() != rhs.bias_to_accumulator.size() ||
        lhs.accumulator_to_output.size() != rhs.accumulator_to_output.size()) {
        return false;
    }
    for (std::size_t index = 0; index < lhs.weights.size(); ++index) {
        if (!samePoint(lhs.weights[index], rhs.weights[index]) ||
            !sameRescale(lhs.accumulator_to_output[index], rhs.accumulator_to_output[index])) {
            return false;
        }
    }
    for (std::size_t index = 0; index < lhs.biases.size(); ++index) {
        if (!samePoint(lhs.biases[index], rhs.biases[index]) ||
            !sameRescale(lhs.bias_to_accumulator[index], rhs.bias_to_accumulator[index])) {
            return false;
        }
    }
    return true;
}

void validateExecutionParams(const LstmShape& shape, const LstmOperatorQuantConfig& config,
                             const LstmQuantParams& quant_params,
                             const LstmExecutionParams& actual) {
    config.validate();
    quant_params.validate(config);
    if (shape.input_size != actual.input_size || shape.hidden_size != actual.hidden_size ||
        shape.hidden_size != quant_params.hidden_size) {
        throw std::invalid_argument("shape、LstmQuantParams 与 LstmExecutionParams 不一致");
    }

    const LstmExecutionParams expected =
        deriveLstmExecutionParams(config, quant_params, shape.input_size);
    if (!sameLinear(actual.input_hidden_linear, expected.input_hidden_linear) ||
        !sameLinear(actual.hidden_hidden_linear, expected.hidden_hidden_linear)) {
        throw std::invalid_argument("Linear 执行参数不是 resolved 参数的派生结果");
    }
    for (std::size_t gate = 0; gate < kGateCount; ++gate) {
        const auto& lhs = actual.gates[gate];
        const auto& rhs = expected.gates[gate];
        if (!samePoint(lhs.input_hidden_linear, rhs.input_hidden_linear) ||
            !samePoint(lhs.hidden_hidden_linear, rhs.hidden_hidden_linear) ||
            !samePoint(lhs.gate_input, rhs.gate_input) ||
            !samePoint(lhs.gate_output, rhs.gate_output) ||
            !sameRescale(lhs.input_hidden_to_gate, rhs.input_hidden_to_gate) ||
            !sameRescale(lhs.hidden_hidden_to_gate, rhs.hidden_hidden_to_gate)) {
            throw std::invalid_argument("Gate 执行参数不是 resolved 参数的派生结果");
        }
    }
    const auto& lhs_cell = actual.cell;
    const auto& rhs_cell = expected.cell;
    if (!samePoint(lhs_cell.forget_gate, rhs_cell.forget_gate) ||
        !samePoint(lhs_cell.old_cell, rhs_cell.old_cell) ||
        !samePoint(lhs_cell.input_gate, rhs_cell.input_gate) ||
        !samePoint(lhs_cell.cell_gate, rhs_cell.cell_gate) ||
        !samePoint(lhs_cell.new_cell, rhs_cell.new_cell) ||
        lhs_cell.forget_scale.multiplier != rhs_cell.forget_scale.multiplier ||
        lhs_cell.input_scale.multiplier != rhs_cell.input_scale.multiplier) {
        throw std::invalid_argument("Cell Q31 执行参数不是 resolved 参数的派生结果");
    }
    if (!samePoint(actual.hidden.output_gate, expected.hidden.output_gate) ||
        !samePoint(actual.hidden.cell_tanh, expected.hidden.cell_tanh) ||
        !samePoint(actual.hidden.output, expected.hidden.output) ||
        !sameRescale(actual.hidden.product_to_output, expected.hidden.product_to_output)) {
        throw std::invalid_argument("Hidden 执行参数不是 resolved 参数的派生结果");
    }
}

DeviceQuantPoint packPoint(const QuantizedPoint& point) {
    point.param.validate(point.type);
    const auto range = point.type.range();
    return {point.param.scale, static_cast<float>(point.param.zero_point),
            static_cast<float>(range.minimum), static_cast<float>(range.maximum)};
}

DeviceRescale packRescale(const ExecutionRescale& value) {
    if (value.kind == ExecutionRescaleKind::MShift) {
        if (value.m_shift.multiplier < 32768U) {
            throw std::invalid_argument("M+shift multiplier 未规范化");
        }
        return {DeviceRescaleKind::MShift, value.m_shift.multiplier, value.m_shift.shift};
    }
    if (value.kind == ExecutionRescaleKind::Pot2) {
        return {DeviceRescaleKind::Pot2, 0, value.pot2.shift};
    }
    throw std::invalid_argument("ExecutionRescaleKind 枚举值非法");
}

std::vector<DeviceLinearChannelParams> packLinearParameters(const LstmExecutionParams& params,
                                                            bool bias_enabled) {
    const std::size_t channels = params.input_hidden_linear.weights.size();
    std::vector<DeviceLinearChannelParams> packed(channels);
    for (std::size_t channel = 0; channel < channels; ++channel) {
        auto& destination = packed[channel];
        destination.weight_ih = packPoint(params.input_hidden_linear.weights[channel]);
        destination.weight_hh = packPoint(params.hidden_hidden_linear.weights[channel]);
        destination.input_linear = packPoint(params.input_hidden_linear.output);
        destination.recurrent_linear = packPoint(params.hidden_hidden_linear.output);
        destination.input_accumulator_to_linear =
            packRescale(params.input_hidden_linear.accumulator_to_output[channel]);
        destination.recurrent_accumulator_to_linear =
            packRescale(params.hidden_hidden_linear.accumulator_to_output[channel]);
        if (bias_enabled) {
            destination.bias_ih = packPoint(params.input_hidden_linear.biases[channel]);
            destination.bias_hh = packPoint(params.hidden_hidden_linear.biases[channel]);
            destination.bias_ih_to_accumulator =
                packRescale(params.input_hidden_linear.bias_to_accumulator[channel]);
            destination.bias_hh_to_accumulator =
                packRescale(params.hidden_hidden_linear.bias_to_accumulator[channel]);
        } else {
            destination.bias_ih = {1.0F, 0.0F, 0.0F, 0.0F};
            destination.bias_hh = {1.0F, 0.0F, 0.0F, 0.0F};
            destination.bias_ih_to_accumulator = {DeviceRescaleKind::Pot2, 0, 0};
            destination.bias_hh_to_accumulator = {DeviceRescaleKind::Pot2, 0, 0};
        }
    }
    return packed;
}

constexpr std::uint64_t kSignatureOffset = 1469598103934665603ULL;
constexpr std::uint64_t kSignaturePrime = 1099511628211ULL;

void appendSignatureBytes(std::uint64_t* result, const void* data, std::size_t count) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(data);
    for (std::size_t index = 0; index < count; ++index) {
        *result ^= bytes[index];
        *result *= kSignaturePrime;
    }
}

template <typename T>
void appendSignatureValue(std::uint64_t* result, const T& value) {
    appendSignatureBytes(result, &value, sizeof(value));
}

void appendPointSignature(std::uint64_t* result, const DeviceQuantPoint& point) {
    appendSignatureValue(result, point.scale);
    appendSignatureValue(result, point.zero_point);
    appendSignatureValue(result, point.minimum);
    appendSignatureValue(result, point.maximum);
}

void appendRescaleSignature(std::uint64_t* result, const DeviceRescale& rescale) {
    const auto kind = static_cast<std::uint8_t>(rescale.kind);
    appendSignatureValue(result, kind);
    appendSignatureValue(result, rescale.multiplier);
    appendSignatureValue(result, rescale.shift);
}

std::uint64_t parameterSignature(const std::vector<DeviceLinearChannelParams>& params) {
    std::uint64_t result = kSignatureOffset;
    for (const auto& value : params) {
        appendPointSignature(&result, value.weight_ih);
        appendPointSignature(&result, value.weight_hh);
        appendPointSignature(&result, value.bias_ih);
        appendPointSignature(&result, value.bias_hh);
        appendPointSignature(&result, value.input_linear);
        appendPointSignature(&result, value.recurrent_linear);
        appendRescaleSignature(&result, value.bias_ih_to_accumulator);
        appendRescaleSignature(&result, value.input_accumulator_to_linear);
        appendRescaleSignature(&result, value.bias_hh_to_accumulator);
        appendRescaleSignature(&result, value.recurrent_accumulator_to_linear);
    }
    return result == 0 ? 1 : result;
}

std::uint64_t staticParameterSignature(const std::vector<DeviceLinearChannelParams>& params,
                                       const LstmFloatWeights& weights, const LstmShape& shape,
                                       std::uint64_t cache_key, bool bias_enabled) {
    std::uint64_t result = parameterSignature(params);
    constexpr std::uint64_t kPrime = 1099511628211ULL;
    const auto append = [&result](std::uint64_t value) {
        for (int index = 0; index < 8; ++index) {
            result ^= static_cast<unsigned char>(value & 0xFFU);
            result *= kPrime;
            value >>= 8;
        }
    };
    append(cache_key);
    append(reinterpret_cast<std::uintptr_t>(weights.weight_ih));
    append(reinterpret_cast<std::uintptr_t>(weights.weight_hh));
    append(reinterpret_cast<std::uintptr_t>(weights.bias_ih));
    append(reinterpret_cast<std::uintptr_t>(weights.bias_hh));
    append(static_cast<std::uint64_t>(shape.input_size));
    append(static_cast<std::uint64_t>(shape.hidden_size));
    append(bias_enabled ? 1U : 0U);
    return result == 0 ? 1 : result;
}

DeviceScalarParams packScalarParameters(const LstmExecutionParams& params, bool bias_enabled) {
    DeviceScalarParams result{};
    result.input = packPoint(params.input_hidden_linear.input);
    result.hidden = packPoint(params.hidden.output);
    result.cell = packPoint(params.cell.old_cell);
    result.cell_tanh = packPoint(params.hidden.cell_tanh);
    for (std::size_t gate = 0; gate < kGateCount; ++gate) {
        result.gates[gate] = {packPoint(params.gates[gate].gate_input),
                              packPoint(params.gates[gate].gate_output),
                              packRescale(params.gates[gate].input_hidden_to_gate),
                              packRescale(params.gates[gate].hidden_hidden_to_gate)};
    }
    result.hidden_product_to_output = packRescale(params.hidden.product_to_output);
    if (params.cell.forget_scale.multiplier < 0 || params.cell.input_scale.multiplier < 0) {
        throw std::invalid_argument("Cell Q31 multiplier 不能为负");
    }
    result.cell_forget_q31 = params.cell.forget_scale.multiplier;
    result.cell_input_q31 = params.cell.input_scale.multiplier;
    result.bias_enabled = bias_enabled ? 1 : 0;
    return result;
}

void validateStream(cudaStream_t stream, const char* name) {
    if (stream == nullptr) {
        throw std::invalid_argument(std::string(name) + " 不能为空");
    }
    unsigned int flags = 0;
    checkCuda(cudaStreamGetFlags(stream, &flags), name);
}

void validateEvent(cudaEvent_t event, const char* name) {
    if (event == nullptr) {
        throw std::invalid_argument(std::string(name) + " 不能为空");
    }
    const cudaError_t status = cudaEventQuery(event);
    if (status != cudaSuccess && status != cudaErrorNotReady) {
        checkCuda(status, name);
    }
}

void validateHandle(cublasHandle_t handle, const char* name) {
    if (handle == nullptr) {
        throw std::invalid_argument(std::string(name) + " 不能为空");
    }
    int version = 0;
    checkCublas(cublasGetVersion(handle, &version), name);
}

void validateContext(const LstmQuantizedFpCudaContext& context) {
    if (context.handles[0] == context.handles[1] || context.streams[0] == context.streams[1] ||
        context.events[0] == context.events[1]) {
        throw std::invalid_argument("CUDA context 必须提供两个不同的 handle/stream/event");
    }
    validateHandle(context.handles[0], "recurrent cuBLAS handle");
    validateHandle(context.handles[1], "input cuBLAS handle");
    validateStream(context.streams[0], "recurrent CUDA stream");
    validateStream(context.streams[1], "input CUDA stream");
    validateEvent(context.events[0], "input-ready CUDA event");
    validateEvent(context.events[1], "completion CUDA event");
}

void validateTimingEvents(const LstmQuantizedFpCudaTimingEvents* timing_events) {
    if (timing_events == nullptr) {
        return;
    }
    const std::array<cudaEvent_t, 6> events{timing_events->start,
                                            timing_events->input_quantized,
                                            timing_events->recurrent_quantized,
                                            timing_events->quantization_complete,
                                            timing_events->core_complete,
                                            timing_events->complete};
    for (std::size_t index = 0; index < events.size(); ++index) {
        validateEvent(events[index], "timing CUDA event");
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (events[index] == events[previous]) {
                throw std::invalid_argument("timing CUDA events 必须互不相同");
            }
        }
    }
}

void validateDeviceSpan(const void* pointer, std::size_t bytes, int expected_device,
                        const char* name) {
    if (pointer == nullptr) {
        throw std::invalid_argument(std::string(name) + " 不能为空");
    }
    if ((reinterpret_cast<std::uintptr_t>(pointer) % alignof(float)) != 0) {
        throw std::invalid_argument(std::string(name) + " 未按 float 对齐");
    }
    cudaPointerAttributes attributes{};
    checkCuda(cudaPointerGetAttributes(&attributes, pointer), name);
#if CUDART_VERSION >= 10000
    if (attributes.type != cudaMemoryTypeDevice) {
#else
    if (attributes.memoryType != cudaMemoryTypeDevice) {
#endif
        throw std::invalid_argument(std::string(name) + " 必须是 CUDA device pointer");
    }
    if (attributes.device != expected_device) {
        throw std::invalid_argument(std::string(name) + " 不属于当前 CUDA device");
    }
    // CUDA Runtime 不暴露 allocation span 查询；容量由公开 shape（以及
    // workspace.bytes）契约约束，这里仍严格校验指针类型、对齐和 device。
    (void)bytes;
}

void validateOptionalDeviceSpan(const void* pointer, std::size_t bytes, int expected_device,
                                const char* name) {
    if (pointer != nullptr) {
        validateDeviceSpan(pointer, bytes, expected_device, name);
    }
}

template <typename T>
T* region(void* base, std::size_t offset) {
    return reinterpret_cast<T*>(static_cast<unsigned char*>(base) + offset);
}

unsigned int oneDimensionalBlocks(std::size_t elements) {
    const std::size_t needed =
        (elements + static_cast<std::size_t>(kThreads) - 1) / static_cast<std::size_t>(kThreads);
    return static_cast<unsigned int>(std::min<std::size_t>(needed, kMaximumOneDimensionalBlocks));
}

__device__ __forceinline__ float clampDevice(float value, const DeviceQuantPoint& point,
                                             std::uint8_t* clamped = nullptr) {
    if (clamped != nullptr) {
        *clamped = value < point.minimum || value > point.maximum;
    }
    return fminf(point.maximum, fmaxf(point.minimum, value));
}

__device__ __forceinline__ float quantizeMasterDevice(float value, const DeviceQuantPoint& point,
                                                      std::uint8_t* clamped = nullptr) {
    const double translated = static_cast<double>(value) / static_cast<double>(point.scale) +
                              static_cast<double>(point.zero_point);
    if (clamped != nullptr) {
        *clamped = translated < static_cast<double>(point.minimum) ||
                   translated > static_cast<double>(point.maximum);
    }
    if (translated <= static_cast<double>(point.minimum)) {
        return point.minimum;
    }
    if (translated >= static_cast<double>(point.maximum)) {
        return point.maximum;
    }
    return static_cast<float>(quantization::roundToNearestEven(translated));
}

__device__ __forceinline__ float dequantizeDevice(float value, const DeviceQuantPoint& point) {
    return static_cast<float>(static_cast<double>(value - point.zero_point) *
                              static_cast<double>(point.scale));
}

__device__ __forceinline__ float applyRescaleDevice(float value, const DeviceRescale& encoded) {
    return cuda_detail::applyEncodedRescaleCore(value, encoded.kind == DeviceRescaleKind::MShift,
                                                encoded.multiplier, encoded.shift);
}

__device__ __forceinline__ float realActivationDevice(float quantized_input,
                                                      const DeviceGateParams& params,
                                                      bool tanh_activation,
                                                      std::uint8_t* clamped = nullptr) {
    bool activation_clamped = false;
    const float result = cuda_detail::realActivationCore(
        quantized_input, params.input.scale, static_cast<std::int32_t>(params.input.zero_point),
        params.output.scale, static_cast<std::int32_t>(params.output.zero_point),
        static_cast<std::int32_t>(params.output.minimum),
        static_cast<std::int32_t>(params.output.maximum),
        tanh_activation ? quantization::RealActivationKind::Tanh
                        : quantization::RealActivationKind::Sigmoid,
        &activation_clamped);
    if (clamped != nullptr) {
        *clamped = activation_clamped;
    }
    return result;
}

__global__ void quantizeScalarKernel(const float* source, float* destination, float* saved,
                                     std::uint8_t* clamped, std::size_t count,
                                     DeviceQuantPoint point) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        std::uint8_t value_clamped = 0;
        const float quantized = quantizeMasterDevice(source[index], point, &value_clamped);
        destination[index] = quantized;
        if (saved != nullptr) {
            saved[index] = quantized;
        }
        if (clamped != nullptr) {
            clamped[index] = value_clamped;
        }
    }
}

__global__ void quantizeInitialStateKernel(const float* source, float* destination, float* saved,
                                           std::uint8_t* clamped, std::size_t count,
                                           DeviceQuantPoint point) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        std::uint8_t value_clamped = 0;
        const float quantized = source == nullptr
                                    ? point.zero_point
                                    : quantizeMasterDevice(source[index], point, &value_clamped);
        destination[index] = quantized;
        if (saved != nullptr) {
            saved[index] = quantized;
        }
        if (clamped != nullptr) {
            clamped[index] = value_clamped;
        }
    }
}

template <bool InputWeight>
__global__ void quantizeWeightKernel(const float* source, float* destination, std::size_t count,
                                     std::size_t reduction,
                                     const DeviceLinearChannelParams* channel_params, float* saved,
                                     std::uint8_t* clamped) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        const std::size_t channel = index / reduction;
        const DeviceQuantPoint point =
            InputWeight ? channel_params[channel].weight_ih : channel_params[channel].weight_hh;
        std::uint8_t value_clamped = 0;
        const float quantized = quantizeMasterDevice(source[index], point, &value_clamped);
        destination[index] = quantized;
        if (saved != nullptr) {
            saved[index] = quantized;
        }
        if (clamped != nullptr) {
            clamped[index] = value_clamped;
        }
    }
}

template <bool InputBias>
__global__ void quantizeBiasKernel(const float* source, float* destination, std::size_t channels,
                                   const DeviceLinearChannelParams* channel_params, float* saved,
                                   std::uint8_t* clamped) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t channel = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         channel < channels; channel += stride) {
        const DeviceQuantPoint point =
            InputBias ? channel_params[channel].bias_ih : channel_params[channel].bias_hh;
        std::uint8_t value_clamped = 0;
        const float quantized = quantizeMasterDevice(source[channel], point, &value_clamped);
        destination[channel] = quantized;
        if (saved != nullptr) {
            saved[channel] = quantized;
        }
        if (clamped != nullptr) {
            clamped[channel] = value_clamped;
        }
    }
}

__global__ void dequantizeQCarrierKernel(const float* source, float* destination, std::size_t count,
                                         std::size_t elements_per_channel, const float* scales,
                                         const float* zero_points, std::size_t parameter_count) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        const std::size_t parameter_index = parameter_count == 1 ? 0 : index / elements_per_channel;
        destination[index] =
            static_cast<float>(static_cast<double>(source[index] - zero_points[parameter_index]) *
                               static_cast<double>(scales[parameter_index]));
    }
}

__global__ void weightSumKernel(const float* weights, float* sums, std::size_t channels,
                                std::size_t reduction) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t channel = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         channel < channels; channel += stride) {
        float sum = 0.0F;
        const float* row = weights + channel * reduction;
        for (std::size_t index = 0; index < reduction; ++index) {
            sum += row[index];
        }
        sums[channel] = sum;
    }
}

__global__ void fusedLstmPointwiseKernel(std::size_t batch, std::size_t hidden, std::size_t time,
                                         const float* input_gemm, const float* recurrent_gemm,
                                         const float* bias_ih, const float* bias_hh,
                                         const float* weight_ih_sums, const float* weight_hh_sums,
                                         const DeviceLinearChannelParams* channel_params,
                                         DeviceScalarParams scalar_params, float* hidden_state,
                                         float* cell_state, float* output,
                                         LstmQuantizedFpCudaCheckpoints checkpoints) {
    __shared__ float shared_bias_ih[32][4];
    __shared__ float shared_bias_hh[32][4];

    const std::size_t hidden_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (threadIdx.y == 0) {
#pragma unroll
        for (int gate = 0; gate < 4; ++gate) {
            float input_bias = 0.0F;
            float recurrent_bias = 0.0F;
            if (hidden_index < hidden && scalar_params.bias_enabled != 0) {
                const std::size_t channel = static_cast<std::size_t>(gate) * hidden + hidden_index;
                input_bias = bias_ih[channel];
                recurrent_bias = bias_hh[channel];
            }
            shared_bias_ih[threadIdx.x][gate] = input_bias;
            shared_bias_hh[threadIdx.x][gate] = recurrent_bias;
        }
    }
    __syncthreads();

    // 部分 block 的无效 row 只能在 shared 协作和同步完成后退出。
    if (hidden_index >= hidden) {
        return;
    }

    const std::size_t channels = 4 * hidden;
    const std::size_t batch_stride = static_cast<std::size_t>(gridDim.y) * blockDim.y;
    for (std::size_t batch_index = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
         batch_index < batch; batch_index += batch_stride) {
        float gate_outputs[4];
#pragma unroll
        for (int gate = 0; gate < 4; ++gate) {
            const std::size_t channel = static_cast<std::size_t>(gate) * hidden + hidden_index;
            const std::size_t linear_index = batch_index * channels + channel;
            const DeviceLinearChannelParams& linear = channel_params[channel];

            float input_accumulator =
                input_gemm[linear_index] - weight_ih_sums[channel] * scalar_params.input.zero_point;
            float recurrent_accumulator = recurrent_gemm[linear_index] -
                                          weight_hh_sums[channel] * scalar_params.hidden.zero_point;
            if (scalar_params.bias_enabled != 0) {
                input_accumulator += applyRescaleDevice(
                    shared_bias_ih[threadIdx.x][gate] - linear.bias_ih.zero_point,
                    linear.bias_ih_to_accumulator);
                recurrent_accumulator += applyRescaleDevice(
                    shared_bias_hh[threadIdx.x][gate] - linear.bias_hh.zero_point,
                    linear.bias_hh_to_accumulator);
            }

            std::uint8_t input_linear_clamped = 0;
            std::uint8_t recurrent_linear_clamped = 0;
            std::uint8_t gate_input_clamped = 0;
            std::uint8_t gate_output_clamped = 0;
            const float input_linear = clampDevice(
                applyRescaleDevice(input_accumulator, linear.input_accumulator_to_linear) +
                    linear.input_linear.zero_point,
                linear.input_linear, &input_linear_clamped);
            const float recurrent_linear = clampDevice(
                applyRescaleDevice(recurrent_accumulator, linear.recurrent_accumulator_to_linear) +
                    linear.recurrent_linear.zero_point,
                linear.recurrent_linear, &recurrent_linear_clamped);
            const DeviceGateParams& gate_params = scalar_params.gates[gate];
            const float gate_input = clampDevice(
                applyRescaleDevice(input_linear - linear.input_linear.zero_point,
                                   gate_params.input_linear_to_gate) +
                    applyRescaleDevice(recurrent_linear - linear.recurrent_linear.zero_point,
                                       gate_params.recurrent_linear_to_gate) +
                    gate_params.input.zero_point,
                gate_params.input, &gate_input_clamped);
            const float gate_output =
                realActivationDevice(gate_input, gate_params, gate == 2, &gate_output_clamped);
            gate_outputs[gate] = gate_output;

            const std::size_t checkpoint_index = (time * batch + batch_index) * channels + channel;
            if (checkpoints.weight_ih_linear != nullptr) {
                checkpoints.weight_ih_linear[checkpoint_index] = input_linear;
            }
            if (checkpoints.weight_hh_linear != nullptr) {
                checkpoints.weight_hh_linear[checkpoint_index] = recurrent_linear;
            }
            if (checkpoints.gate_inputs != nullptr) {
                checkpoints.gate_inputs[checkpoint_index] = gate_input;
            }
            if (checkpoints.gate_outputs != nullptr) {
                checkpoints.gate_outputs[checkpoint_index] = gate_output;
            }
            if (checkpoints.weight_ih_linear_clamped != nullptr) {
                checkpoints.weight_ih_linear_clamped[checkpoint_index] = input_linear_clamped;
            }
            if (checkpoints.weight_hh_linear_clamped != nullptr) {
                checkpoints.weight_hh_linear_clamped[checkpoint_index] = recurrent_linear_clamped;
            }
            if (checkpoints.gate_inputs_clamped != nullptr) {
                checkpoints.gate_inputs_clamped[checkpoint_index] = gate_input_clamped;
            }
            if (checkpoints.gate_outputs_clamped != nullptr) {
                checkpoints.gate_outputs_clamped[checkpoint_index] = gate_output_clamped;
            }
        }

        const std::size_t state_index = batch_index * hidden + hidden_index;
        const float old_cell = cell_state[state_index];
        const cuda_detail::QuantizedCellFpCoreParams cell_params{
            scalar_params.gates[1].output.zero_point,
            scalar_params.cell.zero_point,
            scalar_params.gates[0].output.zero_point,
            scalar_params.gates[2].output.zero_point,
            scalar_params.cell_forget_q31,
            scalar_params.cell_input_q31,
            scalar_params.cell.zero_point,
            scalar_params.cell.minimum,
            scalar_params.cell.maximum};
        const auto cell_result = cuda_detail::computeQuantizedCellFpCore(
            gate_outputs[1], old_cell, gate_outputs[0], gate_outputs[2], cell_params);
        const float cell_pre_clamp =
            quantization::roundToNearestEven(cell_result.diagnostics.pre_round_sum) +
            scalar_params.cell.zero_point;
        const std::uint8_t cell_clamped = cell_pre_clamp < scalar_params.cell.minimum ||
                                          cell_pre_clamp > scalar_params.cell.maximum;
        const float next_cell = cell_result.value;
        cell_state[state_index] = next_cell;

        DeviceGateParams cell_tanh_params{scalar_params.cell,
                                          scalar_params.cell_tanh,
                                          {DeviceRescaleKind::Pot2, 0, 0},
                                          {DeviceRescaleKind::Pot2, 0, 0}};
        std::uint8_t cell_tanh_clamped = 0;
        const float cell_tanh =
            realActivationDevice(next_cell, cell_tanh_params, true, &cell_tanh_clamped);
        const auto hidden_result = cuda_detail::computeQuantizedHiddenFpEncodedCore(
            gate_outputs[3], scalar_params.gates[3].output.zero_point, cell_tanh,
            scalar_params.cell_tanh.zero_point,
            scalar_params.hidden_product_to_output.kind == DeviceRescaleKind::MShift,
            scalar_params.hidden_product_to_output.multiplier,
            scalar_params.hidden_product_to_output.shift, scalar_params.hidden.zero_point,
            scalar_params.hidden.minimum, scalar_params.hidden.maximum);
        const float hidden_pre_clamp = applyRescaleDevice(hidden_result.diagnostics.raw_product,
                                                          scalar_params.hidden_product_to_output) +
                                       scalar_params.hidden.zero_point;
        const std::uint8_t hidden_clamped = hidden_pre_clamp < scalar_params.hidden.minimum ||
                                            hidden_pre_clamp > scalar_params.hidden.maximum;
        const float next_hidden = hidden_result.value;
        hidden_state[state_index] = next_hidden;

        const std::size_t output_index = (time * batch + batch_index) * hidden + hidden_index;
        output[output_index] = dequantizeDevice(next_hidden, scalar_params.hidden);
        if (checkpoints.cell_states != nullptr) {
            checkpoints.cell_states[output_index] = next_cell;
        }
        if (checkpoints.cell_tanh_outputs != nullptr) {
            checkpoints.cell_tanh_outputs[output_index] = cell_tanh;
        }
        if (checkpoints.hidden_outputs != nullptr) {
            checkpoints.hidden_outputs[output_index] = next_hidden;
        }
        if (checkpoints.cell_states_clamped != nullptr) {
            checkpoints.cell_states_clamped[output_index] = cell_clamped;
        }
        if (checkpoints.cell_tanh_outputs_clamped != nullptr) {
            checkpoints.cell_tanh_outputs_clamped[output_index] = cell_tanh_clamped;
        }
        if (checkpoints.hidden_outputs_clamped != nullptr) {
            checkpoints.hidden_outputs_clamped[output_index] = hidden_clamped;
        }
    }
}

__global__ void dequantizeFinalStatesKernel(const float* hidden_state, const float* cell_state,
                                            float* final_hidden, float* final_cell,
                                            std::size_t count, DeviceQuantPoint hidden_point,
                                            DeviceQuantPoint cell_point) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        final_hidden[index] = dequantizeDevice(hidden_state[index], hidden_point);
        final_cell[index] = dequantizeDevice(cell_state[index], cell_point);
    }
}

void checkKernel(const char* name) { checkCuda(cudaGetLastError(), name); }

}  // namespace

LstmQuantizedFpCudaWorkspaceBreakdown lstmQuantizedFpCudaWorkspaceBreakdown(const LstmShape& shape,
                                                                            bool bias_enabled) {
    return makeWorkspaceLayout(shape, bias_enabled).breakdown;
}

std::size_t lstmQuantizedFpCudaWorkspaceBytes(const LstmShape& shape, bool bias_enabled) {
    return makeWorkspaceLayout(shape, bias_enabled).breakdown.total_bytes;
}

std::size_t lstmQuantizedFpCudaStaticParameterBytes(const LstmShape& shape, bool bias_enabled) {
    return makeStaticParameterLayout(shape, bias_enabled).total_bytes;
}

void lstmForwardQuantizedFpCuda(
    const LstmShape& shape, const LstmFloatWeights& master_weights, const float* input,
    const float* initial_hidden, const float* initial_cell,
    const LstmOperatorQuantConfig& resolved_config, const LstmQuantParams& quant_params,
    const LstmExecutionParams& execution_params, float* output, float* final_hidden,
    float* final_cell, LstmQuantizedFpCudaContext& context, LstmQuantizedFpCudaMathMode math_mode,
    LstmQuantizedFpCudaWorkspace workspace, const LstmQuantizedFpCudaCheckpoints* checkpoints,
    LstmQuantizedFpCudaStats* stats, const LstmQuantizedFpCudaTimingEvents* timing_events,
    std::uint64_t static_parameter_cache_key, const LstmQuantizedFpCudaMasters* masters) {
    const WorkspaceLayout layout = makeWorkspaceLayout(shape, quant_params.bias_enabled);
    const StaticParameterLayout static_layout =
        makeStaticParameterLayout(shape, quant_params.bias_enabled);
    validateExecutionParams(shape, resolved_config, quant_params, execution_params);
    validateContext(context);
    validateTimingEvents(timing_events);
    if (math_mode != LstmQuantizedFpCudaMathMode::Pedantic &&
        math_mode != LstmQuantizedFpCudaMathMode::Tf32) {
        throw std::invalid_argument("CUDA math mode 枚举值非法");
    }
    if ((initial_hidden == nullptr) != (initial_cell == nullptr)) {
        throw std::invalid_argument("initial_hidden/initial_cell 必须同时提供或省略");
    }
    if (master_weights.weight_ih == nullptr || master_weights.weight_hh == nullptr ||
        input == nullptr || output == nullptr || final_hidden == nullptr || final_cell == nullptr) {
        throw std::invalid_argument("CUDA 量化前向必需指针不能为空");
    }
    if (quant_params.bias_enabled &&
        (master_weights.bias_ih == nullptr || master_weights.bias_hh == nullptr)) {
        throw std::invalid_argument("启用 bias 时双 bias 指针不能为空");
    }
    if (!quant_params.bias_enabled &&
        (master_weights.bias_ih != nullptr || master_weights.bias_hh != nullptr)) {
        throw std::invalid_argument("禁用 bias 时双 bias 指针必须为空");
    }
    if ((workspace.data == nullptr) != (workspace.bytes == 0)) {
        throw std::invalid_argument("workspace pointer/bytes 必须同时提供或同时省略");
    }
    if (workspace.data != nullptr && workspace.bytes < layout.breakdown.total_bytes) {
        throw std::invalid_argument("caller workspace bytes 不足");
    }
    if ((context.device_execution_params == nullptr) !=
        (context.device_execution_params_bytes == 0)) {
        throw std::invalid_argument(
            "context device execution params pointer/bytes 必须同时提供或省略");
    }
    if (context.device_execution_params != nullptr &&
        context.device_execution_params_bytes < layout.breakdown.device_parameter_bytes) {
        throw std::invalid_argument("context device execution params 容量不足");
    }
    if ((context.device_static_parameters == nullptr) !=
        (context.device_static_parameters_bytes == 0)) {
        throw std::invalid_argument("context static parameters pointer/bytes 必须同时提供或省略");
    }
    if (static_parameter_cache_key != 0 && context.device_static_parameters == nullptr) {
        throw std::invalid_argument("非零 static parameter cache key 要求提供 context cache");
    }
    if (context.device_static_parameters != nullptr &&
        context.device_static_parameters_bytes < static_layout.total_bytes) {
        throw std::invalid_argument("context static parameters 容量不足");
    }

    const int sequence_length =
        cuda_detail::checkedCublasInt(shape.sequence_length, "sequence_length");
    const int batch_size = cuda_detail::checkedCublasInt(shape.batch_size, "batch_size");
    const int input_size = cuda_detail::checkedCublasInt(shape.input_size, "input_size");
    const int hidden_size = cuda_detail::checkedCublasInt(shape.hidden_size, "hidden_size");
    if (hidden_size > std::numeric_limits<int>::max() / 4) {
        throw std::invalid_argument("4H 超出 CUDA/cuBLAS int 范围");
    }
    if (batch_size > std::numeric_limits<int>::max() / sequence_length) {
        throw std::invalid_argument("T*B 超出 cuBLAS int 范围");
    }
    const int sequence_batch = sequence_length * batch_size;
    const int channels_int = 4 * hidden_size;

    const std::size_t steps = static_cast<std::size_t>(sequence_length);
    const std::size_t batch = static_cast<std::size_t>(batch_size);
    const std::size_t input_width = static_cast<std::size_t>(input_size);
    const std::size_t hidden = static_cast<std::size_t>(hidden_size);
    const std::size_t channels = static_cast<std::size_t>(channels_int);
    const std::size_t sequence_batch_size = static_cast<std::size_t>(sequence_batch);
    const std::size_t states = checkedMul(batch, hidden, "B*H");
    const std::size_t sequence_states =
        checkedMul(checkedMul(steps, batch, "T*B"), hidden, "T*B*H");
    const std::size_t sequence_linears =
        checkedMul(checkedMul(steps, batch, "T*B"), channels, "T*B*4H");

    int current_device = 0;
    checkCuda(cudaGetDevice(&current_device), "cudaGetDevice");
    validateDeviceSpan(input,
                       floatBytes(checkedMul(sequence_batch_size, input_width, "T*B*I"), "input"),
                       current_device, "input");
    validateDeviceSpan(master_weights.weight_ih,
                       floatBytes(checkedMul(channels, input_width, "4H*I"), "weight_ih"),
                       current_device, "weight_ih");
    validateDeviceSpan(master_weights.weight_hh,
                       floatBytes(checkedMul(channels, hidden, "4H*H"), "weight_hh"),
                       current_device, "weight_hh");
    if (quant_params.bias_enabled) {
        validateDeviceSpan(master_weights.bias_ih, floatBytes(channels, "bias_ih"), current_device,
                           "bias_ih");
        validateDeviceSpan(master_weights.bias_hh, floatBytes(channels, "bias_hh"), current_device,
                           "bias_hh");
    }
    if (initial_hidden != nullptr) {
        validateDeviceSpan(initial_hidden, floatBytes(states, "h_0"), current_device,
                           "initial_hidden");
        validateDeviceSpan(initial_cell, floatBytes(states, "c_0"), current_device, "initial_cell");
    }
    validateDeviceSpan(output, floatBytes(sequence_states, "output"), current_device, "output");
    validateDeviceSpan(final_hidden, floatBytes(states, "h_n"), current_device, "final_hidden");
    validateDeviceSpan(final_cell, floatBytes(states, "c_n"), current_device, "final_cell");
    if (workspace.data != nullptr) {
        validateDeviceSpan(workspace.data, layout.breakdown.total_bytes, current_device,
                           "workspace");
        if ((reinterpret_cast<std::uintptr_t>(workspace.data) % kWorkspaceAlignment) != 0) {
            throw std::invalid_argument("caller workspace 必须至少 256-byte 对齐");
        }
    }
    if (context.device_execution_params != nullptr) {
        validateDeviceSpan(context.device_execution_params, layout.breakdown.device_parameter_bytes,
                           current_device, "context device execution params");
    }
    if (context.device_static_parameters != nullptr) {
        validateDeviceSpan(context.device_static_parameters, static_layout.total_bytes,
                           current_device, "context static parameters");
    }

    LstmQuantizedFpCudaCheckpoints device_checkpoints{};
    if (checkpoints != nullptr) {
        device_checkpoints = *checkpoints;
        const std::size_t linear_bytes = floatBytes(sequence_linears, "linear checkpoint");
        const std::size_t state_bytes = floatBytes(sequence_states, "state checkpoint");
        validateOptionalDeviceSpan(checkpoints->weight_ih_linear, linear_bytes, current_device,
                                   "checkpoint weight_ih_linear");
        validateOptionalDeviceSpan(checkpoints->weight_hh_linear, linear_bytes, current_device,
                                   "checkpoint weight_hh_linear");
        validateOptionalDeviceSpan(checkpoints->gate_inputs, linear_bytes, current_device,
                                   "checkpoint gate_inputs");
        validateOptionalDeviceSpan(checkpoints->gate_outputs, linear_bytes, current_device,
                                   "checkpoint gate_outputs");
        validateOptionalDeviceSpan(checkpoints->cell_states, state_bytes, current_device,
                                   "checkpoint cell_states");
        validateOptionalDeviceSpan(checkpoints->cell_tanh_outputs, state_bytes, current_device,
                                   "checkpoint cell_tanh_outputs");
        validateOptionalDeviceSpan(checkpoints->hidden_outputs, state_bytes, current_device,
                                   "checkpoint hidden_outputs");
        validateOptionalDeviceSpan(checkpoints->weight_ih_linear_clamped, sequence_linears,
                                   current_device, "checkpoint weight_ih_linear_clamped");
        validateOptionalDeviceSpan(checkpoints->weight_hh_linear_clamped, sequence_linears,
                                   current_device, "checkpoint weight_hh_linear_clamped");
        validateOptionalDeviceSpan(checkpoints->gate_inputs_clamped, sequence_linears,
                                   current_device, "checkpoint gate_inputs_clamped");
        validateOptionalDeviceSpan(checkpoints->gate_outputs_clamped, sequence_linears,
                                   current_device, "checkpoint gate_outputs_clamped");
        validateOptionalDeviceSpan(checkpoints->cell_states_clamped, sequence_states,
                                   current_device, "checkpoint cell_states_clamped");
        validateOptionalDeviceSpan(checkpoints->cell_tanh_outputs_clamped, sequence_states,
                                   current_device, "checkpoint cell_tanh_outputs_clamped");
        validateOptionalDeviceSpan(checkpoints->hidden_outputs_clamped, sequence_states,
                                   current_device, "checkpoint hidden_outputs_clamped");
    }

    LstmQuantizedFpCudaMasters device_masters{};
    if (masters != nullptr) {
        device_masters = *masters;
        const std::size_t input_elements = checkedMul(sequence_batch_size, input_width, "T*B*I");
        const std::size_t weight_ih_elements = checkedMul(channels, input_width, "4H*I");
        const std::size_t weight_hh_elements = checkedMul(channels, hidden, "4H*H");
        const auto require_span = [&](const void* pointer, std::size_t bytes, const char* name) {
            if (pointer == nullptr) {
                throw std::invalid_argument(std::string(name) + " 不能为空");
            }
            validateDeviceSpan(pointer, bytes, current_device, name);
        };
        require_span(masters->input, floatBytes(input_elements, "saved input"), "saved input");
        require_span(masters->weight_ih, floatBytes(weight_ih_elements, "saved weight_ih"),
                     "saved weight_ih");
        require_span(masters->weight_hh, floatBytes(weight_hh_elements, "saved weight_hh"),
                     "saved weight_hh");
        require_span(masters->initial_hidden, floatBytes(states, "saved h_0"), "saved h_0");
        require_span(masters->initial_cell, floatBytes(states, "saved c_0"), "saved c_0");
        require_span(masters->input_clamped, input_elements, "saved input mask");
        require_span(masters->weight_ih_clamped, weight_ih_elements, "saved weight_ih mask");
        require_span(masters->weight_hh_clamped, weight_hh_elements, "saved weight_hh mask");
        require_span(masters->initial_hidden_clamped, states, "saved h_0 mask");
        require_span(masters->initial_cell_clamped, states, "saved c_0 mask");
        if (quant_params.bias_enabled) {
            require_span(masters->bias_ih, floatBytes(channels, "saved bias_ih"), "saved bias_ih");
            require_span(masters->bias_hh, floatBytes(channels, "saved bias_hh"), "saved bias_hh");
            require_span(masters->bias_ih_clamped, channels, "saved bias_ih mask");
            require_span(masters->bias_hh_clamped, channels, "saved bias_hh mask");
        } else if (masters->bias_ih != nullptr || masters->bias_hh != nullptr ||
                   masters->bias_ih_clamped != nullptr || masters->bias_hh_clamped != nullptr) {
            throw std::invalid_argument("bias disabled 时 saved bias 与 mask 必须为空");
        }
    }

    cudaDeviceProp properties{};
    checkCuda(cudaGetDeviceProperties(&properties, current_device), "cudaGetDeviceProperties");
    const dim3 pointwise_block(32, 8);
    const unsigned int grid_x =
        static_cast<unsigned int>((hidden + pointwise_block.x - 1) / pointwise_block.x);
    const unsigned int requested_grid_y = static_cast<unsigned int>(
        std::min<std::size_t>((batch + pointwise_block.y - 1) / pointwise_block.y,
                              static_cast<std::size_t>(properties.maxGridSize[1])));
    if (grid_x == 0 || grid_x > static_cast<unsigned int>(properties.maxGridSize[0]) ||
        requested_grid_y == 0) {
        throw std::invalid_argument("shape 超出 pointwise CUDA grid 范围");
    }
    const dim3 pointwise_grid(grid_x, requested_grid_y);

    const std::vector<DeviceLinearChannelParams> host_channel_params =
        packLinearParameters(execution_params, quant_params.bias_enabled);
    if (host_channel_params.size() != channels) {
        throw std::invalid_argument("device Linear 参数必须恰好展开为 4H");
    }
    const DeviceScalarParams scalar_params =
        packScalarParameters(execution_params, quant_params.bias_enabled);

    const bool uses_internal_workspace = workspace.data == nullptr;
    cuda_detail::OwnedMultiStreamWorkspace owned_workspace(
        uses_internal_workspace ? layout.breakdown.total_bytes : 0, context.streams[0],
        context.streams[1], context.events[0]);
    if (uses_internal_workspace) {
        workspace.data = owned_workspace.get();
        workspace.bytes = layout.breakdown.total_bytes;
    }

    float* quantized_input = region<float>(workspace.data, layout.quantized_input);
    const bool static_cache_enabled = static_parameter_cache_key != 0;
    void* static_base = context.device_static_parameters;
    float* quantized_weight_ih = static_cache_enabled
                                     ? region<float>(static_base, static_layout.quantized_weight_ih)
                                     : region<float>(workspace.data, layout.quantized_weight_ih);
    float* quantized_weight_hh = static_cache_enabled
                                     ? region<float>(static_base, static_layout.quantized_weight_hh)
                                     : region<float>(workspace.data, layout.quantized_weight_hh);
    float* quantized_bias_ih =
        quant_params.bias_enabled
            ? (static_cache_enabled ? region<float>(static_base, static_layout.quantized_bias)
                                    : region<float>(workspace.data, layout.quantized_bias))
            : nullptr;
    float* quantized_bias_hh = quant_params.bias_enabled ? quantized_bias_ih + channels : nullptr;
    float* quantized_hidden = region<float>(workspace.data, layout.quantized_state);
    float* quantized_cell = quantized_hidden + states;
    float* input_linear = region<float>(workspace.data, layout.input_linear);
    float* recurrent_linear = region<float>(workspace.data, layout.recurrent_linear);
    float* weight_ih_sums = static_cache_enabled
                                ? region<float>(static_base, static_layout.weight_sum)
                                : region<float>(workspace.data, layout.weight_sum);
    float* weight_hh_sums = weight_ih_sums + channels;
    auto* device_channel_params =
        context.device_execution_params != nullptr
            ? static_cast<DeviceLinearChannelParams*>(context.device_execution_params)
            : region<DeviceLinearChannelParams>(workspace.data, layout.device_parameters);

    const cuda_detail::CublasMathMode shared_math_mode =
        math_mode == LstmQuantizedFpCudaMathMode::Pedantic ? cuda_detail::CublasMathMode::Pedantic
                                                           : cuda_detail::CublasMathMode::Tf32;
    cuda_detail::ScopedCublasSettings recurrent_settings(context.handles[0], context.streams[0],
                                                         shared_math_mode);
    cuda_detail::ScopedCublasSettings input_settings(context.handles[1], context.streams[1],
                                                     shared_math_mode);
    const std::uint64_t signature = parameterSignature(host_channel_params);
    const bool parameters_cached = context.device_execution_params != nullptr &&
                                   context.cached_execution_signature == signature;
    const std::uint64_t static_signature =
        static_cache_enabled
            ? staticParameterSignature(host_channel_params, master_weights, shape,
                                       static_parameter_cache_key, quant_params.bias_enabled)
            : 0;
    const bool static_parameters_cached =
        static_cache_enabled && context.cached_static_parameter_signature == static_signature;
    const bool reuse_static_parameters = static_parameters_cached && masters == nullptr;

    if (timing_events != nullptr) {
        checkCuda(cudaEventRecord(timing_events->start, context.streams[0]),
                  "record quantized FP timing start");
        checkCuda(cudaStreamWaitEvent(context.streams[1], timing_events->start, 0),
                  "input stream wait timing start");
    }

    if (!parameters_cached) {
        // 同步只保护临时 host flatten buffer 的生命周期；启用 context cache
        // 后，相同执行参数的后续调用不会重复上传。
        checkCuda(cudaMemcpyAsync(device_channel_params, host_channel_params.data(),
                                  layout.breakdown.device_parameter_bytes, cudaMemcpyHostToDevice,
                                  context.streams[0]),
                  "copy flattened 4H device parameters");
        checkCuda(cudaStreamSynchronize(context.streams[0]), "wait flattened 4H device parameters");
        if (context.device_execution_params != nullptr) {
            context.cached_execution_signature = signature;
        }
    }
    checkCuda(cudaEventRecord(context.events[1], context.streams[0]),
              "record device-parameter-ready event");
    checkCuda(cudaStreamWaitEvent(context.streams[1], context.events[1], 0),
              "input stream wait device parameters");
    const std::size_t input_elements = checkedMul(sequence_batch_size, input_width, "T*B*I");
    quantizeScalarKernel<<<oneDimensionalBlocks(input_elements), kThreads, 0, context.streams[1]>>>(
        input, quantized_input, device_masters.input, device_masters.input_clamped, input_elements,
        scalar_params.input);
    checkKernel("quantize input kernel");

    if (!reuse_static_parameters) {
        const std::size_t weight_ih_elements = checkedMul(channels, input_width, "4H*I");
        quantizeWeightKernel<true>
            <<<oneDimensionalBlocks(weight_ih_elements), kThreads, 0, context.streams[1]>>>(
                master_weights.weight_ih, quantized_weight_ih, weight_ih_elements, input_width,
                device_channel_params, device_masters.weight_ih, device_masters.weight_ih_clamped);
        checkKernel("quantize weight_ih kernel");
        weightSumKernel<<<oneDimensionalBlocks(channels), kThreads, 0, context.streams[1]>>>(
            quantized_weight_ih, weight_ih_sums, channels, input_width);
        checkKernel("weight_ih sum kernel");
    }
    if (timing_events != nullptr) {
        checkCuda(cudaEventRecord(timing_events->input_quantized, context.streams[1]),
                  "record input quantization complete");
    }

    if (!reuse_static_parameters) {
        const std::size_t weight_hh_elements = checkedMul(channels, hidden, "4H*H");
        quantizeWeightKernel<false>
            <<<oneDimensionalBlocks(weight_hh_elements), kThreads, 0, context.streams[0]>>>(
                master_weights.weight_hh, quantized_weight_hh, weight_hh_elements, hidden,
                device_channel_params, device_masters.weight_hh, device_masters.weight_hh_clamped);
        checkKernel("quantize weight_hh kernel");
        weightSumKernel<<<oneDimensionalBlocks(channels), kThreads, 0, context.streams[0]>>>(
            quantized_weight_hh, weight_hh_sums, channels, hidden);
        checkKernel("weight_hh sum kernel");

        if (quant_params.bias_enabled) {
            quantizeBiasKernel<true>
                <<<oneDimensionalBlocks(channels), kThreads, 0, context.streams[0]>>>(
                    master_weights.bias_ih, quantized_bias_ih, channels, device_channel_params,
                    device_masters.bias_ih, device_masters.bias_ih_clamped);
            checkKernel("quantize bias_ih kernel");
            quantizeBiasKernel<false>
                <<<oneDimensionalBlocks(channels), kThreads, 0, context.streams[0]>>>(
                    master_weights.bias_hh, quantized_bias_hh, channels, device_channel_params,
                    device_masters.bias_hh, device_masters.bias_hh_clamped);
            checkKernel("quantize bias_hh kernel");
        }
    }
    if (static_cache_enabled) {
        context.cached_static_parameter_signature = static_signature;
    }

    quantizeInitialStateKernel<<<oneDimensionalBlocks(states), kThreads, 0, context.streams[0]>>>(
        initial_hidden, quantized_hidden, device_masters.initial_hidden,
        device_masters.initial_hidden_clamped, states, scalar_params.hidden);
    checkKernel("quantize initial hidden kernel");
    quantizeInitialStateKernel<<<oneDimensionalBlocks(states), kThreads, 0, context.streams[0]>>>(
        initial_cell, quantized_cell, device_masters.initial_cell,
        device_masters.initial_cell_clamped, states, scalar_params.cell);
    checkKernel("quantize initial cell kernel");
    if (timing_events != nullptr) {
        checkCuda(cudaEventRecord(timing_events->recurrent_quantized, context.streams[0]),
                  "record recurrent quantization complete");
        checkCuda(cudaStreamWaitEvent(context.streams[0], timing_events->input_quantized, 0),
                  "recurrent stream wait input quantization");
        checkCuda(cudaEventRecord(timing_events->quantization_complete, context.streams[0]),
                  "record combined quantization complete");
        checkCuda(cudaStreamWaitEvent(context.streams[1], timing_events->quantization_complete, 0),
                  "input stream wait combined quantization");
    }

    cuda_detail::runGemm(context.handles[1], sequence_batch, channels_int, input_size,
                         quantized_input, quantized_weight_ih, input_linear);
    checkCuda(cudaEventRecord(context.events[0], context.streams[1]),
              "record all-time input GEMM event");

    for (int time = 0; time < sequence_length; ++time) {
        cuda_detail::runGemm(context.handles[0], batch_size, channels_int, hidden_size,
                             quantized_hidden, quantized_weight_hh, recurrent_linear);
        if (time == 0) {
            checkCuda(cudaStreamWaitEvent(context.streams[0], context.events[0], 0),
                      "pointwise stream wait all-time input GEMM");
        }
        fusedLstmPointwiseKernel<<<pointwise_grid, pointwise_block, 0, context.streams[0]>>>(
            batch, hidden, static_cast<std::size_t>(time),
            input_linear + static_cast<std::size_t>(time) * batch * channels, recurrent_linear,
            quantized_bias_ih, quantized_bias_hh, weight_ih_sums, weight_hh_sums,
            device_channel_params, scalar_params, quantized_hidden, quantized_cell, output,
            device_checkpoints);
        checkKernel("fused LSTM pointwise kernel");
    }
    if (timing_events != nullptr) {
        checkCuda(cudaEventRecord(timing_events->core_complete, context.streams[0]),
                  "record quantized core complete");
    }

    dequantizeFinalStatesKernel<<<oneDimensionalBlocks(states), kThreads, 0, context.streams[0]>>>(
        quantized_hidden, quantized_cell, final_hidden, final_cell, states, scalar_params.hidden,
        scalar_params.cell);
    checkKernel("dequantize final states kernel");
    if (timing_events != nullptr) {
        checkCuda(cudaEventRecord(timing_events->complete, context.streams[0]),
                  "record quantized FP timing complete");
    }

    input_settings.restore();
    recurrent_settings.restore();
    if (owned_workspace.get() != nullptr) {
        owned_workspace.release();
    }
    checkCuda(cudaEventRecord(context.events[1], context.streams[0]),
              "record quantized FP completion event");
    checkCuda(cudaStreamWaitEvent(context.streams[1], context.events[1], 0),
              "join input stream to completion event");

    if (stats != nullptr) {
        stats->input_gemm_calls = 1;
        stats->recurrent_gemm_calls = static_cast<std::uint64_t>(sequence_length);
        stats->workspace_bytes = layout.breakdown.total_bytes;
        stats->used_internal_workspace = uses_internal_workspace;
        stats->execution_parameter_cache_hit = parameters_cached;
        stats->static_parameter_cache_hit = reuse_static_parameters;
    }
}

void lstmDequantizeQCarrierCuda(const float* source, float* destination, std::size_t count,
                                std::size_t elements_per_channel, const float* scales,
                                const float* zero_points, std::size_t parameter_count,
                                cudaStream_t stream) {
    if (source == nullptr || destination == nullptr || scales == nullptr ||
        zero_points == nullptr) {
        throw std::invalid_argument("q-carrier 反量化指针不能为空");
    }
    if (count == 0 || parameter_count == 0 || elements_per_channel == 0 ||
        (parameter_count != 1 &&
         (count % parameter_count != 0 || count / parameter_count != elements_per_channel))) {
        throw std::invalid_argument("q-carrier 反量化 shape/参数数量不匹配");
    }
    dequantizeQCarrierKernel<<<oneDimensionalBlocks(count), kThreads, 0, stream>>>(
        source, destination, count, elements_per_channel, scales, zero_points, parameter_count);
    checkKernel("dequantize q-carrier kernel");
}

}  // namespace quant_lstm
