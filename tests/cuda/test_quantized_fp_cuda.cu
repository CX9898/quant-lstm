#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "cuda/cuda_test_support.cuh"
#include "golden_fixtures.h"
#include "lstm/forward_quantized_fp_cuda.h"
#include "lstm/gate_layout.h"
#include "lstm/quant_config_loader.h"
#include "quantization/fixed_point_ops.h"

namespace {

using Json = nlohmann::json;
using quant_lstm::test::DeviceBuffer;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

const Json& data(const Json& tensors, std::string_view name) {
    return tensors.at(std::string(name)).at("data");
}

template <typename T>
std::vector<T> values(const Json& tensors, std::string_view name) {
    return data(tensors, name).get<std::vector<T>>();
}

std::vector<float> floatValues(const Json& tensors, std::string_view name) {
    std::vector<float> result;
    for (const Json& item : data(tensors, name)) {
        result.push_back(quant_lstm::parseCanonicalFloat32Value(item.get<std::string>()));
    }
    return result;
}

struct GoldenCase {
    quant_lstm::LstmShape shape{};
    quant_lstm::LstmOperatorQuantConfig config;
    quant_lstm::LstmQuantParams params;
    quant_lstm::LstmExecutionParams execution;
    std::vector<std::int32_t> input;
    std::vector<std::int32_t> h0;
    std::vector<std::int32_t> c0;
    std::vector<std::int32_t> weight_ih;
    std::vector<std::int32_t> weight_hh;
    std::vector<std::int32_t> bias_ih;
    std::vector<std::int32_t> bias_hh;
};

GoldenCase loadCase(const Json& tensors) {
    GoldenCase result;
    const auto shape = values<std::int64_t>(tensors, "shape");
    require(shape.size() == 4, "Golden shape 必须为 T,B,I,H");
    result.shape = {shape[0], shape[1], shape[2], shape[3]};
    result.config.scale_mode =
        quant_lstm::parseScaleMode(data(tensors, "scale_mode").at(0).get<std::string>());
    const auto names = values<std::string>(tensors, "operator_names");
    const auto bitwidth = values<std::uint8_t>(tensors, "operator_bitwidth");
    const auto is_unsigned = values<bool>(tensors, "operator_is_unsigned");
    const auto is_symmetric = values<bool>(tensors, "operator_is_symmetric");
    const auto granularities = values<std::string>(tensors, "operator_granularity");
    const auto offsets = values<std::int32_t>(tensors, "quant_param_value_offsets");
    const auto group_counts = values<std::int32_t>(tensors, "quant_param_group_counts");
    const auto scales = floatValues(tensors, "quant_param_scales");
    const auto zero_points = values<std::int32_t>(tensors, "quant_param_zero_points");
    require(names.size() == quant_lstm::kQuantOperatorCount &&
                offsets.size() == quant_lstm::kQuantOperatorCount + 1 &&
                scales.size() == zero_points.size(),
            "Golden 量化元数据长度非法");

    result.params.hidden_size = result.shape.hidden_size;
    result.params.bias_enabled = data(tensors, "bias_enabled").at(0).get<bool>();
    for (std::size_t index = 0; index < quant_lstm::kQuantOperatorCount; ++index) {
        const auto id = static_cast<quant_lstm::QuantOperator>(index);
        require(names[index] == quant_lstm::quantOperatorName(id), "Golden operator 顺序非法");
        auto& operator_config = result.config.at(id);
        operator_config.type = {bitwidth.at(index), is_unsigned.at(index), is_symmetric.at(index)};
        operator_config.granularity = quant_lstm::parseGranularity(granularities.at(index));
        auto& operator_params = result.params.operators[index];
        operator_params.source_granularity = operator_config.granularity;
        for (std::int32_t offset = offsets.at(index); offset < offsets.at(index + 1); ++offset) {
            operator_params.values.push_back({scales.at(static_cast<std::size_t>(offset)),
                                              zero_points.at(static_cast<std::size_t>(offset))});
        }
        operator_params.group_diagnostics.resize(static_cast<std::size_t>(group_counts.at(index)));
    }
    result.config.validate();
    result.params.validate(result.config);
    result.execution = quant_lstm::deriveLstmExecutionParams(result.config, result.params,
                                                             result.shape.input_size);
    result.input = values<std::int32_t>(tensors, "input");
    result.h0 = values<std::int32_t>(tensors, "initial_hidden");
    result.c0 = values<std::int32_t>(tensors, "initial_cell");
    result.weight_ih = values<std::int32_t>(tensors, "weight_ih");
    result.weight_hh = values<std::int32_t>(tensors, "weight_hh");
    result.bias_ih = values<std::int32_t>(tensors, "bias_ih");
    result.bias_hh = values<std::int32_t>(tensors, "bias_hh");
    return result;
}

std::vector<float> dequantize(const std::vector<std::int32_t>& source, quant_lstm::QuantOperator id,
                              const GoldenCase& fixture, std::size_t row_width = 0) {
    std::vector<float> result(source.size());
    const auto& points = fixture.params.at(id).values;
    for (std::size_t index = 0; index < source.size(); ++index) {
        const std::size_t parameter_index = row_width == 0 ? 0 : index / row_width;
        result[index] = quant_lstm::quantization::dequantize(
            source[index], points.at(parameter_index), fixture.config.at(id).type);
    }
    return result;
}

template <typename Expected>
void compareExact(const std::vector<float>& actual, const std::vector<Expected>& expected,
                  const std::string& field) {
    require(actual.size() == expected.size(), field + " 长度不一致");
    for (std::size_t index = 0; index < actual.size(); ++index) {
        require(actual[index] == static_cast<float>(expected[index]),
                field + "[" + std::to_string(index) + "] 不一致");
    }
}

std::vector<std::int32_t> flattenGateCheckpoints(const Json& checkpoints, std::string_view suffix,
                                                 const quant_lstm::LstmShape& shape) {
    const std::size_t steps = static_cast<std::size_t>(shape.sequence_length);
    const std::size_t batch = static_cast<std::size_t>(shape.batch_size);
    const std::size_t hidden = static_cast<std::size_t>(shape.hidden_size);
    std::array<std::vector<std::int32_t>, 4> gates;
    constexpr std::array<const char*, 4> kNames{"input", "forget", "cell", "output"};
    for (std::size_t gate = 0; gate < gates.size(); ++gate) {
        gates[gate] = values<std::int32_t>(checkpoints,
                                           std::string(kNames[gate]) + "_" + std::string(suffix));
    }
    std::vector<std::int32_t> result(steps * batch * 4 * hidden);
    for (std::size_t step = 0; step < steps; ++step) {
        for (std::size_t row = 0; row < batch; ++row) {
            for (std::size_t gate = 0; gate < 4; ++gate) {
                for (std::size_t column = 0; column < hidden; ++column) {
                    const std::size_t source = (step * batch + row) * hidden + column;
                    const std::size_t target =
                        (step * batch + row) * 4 * hidden + gate * hidden + column;
                    result[target] = gates[gate][source];
                }
            }
        }
    }
    return result;
}

void runGolden(const Json& document, bool caller_workspace, bool test_small_workspace) {
    const std::string case_id = document.at("case_id");
    const Json& inputs = document.at("inputs");
    const Json& expected = document.at("expected").at("checkpoints");
    const GoldenCase fixture = loadCase(inputs);
    const std::size_t steps = static_cast<std::size_t>(fixture.shape.sequence_length);
    const std::size_t batch = static_cast<std::size_t>(fixture.shape.batch_size);
    const std::size_t input_size = static_cast<std::size_t>(fixture.shape.input_size);
    const std::size_t hidden = static_cast<std::size_t>(fixture.shape.hidden_size);
    const std::size_t output_count = steps * batch * hidden;
    const std::size_t state_count = batch * hidden;
    const std::size_t gate_count = steps * batch * 4 * hidden;

    const auto input = dequantize(fixture.input, quant_lstm::QuantOperator::Input, fixture);
    const auto weight_ih = dequantize(
        fixture.weight_ih, quant_lstm::QuantOperator::WeightInputHidden, fixture, input_size);
    const auto weight_hh = dequantize(
        fixture.weight_hh, quant_lstm::QuantOperator::WeightHiddenHidden, fixture, hidden);
    const auto bias_ih =
        dequantize(fixture.bias_ih, quant_lstm::QuantOperator::BiasInputHidden, fixture, 1);
    const auto bias_hh =
        dequantize(fixture.bias_hh, quant_lstm::QuantOperator::BiasHiddenHidden, fixture, 1);
    const auto h0 = dequantize(fixture.h0, quant_lstm::QuantOperator::Output, fixture);
    const auto c0 = dequantize(fixture.c0, quant_lstm::QuantOperator::CellState, fixture);

    DeviceBuffer<float> d_input(input.size());
    DeviceBuffer<float> d_weight_ih(weight_ih.size());
    DeviceBuffer<float> d_weight_hh(weight_hh.size());
    DeviceBuffer<float> d_bias_ih(bias_ih.size());
    DeviceBuffer<float> d_bias_hh(bias_hh.size());
    DeviceBuffer<float> d_h0(h0.size());
    DeviceBuffer<float> d_c0(c0.size());
    DeviceBuffer<float> d_output(output_count);
    DeviceBuffer<float> d_hn(state_count);
    DeviceBuffer<float> d_cn(state_count);
    d_input.copyFrom(input);
    d_weight_ih.copyFrom(weight_ih);
    d_weight_hh.copyFrom(weight_hh);
    d_bias_ih.copyFrom(bias_ih);
    d_bias_hh.copyFrom(bias_hh);
    d_h0.copyFrom(h0);
    d_c0.copyFrom(c0);

    DeviceBuffer<float> d_linear_ih(gate_count);
    DeviceBuffer<float> d_linear_hh(gate_count);
    DeviceBuffer<float> d_gate_input(gate_count);
    DeviceBuffer<float> d_gate_output(gate_count);
    DeviceBuffer<float> d_cell(output_count);
    DeviceBuffer<float> d_cell_tanh(output_count);
    DeviceBuffer<float> d_hidden(output_count);
    quant_lstm::LstmQuantizedFpCudaCheckpoints checkpoints{
        d_linear_ih.get(), d_linear_hh.get(), d_gate_input.get(), d_gate_output.get(),
        d_cell.get(),      d_cell_tanh.get(), d_hidden.get()};
    const std::size_t workspace_bytes =
        quant_lstm::lstmQuantizedFpCudaWorkspaceBytes(fixture.shape, fixture.params.bias_enabled);
    const auto workspace_breakdown = quant_lstm::lstmQuantizedFpCudaWorkspaceBreakdown(
        fixture.shape, fixture.params.bias_enabled);
    const std::size_t static_parameter_bytes = quant_lstm::lstmQuantizedFpCudaStaticParameterBytes(
        fixture.shape, fixture.params.bias_enabled);
    DeviceBuffer<std::byte> workspace(caller_workspace ? workspace_bytes : 0);
    quant_lstm::test::QuantizedCudaContextOwner context(workspace_breakdown.device_parameter_bytes,
                                                        static_parameter_bytes);

    if (test_small_workspace) {
        bool rejected = false;
        try {
            quant_lstm::lstmForwardQuantizedFpCuda(
                fixture.shape,
                {d_weight_ih.get(), d_weight_hh.get(), d_bias_ih.get(), d_bias_hh.get()},
                d_input.get(), d_h0.get(), d_c0.get(), fixture.config, fixture.params,
                fixture.execution, d_output.get(), d_hn.get(), d_cn.get(), context.get(),
                quant_lstm::LstmQuantizedFpCudaMathMode::Pedantic,
                {workspace.get(), workspace_bytes - 1});
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "过小 caller workspace 必须被拒绝");
    }

    quant_lstm::LstmQuantizedFpCudaStats stats;
    quant_lstm::lstmForwardQuantizedFpCuda(
        fixture.shape, {d_weight_ih.get(), d_weight_hh.get(), d_bias_ih.get(), d_bias_hh.get()},
        d_input.get(), d_h0.get(), d_c0.get(), fixture.config, fixture.params, fixture.execution,
        d_output.get(), d_hn.get(), d_cn.get(), context.get(),
        quant_lstm::LstmQuantizedFpCudaMathMode::Pedantic, {workspace.get(), workspace.bytes()},
        &checkpoints, &stats, nullptr, 1);
    context.synchronize();
    require(stats.input_gemm_calls == 1 && stats.recurrent_gemm_calls == steps,
            case_id + " GEMM 次数非法");
    require(!stats.static_parameter_cache_hit, case_id + " 首次静态参数缓存必须 miss");

    quant_lstm::LstmQuantizedFpCudaStats cached_stats;
    quant_lstm::lstmForwardQuantizedFpCuda(
        fixture.shape, {d_weight_ih.get(), d_weight_hh.get(), d_bias_ih.get(), d_bias_hh.get()},
        d_input.get(), d_h0.get(), d_c0.get(), fixture.config, fixture.params, fixture.execution,
        d_output.get(), d_hn.get(), d_cn.get(), context.get(),
        quant_lstm::LstmQuantizedFpCudaMathMode::Pedantic, {workspace.get(), workspace.bytes()},
        &checkpoints, &cached_stats, nullptr, 1);
    context.synchronize();
    require(cached_stats.static_parameter_cache_hit, case_id + " 第二次静态参数缓存必须 hit");
    quant_lstm::LstmQuantizedFpCudaStats invalidated_stats;
    quant_lstm::lstmForwardQuantizedFpCuda(
        fixture.shape, {d_weight_ih.get(), d_weight_hh.get(), d_bias_ih.get(), d_bias_hh.get()},
        d_input.get(), d_h0.get(), d_c0.get(), fixture.config, fixture.params, fixture.execution,
        d_output.get(), d_hn.get(), d_cn.get(), context.get(),
        quant_lstm::LstmQuantizedFpCudaMathMode::Pedantic, {workspace.get(), workspace.bytes()},
        &checkpoints, &invalidated_stats, nullptr, 2);
    context.synchronize();
    require(!invalidated_stats.static_parameter_cache_hit,
            case_id + " generation key 变化后必须 cache miss");
    compareExact(d_linear_ih.copyToHost(), values<std::int32_t>(expected, "weight_ih_linear"),
                 case_id + " weight_ih_linear");
    compareExact(d_linear_hh.copyToHost(), values<std::int32_t>(expected, "weight_hh_linear"),
                 case_id + " weight_hh_linear");
    compareExact(d_gate_input.copyToHost(),
                 flattenGateCheckpoints(expected, "gate_input", fixture.shape),
                 case_id + " gate_input");
    compareExact(d_gate_output.copyToHost(),
                 flattenGateCheckpoints(expected, "gate_output", fixture.shape),
                 case_id + " gate_output");
    compareExact(d_cell.copyToHost(), values<std::int32_t>(expected, "cell"), case_id + " cell");
    compareExact(d_cell_tanh.copyToHost(), values<std::int32_t>(expected, "cell_tanh"),
                 case_id + " cell_tanh");
    compareExact(d_hidden.copyToHost(), values<std::int32_t>(expected, "output"),
                 case_id + " output q");
    compareExact(d_hn.copyToHost(),
                 dequantize(values<std::int32_t>(expected, "h_n"),
                            quant_lstm::QuantOperator::Output, fixture),
                 case_id + " h_n real");
    compareExact(d_cn.copyToHost(),
                 dequantize(values<std::int32_t>(expected, "c_n"),
                            quant_lstm::QuantOperator::CellState, fixture),
                 case_id + " c_n real");
}

}  // namespace

int main() {
    try {
        std::size_t checked = 0;
        for (const auto& fixture : quant_lstm::test::kGoldenDocuments) {
            if (fixture.execution_model != "cpu_fp32" ||
                (fixture.kind != "cell" && fixture.kind != "recurrent")) {
                continue;
            }
            runGolden(Json::parse(fixture.json), checked % 2 == 0, checked == 0);
            ++checked;
        }
        require(checked == 2, "CUDA Golden 必须复用两个 cpu_fp32 reference 用例");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
