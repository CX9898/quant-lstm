#include "common/numeric_metrics.h"
#include "cuda/cuda_test_support.cuh"
#include "lstm/calibration.h"
#include "lstm/forward_quantized_fp_cuda.h"
#include "lstm/quant_config_loader.h"
#include "lstm/quant_params_io.h"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using quant_lstm::test::DeviceBuffer;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    try {
        using namespace quant_lstm;
        const LstmShape shape{3, 1, 2, 2};
        const std::vector<float> input{
            0.20F, -0.10F, -0.40F, 0.30F, 0.15F, 0.25F};
        const std::vector<float> weight_ih{
            0.10F, -0.20F, 0.05F, 0.15F, -0.12F, 0.08F, 0.20F, -0.05F,
            0.07F, 0.11F, -0.09F, 0.14F, 0.03F, -0.06F, 0.13F, 0.04F};
        const std::vector<float> weight_hh{
            0.04F, -0.03F, 0.02F, 0.05F, -0.06F, 0.01F, 0.03F, -0.02F,
            0.05F, 0.02F, -0.04F, 0.06F, 0.01F, -0.05F, 0.02F, 0.03F};
        const std::vector<float> bias_ih{
            0.01F, -0.02F, 0.03F, 0.04F, -0.01F, 0.02F, 0.01F, -0.03F};
        const std::vector<float> bias_hh{
            -0.01F, 0.01F, 0.02F, -0.02F, 0.01F, 0.00F, -0.01F, 0.02F};
        const std::vector<float> h0{0.05F, -0.04F};
        const std::vector<float> c0{0.10F, -0.08F};
        const LstmFloatWeights host_weights{
            weight_ih.data(), weight_hh.data(), bias_ih.data(), bias_hh.data()};
        const auto config = resolveQuantConfigFiles(
            std::filesystem::path(QUANT_LSTM_SOURCE_DIR) /
            "config/defaults/lstm_quant_default_v1.json");

        LstmCalibrationSession session(config, shape.input_size,
                                       shape.hidden_size, true);
        session.collect(shape, host_weights, input.data(), h0.data(),
                        c0.data());
        const auto& calibrated = session.finalize();
        const LstmQuantParamsBundle bundle{
            1, shape.input_size, config, calibrated.quant_params};
        const std::string encoded = exportQuantParamsBundle(bundle);
        const auto imported = importQuantParamsBundle(encoded, true);
        const auto imported_execution = auditQuantParamsBundle(imported);

        DeviceBuffer<float> device_input(input.size());
        DeviceBuffer<float> device_weight_ih(weight_ih.size());
        DeviceBuffer<float> device_weight_hh(weight_hh.size());
        DeviceBuffer<float> device_bias_ih(bias_ih.size());
        DeviceBuffer<float> device_bias_hh(bias_hh.size());
        DeviceBuffer<float> device_h0(h0.size());
        DeviceBuffer<float> device_c0(c0.size());
        DeviceBuffer<float> device_output(
            static_cast<std::size_t>(shape.sequence_length *
                                     shape.batch_size * shape.hidden_size));
        DeviceBuffer<float> device_hn(h0.size());
        DeviceBuffer<float> device_cn(c0.size());
        device_input.copyFrom(input);
        device_weight_ih.copyFrom(weight_ih);
        device_weight_hh.copyFrom(weight_hh);
        device_bias_ih.copyFrom(bias_ih);
        device_bias_hh.copyFrom(bias_hh);
        device_h0.copyFrom(h0);
        device_c0.copyFrom(c0);

        const LstmFloatWeights device_weights{
            device_weight_ih.get(), device_weight_hh.get(),
            device_bias_ih.get(), device_bias_hh.get()};
        const auto workspace_breakdown =
            lstmQuantizedFpCudaWorkspaceBreakdown(shape, true);
        quant_lstm::test::QuantizedCudaContextOwner context(
            workspace_breakdown.device_parameter_bytes);

        const auto run = [&](const LstmQuantParams& params,
                             const LstmExecutionParams& execution) {
            lstmForwardQuantizedFpCuda(
                shape, device_weights, device_input.get(), device_h0.get(),
                device_c0.get(), config, params, execution,
                device_output.get(), device_hn.get(), device_cn.get(),
                context.get(), LstmQuantizedFpCudaMathMode::Pedantic);
            context.synchronize();
            return std::array<std::vector<float>, 3>{
                device_output.copyToHost(), device_hn.copyToHost(),
                device_cn.copyToHost()};
        };
        const auto before = run(calibrated.quant_params,
                                calibrated.execution_params);
        const auto after = run(imported.quant_params, imported_execution);
        require(before == after,
                "export/import must preserve CUDA output, h_n and c_n exactly");

        std::vector<float> float_output(before[0].size());
        std::vector<float> float_hn(h0.size());
        std::vector<float> float_cn(c0.size());
        lstmForwardFloatCpu(shape, host_weights, input.data(), h0.data(),
                            c0.data(), float_output.data(), float_hn.data(),
                            float_cn.data());
        const auto metrics = quant_lstm::test::computeNumericMetrics(
            after[0].data(), float_output.data(), after[0].size());
        require(metrics.mean_absolute_error < 0.02 &&
                    metrics.mean_squared_error < 0.001 &&
                    !metrics.one_sided_zero_norm &&
                    metrics.cosine_similarity >= 0.999,
                "calibrated CUDA round-trip precision threshold");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
