#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/deterministic_rng.h"
#include "lstm/backward_float_cuda.h"
#include "lstm/forward_float.h"
#include "lstm/forward_float_cuda.h"

namespace {

struct BackwardResult {
    std::vector<float> input;
    std::vector<float> weight_ih;
    std::vector<float> weight_hh;
    std::vector<float> bias_ih;
    std::vector<float> bias_hh;
    std::vector<float> initial_hidden;
    std::vector<float> initial_cell;
};

BackwardResult backwardReference(const quant_lstm::LstmShape& shape,
                                 const quant_lstm::LstmFloatWeights& weights, const float* input,
                                 const float* initial_hidden, const float* initial_cell,
                                 const quant_lstm::LstmFloatReferenceTrace& trace,
                                 const float* grad_output, const float* grad_final_hidden,
                                 const float* grad_final_cell,
                                 const quant_lstm::LstmFloatCudaBackwardMasks* masks = nullptr) {
    const std::size_t steps = static_cast<std::size_t>(shape.sequence_length);
    const std::size_t batch = static_cast<std::size_t>(shape.batch_size);
    const std::size_t input_size = static_cast<std::size_t>(shape.input_size);
    const std::size_t hidden = static_cast<std::size_t>(shape.hidden_size);
    const std::size_t state_elements = batch * hidden;
    const std::size_t gates = 4 * hidden;
    BackwardResult result{
        std::vector<float>(steps * batch * input_size, 0.0F),
        std::vector<float>(gates * input_size, 0.0F),
        std::vector<float>(gates * hidden, 0.0F),
        std::vector<float>(gates, 0.0F),
        std::vector<float>(gates, 0.0F),
        std::vector<float>(state_elements, 0.0F),
        std::vector<float>(state_elements, 0.0F),
    };
    std::copy_n(grad_final_hidden, state_elements, result.initial_hidden.data());
    std::copy_n(grad_final_cell, state_elements, result.initial_cell.data());
    std::vector<float> input_linear_gradients(batch * gates, 0.0F);
    std::vector<float> recurrent_linear_gradients(batch * gates, 0.0F);
    std::vector<float> previous_hidden_gradient(state_elements, 0.0F);
    const auto keep = [](const std::uint8_t* mask, std::size_t index) {
        return mask == nullptr || mask[index] == 0 ? 1.0F : 0.0F;
    };

    for (std::size_t reverse_time = 0; reverse_time < steps; ++reverse_time) {
        const std::size_t time = steps - reverse_time - 1;
        const std::size_t state_offset = time * state_elements;
        const std::size_t gate_offset = time * batch * gates;
        for (std::size_t batch_index = 0; batch_index < batch; ++batch_index) {
            for (std::size_t hidden_index = 0; hidden_index < hidden; ++hidden_index) {
                const std::size_t state_index = batch_index * hidden + hidden_index;
                const std::size_t gate_base = batch_index * gates + hidden_index;
                const float input_gate = trace.gate_outputs[gate_offset + gate_base];
                const float forget_gate = trace.gate_outputs[gate_offset + gate_base + hidden];
                const float cell_gate = trace.gate_outputs[gate_offset + gate_base + 2 * hidden];
                const float output_gate = trace.gate_outputs[gate_offset + gate_base + 3 * hidden];
                const float cell_tanh = trace.cell_tanh_outputs[state_offset + state_index];
                const float previous_cell =
                    time == 0 ? initial_cell[state_index]
                              : trace.cell_states[state_offset - state_elements + state_index];
                const float dh =
                    (result.initial_hidden[state_index] + grad_output[state_offset + state_index]) *
                    keep(masks == nullptr ? nullptr : masks->hidden_outputs,
                         state_offset + state_index);
                const float grad_output_gate = dh * cell_tanh;
                const float grad_cell_tanh =
                    dh * output_gate *
                    keep(masks == nullptr ? nullptr : masks->cell_tanh_outputs,
                         state_offset + state_index);
                const float dc_unmasked = result.initial_cell[state_index] +
                                          grad_cell_tanh * (1.0F - cell_tanh * cell_tanh);
                const float dc = dc_unmasked * keep(masks == nullptr ? nullptr : masks->cell_states,
                                                    state_offset + state_index);
                const float gate_output_gradients[4]{
                    dc * cell_gate,
                    dc * previous_cell,
                    dc * input_gate,
                    grad_output_gate,
                };
                const float derivatives[4]{
                    input_gate * (1.0F - input_gate),
                    forget_gate * (1.0F - forget_gate),
                    1.0F - cell_gate * cell_gate,
                    output_gate * (1.0F - output_gate),
                };
                for (std::size_t gate = 0; gate < 4; ++gate) {
                    const std::size_t index = gate_base + gate * hidden;
                    const std::size_t global_index = gate_offset + index;
                    const float gradient =
                        gate_output_gradients[gate] *
                        keep(masks == nullptr ? nullptr : masks->gate_outputs, global_index) *
                        derivatives[gate] *
                        keep(masks == nullptr ? nullptr : masks->gate_inputs, global_index);
                    input_linear_gradients[index] =
                        gradient *
                        keep(masks == nullptr ? nullptr : masks->weight_ih_linear, global_index);
                    recurrent_linear_gradients[index] =
                        gradient *
                        keep(masks == nullptr ? nullptr : masks->weight_hh_linear, global_index);
                }
                result.initial_cell[state_index] = dc * forget_gate;
            }
        }

        std::fill(previous_hidden_gradient.begin(), previous_hidden_gradient.end(), 0.0F);
        for (std::size_t batch_index = 0; batch_index < batch; ++batch_index) {
            const float* input_row = input + (time * batch + batch_index) * input_size;
            const float* previous_hidden =
                time == 0
                    ? initial_hidden + batch_index * hidden
                    : trace.hidden_outputs.data() + (time * batch + batch_index - batch) * hidden;
            for (std::size_t gate = 0; gate < gates; ++gate) {
                const float input_gradient = input_linear_gradients[batch_index * gates + gate];
                const float recurrent_gradient =
                    recurrent_linear_gradients[batch_index * gates + gate];
                result.bias_ih[gate] += input_gradient;
                result.bias_hh[gate] += recurrent_gradient;
                for (std::size_t index = 0; index < input_size; ++index) {
                    result.input[(time * batch + batch_index) * input_size + index] +=
                        input_gradient * weights.weight_ih[gate * input_size + index];
                    result.weight_ih[gate * input_size + index] +=
                        input_gradient * input_row[index];
                }
                for (std::size_t index = 0; index < hidden; ++index) {
                    previous_hidden_gradient[batch_index * hidden + index] +=
                        recurrent_gradient * weights.weight_hh[gate * hidden + index];
                    result.weight_hh[gate * hidden + index] +=
                        recurrent_gradient * previous_hidden[index];
                }
            }
        }
        result.initial_hidden.swap(previous_hidden_gradient);
    }
    if (masks != nullptr) {
        const auto apply = [&](std::vector<float>& values, const std::uint8_t* mask) {
            if (mask == nullptr) {
                return;
            }
            for (std::size_t index = 0; index < values.size(); ++index) {
                values[index] *= keep(mask, index);
            }
        };
        apply(result.input, masks->input);
        apply(result.weight_ih, masks->weight_ih);
        apply(result.weight_hh, masks->weight_hh);
        apply(result.bias_ih, masks->bias_ih);
        apply(result.bias_hh, masks->bias_hh);
        apply(result.initial_hidden, masks->initial_hidden);
        apply(result.initial_cell, masks->initial_cell);
    }
    return result;
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
   public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)), "cudaMalloc");
    }
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return data_; }
    void copyFrom(const std::vector<T>& source) {
        if (source.size() != count_) {
            throw std::invalid_argument("host/device 元素数量不匹配");
        }
        checkCuda(cudaMemcpy(data_, source.data(), count_ * sizeof(T), cudaMemcpyHostToDevice),
                  "cudaMemcpy H2D");
    }
    std::vector<T> copyToHost() const {
        std::vector<T> result(count_);
        checkCuda(cudaMemcpy(result.data(), data_, count_ * sizeof(T), cudaMemcpyDeviceToHost),
                  "cudaMemcpy D2H");
        return result;
    }

   private:
    T* data_ = nullptr;
    std::size_t count_;
};

}  // namespace

int main() {
    try {
        constexpr quant_lstm::LstmShape shape{3, 2, 5, 7};
        const std::size_t input_count = shape.sequence_length * shape.batch_size * shape.input_size;
        const std::size_t state_count = shape.batch_size * shape.hidden_size;
        const std::size_t weight_ih_count = 4 * shape.hidden_size * shape.input_size;
        const std::size_t weight_hh_count = 4 * shape.hidden_size * shape.hidden_size;
        const std::size_t bias_count = 4 * shape.hidden_size;
        const std::size_t output_count = shape.sequence_length * state_count;
        const std::size_t gate_count =
            shape.sequence_length * shape.batch_size * 4 * shape.hidden_size;

        std::vector<float> input(input_count);
        std::vector<float> initial_hidden(state_count);
        std::vector<float> initial_cell(state_count);
        std::vector<float> weight_ih(weight_ih_count);
        std::vector<float> weight_hh(weight_hh_count);
        std::vector<float> bias_ih(bias_count);
        std::vector<float> bias_hh(bias_count);
        using quant_lstm::test::TensorStream;
        quant_lstm::test::fillNormalLike(input.data(), input.size(), 0, TensorStream::Input);
        quant_lstm::test::fillNormalLike(initial_hidden.data(), initial_hidden.size(), 0,
                                         TensorStream::InitialHidden);
        quant_lstm::test::fillNormalLike(initial_cell.data(), initial_cell.size(), 0,
                                         TensorStream::InitialCell);
        quant_lstm::test::fillLstmParameter(weight_ih.data(), weight_ih.size(), shape.hidden_size,
                                            3004, TensorStream::WeightInputHidden);
        quant_lstm::test::fillLstmParameter(weight_hh.data(), weight_hh.size(), shape.hidden_size,
                                            3004, TensorStream::WeightHiddenHidden);
        quant_lstm::test::fillLstmParameter(bias_ih.data(), bias_ih.size(), shape.hidden_size, 3004,
                                            TensorStream::BiasInputHidden);
        quant_lstm::test::fillLstmParameter(bias_hh.data(), bias_hh.size(), shape.hidden_size, 3004,
                                            TensorStream::BiasHiddenHidden);

        std::vector<float> expected_output(output_count);
        std::vector<float> expected_hidden(state_count);
        std::vector<float> expected_cell(state_count);
        quant_lstm::LstmFloatReferenceTrace expected_trace;
        const quant_lstm::LstmFloatWeights cpu_weights{weight_ih.data(), weight_hh.data(),
                                                       bias_ih.data(), bias_hh.data()};
        quant_lstm::lstmForwardFloatCpu(
            shape, cpu_weights, input.data(), initial_hidden.data(), initial_cell.data(),
            expected_output.data(), expected_hidden.data(), expected_cell.data(), &expected_trace);

        std::vector<float> grad_output(output_count);
        std::vector<float> grad_final_hidden(state_count);
        std::vector<float> grad_final_cell(state_count);
        for (std::size_t index = 0; index < grad_output.size(); ++index) {
            grad_output[index] = static_cast<float>(static_cast<int>(index % 11) - 5) * 0.013F;
        }
        for (std::size_t index = 0; index < state_count; ++index) {
            grad_final_hidden[index] = static_cast<float>(static_cast<int>(index % 7) - 3) * 0.017F;
            grad_final_cell[index] = static_cast<float>(static_cast<int>(index % 5) - 2) * 0.019F;
        }
        const BackwardResult expected_gradients = backwardReference(
            shape, cpu_weights, input.data(), initial_hidden.data(), initial_cell.data(),
            expected_trace, grad_output.data(), grad_final_hidden.data(), grad_final_cell.data());
        std::vector<std::uint8_t> input_mask(input_count, 0);
        std::vector<std::uint8_t> weight_ih_mask(weight_ih_count, 0);
        std::vector<std::uint8_t> weight_hh_mask(weight_hh_count, 0);
        std::vector<std::uint8_t> bias_ih_mask(bias_count, 0);
        std::vector<std::uint8_t> bias_hh_mask(bias_count, 0);
        std::vector<std::uint8_t> initial_hidden_mask(state_count, 0);
        std::vector<std::uint8_t> initial_cell_mask(state_count, 0);
        std::vector<std::uint8_t> weight_ih_linear_mask(gate_count, 0);
        std::vector<std::uint8_t> weight_hh_linear_mask(gate_count, 0);
        std::vector<std::uint8_t> gate_input_mask(gate_count, 0);
        std::vector<std::uint8_t> gate_output_mask(gate_count, 0);
        std::vector<std::uint8_t> cell_state_mask(output_count, 0);
        std::vector<std::uint8_t> cell_tanh_output_mask(output_count, 0);
        std::vector<std::uint8_t> hidden_output_mask(output_count, 0);
        input_mask[3] = 1;
        weight_ih_mask[5] = 1;
        weight_hh_mask[7] = 1;
        bias_ih_mask[9] = 1;
        bias_hh_mask[11] = 1;
        initial_hidden_mask[2] = 1;
        initial_cell_mask[4] = 1;
        weight_ih_linear_mask[13] = 1;
        weight_hh_linear_mask[17] = 1;
        gate_input_mask[19] = 1;
        gate_output_mask[23] = 1;
        cell_state_mask[6] = 1;
        cell_tanh_output_mask[8] = 1;
        hidden_output_mask[10] = 1;
        const quant_lstm::LstmFloatCudaBackwardMasks host_masks{
            input_mask.data(),
            weight_ih_mask.data(),
            weight_hh_mask.data(),
            bias_ih_mask.data(),
            bias_hh_mask.data(),
            initial_hidden_mask.data(),
            initial_cell_mask.data(),
            weight_ih_linear_mask.data(),
            weight_hh_linear_mask.data(),
            gate_input_mask.data(),
            gate_output_mask.data(),
            cell_state_mask.data(),
            cell_tanh_output_mask.data(),
            hidden_output_mask.data(),
        };
        const BackwardResult expected_masked_gradients =
            backwardReference(shape, cpu_weights, input.data(), initial_hidden.data(),
                              initial_cell.data(), expected_trace, grad_output.data(),
                              grad_final_hidden.data(), grad_final_cell.data(), &host_masks);

        DeviceBuffer<float> device_input(input_count);
        DeviceBuffer<float> device_initial_hidden(state_count);
        DeviceBuffer<float> device_initial_cell(state_count);
        DeviceBuffer<float> device_weight_ih(weight_ih_count);
        DeviceBuffer<float> device_weight_hh(weight_hh_count);
        DeviceBuffer<float> device_bias_ih(bias_count);
        DeviceBuffer<float> device_bias_hh(bias_count);
        DeviceBuffer<float> device_output(output_count);
        DeviceBuffer<float> device_hidden(state_count);
        DeviceBuffer<float> device_cell(state_count);
        DeviceBuffer<float> device_gate_outputs(gate_count);
        DeviceBuffer<float> device_cell_states(output_count);
        DeviceBuffer<float> device_cell_tanh_outputs(output_count);
        DeviceBuffer<float> device_grad_output(output_count);
        DeviceBuffer<float> device_grad_final_hidden(state_count);
        DeviceBuffer<float> device_grad_final_cell(state_count);
        DeviceBuffer<float> device_grad_input(input_count);
        DeviceBuffer<float> device_grad_weight_ih(weight_ih_count);
        DeviceBuffer<float> device_grad_weight_hh(weight_hh_count);
        DeviceBuffer<float> device_grad_bias_ih(bias_count);
        DeviceBuffer<float> device_grad_bias_hh(bias_count);
        DeviceBuffer<float> device_grad_initial_hidden(state_count);
        DeviceBuffer<float> device_grad_initial_cell(state_count);
        DeviceBuffer<float> device_masked_grad_input(input_count);
        DeviceBuffer<float> device_masked_grad_weight_ih(weight_ih_count);
        DeviceBuffer<float> device_masked_grad_weight_hh(weight_hh_count);
        DeviceBuffer<float> device_masked_grad_bias_ih(bias_count);
        DeviceBuffer<float> device_masked_grad_bias_hh(bias_count);
        DeviceBuffer<float> device_masked_grad_initial_hidden(state_count);
        DeviceBuffer<float> device_masked_grad_initial_cell(state_count);
        DeviceBuffer<std::uint8_t> device_input_mask(input_count);
        DeviceBuffer<std::uint8_t> device_weight_ih_mask(weight_ih_count);
        DeviceBuffer<std::uint8_t> device_weight_hh_mask(weight_hh_count);
        DeviceBuffer<std::uint8_t> device_bias_ih_mask(bias_count);
        DeviceBuffer<std::uint8_t> device_bias_hh_mask(bias_count);
        DeviceBuffer<std::uint8_t> device_initial_hidden_mask(state_count);
        DeviceBuffer<std::uint8_t> device_initial_cell_mask(state_count);
        DeviceBuffer<std::uint8_t> device_weight_ih_linear_mask(gate_count);
        DeviceBuffer<std::uint8_t> device_weight_hh_linear_mask(gate_count);
        DeviceBuffer<std::uint8_t> device_gate_input_mask(gate_count);
        DeviceBuffer<std::uint8_t> device_gate_output_mask(gate_count);
        DeviceBuffer<std::uint8_t> device_cell_state_mask(output_count);
        DeviceBuffer<std::uint8_t> device_cell_tanh_output_mask(output_count);
        DeviceBuffer<std::uint8_t> device_hidden_output_mask(output_count);
        device_input.copyFrom(input);
        device_initial_hidden.copyFrom(initial_hidden);
        device_initial_cell.copyFrom(initial_cell);
        device_weight_ih.copyFrom(weight_ih);
        device_weight_hh.copyFrom(weight_hh);
        device_bias_ih.copyFrom(bias_ih);
        device_bias_hh.copyFrom(bias_hh);
        device_grad_output.copyFrom(grad_output);
        device_grad_final_hidden.copyFrom(grad_final_hidden);
        device_grad_final_cell.copyFrom(grad_final_cell);
        device_input_mask.copyFrom(input_mask);
        device_weight_ih_mask.copyFrom(weight_ih_mask);
        device_weight_hh_mask.copyFrom(weight_hh_mask);
        device_bias_ih_mask.copyFrom(bias_ih_mask);
        device_bias_hh_mask.copyFrom(bias_hh_mask);
        device_initial_hidden_mask.copyFrom(initial_hidden_mask);
        device_initial_cell_mask.copyFrom(initial_cell_mask);
        device_weight_ih_linear_mask.copyFrom(weight_ih_linear_mask);
        device_weight_hh_linear_mask.copyFrom(weight_hh_linear_mask);
        device_gate_input_mask.copyFrom(gate_input_mask);
        device_gate_output_mask.copyFrom(gate_output_mask);
        device_cell_state_mask.copyFrom(cell_state_mask);
        device_cell_tanh_output_mask.copyFrom(cell_tanh_output_mask);
        device_hidden_output_mask.copyFrom(hidden_output_mask);

        cublasHandle_t handle = nullptr;
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasCreate 失败");
        }
        const quant_lstm::LstmFloatWeights cuda_weights{device_weight_ih.get(),
                                                        device_weight_hh.get(),
                                                        device_bias_ih.get(), device_bias_hh.get()};
        quant_lstm::LstmFloatCudaTrace cuda_trace{
            device_gate_outputs.get(), device_cell_states.get(), device_cell_tanh_outputs.get()};
        quant_lstm::lstmForwardFloatCuda(shape, cuda_weights, device_input.get(),
                                         device_initial_hidden.get(), device_initial_cell.get(),
                                         device_output.get(), device_hidden.get(),
                                         device_cell.get(), handle, nullptr, nullptr, &cuda_trace);
        const quant_lstm::LstmFloatCudaBackwardTrace backward_trace{
            device_gate_outputs.get(), device_cell_states.get(), device_cell_tanh_outputs.get(),
            device_output.get()};
        const quant_lstm::LstmFloatCudaGradients cuda_gradients{
            device_grad_input.get(),       device_grad_weight_ih.get(),
            device_grad_weight_hh.get(),   device_grad_bias_ih.get(),
            device_grad_bias_hh.get(),     device_grad_initial_hidden.get(),
            device_grad_initial_cell.get()};
        quant_lstm::lstmBackwardFloatCuda(
            shape, cuda_weights, device_input.get(), device_initial_hidden.get(),
            device_initial_cell.get(), backward_trace, device_grad_output.get(),
            device_grad_final_hidden.get(), device_grad_final_cell.get(), cuda_gradients, handle,
            nullptr);
        const quant_lstm::LstmFloatCudaGradients cuda_masked_gradients{
            device_masked_grad_input.get(),       device_masked_grad_weight_ih.get(),
            device_masked_grad_weight_hh.get(),   device_masked_grad_bias_ih.get(),
            device_masked_grad_bias_hh.get(),     device_masked_grad_initial_hidden.get(),
            device_masked_grad_initial_cell.get()};
        const quant_lstm::LstmFloatCudaBackwardMasks cuda_masks{
            device_input_mask.get(),
            device_weight_ih_mask.get(),
            device_weight_hh_mask.get(),
            device_bias_ih_mask.get(),
            device_bias_hh_mask.get(),
            device_initial_hidden_mask.get(),
            device_initial_cell_mask.get(),
            device_weight_ih_linear_mask.get(),
            device_weight_hh_linear_mask.get(),
            device_gate_input_mask.get(),
            device_gate_output_mask.get(),
            device_cell_state_mask.get(),
            device_cell_tanh_output_mask.get(),
            device_hidden_output_mask.get(),
        };
        quant_lstm::lstmBackwardFloatCuda(
            shape, cuda_weights, device_input.get(), device_initial_hidden.get(),
            device_initial_cell.get(), backward_trace, device_grad_output.get(),
            device_grad_final_hidden.get(), device_grad_final_cell.get(), cuda_masked_gradients,
            handle, nullptr, nullptr, &cuda_masks);
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        cublasDestroy(handle);

        const auto actual_output = device_output.copyToHost();
        const auto actual_hidden = device_hidden.copyToHost();
        const auto actual_cell = device_cell.copyToHost();
        const auto check = [](const std::vector<float>& actual, const std::vector<float>& expected,
                              float tolerance = 1.0e-5F) {
            for (std::size_t index = 0; index < actual.size(); ++index) {
                if (std::abs(actual[index] - expected[index]) > tolerance) {
                    return false;
                }
            }
            return true;
        };
        if (!check(actual_output, expected_output) || !check(actual_hidden, expected_hidden) ||
            !check(actual_cell, expected_cell)) {
            std::cerr << "CUDA 与 CPU FP32 reference 不一致\n";
            return EXIT_FAILURE;
        }
        if (!check(device_gate_outputs.copyToHost(), expected_trace.gate_outputs) ||
            !check(device_cell_states.copyToHost(), expected_trace.cell_states) ||
            !check(device_cell_tanh_outputs.copyToHost(), expected_trace.cell_tanh_outputs)) {
            std::cerr << "CUDA forward checkpoint 与 CPU reference 不一致\n";
            return EXIT_FAILURE;
        }
        if (!check(device_grad_input.copyToHost(), expected_gradients.input, 2.0e-5F) ||
            !check(device_grad_weight_ih.copyToHost(), expected_gradients.weight_ih, 2.0e-5F) ||
            !check(device_grad_weight_hh.copyToHost(), expected_gradients.weight_hh, 2.0e-5F) ||
            !check(device_grad_bias_ih.copyToHost(), expected_gradients.bias_ih, 2.0e-5F) ||
            !check(device_grad_bias_hh.copyToHost(), expected_gradients.bias_hh, 2.0e-5F) ||
            !check(device_grad_initial_hidden.copyToHost(), expected_gradients.initial_hidden,
                   2.0e-5F) ||
            !check(device_grad_initial_cell.copyToHost(), expected_gradients.initial_cell,
                   2.0e-5F)) {
            std::cerr << "CUDA backward 与 CPU 公式 reference 不一致\n";
            return EXIT_FAILURE;
        }
        if (!check(device_masked_grad_input.copyToHost(), expected_masked_gradients.input,
                   2.0e-5F) ||
            !check(device_masked_grad_weight_ih.copyToHost(), expected_masked_gradients.weight_ih,
                   2.0e-5F) ||
            !check(device_masked_grad_weight_hh.copyToHost(), expected_masked_gradients.weight_hh,
                   2.0e-5F) ||
            !check(device_masked_grad_bias_ih.copyToHost(), expected_masked_gradients.bias_ih,
                   2.0e-5F) ||
            !check(device_masked_grad_bias_hh.copyToHost(), expected_masked_gradients.bias_hh,
                   2.0e-5F) ||
            !check(device_masked_grad_initial_hidden.copyToHost(),
                   expected_masked_gradients.initial_hidden, 2.0e-5F) ||
            !check(device_masked_grad_initial_cell.copyToHost(),
                   expected_masked_gradients.initial_cell, 2.0e-5F)) {
            std::cerr << "CUDA QAT STE backward 与 CPU mask reference 不一致\n";
            return EXIT_FAILURE;
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
