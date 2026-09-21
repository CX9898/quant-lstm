#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cuda/cuda_common.cuh"
#include "lstm/calibration_cuda.h"
#include "lstm/forward_float_cuda.h"
#include "lstm/gate_layout.h"

namespace quant_lstm {
namespace {

constexpr std::array<QuantOperator, kGateCount> kGateInputOperators{
    QuantOperator::InputGateInput,
    QuantOperator::ForgetGateInput,
    QuantOperator::CellGateInput,
    QuantOperator::OutputGateInput,
};

constexpr std::array<QuantOperator, kGateCount> kGateOutputOperators{
    QuantOperator::InputGateOutput,
    QuantOperator::ForgetGateOutput,
    QuantOperator::CellGateOutput,
    QuantOperator::OutputGateOutput,
};

struct DeviceView {
    const float* data = nullptr;
    std::uint64_t group_count = 0;
    std::uint64_t elements_per_group = 0;
    std::uint64_t slice_width = 0;
    std::uint64_t row_stride = 0;
    std::uint64_t group_stride = 0;
    std::uint64_t offset = 0;
};

struct HostViewStatistics {
    std::vector<CalibrationRange> ranges;
    std::vector<quantization::Histogram> histograms;
};

template <typename T>
class DeviceBuffer {
   public:
    DeviceBuffer(std::size_t count, cudaStream_t stream) : stream_(stream) {
        if (count != 0) {
            if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
                throw std::invalid_argument("CUDA 校准 buffer 字节数溢出");
            }
            cuda_detail::checkCuda(
                cudaMallocAsync(reinterpret_cast<void**>(&data_), count * sizeof(T), stream_),
                "cudaMallocAsync calibration buffer");
        }
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFreeAsync(data_, stream_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const noexcept { return data_; }

   private:
    T* data_ = nullptr;
    cudaStream_t stream_ = nullptr;
};

__device__ std::uint64_t viewIndex(const DeviceView& view, std::uint64_t group,
                                   std::uint64_t element) {
    const std::uint64_t row = element / view.slice_width;
    const std::uint64_t column = element - row * view.slice_width;
    return view.offset + group * view.group_stride + row * view.row_stride + column;
}

__global__ void reduceViewKernel(DeviceView view, float* minimums, float* maximums,
                                 unsigned long long* finite_counts) {
    extern __shared__ unsigned char shared_bytes[];
    auto* shared_minimums = reinterpret_cast<float*>(shared_bytes);
    auto* shared_maximums = shared_minimums + blockDim.x;
    auto* shared_counts = reinterpret_cast<unsigned long long*>(shared_maximums + blockDim.x);
    const std::uint64_t group = blockIdx.x;

    float local_minimum = FLT_MAX;
    float local_maximum = -FLT_MAX;
    unsigned long long local_count = 0;
    for (std::uint64_t element = threadIdx.x; element < view.elements_per_group;
         element += blockDim.x) {
        const float value = view.data[viewIndex(view, group, element)];
        if (isfinite(value)) {
            local_minimum = fminf(local_minimum, value);
            local_maximum = fmaxf(local_maximum, value);
            ++local_count;
        }
    }
    shared_minimums[threadIdx.x] = local_minimum;
    shared_maximums[threadIdx.x] = local_maximum;
    shared_counts[threadIdx.x] = local_count;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1U) {
        if (threadIdx.x < stride) {
            shared_minimums[threadIdx.x] =
                fminf(shared_minimums[threadIdx.x], shared_minimums[threadIdx.x + stride]);
            shared_maximums[threadIdx.x] =
                fmaxf(shared_maximums[threadIdx.x], shared_maximums[threadIdx.x + stride]);
            shared_counts[threadIdx.x] += shared_counts[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        minimums[group] = shared_minimums[0];
        maximums[group] = shared_maximums[0];
        finite_counts[group] = shared_counts[0];
    }
}

__global__ void histogramViewKernel(DeviceView view, const float* minimums, const float* maximums,
                                    std::uint64_t bin_count, unsigned long long* counts) {
    const std::uint64_t logical_count = view.group_count * view.elements_per_group;
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t logical = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         logical < logical_count; logical += stride) {
        const std::uint64_t group = logical / view.elements_per_group;
        const std::uint64_t element = logical - group * view.elements_per_group;
        const float value = view.data[viewIndex(view, group, element)];
        const float minimum = minimums[group];
        const float maximum = maximums[group];
        std::uint64_t bin = 0;
        if (value >= maximum) {
            bin = bin_count - 1;
        } else if (value > minimum) {
            const double normalized =
                (static_cast<double>(value) - minimum) / (static_cast<double>(maximum) - minimum);
            bin = min(static_cast<std::uint64_t>(normalized * static_cast<double>(bin_count)),
                      bin_count - 1);
        }
        atomicAdd(counts + group * bin_count + bin, 1ULL);
    }
}

__global__ void computeContributionsKernel(std::uint64_t state_count,
                                           std::uint64_t state_elements_per_step,
                                           std::uint64_t hidden_size, const float* initial_cell,
                                           const float* gate_outputs, const float* cell_states,
                                           const float* cell_tanh_outputs, float* contributions) {
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < state_count; index += stride) {
        const std::uint64_t state_row = index / hidden_size;
        const std::uint64_t channel = index - state_row * hidden_size;
        const std::uint64_t gate_base = state_row * 4 * hidden_size + channel;
        const std::uint64_t previous_index =
            index < state_elements_per_step ? index : index - state_elements_per_step;
        const float previous_cell =
            index < state_elements_per_step ? initial_cell[index] : cell_states[previous_index];
        const float input_gate = gate_outputs[gate_base];
        const float forget_gate = gate_outputs[gate_base + hidden_size];
        const float cell_gate = gate_outputs[gate_base + 2 * hidden_size];
        const float output_gate = gate_outputs[gate_base + 3 * hidden_size];
        contributions[index] = forget_gate * previous_cell;
        contributions[state_count + index] = input_gate * cell_gate;
        contributions[2 * state_count + index] = output_gate * cell_tanh_outputs[index];
    }
}

std::pair<float, float> nonDegenerateRange(float minimum, float maximum) {
    if (minimum < maximum) {
        return {minimum, maximum};
    }
    const float padding =
        std::max(std::abs(minimum) * std::numeric_limits<float>::epsilon(), 1.0e-6F);
    return {minimum - padding, maximum + padding};
}

std::size_t checkedSize(std::uint64_t value, const char* name) {
    if (value == 0 || value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(std::string(name) + " 非法或溢出");
    }
    return static_cast<std::size_t>(value);
}

DeviceView contiguousView(const float* data, std::uint64_t group_count,
                          std::uint64_t elements_per_group) {
    return {data,
            group_count,
            elements_per_group,
            elements_per_group,
            elements_per_group,
            elements_per_group,
            0};
}

DeviceView gateView(const float* data, std::uint64_t row_count, std::uint64_t hidden_size,
                    std::uint64_t gate) {
    return {data, 1, row_count * hidden_size, hidden_size, 4 * hidden_size, 0, gate * hidden_size};
}

HostViewStatistics collectView(const DeviceView& view, bool collect_histograms,
                               std::size_t histogram_bin_count, cudaStream_t stream) {
    if (view.data == nullptr || view.group_count == 0 || view.elements_per_group == 0 ||
        view.slice_width == 0) {
        throw std::invalid_argument("CUDA 校准 view 非法");
    }
    const std::size_t group_count = checkedSize(view.group_count, "CUDA 校准 group_count");
    constexpr unsigned int threads = 256;
    const std::size_t shared_bytes = threads * (2 * sizeof(float) + sizeof(unsigned long long));
    DeviceBuffer<float> device_minimums(group_count, stream);
    DeviceBuffer<float> device_maximums(group_count, stream);
    DeviceBuffer<unsigned long long> device_finite_counts(group_count, stream);
    reduceViewKernel<<<static_cast<unsigned int>(group_count), threads, shared_bytes, stream>>>(
        view, device_minimums.get(), device_maximums.get(), device_finite_counts.get());
    cuda_detail::checkCuda(cudaGetLastError(), "reduceViewKernel");

    std::vector<float> minimums(group_count);
    std::vector<float> maximums(group_count);
    std::vector<unsigned long long> finite_counts(group_count);
    cuda_detail::checkCuda(
        cudaMemcpyAsync(minimums.data(), device_minimums.get(), group_count * sizeof(float),
                        cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync calibration minimums");
    cuda_detail::checkCuda(
        cudaMemcpyAsync(maximums.data(), device_maximums.get(), group_count * sizeof(float),
                        cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync calibration maximums");
    cuda_detail::checkCuda(
        cudaMemcpyAsync(finite_counts.data(), device_finite_counts.get(),
                        group_count * sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync calibration finite counts");
    cuda_detail::checkCuda(cudaStreamSynchronize(stream),
                           "cudaStreamSynchronize calibration ranges");

    HostViewStatistics result;
    result.ranges.resize(group_count);
    for (std::size_t group = 0; group < group_count; ++group) {
        if (finite_counts[group] != view.elements_per_group) {
            throw std::invalid_argument("CUDA 校准值必须全部为有限 FP32");
        }
        result.ranges[group] = {minimums[group], maximums[group], view.elements_per_group};
    }
    if (!collect_histograms) {
        return result;
    }

    std::vector<float> normalized_minimums(group_count);
    std::vector<float> normalized_maximums(group_count);
    for (std::size_t group = 0; group < group_count; ++group) {
        const auto normalized = nonDegenerateRange(minimums[group], maximums[group]);
        normalized_minimums[group] = normalized.first;
        normalized_maximums[group] = normalized.second;
    }
    cuda_detail::checkCuda(
        cudaMemcpyAsync(device_minimums.get(), normalized_minimums.data(),
                        group_count * sizeof(float), cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync normalized minimums");
    cuda_detail::checkCuda(
        cudaMemcpyAsync(device_maximums.get(), normalized_maximums.data(),
                        group_count * sizeof(float), cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync normalized maximums");

    if (histogram_bin_count > std::numeric_limits<std::size_t>::max() / group_count) {
        throw std::invalid_argument("CUDA 校准 histogram 元素数量溢出");
    }
    const std::size_t histogram_count = group_count * histogram_bin_count;
    DeviceBuffer<unsigned long long> device_histograms(histogram_count, stream);
    cuda_detail::checkCuda(cudaMemsetAsync(device_histograms.get(), 0,
                                           histogram_count * sizeof(unsigned long long), stream),
                           "cudaMemsetAsync calibration histograms");
    const std::uint64_t logical_count = view.group_count * view.elements_per_group;
    const unsigned int blocks = static_cast<unsigned int>(
        std::min<std::uint64_t>((logical_count + threads - 1) / threads, 65535));
    histogramViewKernel<<<blocks, threads, 0, stream>>>(view, device_minimums.get(),
                                                        device_maximums.get(), histogram_bin_count,
                                                        device_histograms.get());
    cuda_detail::checkCuda(cudaGetLastError(), "histogramViewKernel");

    std::vector<unsigned long long> counts(histogram_count);
    cuda_detail::checkCuda(cudaMemcpyAsync(counts.data(), device_histograms.get(),
                                           histogram_count * sizeof(unsigned long long),
                                           cudaMemcpyDeviceToHost, stream),
                           "cudaMemcpyAsync calibration histograms");
    cuda_detail::checkCuda(cudaStreamSynchronize(stream),
                           "cudaStreamSynchronize calibration histograms");
    result.histograms.resize(group_count);
    for (std::size_t group = 0; group < group_count; ++group) {
        auto& histogram = result.histograms[group];
        histogram.minimum = normalized_minimums[group];
        histogram.maximum = normalized_maximums[group];
        histogram.total_count = view.elements_per_group;
        histogram.counts.resize(histogram_bin_count);
        for (std::size_t bin = 0; bin < histogram_bin_count; ++bin) {
            histogram.counts[bin] = static_cast<double>(counts[group * histogram_bin_count + bin]);
        }
    }
    return result;
}

void mergeOperatorStatistics(LstmCalibrationBatch* batch, QuantOperator id,
                             HostViewStatistics statistics) {
    const std::size_t index = static_cast<std::size_t>(id);
    auto& ranges = batch->ranges.operators[index];
    if (ranges.size() != statistics.ranges.size()) {
        throw std::invalid_argument("CUDA 校准 operator range group 数量不匹配");
    }
    for (std::size_t group = 0; group < ranges.size(); ++group) {
        ranges[group].merge(statistics.ranges[group]);
    }
    if (statistics.histograms.empty()) {
        return;
    }
    auto& histograms = batch->histograms[index];
    if (histograms.size() != statistics.histograms.size()) {
        throw std::invalid_argument("CUDA 校准 operator histogram group 数量不匹配");
    }
    for (std::size_t group = 0; group < histograms.size(); ++group) {
        histograms[group].merge(statistics.histograms[group]);
    }
}

void collectOperator(LstmCalibrationBatch* batch, QuantOperator id, const DeviceView& view,
                     const LstmCalibrationCollector& collector, cudaStream_t stream) {
    mergeOperatorStatistics(
        batch, id,
        collectView(view, collector.collectsHistograms(), collector.histogramBinCount(), stream));
}

DeviceView parameterView(QuantOperator id, const float* data, std::uint64_t row_width,
                         const LstmCalibrationCollector& collector) {
    const std::uint64_t hidden = collector.hiddenSize();
    const auto granularity = collector.config().at(id).granularity;
    if (granularity == QuantGranularity::PerTensor) {
        return contiguousView(data, 1, 4 * hidden * row_width);
    }
    if (granularity == QuantGranularity::PerGate) {
        return contiguousView(data, 4, hidden * row_width);
    }
    return contiguousView(data, 4 * hidden, row_width);
}

LstmCalibrationBatch createBatch(const LstmCalibrationCollector& collector) {
    LstmCalibrationBatch batch;
    batch.ranges.reset(collector.config(), collector.hiddenSize(), collector.biasEnabled());
    if (!collector.collectsHistograms()) {
        return batch;
    }
    for (std::size_t index = 0; index < kQuantOperatorCount; ++index) {
        const auto id = static_cast<QuantOperator>(index);
        if (!collector.biasEnabled() && isBiasOperator(id)) {
            continue;
        }
        const std::size_t group_count = batch.ranges.operators[index].size();
        batch.histograms[index].assign(
            group_count, quantization::HistogramCollector(collector.histogramBinCount()));
    }
    return batch;
}

}  // namespace

void lstmForwardCalibrateCuda(const LstmShape& shape, const LstmFloatWeights& weights,
                              const float* input, const float* initial_hidden,
                              const float* initial_cell, float* output, float* final_hidden,
                              float* final_cell, cublasHandle_t handle, cudaStream_t stream,
                              LstmCalibrationSession& session) {
    const auto& collector = session.collector();
    if (session.state() == CalibrationState::Locked) {
        throw std::logic_error("Locked 校准会话拒绝继续采集");
    }
    if (shape.input_size != collector.inputSize() || shape.hidden_size != collector.hiddenSize()) {
        throw std::invalid_argument("CUDA 校准 tensor shape 与 session 不匹配");
    }
    if ((weights.bias_ih != nullptr) != collector.biasEnabled()) {
        throw std::invalid_argument("CUDA 校准 bias 状态与 session 不匹配");
    }

    const std::uint64_t steps = checkedSize(shape.sequence_length, "sequence_length");
    const std::uint64_t batch_size = checkedSize(shape.batch_size, "batch_size");
    const std::uint64_t input_size = checkedSize(shape.input_size, "input_size");
    const std::uint64_t hidden = checkedSize(shape.hidden_size, "hidden_size");
    const std::uint64_t state_elements = batch_size * hidden;
    const std::uint64_t state_trace_elements = steps * state_elements;
    const std::uint64_t gate_trace_elements = state_trace_elements * kGateCount;

    DeviceBuffer<float> forward_workspace(
        checkedSize(cudaWorkspaceElementCount(shape), "CUDA forward workspace"), stream);
    DeviceBuffer<float> gate_outputs(checkedSize(gate_trace_elements, "gate trace"), stream);
    DeviceBuffer<float> cell_states(checkedSize(state_trace_elements, "cell trace"), stream);
    DeviceBuffer<float> cell_tanh_outputs(checkedSize(state_trace_elements, "cell tanh trace"),
                                          stream);
    DeviceBuffer<float> input_linear(checkedSize(gate_trace_elements, "input linear trace"),
                                     stream);
    DeviceBuffer<float> recurrent_linear(checkedSize(gate_trace_elements, "recurrent linear trace"),
                                         stream);
    DeviceBuffer<float> gate_inputs(checkedSize(gate_trace_elements, "gate input trace"), stream);
    DeviceBuffer<float> zero_state(checkedSize(state_elements, "zero state"), stream);
    cuda_detail::checkCuda(
        cudaMemsetAsync(zero_state.get(), 0,
                        checkedSize(state_elements, "zero state") * sizeof(float), stream),
        "cudaMemsetAsync calibration zero state");

    LstmFloatCudaTrace trace{gate_outputs.get(), cell_states.get(),      cell_tanh_outputs.get(),
                             input_linear.get(), recurrent_linear.get(), gate_inputs.get()};
    lstmForwardFloatCuda(shape, weights, input, initial_hidden, initial_cell, output, final_hidden,
                         final_cell, handle, stream, forward_workspace.get(), &trace);

    LstmCalibrationBatch calibration_batch = createBatch(collector);
    collectOperator(&calibration_batch, QuantOperator::Input,
                    contiguousView(input, 1, steps * batch_size * input_size), collector, stream);
    const float* initial_hidden_values =
        initial_hidden == nullptr ? zero_state.get() : initial_hidden;
    const float* initial_cell_values = initial_cell == nullptr ? zero_state.get() : initial_cell;
    collectOperator(&calibration_batch, QuantOperator::Output,
                    contiguousView(initial_hidden_values, 1, state_elements), collector, stream);
    collectOperator(&calibration_batch, QuantOperator::Output,
                    contiguousView(output, 1, state_trace_elements), collector, stream);
    collectOperator(&calibration_batch, QuantOperator::CellState,
                    contiguousView(initial_cell_values, 1, state_elements), collector, stream);
    collectOperator(&calibration_batch, QuantOperator::CellState,
                    contiguousView(cell_states.get(), 1, state_trace_elements), collector, stream);

    collectOperator(
        &calibration_batch, QuantOperator::WeightInputHidden,
        parameterView(QuantOperator::WeightInputHidden, weights.weight_ih, input_size, collector),
        collector, stream);
    collectOperator(
        &calibration_batch, QuantOperator::WeightHiddenHidden,
        parameterView(QuantOperator::WeightHiddenHidden, weights.weight_hh, hidden, collector),
        collector, stream);
    if (collector.biasEnabled()) {
        collectOperator(
            &calibration_batch, QuantOperator::BiasInputHidden,
            parameterView(QuantOperator::BiasInputHidden, weights.bias_ih, 1, collector), collector,
            stream);
        collectOperator(
            &calibration_batch, QuantOperator::BiasHiddenHidden,
            parameterView(QuantOperator::BiasHiddenHidden, weights.bias_hh, 1, collector),
            collector, stream);
    }

    collectOperator(&calibration_batch, QuantOperator::WeightInputHiddenLinear,
                    contiguousView(input_linear.get(), 1, gate_trace_elements), collector, stream);
    collectOperator(&calibration_batch, QuantOperator::WeightHiddenHiddenLinear,
                    contiguousView(recurrent_linear.get(), 1, gate_trace_elements), collector,
                    stream);
    const std::uint64_t gate_rows = steps * batch_size;
    for (std::uint64_t gate = 0; gate < kGateCount; ++gate) {
        collectOperator(&calibration_batch, kGateInputOperators[gate],
                        gateView(gate_inputs.get(), gate_rows, hidden, gate), collector, stream);
        collectOperator(&calibration_batch, kGateOutputOperators[gate],
                        gateView(gate_outputs.get(), gate_rows, hidden, gate), collector, stream);
    }
    collectOperator(&calibration_batch, QuantOperator::CellTanhOutput,
                    contiguousView(cell_tanh_outputs.get(), 1, state_trace_elements), collector,
                    stream);

    DeviceBuffer<float> contributions(checkedSize(3 * state_trace_elements, "contributions"),
                                      stream);
    constexpr unsigned int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>(
        std::min<std::uint64_t>((state_trace_elements + threads - 1) / threads, 65535));
    computeContributionsKernel<<<blocks, threads, 0, stream>>>(
        state_trace_elements, state_elements, hidden, initial_cell_values, gate_outputs.get(),
        cell_states.get(), cell_tanh_outputs.get(), contributions.get());
    cuda_detail::checkCuda(cudaGetLastError(), "computeContributionsKernel");
    calibration_batch.contributions.forget_times_old_cell.merge(
        collectView(contiguousView(contributions.get(), 1, state_trace_elements), false,
                    collector.histogramBinCount(), stream)
            .ranges.front());
    calibration_batch.contributions.input_times_cell.merge(
        collectView(
            contiguousView(contributions.get() + state_trace_elements, 1, state_trace_elements),
            false, collector.histogramBinCount(), stream)
            .ranges.front());
    calibration_batch.contributions.output_times_cell_tanh.merge(
        collectView(
            contiguousView(contributions.get() + 2 * state_trace_elements, 1, state_trace_elements),
            false, collector.histogramBinCount(), stream)
            .ranges.front());
    session.collect(std::move(calibration_batch));
}

}  // namespace quant_lstm
