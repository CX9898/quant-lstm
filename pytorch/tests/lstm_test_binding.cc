#include <torch/extension.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/deterministic_rng.h"

namespace {

quant_lstm::test::TensorStream tensorStreamFromName(const std::string& name) {
    using quant_lstm::test::TensorStream;
    if (name == "input") {
        return TensorStream::Input;
    }
    if (name == "h0") {
        return TensorStream::InitialHidden;
    }
    if (name == "c0") {
        return TensorStream::InitialCell;
    }
    if (name == "weight_ih") {
        return TensorStream::WeightInputHidden;
    }
    if (name == "weight_hh") {
        return TensorStream::WeightHiddenHidden;
    }
    if (name == "bias_ih") {
        return TensorStream::BiasInputHidden;
    }
    if (name == "bias_hh") {
        return TensorStream::BiasHiddenHidden;
    }
    throw std::invalid_argument("未知的 RNG tensor role: " + name);
}

torch::Tensor generateTestTensor(const std::vector<std::int64_t>& shape, std::uint64_t seed,
                                 const std::string& role, bool parameter,
                                 std::int64_t hidden_size) {
    TORCH_CHECK(!shape.empty(), "测试张量 shape 不能为空");
    for (const std::int64_t dimension : shape) {
        TORCH_CHECK(dimension > 0, "测试张量维度必须为正数");
    }
    auto tensor = torch::empty(shape, torch::TensorOptions().dtype(torch::kFloat32));
    const auto stream = tensorStreamFromName(role);
    if (parameter) {
        quant_lstm::test::fillLstmParameter(tensor.data_ptr<float>(), tensor.numel(), hidden_size,
                                            seed, stream);
    } else {
        quant_lstm::test::fillNormalLike(tensor.data_ptr<float>(), tensor.numel(), seed, stream);
    }
    return tensor;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.attr("rng_version") = quant_lstm::test::kRngVersion;
    module.attr("rng_stream_registry") = quant_lstm::test::kRngStreamRegistryVersion;
    module.attr("distribution_profile") = quant_lstm::test::kDistributionProfile;
    module.def("generate_test_tensor", &generateTestTensor,
               "使用版本化 C++ PCG32 生成阶段 1 测试张量", pybind11::arg("shape"),
               pybind11::arg("seed"), pybind11::arg("role"), pybind11::arg("parameter") = false,
               pybind11::arg("hidden_size") = 1);
}
