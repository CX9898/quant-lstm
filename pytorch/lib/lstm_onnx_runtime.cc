// ORT transport only. QuantLSTM owns calibration and recurrent arithmetic.
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace py = pybind11;

namespace {
struct Kernel {
    int64_t identifier;
    explicit Kernel(const OrtKernelInfo* info) {
        identifier = Ort::ConstKernelInfo(info).GetAttribute<int64_t>("context_id");
    }

    OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
        try {
            py::gil_scoped_acquire gil;
            Ort::KernelContext context(raw);
            auto array = [](const Ort::ConstValue& value) -> py::object {
                if (!value) return py::none();
                const auto shape = value.GetTensorTypeAndShapeInfo().GetShape();
                // Own the buffer even if ORT reuses its input allocation.
                py::array_t<float> result(shape);
                std::memcpy(result.mutable_data(), value.GetTensorData<float>(), result.nbytes());
                return result;
            };
            auto values = py::module_::import("quant_lstm_onnx").attr("_run")(
                identifier, array(context.GetInput(0)), array(context.GetInput(1)),
                array(context.GetInput(2))).cast<py::tuple>();
            for (size_t index = 0; index < context.GetOutputCount(); ++index) {
                auto value = values[index].cast<py::array_t<float, py::array::c_style>>();
                std::vector<int64_t> shape(value.shape(), value.shape() + value.ndim());
                auto output = context.GetOutput(index, shape);
                if (output) std::memcpy(output.GetTensorMutableData<float>(), value.data(), value.nbytes());
            }
            return nullptr;
        } catch (const std::exception& error) {
            return Ort::GetApi().CreateStatus(ORT_RUNTIME_EXCEPTION, error.what());
        }
    }
};

struct Operator : Ort::CustomOpBase<Operator, Kernel, true> {
    const char* GetName() const { return "NativeLSTM"; }
    size_t GetInputTypeCount() const { return 3; }
    size_t GetOutputTypeCount() const { return 3; }
    ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
    ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
    OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t index) const {
        return index == 0 ? INPUT_OUTPUT_REQUIRED : INPUT_OUTPUT_OPTIONAL;
    }
    OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t) const { return INPUT_OUTPUT_OPTIONAL; }
    OrtStatus* CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** result) const noexcept {
        try {
            *result = new Kernel(info);
            return nullptr;
        } catch (const std::exception& error) {
            return api.CreateStatus(ORT_RUNTIME_EXCEPTION, error.what());
        }
    }
};
}

extern "C" ORT_EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* base) {
    const auto* api = base->GetApi(ORT_API_VERSION);
    if (!api) return base->GetApi(1)->CreateStatus(ORT_FAIL, "Unsupported ONNX Runtime API version");
    Ort::InitApi(api);
    try {
        static Operator op;
        static std::mutex mutex;
        static std::vector<std::unique_ptr<Ort::CustomOpDomain>> domains;
        auto domain = std::make_unique<Ort::CustomOpDomain>("rx.quant_lstm");
        domain->Add(&op);
        Ort::UnownedSessionOptions(options).Add(*domain);
        std::lock_guard<std::mutex> lock(mutex);
        domains.push_back(std::move(domain));
        return nullptr;
    } catch (const std::exception& error) {
        return api->CreateStatus(ORT_RUNTIME_EXCEPTION, error.what());
    }
}

PYBIND11_MODULE(quant_lstm_onnx_runtime, module) {
    module.doc() = "ONNX Runtime transport for native QuantLSTM";
}
