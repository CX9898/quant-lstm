"""Build the CPU PTQ or CUDA QuantLSTM binding and Python modules."""

import os
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BUILD_DIR = Path(os.environ.get("RX_MET_QUANT_LSTM_BUILD_DIR", ROOT / "build"))
ENABLE_CUDA = os.environ.get("RX_MET_ENABLE_CUDA", "1") != "0"
CORE_LIBRARY = BUILD_DIR / "libquant_lstm.a"
DEFAULT_CONFIG = ROOT / "config" / "defaults" / "lstm_quant_default_v1.json"

if ENABLE_CUDA and CUDA_HOME is None:
    raise RuntimeError("未找到 CUDA toolkit，无法构建 QuantLSTM CUDA binding")
if not CORE_LIBRARY.exists():
    raise RuntimeError(
        f"未找到 {CORE_LIBRARY}；请先执行 `cmake -S . -B build && cmake --build build`"
    )
if not DEFAULT_CONFIG.exists():
    raise RuntimeError(f"未找到默认量化配置: {DEFAULT_CONFIG}")


class BuildPyWithResources(build_py):
    def run(self):
        super().run()
        destination = Path(self.build_lib) / "quant_lstm_resources"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DEFAULT_CONFIG, destination / DEFAULT_CONFIG.name)


onnx_headers = Path(os.environ.get("QUANT_LSTM_ONNXRUNTIME_INCLUDE",
    HERE.parents[2] / "docker/environment/third_party/onnxruntime/include"))
onnx_extensions = []
if (onnx_headers / "onnxruntime_cxx_api.h").is_file():
    onnx_extensions.append(CppExtension(
        name="quant_lstm_onnx_runtime", sources=[str(HERE / "lib/lstm_onnx_runtime.cc")],
        include_dirs=[str(onnx_headers)], extra_compile_args=["-std=c++17", "-O2"],
    ))


binding = CUDAExtension if ENABLE_CUDA else CppExtension
binding_source = "lstm_interface_binding.cc" if ENABLE_CUDA else "lstm_interface_binding_cpu.cc"


setup(
    name="quant-lstm",
    version=os.environ.get("RX_MET_QUANT_LSTM_VERSION", "0.1.0"),
    py_modules=["quant_lstm", "lstm_autograd", "lstm_onnx", "lstm_aimet", "quant_lstm_onnx"],
    packages=["quant_lstm_resources"],
    package_data={"quant_lstm_resources": ["*.json"]},
    zip_safe=False,
    ext_modules=[
        binding(
            name="_quant_lstm",
            sources=[str(HERE / "lib" / binding_source)],
            include_dirs=[str(ROOT / "include")],
            library_dirs=[str(BUILD_DIR)],
            libraries=["quant_lstm", "cublas", "cudart"] if ENABLE_CUDA else ["quant_lstm"],
            extra_compile_args={"cxx": ["-O3", "-std=c++17"], "nvcc": ["-O3", "-std=c++17"]}
            if ENABLE_CUDA else ["-O3", "-std=c++17"],
        )
    ] + onnx_extensions,
    cmdclass={"build_ext": BuildExtension, "build_py": BuildPyWithResources},
)
