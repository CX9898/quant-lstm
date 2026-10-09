"""构建 QuantLSTM CUDA FP 载体 binding 与 Python 模块。"""

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BUILD_DIR = Path(os.environ.get("RX_MET_QUANT_LSTM_BUILD_DIR", ROOT / "build"))
CORE_LIBRARY = BUILD_DIR / "libquant_lstm.a"
DEFAULT_CONFIG = ROOT / "config" / "defaults" / "lstm_quant_default_v1.json"

if CUDA_HOME is None:
    raise RuntimeError("未找到 CUDA toolkit，无法构建 QuantLSTM CUDA binding")
if not CORE_LIBRARY.exists():
    raise RuntimeError(
        f"未找到 {CORE_LIBRARY}；请先执行 `cmake -S . -B build && cmake --build build`"
    )
if not DEFAULT_CONFIG.exists():
    raise RuntimeError(f"未找到默认量化配置: {DEFAULT_CONFIG}")


setup(
    name="quant-lstm",
    version=os.environ.get("RX_MET_QUANT_LSTM_VERSION", "0.1.0"),
    py_modules=["quant_lstm", "lstm_autograd", "lstm_onnx", "lstm_aimet"],
    data_files=[
        ("share/quant-lstm/config", [str(DEFAULT_CONFIG)]),
    ],
    ext_modules=[
        CUDAExtension(
            name="_quant_lstm",
            sources=[str(HERE / "lib" / "lstm_interface_binding.cc")],
            include_dirs=[str(ROOT / "include")],
            library_dirs=[str(BUILD_DIR)],
            libraries=["quant_lstm", "cublas", "cudart"],
            extra_compile_args={"cxx": ["-O3", "-std=c++17"], "nvcc": ["-O3", "-std=c++17"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
