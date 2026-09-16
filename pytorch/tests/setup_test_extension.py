"""构建仅供阶段测试使用的确定性数据扩展。"""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TEST_SUPPORT_LIBRARY = ROOT / "build" / "libquant_lstm_test_support.a"

if not TEST_SUPPORT_LIBRARY.exists():
    raise RuntimeError(
        f"未找到 {TEST_SUPPORT_LIBRARY}；请使用 QUANT_LSTM_BUILD_TESTS=ON 构建 CMake 工程"
    )


setup(
    name="quant-lstm-test-support",
    version="0.1.0",
    ext_modules=[
        CppExtension(
            name="_quant_lstm_test",
            sources=[str(HERE / "lstm_test_binding.cc")],
            include_dirs=[str(ROOT / "tests")],
            library_dirs=[str(ROOT / "build")],
            libraries=["quant_lstm_test_support"],
            extra_compile_args=["-O2", "-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
