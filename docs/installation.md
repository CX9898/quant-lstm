# quant-lstm 安装指南

quant-lstm 提供两个安装界面：PyTorch CUDA 模块和 CMake C++ package。Python
模块面向训练、校准和推理；CMake package 面向独立 C++ 集成，也支持不依赖 CUDA
的 reference-only 安装。

## 1. 环境要求

| 组件 | 要求 | 用途 |
| --- | --- | --- |
| CMake | 3.24 或更高版本 | 配置原生库和安装 package |
| C++ compiler | 支持 C++17 | 编译 C++ 核心和 reference |
| `nlohmann_json` | 3.11.2 或更高版本 | 配置与参数 JSON |
| CUDA Toolkit | 提供 `nvcc`、cuBLAS 和 CUDA Runtime | CUDA C++ 与 Python 模块 |
| Python | 具有开发头文件 | 构建 PyTorch extension |
| PyTorch | CUDA-enabled，且 CUDA ABI 与构建环境兼容 | Python 接口 |

CPU-only C++ package 不要求 CUDA Toolkit 或 PyTorch。Python 模块始终要求 CUDA；
它不会在 CUDA 不可用时回退到 CPU reference。

Ubuntu 可以使用系统包提供 CMake、编译器和 `nlohmann_json`：

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake nlohmann-json3-dev python3-dev
```

CUDA Toolkit、驱动和 CUDA-enabled PyTorch 需要按照目标 GPU 和 PyTorch 官方安装
矩阵选择。`nvcc --version`、`python -c 'import torch; print(torch.version.cuda)'`
和驱动支持的 CUDA 版本需要兼容。

## 2. 安装 PyTorch CUDA 模块

当前 Python package 支持源码可编辑安装。以下命令均从仓库根目录执行，且原生核心
必须构建到固定的 `build/` 目录，因为 `pytorch/setup.py` 从该目录链接
`libquant_lstm.a`。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# 按目标 CUDA 环境安装 CUDA-enabled PyTorch 后执行：
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DQUANT_LSTM_ENABLE_CUDA=ON \
  -DQUANT_LSTM_BUILD_TESTS=OFF \
  -DQUANT_LSTM_BUILD_EXAMPLES=OFF
cmake --build build --parallel
python -m pip install --editable ./pytorch --no-build-isolation
```

`--no-build-isolation` 使 extension 使用当前环境中已经安装且与 CUDA 匹配的
PyTorch。可编辑安装保留 `pytorch/quant_lstm.py` 与仓库中的默认配置文件之间的
路径关系，因此源码目录不能在安装后删除或移动。

在可访问 CUDA GPU 的环境执行验证：

```bash
python - <<'PY'
import torch
from quant_lstm import QuantLSTM

assert torch.cuda.is_available()
module = QuantLSTM(4, 8, batch_first=True, device="cuda").eval()
inputs = torch.randn(2, 3, 4, device="cuda", dtype=torch.float32)
output, (hidden, cell) = module(inputs)
assert output.shape == (2, 3, 8)
assert hidden.shape == cell.shape == (1, 2, 8)
assert torch.isfinite(output).all()
print("QuantLSTM CUDA FP32 forward passed")
PY
```

打印 `QuantLSTM CUDA FP32 forward passed` 且进程退出码为 0 表示 Python 模块、
extension、CUDA Runtime 和 native FP32 forward 已正确加载。该检查使用随机输入，
只验证安装和接口，不用于量化校准。

卸载可编辑 package：

```bash
python -m pip uninstall quant-lstm
```

项目尚未发布 PyPI package，也未生成可脱离源码树分发的 wheel。

## 3. 安装 CUDA C++ package

使用自定义安装前缀可以避免修改系统目录：

```bash
install_prefix=/path/to/quant-lstm-install

cmake -S . -B build-install \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${install_prefix}" \
  -DQUANT_LSTM_ENABLE_CUDA=ON \
  -DQUANT_LSTM_BUILD_TESTS=OFF \
  -DQUANT_LSTM_BUILD_EXAMPLES=OFF
cmake --build build-install --parallel
cmake --install build-install
```

安装结果包括：

```text
<prefix>/include/
<prefix>/lib/libquant_lstm.a
<prefix>/lib/cmake/quant-lstm/
<prefix>/share/quant-lstm/config/
<prefix>/share/doc/quant-lstm/
```

下游项目使用导出的 target，不需要手工拼接 include 和 library 路径：

```cmake
cmake_minimum_required(VERSION 3.24)
project(quant_lstm_consumer LANGUAGES CXX)

find_package(quant-lstm CONFIG REQUIRED)
add_executable(app main.cc)
target_link_libraries(app PRIVATE quant_lstm::quant_lstm)
```

配置并构建下游项目：

```bash
cmake -S /path/to/consumer -B /path/to/consumer/build \
  -DCMAKE_PREFIX_PATH=/path/to/quant-lstm-install
cmake --build /path/to/consumer/build --parallel
```

`find_package()` 会加载 `quant-lstm-config.cmake`，查找 `nlohmann_json`，并在该
package 包含 CUDA 时查找 `CUDAToolkit`。成功生成并运行下游可执行文件是 C++
安装的完成判定。

将 `CMAKE_INSTALL_PREFIX` 设置为 `/usr/local` 可以执行系统级安装，但这不是使用
quant-lstm 的必要条件。自定义 prefix 更易于并存、升级和删除。

## 4. 安装 CPU-only reference package

CPU-only package 包含浮点和 int32 C++ reference，不包含 PyTorch 模块，也不链接
CUDA 或 cuBLAS：

```bash
install_prefix=/path/to/quant-lstm-cpu-install

cmake -S . -B build-cpu-install \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${install_prefix}" \
  -DQUANT_LSTM_ENABLE_CUDA=OFF \
  -DQUANT_LSTM_BUILD_TESTS=OFF \
  -DQUANT_LSTM_BUILD_EXAMPLES=ON
cmake --build build-cpu-install --parallel
cmake --install build-cpu-install
"${install_prefix}/bin/lstm_float_example"
"${install_prefix}/bin/lstm_int32_example"
```

两个示例均退出码为 0 表示 reference package 安装成功。完整的独立 consumer 验收
可以运行：

```bash
tools/run_cpu_only_package_check.sh
```

该脚本会配置独立构建树、运行 CTest、安装到临时 prefix、构建外部
`find_package()` consumer，并确认 consumer 没有链接 CUDA 动态库。

## 5. 使用 Docker 构建环境

仓库提供 CUDA、PyTorch、CMake 和测试依赖的开发镜像。以下命令从仓库根目录执行：

```bash
docker build -f docker/Dockerfile -t quant-lstm:cuda .
docker run --rm -it --gpus all \
  -v "$PWD:/workspace" \
  quant-lstm:cuda
```

容器进入 `/workspace`。挂载目录后，按照第 2 节执行 CMake 构建和可编辑安装。
宿主机需要 NVIDIA Container Toolkit，且驱动需要支持镜像内 CUDA Runtime。

## 6. 常见安装错误

| 错误 | 原因与处理 |
| --- | --- |
| `未找到 CUDA 编译器` | `QUANT_LSTM_ENABLE_CUDA=ON`，但 `nvcc` 不在 `PATH` 或 CUDA Toolkit 未安装 |
| `未找到 build/libquant_lstm.a` | Python extension 构建前未在固定 `build/` 目录完成 CMake 构建 |
| `_quant_lstm 扩展未找到` | 可编辑安装或 `build_ext --inplace` 尚未完成，或当前 Python 环境不是构建环境 |
| `PyTorch ... 只支持 CUDA input` | 输入或状态位于 CPU；将模块、输入和状态移动到 CUDA |
| `Could not find quant-lstm` | 下游没有设置安装 prefix 的 `CMAKE_PREFIX_PATH` |
| `Could not find CUDAToolkit` | 下游正在消费 CUDA package，但 CMake 无法定位 CUDA Toolkit |
