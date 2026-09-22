# quant-lstm

`quant-lstm` 提供单层 LSTM 的 CUDA FP32、CUDA FP32 q-carrier 量化和 QAT
实现。q-carrier 使用 FP32 保存量化后的整数网格值，Linear 由 cuBLAS SGEMM
执行，sigmoid/tanh 保持浮点计算。Python 只负责模块接口、状态管理和 autograd
调度，前向、反向和校准的主要计算位于 C++/CUDA。

## 核心能力

| 能力 | 支持范围 |
| --- | --- |
| PyTorch 接口 | 单层、单向或双向、`bias=True/False`、`batch_first=True/False` |
| 完全浮点训练 | CUDA FP32 native forward/backward |
| 量化训练与推理 | CUDA FP32 q-carrier，8-bit、16-bit 和混合位宽 |
| QAT backward | CUDA checkpoint、Clamp mask 和 mask-aware STE |
| 校准 | CUDA MinMax、SQNR 和 Percentile |
| ONNX | 单个标准 ONNX `LSTM` 节点，opset 18 |
| C++ reference | CPU FP32 q-carrier 和 CPU int32 carrier |
| CMake package | CUDA 或 CPU-only 安装，导出 `quant_lstm::quant_lstm` |

PyTorch 执行路径只接受 CUDA FP32 tensor。CPU 实现用于 C++ Golden、数值验证和
CPU-only package，不作为 Python 运行时 fallback。CPU int32 reference 的
sigmoid/tanh 仍使用浮点函数，因此不属于端到端纯整数实现。

## 快速开始

以下命令从仓库根目录执行。构建要求 CMake 3.24、C++17、CUDA Toolkit、
`nlohmann_json` 3.11.2 及以上版本，以及与本机 CUDA 环境匹配的 PyTorch。

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DQUANT_LSTM_ENABLE_CUDA=ON \
  -DQUANT_LSTM_BUILD_TESTS=OFF \
  -DQUANT_LSTM_BUILD_EXAMPLES=OFF
cmake --build build --parallel
python -m pip install --editable ./pytorch --no-build-isolation
```

安装成功后，在可访问 CUDA GPU 的环境运行：

```bash
python - <<'PY'
import torch
from quant_lstm import QuantLSTM

module = QuantLSTM(16, 32, batch_first=True, device="cuda").eval()
inputs = torch.randn(2, 8, 16, device="cuda")
output, (hidden, cell) = module(inputs)
assert output.shape == (2, 8, 32)
assert hidden.shape == cell.shape == (1, 2, 32)
print("QuantLSTM CUDA FP32 forward passed")
PY
```

上述 Python 安装是源码可编辑安装，源码目录需要保留。C++ 安装、自定义 prefix、
CPU-only package、Docker 环境和卸载方式见[安装指南](docs/installation.md)。

## 文档

- [文档导航](docs/README.md)：按使用者和维护者角色列出文档入口。
- [安装指南](docs/installation.md)：Python、C++、CPU-only 和 Docker 安装。
- [配置与校准](docs/configuration.md)：配置字段、默认值、校准和参数导入导出。
- [系统架构](docs/architecture.md)：模块边界、数据流、制品和验证策略。
- [量化执行规格](docs/quantized-execution-spec.md)：量化语义的权威规格。
- [量化公式推导](docs/lstm-quantization-formula-derivation.md)：公式、舍入和误差来源。
- [ONNX 导出](docs/onnx-export.md)：标准 ONNX `LSTM` 导出接口。
- [CUDA 性能验收](docs/cuda-performance.md)：测量条件、结果和版本化阈值。
- [贡献指南](CONTRIBUTING.md)：开发环境、测试范围和提交要求。
- [变更记录](CHANGELOG.md)：未发布和正式版本的用户可见变化。

## 状态与限制

- Python package 尚未发布到 PyPI，也未提供脱离源码树的独立 wheel。
- PyTorch 接口仅支持 `num_layers=1`、`dropout=0` 和 `torch.float32`。
- ONNX 导出表达浮点 LSTM 语义，不携带 q-carrier 参数或 QAT Clamp mask。
- CUDA int32 backend 和整数 sigmoid/tanh LUT 尚未实现。
- 真实网络测试需要 Speech Commands v0.02，不进入默认 CI。

## 许可证

仓库尚未提供 `LICENSE` 文件。代码公开可见不等同于获得开源使用许可；正式发布前
需要由项目维护者选择并加入许可证，同时核对第三方来源和许可证声明。
