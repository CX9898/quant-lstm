# QuantLSTM

QuantLSTM 提供 PyTorch LSTM（长短期记忆网络）的量化实现，基于 CUDA 和 PyTorch，
包含 CUDA 浮点计算、训练后量化（PTQ）、量化感知训练（QAT）和 ONNX 导出。
模型参数命名与支持范围内的单层 `torch.nn.LSTM` 对齐。

## 支持范围

- 浮点和量化两种模式，通过 `use_quantization` 切换；量化模式需要先校准或加载量化参数。
- 模型使用 `num_layers=1`、`dropout=0`，支持单向和双向计算、可选 bias。
- 输入为三维序列张量，`batch_first=True` 对应 `[B, T, I]`，默认布局为 `[T, B, I]`。
- 常规前向和校准使用 CUDA float32 张量，返回 `output, (hidden, cell)`。
- 隐藏状态和细胞状态形状均为 `[D, B, H]`，其中单向 `D=1`、双向 `D=2`。
- 校准方法包括 MinMax、SQNR 和 Percentile，默认使用 MinMax。
- 权重和偏置采用对称量化，支持 per-tensor、per-gate 和 per-channel 粒度。
- 位宽支持 8 和 16，各量化点可独立配置；激活和状态固定使用 per-tensor 粒度。

当前 CUDA 量化路径使用 FP32 承载量化后的整数网格值（q-carrier），线性运算由
cuBLAS SGEMM 执行，sigmoid/tanh 使用浮点计算。CPU 浮点和 int32 reference
用于数值验证与 C++ 集成；Python 运行时只支持 CUDA，CUDA int32 后端尚未实现。

## 环境与安装

本仓库提供源码构建、Python wheel 和 CMake package，环境与 Docker 使用说明见
[安装指南](docs/installation.md)。项目尚未发布到 PyPI。

源码构建需要 C++17 编译器、CMake 3.24 及以上版本、`nlohmann_json` 3.11.2 及以上
版本、CUDA Toolkit 和 CUDA 版 PyTorch。Python 需要开发头文件、`pip`、`setuptools`
和 `wheel`，编译工具链与 PyTorch CUDA 版本保持兼容。

以下命令从本仓库根目录执行：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DQUANT_LSTM_ENABLE_CUDA=ON \
  -DQUANT_LSTM_BUILD_TESTS=OFF \
  -DQUANT_LSTM_BUILD_EXAMPLES=OFF
cmake --build build --parallel 2
python -m pip install ./pytorch --no-deps --no-build-isolation
```

CMake 生成 `build/libquant_lstm.a`，pip 构建 `_quant_lstm` 扩展并安装 `quant_lstm`
和默认配置。Python 构建从固定的 `build/` 目录链接原生库，安装完成后运行不依赖
源码目录。CPU-only C++ 安装不需要 CUDA 或 PyTorch，步骤见[安装指南](docs/installation.md)。

## 最小示例

以下示例检查 CUDA 模块加载、浮点前向接口和输出形状：

```python
import torch
from quant_lstm import QuantLSTM

lstm = QuantLSTM(input_size=64, hidden_size=128, batch_first=True).cuda().eval()
inputs = torch.randn(2, 10, 64, device="cuda")
with torch.no_grad():
    output, (hidden, cell) = lstm(inputs)

assert output.shape == (2, 10, 128)
assert hidden.shape == cell.shape == (1, 2, 128)
assert torch.isfinite(output).all()
print("QuantLSTM forward OK")
```

完成断言并打印 `QuantLSTM forward OK` 表示前向冒烟检查通过。示例中的随机输入
用于接口检查，模型量化使用代表性校准数据。

## 量化流程

推荐流程为 **浮点预训练 → PTQ 校准与评估 → 固定量化参数的 QAT 微调 → 推理**。
已有单层、无投影、`dropout=0` 的 `torch.nn.LSTM` 可通过
`lstm.load_state_dict(pretrained_lstm.state_dict())` 加载权重，模型形状、方向和 bias
配置应保持一致。

以下代码接续已加载浮点权重的 `lstm`，`calibration_loader` 每次返回形状为
`[B, T, 64]` 的代表性输入 tensor。示例为每个 batch 使用默认的零初始状态：

```python
lstm.set_all_bitwidth(8)
lstm.reset_calibration()
lstm.calibrating = True
with torch.no_grad():
    for inputs in calibration_loader:
        lstm(inputs.to(device="cuda", dtype=torch.float32))
lstm.calibrating = False
lstm.finalize_calibration()
lstm.use_quantization = True
```

默认配置使用 `scale_mode="affine"`，`scale_mode="pot2"` 将 scale 限制为 2 的整数
次幂。构造函数的 `quant_config` 接受配置字典、JSON 文本或文件路径；逐量化点
调整使用 `adjust_quant_config()`。配置字段、默认值与失效规则见
[配置说明](docs/configuration.md)。

QAT 在 PTQ 评估后使用 `lstm.train()` 和常规 PyTorch 训练循环。
**scale、zero point 和位宽沿用首次 PTQ 校准结果，不按 epoch 重新校准**；浮点主权重
继续更新，每次前向按固定参数量化，反向传播使用 STE 和截断 mask。
微调期间保持 `calibrating=False`、`use_quantization=True`，不要重置校准或修改配置。
完整评估流程见[Speech Commands 真实网络测试](tests/real_network/README.md)。

## 保存与重载

模型权重与量化参数共同确定推理结果。以下代码接续已校准的 `lstm`：

```python
torch.save(lstm.state_dict(), "lstm_weights.pth")
lstm.export_quant_params("lstm_encodings.json")

restored = QuantLSTM(64, 128, batch_first=True).cuda().eval()
restored.load_state_dict(
    torch.load("lstm_weights.pth", map_location="cuda", weights_only=True)
)
restored.load_quant_params("lstm_encodings.json")
restored.use_quantization = True
```

重建实例沿用原模型的形状、方向和 bias 配置，并用相同输入比较保存前后的输出。
模型权重与量化文件按同次导出结果配套使用。

## ONNX 导出

导出使用 `export_mode=True` 和 PyTorch legacy exporter，当前使用 opset 18。
安装 `onnx` 后，以下代码接续前面的 `lstm`，省略初始状态时使用零状态：

```python
from quant_lstm import ensure_quant_lstm_onnx_registered

ensure_quant_lstm_onnx_registered(opset=18)
lstm.cpu().eval()
try:
    lstm.export_mode = True
    torch.onnx.export(
        lstm,
        torch.zeros(1, 10, 64),
        "lstm.onnx",
        opset_version=18,
        dynamo=False,
        input_names=["input"],
        output_names=["output", "hidden", "cell"],
    )
finally:
    lstm.export_mode = False
    lstm.cuda()
```

产物使用标准 ONNX `LSTM` 节点，表达浮点 LSTM 语义，量化参数单独导出。
显式初始状态、双向模型和 ONNX Runtime 一致性验证见[ONNX 导出说明](docs/onnx-export.md)。

## 接口与验证

| 接口 | 用途 |
| --- | --- |
| `forward(input, hx=None)` | 按量化开关执行浮点或量化计算，`hx` 为 `(h_0, c_0)` |
| `set_all_bitwidth()` | 将所有量化点设为 8-bit 或 16-bit |
| `get_quant_config()`、`adjust_quant_config()` | 查询或调整量化点的配置 |
| `finalize_calibration()`、`reset_calibration()` | 生成并锁定量化参数，或清除校准状态 |
| `is_calibrated()`、`calibration_state()` | 查询校准是否完成及会话状态 |
| `export_quant_params()`、`load_quant_params()` | 导出或加载量化参数 JSON |

[端到端测试](tools/run_end_to_end_test.sh) 覆盖 CUDA、PyTorch、QAT 和 ONNX；
[CPU package 测试](tools/run_cpu_only_package_check.sh) 检查 reference、配置、安装和
外部 CMake 项目集成；[真实网络测试](tests/real_network/README.md) 比较浮点、PTQ
及固定参数 QAT。命令从仓库根目录执行：

```bash
tools/run_cpu_only_package_check.sh
tools/run_end_to_end_test.sh
```

当前 [GitHub Actions CI](.github/workflows/ci.yml) 运行 CPU-only 验证。CUDA 测试
需要可用 GPU，真实网络测试还需要 Speech Commands v0.02 数据集。测试依赖、
数据规模与通过阈值见[测试说明](tests/README.md)及相应测试脚本。

## 文档与源码

| 路径 | 职责 |
| --- | --- |
| [config/defaults/](config/defaults/) | 用户可调整的默认配置，`comment` 解释各量化点 |
| [schemas/](schemas/) | 配置和参数文件的格式规范，日常使用无需修改 |
| [docs/configuration.md](docs/configuration.md) | 配置字段、校准流程、参数交换与失效规则 |
| [docs/lstm-quantization-formula-derivation.md](docs/lstm-quantization-formula-derivation.md) | 逐算子公式、舍入、重缩放与 STE |
| [docs/quantized-execution-spec.md](docs/quantized-execution-spec.md) | 量化执行语义和约束 |
| [tests/real_network/](tests/real_network/README.md) | 浮点预训练、PTQ 与固定参数 QAT 质量门禁 |
| [example/](example/README.md) | C++ 浮点与 int32 reference 示例 |
| `include/`、`src/` | C++/CUDA 计算、校准和量化实现 |
| `pytorch/` | Python 模块、autograd 和扩展 binding |
| [CHANGELOG.md](CHANGELOG.md) | 功能变化与迁移记录 |

完整入口见[文档导航](docs/README.md)。提交问题时请附最小复现和运行环境；
代码贡献遵循[开发约定](docs/README.md#开发约定)，并运行与改动范围对应的测试。

## 许可证

QuantLSTM 仓库目前尚未提供 `LICENSE` 文件，尚未声明开源许可证。
