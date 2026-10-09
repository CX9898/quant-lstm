# PyTorch 接口

本目录提供 CUDA-only `QuantLSTM` Python API 和 `_quant_lstm` extension。

| 文件 | 职责 |
| --- | --- |
| `quant_lstm.py` | 模块状态、配置、校准和参数导入导出 |
| `lstm_autograd.py` | autograd Function 与 extension 调度 |
| `lstm_aimet.py` | 通用循环算子接入接口：配置、校准生命周期、Po2、AIMET 编码与 ONNX 参数对齐 |
| `lstm_onnx.py` | 标准 ONNX `LSTM` symbolic |
| `lib/lstm_interface_binding.cc` | PyTorch 与 C++/CUDA 核心边界 |
| `tests/` | Python 接口、backward、STE、双向和 ONNX 验证 |

Python 层不实现生产 LSTM 公式，也不提供 CPU fallback。普通安装和 wheel 构建见
[安装指南](../docs/installation.md#2-安装-pytorch-cuda-模块)，用户配置与校准流程见
[配置参考](../docs/configuration.md)。

正确性 E2E 包含 `tests.test_qat_independent`：从 master 参数和 standard scale/zp
独立编码 rescale、计算完整前向轨迹与所有 Clamp mask，再通过 PyTorch autograd
核验七类梯度。参考模型不消费 native checkpoint/mask；激活导数遵循冻结规格
第 9.4 节，在反量化后的 q 上求值。CUDA FP32 激活与 CPU 激活在舍入半格附近可能
相差一个量化级，因此逐位核验使用独立 PyTorch CUDA 激活函数。

覆盖 INT8/INT16、混合位宽、signed/unsigned、三种参数粒度、affine/POT2、两种布局、
饱和输入、有无 bias，以及多次前向后反向、非连续 tensor、分段状态和非默认 stream。
这些数值测试不能代替 Speech Commands 的训练质量门禁。

AIMET/rx-met 接入方法与编码约定见 [循环算子集成接口](../docs/aimet_integration.md)。
