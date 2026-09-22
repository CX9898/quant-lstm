# PyTorch 接口

本目录提供 CUDA-only `QuantLSTM` Python API 和 `_quant_lstm` extension。

| 文件 | 职责 |
| --- | --- |
| `quant_lstm.py` | 模块状态、配置、校准和参数导入导出 |
| `lstm_autograd.py` | autograd Function 与 extension 调度 |
| `lstm_onnx.py` | 标准 ONNX `LSTM` symbolic |
| `lib/lstm_interface_binding.cc` | PyTorch 与 C++/CUDA 核心边界 |
| `tests/` | Python 接口、backward、STE、双向和 ONNX 验证 |

Python 层不实现生产 LSTM 公式，也不提供 CPU fallback。源码可编辑安装方式见
[安装指南](../docs/installation.md#2-安装-pytorch-cuda-模块)，用户配置与校准流程见
[配置参考](../docs/configuration.md)。
