# C++ 与 CUDA 实现

本目录包含 `quant_lstm` 原生库实现。

- `lstm/` 实现 CPU/CUDA forward/backward、校准、配置解析和参数 I/O。
- `quantization/` 实现共享量化原语和数值安全分析。

CUDA 生产路径与 CPU reference 可以使用不同载体，但共享量化点、standard scale/zp、
Round/Clamp 位置和融合计算图。跨模块语义变更需要先更新
[量化执行规格](../docs/quantized-execution-spec.md)。
