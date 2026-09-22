# 架构决策记录

本目录记录已经接受且持续影响公共接口、模块边界或运行语义的设计决策。量化公式的
权威定义仍位于[量化执行规格](../quantized-execution-spec.md)。

| ADR | 状态 | 决策 |
| --- | --- | --- |
| [0001](0001-cuda-only-python-runtime.md) | Accepted | PyTorch runtime 仅执行 CUDA backend，CPU 实现作为显式 C++ reference |
