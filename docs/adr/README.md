# 架构决策记录

本目录记录已经接受且持续影响公共接口、模块边界或运行语义的设计决策。量化公式的
权威定义仍位于[量化执行规格](../quantized-execution-spec.md)。

| ADR | 状态 | 决策 |
| --- | --- | --- |
| [0001](0001-cuda-only-python-runtime.md) | Superseded in part by 0002 | 原始 CUDA-only Python 边界；训练和 QAT 仍适用 |
| [0002](0002-cpu-calibration-and-ptq.md) | Accepted, revised in 1.1.1 | 显式 CPU wheel 支持 eval 浮点前向、校准与 PTQ；反向传播明确报错 |
