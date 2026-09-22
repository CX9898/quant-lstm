# 文档导航

本目录维护 quant-lstm 的公开用户文档、架构说明、量化规格和可复现验证记录。
代码接口以当前源码为准，量化数学和执行边界以
[量化执行规格](quantized-execution-spec.md)为准。

## 使用者文档

| 文档 | 内容 |
| --- | --- |
| [安装指南](installation.md) | Python、C++、CPU-only 和 Docker 安装 |
| [配置与校准](configuration.md) | 配置字段、默认值、校准和参数导入导出 |
| [ONNX 导出](onnx-export.md) | 标准 ONNX `LSTM` 导出方式和限制 |
| [真实网络测试](../tests/real_network/README.md) | Speech Commands v0.02 运行方法、阈值和结果 |

## 维护者文档

| 文档 | 内容 |
| --- | --- |
| [系统架构](architecture.md) | 模块、接口、数据流、制品、约束和风险 |
| [量化执行规格](quantized-execution-spec.md) | 权威量化语义、载体契约和验证要求 |
| [量化公式推导](lstm-quantization-formula-derivation.md) | 公式、整数编码、STE 和误差来源 |
| [CUDA 性能验收](cuda-performance.md) | 测量环境、性能结果和版本化门禁 |
| [贡献指南](../CONTRIBUTING.md) | 开发、测试、提交和文档维护规则 |
| [架构决策](adr/README.md) | 已接受设计决策及其影响 |

开发过程和已完成的阶段计划由 Git 历史保存。尚未实现的 CUDA integer backend 与
整数激活 LUT 在[系统架构](architecture.md#约束风险与扩展)中记录为条件性扩展，
不作为当前能力描述。
