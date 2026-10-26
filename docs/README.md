# 文档导航

本目录维护 quant-lstm 的公开用户文档、架构说明、量化规格和可复现验证记录。
代码接口以当前源码为准，量化数学和执行边界以
[量化执行规格](quantized-execution-spec.md)为准。

## 使用者文档

| 文档 | 内容 |
| --- | --- |
| [安装指南](installation.md) | Python、C++、CPU-only 和 Docker 安装 |
| [配置与校准](configuration.md) | 配置字段、默认值、校准和参数导入导出 |
| [ONNX 导出](onnx-export.md) | 标准 ONNX `LSTM` 导出方式和配套编码 |
| [AIMET/rx-met 集成](aimet_integration.md) | 通用接口、Po2 转换、量化开关、阶段恢复及部署编码格式 |
| [真实网络测试](../tests/real_network/README.md) | Speech Commands v0.02 运行方法、阈值和结果 |

## 维护者文档

| 文档 | 内容 |
| --- | --- |
| [系统架构](architecture.md) | 模块、接口、数据流、制品、约束和风险 |
| [量化执行规格](quantized-execution-spec.md) | 权威量化语义、载体契约和验证要求 |
| [量化公式推导](lstm-quantization-formula-derivation.md) | 公式、整数编码、STE 和误差来源 |
| [CUDA 性能验收](cuda-performance.md) | 测量环境、性能结果和版本化门禁 |
| [测试说明](../tests/README.md) | 测试层级、验证命令和改动对应的必跑范围 |
| [架构决策](adr/README.md) | 已接受设计决策及其影响 |

开发过程和已完成的阶段计划由 Git 历史保存。尚未实现的 CUDA integer backend 与
整数激活 LUT 在[系统架构](architecture.md#8-约束风险与扩展)中记录为条件性扩展，
不作为当前能力描述。

## 开发约定

开发环境与 Docker 用法见[安装指南](installation.md)，提交前按
[测试说明](../tests/README.md#按改动范围验证)运行对应验证。

- 运算逻辑位于 C++/CUDA 核心；Python 层负责公共模块接口、配置传递、布局整理和
  扩展调度。测试 oracle 可以使用独立 PyTorch 公式。
- C++ 和 CUDA 使用根目录的 `.clang-format`。修改保持在所属模块内，公共量化
  语义集中在已有公共原语中。
- 不提交构建目录、Python cache、扩展 `.so`、测试报告或数据集 cache。
- commit message 使用英文 Conventional Commits，例如
  `fix(calibration): handle degenerate POT2 ranges`。
- 功能实现和测试可以拆分提交，但每个提交都应保持可构建，并明确对应行为。
- 用户可见行为、配置、依赖、命令、制品、量化公式或测试阈值变化时，同步更新
  对应权威文档和 [CHANGELOG](../CHANGELOG.md)。根 README 只保留摘要和入口，
  完整内容写入专题文档。

提交文档前运行 `git diff --check`，并检查 Markdown 相对链接、代码围栏、
JSON/Python/Shell 示例、公开路径和敏感信息。性能或精度结论必须包含可复现条件，
不得把单次本地结果写成通用保证。
