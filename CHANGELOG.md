# Changelog

本文档记录 quant-lstm 的用户可见功能、修复、兼容性变化和迁移要求。开发过程与临时
调试记录由 Git 历史保存。

## [Unreleased]

- LSTM 的 AIMET/RX 部署编码对齐 QuantGRU：补充 schema_version 3 和 bitwidth，
  使用字符串 is_symmetric、I/O 端口字典、internal_ops.output 列表及扁平参数数组。
  双向 h/c 网格以各方向的 PER_TENSOR 标量编码保留，分别使用 output / cell_state
  量化点；权重和 bias 的部署编码展开为一维 per-channel 数组。公共编码可直接回读，
  原生参数检查点保留独立格式。
- 公共编码增加与 scale 同形的 `n = round(-log2(scale))`；加载时校验已提供的 n，
  并兼容旧版单元素 I/O 编码列表。
- 显式 CPU Python 构建支持 eval FP32 前向、MinMax/Percentile/SQNR 校准和
  FP32 q-carrier PTQ；普通前向无需额外关闭梯度，实际反向传播明确报错。
  训练和 QAT 仍要求 CUDA，CUDA wheel 不提供隐式 CPU fallback。

- 增加与 QuantGRU 对齐的循环算子集成接口，配置、校准生命周期、Po2 和 AIMET/ONNX
  编码处理均由本库负责，无 AIMET 依赖。
- 增加 Percentile 数值、参数锁及模型复制接口；per-gate ONNX 编码按原生展开形式重排。
- 构建支持外部目录、显式 CUDA 架构和构建期 JSON 头文件 wheel。

### Added

- 不依赖 native checkpoint/mask 的量化前向与 QAT autograd oracle，加入默认 E2E；
  增加配置矩阵、图与状态生命周期、非连续 tensor 和 CUDA stream 正确性验证。

- 单层单向和双向 `QuantLSTM` CUDA FP32 forward/backward。
- CUDA FP32 q-carrier INT8、INT16 和混合位宽量化前向与 QAT backward。
- CUDA MinMax、SQNR 和 Percentile 校准，以及 GRU-compatible v1 参数交换格式。
- CPU FP32 q-carrier 与 CPU int32 carrier reference。
- 标准 ONNX `LSTM` 导出、CMake package 和 CPU-only 外部 consumer 验证。
- Speech Commands v0.02 快速与完整数据集训练门禁。
- 可脱离源码目录安装的 Python wheel，包含 native extension 和默认量化配置。

### Changed

- 普通源码安装默认构建 Release CUDA 库，测试和示例改为显式开启；安装命令简化为
  `cmake -S . -B build`。已有 CMake 缓存和显式构建选项继续生效。

- README 按 QuantGRU 的使用流程组织，补充保存重载、ONNX 导出示例及接口速查。

- 重构 README，补充安装、PTQ/QAT 使用、目录、测试和贡献入口；配置文档同步明确
  QAT 沿用首次 PTQ 校准参数，不按 epoch 重新校准。

- 公共 JSON Schema 从 `config/schema/` 移至根目录 `schemas/`，与用户配置分开；
  CMake 安装后的规范文件位于 `share/quant-lstm/schemas/`，默认配置仍位于
  `share/quant-lstm/config/`。

- 开发约定和测试范围合并至文档导航与测试说明，安装文档同步调整。
- 公共量化参数统一使用 `model_info`、`operators`、可选
  `operators_reverse` 和 operator 级 standard scale/zp。
- POT2 CoverRange 对退化校准范围复用统一 minimum-scale fallback。

### Fixed

- CPU eval 前向与校准接受默认梯度上下文，覆盖全部输出及输入、状态和参数依赖；
  不保留原生推理缓存，反向传播明确拒绝，兼容原地激活与混合线性网络。
- 校准后执行 `enable_pot2()` 时，按新 scale 重新计算非对称 zero point 和实数范围；
  内存中保留校准下界以避免额外舍入偏差，深拷贝保留这些信息。文件重载后以现有
  网格下界转换，导出格式不增加诊断字段。更新需配套重建 `_quant_lstm` extension。
- rx-met 集成保存与导出时遵守 GRU/LSTM 的量化开关：关闭的层保留浮点 ONNX 节点，
  不写入量化 encoding。相同配置的阶段恢复保持浮点状态，严格加载仍检查启用层。
  此开关策略位于 rx-met 公共流程，算子的显式编码导入导出接口保持原有行为。
- 有符号量化与 QAT 使用完整 INT8/INT16 整数范围；对称校准仍使用 `max_abs/qmax`。
  `real_min/real_max` 由最终 scale、zero point 和整数上下界计算，保留最小负值。

- CUDA 浮点测试比较器拒绝 NaN、Infinity 和长度不匹配；真实网络 runner 在无
  可用 CUDA GPU 时失败退出，避免将跳过训练误报为验收成功。

- 修正测试 oracle 中残留的先加 zero point 再舍入公式，并增加正负 half tie 回归。

- CPU/CUDA standard-scale 量化统一先执行 RNE 再加 zero point，修复非对称量化
  在奇数 zero point 的 half tie 处偏差一个量化级的问题。
- QAT 激活 Clamp mask 改为检查舍入后的量化值，保留边界外半个 LSB 内未真正
  截断的梯度，并继续屏蔽实际越界值的梯度。
  此修复会改变 QAT 训练轨迹：完整 35 类 Speech Commands 门禁通过，但四类快速
  profile 的原有质量门禁尚未通过；未修改训练配方或放宽验收阈值。

### Known limitations

- Python runtime 只支持 FP32 tensor、单层 LSTM 和 `dropout=0`；CPU 仅支持显式
  CPU wheel 的 eval 校准与 PTQ，浮点训练、反向传播和 QAT 仍要求 CUDA。
- Python package 尚未发布到 PyPI；wheel 需要在兼容的 Python、PyTorch 和 CUDA
  构建环境中生成。
- CPU int32 reference 的 sigmoid/tanh 使用浮点函数；CUDA int32 backend 尚未实现。
- 仓库尚未声明许可证。
