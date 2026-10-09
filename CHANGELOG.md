# Changelog

本文档记录 quant-lstm 的用户可见功能、修复、兼容性变化和迁移要求。开发过程与临时
调试记录由 Git 历史保存。

## [Unreleased]

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

- Python runtime 只支持 CUDA FP32 tensor、单层 LSTM 和 `dropout=0`。
- Python package 尚未发布到 PyPI；wheel 需要在兼容的 Python、PyTorch 和 CUDA
  构建环境中生成。
- CPU int32 reference 的 sigmoid/tanh 使用浮点函数；CUDA int32 backend 尚未实现。
- 仓库尚未声明许可证。
