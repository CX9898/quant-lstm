# Changelog

本文档记录 quant-lstm 的用户可见功能、修复、兼容性变化和迁移要求。开发过程与临时
调试记录由 Git 历史保存。

## [Unreleased]

### Added

- 单层单向和双向 `QuantLSTM` CUDA FP32 forward/backward。
- CUDA FP32 q-carrier INT8、INT16 和混合位宽量化前向与 QAT backward。
- CUDA MinMax、SQNR 和 Percentile 校准，以及 GRU-compatible v1 参数交换格式。
- CPU FP32 q-carrier 与 CPU int32 carrier reference。
- 标准 ONNX `LSTM` 导出、CMake package 和 CPU-only 外部 consumer 验证。
- Speech Commands v0.02 快速与完整数据集训练门禁。

### Changed

- 公共量化参数统一使用 `model_info`、`operators`、可选
  `operators_reverse` 和 operator 级 standard scale/zp。
- POT2 CoverRange 对退化校准范围复用统一 minimum-scale fallback。

### Known limitations

- Python runtime 只支持 CUDA FP32 tensor、单层 LSTM 和 `dropout=0`。
- Python package 目前只支持源码可编辑安装，尚未发布独立 wheel。
- CPU int32 reference 的 sigmoid/tanh 使用浮点函数；CUDA int32 backend 尚未实现。
- 仓库尚未声明许可证。
