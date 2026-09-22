# QuantLSTM 配置与参数文档

## 1. 配置入口

运行时配置只有两种表示：

- 稀疏 override：遵循 `config/schema/lstm_quant_override.schema.json`。
- 完整 resolved config：遵循 `config/schema/lstm_quant_resolved.schema.json`。

默认值来自 `config/defaults/lstm_quant_default_v1.json`。Python 接口不自行合并
配置；`QuantLSTM` 会把 override 原文交给唯一 C++ resolver，并由
`get_quant_config()` 返回完整 canonical resolved config。

`set_all_bitwidth(8|16)` 会修改全部 18 个真实量化点。
`adjust_quant_config(name, ...)` 只接受以下字段：

- 所有量化点：`bitwidth`、`is_unsigned`、`is_symmetric`。
- `weight_ih/weight_hh/bias_ih/bias_hh`：另可设置
  `granularity=per_tensor|per_gate|per_channel`。

非参数量化点固定为 `per_tensor`。四组 weight/bias 固定 signed symmetric；
非法位宽、未知量化点、未知字段、`null` 和不合法固定值都会在 forward 前失败。
配置变化会使现有校准参数失效，必须重新校准或重新导入参数。

## 2. 校准

支持 `minmax`、`sqnr` 和 `percentile`。典型流程：

```python
module.calibrating = True
module(calibration_batch_0, state_0)
module(calibration_batch_1, state_1)
module.calibrating = False
report = module.finalize_calibration()
module.use_quantization = True
```

校准 batch、参数和状态必须位于 CUDA。一次 CUDA 浮点 forward 同时生成输出和完整
checkpoint；range、per-channel/per-gate 参数统计、直方图以及三个 cell contribution
诊断都直接在设备端计算。只有每组 min/max 和可选 histogram bins 会回传给 C++
finalization，原始 tensor 不复制到 CPU，也不会额外执行第二次 forward。
`finalize_calibration()` 将会话锁定，并生成 standard scale/zp 和执行编码。
`reset_calibration()` 是开始新会话的唯一入口。

CPU 标量 collector 仅作为独立 C++ reference，CUDA 校准测试会逐组比较两者的范围、
样本数和最终参数；它不属于 PyTorch 校准路径或运行时 fallback。

双向模块为 forward/reverse 分别保存参数统计。两个方向使用同一 resolved config，
并强制 input 的完整 standard scale/zp、位宽和对称性逐值相同；h/c、Linear、门和
参数统计保持方向独立。

## 3. 参数文档

公共导入导出统一使用 GRU-compatible PyTorch 参数文档 v1：

- 单向 schema：`lstm_pytorch_quant_params.schema.json`。
- 双向 schema：`lstm_pytorch_bidirectional_quant_params.schema.json`。
- 公共根字段与 GRU 一致，使用 `model_info`、`operators`，双向另含
  `operators_reverse`。
- LSTM 额外保留 `schema_version=1` 和 `execution_metadata`；相同信息仍使用 GRU
  的字段名和 JSON 类型。

`model_info` 包含 `input_size`、`hidden_size`、`bias`、`batch_first`、
`bidirectional` 和 `use_pot2_scale`。每个 operator 使用以下 GRU 字段：

```json
{
  "dtype": "INT8",
  "symmetric": true,
  "scale": 0.01,
  "zero_point": 0,
  "enc_type": "PER_TENSOR",
  "real_min": -1.27,
  "real_max": 1.27
}
```

单组参数使用 JSON number/integer；多组参数使用对应数组。四组 weight/bias 的
`scale/zero_point/real_min/real_max` 始终是完整 `4H` 数组，`enc_type` 记录原始
`PER_TENSOR|PER_GATE|PER_CHANNEL` 粒度；不存在 1/4 元素 compact 副本。
`bias=False` 时 bias operator 必须缺失。

公共文档只保存 standard scale/zp，不保存 raw ratio、M+shift、POT2 shift 或 Q31
编码。Python 边界 adapter 将 GRU-compatible 数值字段转换成私有 canonical bundle；
该 bundle 仍遵循 `lstm_quant_params_bundle.schema.json`，其中 scale 使用最短可往返
FP32 字符串。随后 C++ 严格审计参数并派生全部执行编码。私有 bundle 不是第二套
公共导入导出格式。

执行元数据固定记录：

- `carrier=cuda_fp32_qcarrier`
- `activation_mode=real_sigmoid_tanh`
- `cublas_math_mode=pedantic|tf32`
- `standard_scale_mode=affine|pot2`

双向属性记录在 `model_info.bidirectional=true`。

## 4. 错误与告警

以下情况直接报错：未校准量化推理、CPU tensor 进入任意 PyTorch 执行路径、shape/bias
不匹配、方向参数缺失、双向 input 网格不一致、compact 参数向量、非法 canonical
FP32 scale，以及 `require_exact_accumulation=True` 时出现 FP32 精度风险。

允许继续执行的 `precision_risk` 会写入 safety report 并发出 Python warning；
接口不会静默切换到 int32 后端。
