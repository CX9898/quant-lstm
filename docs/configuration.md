# QuantLSTM 配置与校准

本文是 `QuantLSTM` 用户配置、校准状态和量化参数交换格式的参考。量化公式与
Round/Clamp 位置以[量化执行规格](quantized-execution-spec.md)为准。

## 1. 配置来源与优先级

运行时配置具有两种 JSON 表示：

| 表示 | Schema | 用途 |
| --- | --- | --- |
| sparse override | `config/schema/lstm_quant_override.schema.json` | 构造模块或修改配置时提供用户覆盖项 |
| resolved config | `config/schema/lstm_quant_resolved.schema.json` | C++ resolver 生成的完整执行配置 |

默认值的唯一来源是 `config/defaults/lstm_quant_default_v1.json`。解析顺序为：

```text
default config <- user override -> canonical resolved config
```

Python 不补充默认值。`QuantLSTM` 构造函数把 override 原文交给 C++ resolver，
`get_quant_config()` 返回 resolver 产生的完整 resolved config。

三类值需要分别理解：

- 新建实例没有提供 `quant_config` 时，使用只包含 `schema_version=1` 的空 override，
  因而得到默认文件中的全部值。
- override 省略某个字段时，该字段使用默认文件中的值。
- 示例显式传入的字段只影响该实例，不改变仓库默认文件。

## 2. 顶层字段

最小有效 override 为：

```json
{
  "schema_version": 1
}
```

完整顶层结构为：

| 字段 | 类型 | 必需 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `schema_version` | integer | 是 | 无 | 当前只接受 `1` |
| `scale_mode` | string | 否 | `affine` | 可选 `affine` 或 `pot2` |
| `operators` | object | 否 | 18 个默认量化点 | key 是真实量化点名称 |

未知字段、重复 JSON key、`null`、错误类型和非法枚举会直接失败。resolver 不忽略
无法识别的配置。

## 3. Operator 字段

每个 operator 可配置：

| 字段 | 类型 | 取值 | 默认值 |
| --- | --- | --- | --- |
| `bitwidth` | integer | `8` 或 `16` | `8` |
| `is_unsigned` | boolean | `true` 或 `false` | 见下表 |
| `is_symmetric` | boolean | `true` 或 `false` | `true` |
| `granularity` | string | `per_tensor`、`per_gate`、`per_channel` | 参数为 `per_channel`，其他为 `per_tensor` |

只有 `weight_ih`、`weight_hh`、`bias_ih` 和 `bias_hh` 可以修改 `granularity`。
这四组参数固定为 signed symmetric；设置 unsigned、asymmetric 或非零 zero point
会失败。其余量化点固定为 `per_tensor`。

默认 signed/unsigned 设置为：

| 量化点 | `is_unsigned` | `is_symmetric` |
| --- | ---: | ---: |
| `input_gate_output` | `true` | `true` |
| `forget_gate_output` | `true` | `true` |
| `output_gate_output` | `true` | `true` |
| 其他 15 个量化点 | `false` | `true` |

18 个真实量化点的完整列表见
[量化执行规格的真实量化点](quantized-execution-spec.md#3-真实量化点)。

将全部量化点设为 INT16：

```python
from quant_lstm import QuantLSTM

module = QuantLSTM(16, 32, device="cuda")
module.set_all_bitwidth(16)
assert module.get_quant_config("input")["bitwidth"] == 16
```

修改单个量化点：

```python
module.adjust_quant_config(
    "cell_state",
    bitwidth=16,
    is_unsigned=False,
    is_symmetric=True,
)
```

`set_all_bitwidth()` 和 `adjust_quant_config()` 会立即重新运行 resolver，并使已有
校准参数失效。修改后必须重新校准或加载与新配置一致的参数文档。

## 4. 校准

### 4.1 方法

构造参数 `calibration_method` 接受：

| 方法 | 行为 |
| --- | --- |
| `minmax` | 使用全部观测值的最小值和最大值 |
| `percentile` | 使用 2,048-bin histogram 的中心 `99.99%` 范围 |
| `sqnr` | 在 histogram 候选范围中选择量化噪声最小的范围 |

候选范围仍通过统一 MinMax、minimum-scale fallback、可选 POT2 CoverRange 和执行参数
派生链生成最终 standard scale/zp。方法不改变外部参数格式。

正式校准需要使用来自目标训练或推理分布的代表性数据。随机 tensor 只适用于接口
冒烟测试，不能作为模型精度校准数据。

### 4.2 状态与流程

校准会话状态为：

```text
empty --collect--> dirty --finalize--> locked
  ^                                  |
  +------------- reset -------------+
```

典型单向流程：

```python
module.reset_calibration()
module.calibrating = True

with torch.no_grad():
    for inputs, state in calibration_batches:
        module(inputs, state)

module.calibrating = False
report = module.finalize_calibration()
assert report["batch_count"] > 0
assert module.is_calibrated()

module.use_quantization = True
output, (hidden, cell) = module(inference_input, inference_state)
```

所有 module 参数、calibration batch 和状态必须是 CUDA FP32 tensor。一次 CUDA
浮点 forward 同时生成输出和完整 checkpoint；range、参数分组统计、直方图和三个
cell contribution 诊断在设备端计算。只有每组 min/max 和可选 histogram bins 返回
C++ finalization，原始 tensor 不复制到 CPU，也不会执行第二次 forward。

`finalize_calibration()` 锁定会话并生成 standard scale/zp、内部执行编码和
`NumericSafetyReport`。locked 状态拒绝继续 collect；`reset_calibration()` 是开始
新会话的入口。双向模块分别维护 forward/reverse 状态，且两个方向必须共享完整
input 网格。

CPU collector 是独立 C++ reference，仅用于 CUDA 校准一致性测试，不属于 Python
校准路径或运行时 fallback。

## 5. 参数保存与加载

校准完成后可以保存公共参数文档：

```python
module.export_quant_params("/path/to/quant_params.json")
```

新实例可以加载并直接启用量化：

```python
restored = QuantLSTM(
    16,
    32,
    batch_first=True,
    bidirectional=False,
    device="cuda",
)
restored.load_quant_params("/path/to/quant_params.json")
restored.use_quantization = True
```

公共格式使用 GRU-compatible schema v1：

- 单向 schema：`config/schema/lstm_pytorch_quant_params.schema.json`。
- 双向 schema：`config/schema/lstm_pytorch_bidirectional_quant_params.schema.json`。
- 根字段为 `schema_version`、`model_info`、`execution_metadata`、`operators`，双向
  文档另含 `operators_reverse`。

`model_info` 记录 `input_size`、`hidden_size`、`bias`、`batch_first`、
`bidirectional` 和 `use_pot2_scale`。`execution_metadata` 记录：

```json
{
  "carrier": "cuda_fp32_qcarrier",
  "activation_mode": "real_sigmoid_tanh",
  "cublas_math_mode": "pedantic",
  "standard_scale_mode": "affine"
}
```

每个 operator 使用：

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

单组参数使用 JSON number/integer，多组参数使用数组。四组 weight/bias 的
`scale/zero_point/real_min/real_max` 始终是完整 `4H` 数组，`enc_type` 记录
`PER_TENSOR`、`PER_GATE` 或 `PER_CHANNEL` 来源。`bias=False` 时 bias operator
必须缺失。

公共文档只保存 standard scale/zp，不保存 raw ratio、M+shift、POT2 shift 或 Q31
编码。加载时 Python adapter 转换为内部 canonical bundle，C++ 随后审计 shape、
数值范围、scale、zero point 和执行安全性，并重新派生执行编码。

## 6. 加载时机与失效规则

| 操作 | 校验时机 | 对已有校准参数的影响 |
| --- | --- | --- |
| 构造 `QuantLSTM` | 构造期间解析 override | 创建空校准状态 |
| `adjust_quant_config()` | 调用期间重新运行 resolver | 失效 |
| `set_all_bitwidth()` | 调用期间重新运行 resolver | 失效 |
| 修改 `calibration_method` | property 赋值期间验证 | 失效 |
| `reset_calibration()` | 立即执行 | 清除参数和 safety report |
| `finalize_calibration()` | finalization 期间派生并审计 | 生成并锁定参数 |
| `load_quant_params()` | 加载期间完整审计 | 用导入参数替换会话 |

修改模型 weight 不会自动刷新量化参数。QAT 训练需要按照训练流程定期使用代表性训练
数据重新校准；静态参数缓存的调用方还需要在 master 参数变化后更新 generation key。

## 7. 错误与告警

以下情况直接报错：

- 量化 forward 前没有完成校准或加载参数；
- CPU tensor、非 FP32 tensor、shape 或 bias 不匹配；
- 未知 operator/字段、非法位宽、非法固定值或 compact 参数向量；
- 双向文档缺失 reverse 参数或两个方向的 input 网格不同；
- scale 非有限、非正或不是 canonical FP32；
- 内部 shift 越界、整数载体无法证明安全或可能产生 Inf/NaN；
- `require_exact_accumulation=True` 时任一 FP32 整数运算超过精确范围。

普通模式下，超过 FP32 精确整数范围但仍有限的配置允许继续执行。接口会发出
`RuntimeWarning`，并在 safety report 中记录 `precision_risk`。该告警不会切换到
CPU、int32 或浮点 LSTM fallback；调用方需要通过精度门禁判断该配置是否可接受。
