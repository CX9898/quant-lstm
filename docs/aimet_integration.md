# 通用循环算子集成接口

QuantLSTM 的集成逻辑在本仓库维护，`pytorch/lstm_aimet.py` 通过 mixin 提供
QuantGRU 同名的公共方法，不导入 `aimet_torch` 或 `aimet_common`。工具包只需注册
类、ONNX 类型和导出 hook，遍历模块后调用公共方法，不解析 LSTM 内部量化点。

| 接口 | 职责 |
| --- | --- |
| `load_bitwidth_config(config_file, verbose=False)` | 从完整阶段 JSON 文件/字典读取 `LSTM_config`，复用原生配置校验 |
| `calibration_context()` | 开始新一轮收集，成功退出时 finalize，异常恢复标志；跳过锁定参数和未执行分支 |
| `enable_pot2(method="cover_range", tolerance=0.02)` | 转换 scale、重算非对称 zero point 和实数范围，通过原生加载器重建执行参数并做数值审计 |
| `export_quant_params_to_aimet_format(encodings_dict, module_name=None, verbose=False, for_onnx=True)` | 仅向公共 activation_encodings / param_encodings 合并部署参数 |
| `load_quant_params_from_aimet_format(encodings_dict, module_name=None, verbose=False)` | 从公共编码恢复量化网格；缺少当前模块时返回 False；不修改量化开关 |
| `set_module_name(name)` | 设置编码所用模块路径 |
| `set_quant_params_locked(locked=True)` | 锁定已有网格，阻止集成校准覆盖 |
| `normalize_quant_lstm_onnx(onnx_path)` | 标准 ONNX LSTM 节点及 W/R/B initializer 名称对齐 |

`owns_quantization=True` 表示量化由算子内部负责，工具包不能再附加输入、输出、权重
量化器。`calibration_method`、`percentile_value`、`use_quantization`、`export_mode`
保留原有语义；量化方案选择由工具包传入，量化参数计算仍在算子内部。

## 阶段配置

```json
{
  "LSTM_config": {
    "use_quantization": true,
    "quant_config": {
      "schema_version": 1,
      "scale_mode": "affine",
      "operators": {
        "cell_state": {"bitwidth": 16},
        "weight_ih": {"bitwidth": 8, "granularity": "per_gate"}
      }
    }
  }
}
```

未提供 `LSTM_config` 时不改变模块设置。`quant_config` 采用原生 sparse override
格式，相同配置可重复加载；改变已校准配置前必须显式 `reset_calibration()`。
该操作同时清除锁。QAT 沿用已校准参数，不应在每个训练迭代重新进入校准上下文。

```python
module.load_bitwidth_config("stage.json")
with module.calibration_context():
    module(calibration_batch)
encodings = module.export_quant_params_to_aimet_format({}, module_name="encoder.lstm", for_onnx=False)
restored.load_quant_params_from_aimet_format(encodings, module_name="encoder.lstm")
restored.use_quantization = True
restored.set_quant_params_locked(True)
```

## 量化开关与阶段恢复

`use_quantization=False` 控制普通前向使用浮点计算。进入 `calibration_context()`
仍可收集参数，因此“已经校准”和“启用量化”是两个独立状态。

rx-met 的 `save_quantizer_encodings()` 和 `export_onnx_json()` 按量化开关处理：

| 当前模块状态 | 阶段保存 / 部署 encoding | ONNX |
| --- | --- | --- |
| 已关闭量化，无论是否校准 | 不写入该层的 activation / parameter encoding | 保留标准浮点 LSTM 节点 |
| 已启用量化且完成校准 | 写入公共 encoding | 保留标准 LSTM 节点，与 encoding 配套使用 |
| 已启用量化但未校准 | 尚无可保存参数，部署导出报错 | 校准或加载有效参数后再导出 |

恢复阶段文件时，先创建相同结构、应用相同阶段配置并加载配套权重，再调用
`load_quantizer_encodings()`。关闭量化且文件中没有该层 encoding 时保持浮点状态；
`skip_if_not_found=False` 仍会拒绝已启用层缺少 encoding 的情况。
显式加载文件中已有的有效 encoding 会启用该层量化；传入 `allow_overwrite=False`
还会锁定加载的参数，使后续共享校准跳过该层。

上述开关策略由 rx-met 公共流程负责。直接调用算子的
`export_quant_params_to_aimet_format()` 仍可导出已校准但关闭量化的模块；
`load_quant_params_from_aimet_format()` 只恢复参数，不改变 `use_quantization`。
算子自身的原生参数接口也保留这种显式导入导出能力。

## 已校准参数的 Po2 转换

`enable_pot2()` 在校准前调用时选择后续校准的 `pot2` 模式；在校准后调用时转换
已有网格，不重新执行校准前向。已处于 Po2 模式时重复调用不改变参数。
已用 `set_quant_params_locked(True)` 锁定的 affine 参数不能直接转换，应在锁定前
完成转换，或在首次配置中指定 `scale_mode: "pot2"`。

默认 `method="cover_range"`、`tolerance=0.02`；校准后转换也支持 `method="round"`。
这些是 Python 方法参数，不增加 JSON 配置字段。原生 Po2 校准使用固定的
CoverRange 策略，详见[量化执行规格](quantized-execution-spec.md#62-pot2)。

转换得到新 scale 后，对称量化保持 `zero_point=0`；非对称量化重新计算：

```text
zero_point = clamp(round_to_nearest_even(qmin - lower / new_scale), qmin, qmax)
real_min = (qmin - zero_point) * new_scale
real_max = (qmax - zero_point) * new_scale
```

`lower` 优先取校准后保留在内存中的调整后下界，避免使用已取整的 affine 下界
再次引入舍入偏差。深拷贝保留这些范围；重置校准或加载参数文件会清除旧范围。
从原生或公共 encoding 文件恢复的模块没有原始校准统计，转换时使用已加载量化
网格的 `real_min`。两条路径均在转换后重新派生执行参数并完成原生数值审计。
这些诊断范围不写入任何导出 JSON。

例如，INT8 非对称输入校准范围为 `[-1, 9]` 时，affine 的 scale 约为 `10/255`、
zero point 为 `-103`。默认 Po2 转换后 scale 为 `0.0625`、zero point 为 `-112`，
对应实数范围为 `[-1, 14.9375]`。

## 编码与 ONNX

部署编码与 QuantGRU / rx-met `export_onnx_json()` 使用相同的记录结构：

- 顶层结构沿用已有 QuantGRU：`schema_version: 3`、`activation_encodings` 和
  `param_encodings`。rx-met 导出器中的 `version: "1.0.0"` 来自 AIMET；算子不新增或
  改写该字段，不新增 LSTM 专有顶层区段。
- 每条编码包含 `dtype`、整数 `bitwidth`、字符串 `is_symmetric`（`"True"` / `"False"`）、
  `enc_type`、`scale`、`zero_point`、`real_min`、`real_max`；不输出原生 `symmetric` 字段。
- `real_min = (qmin-zero_point)*scale`、`real_max = (qmax-zero_point)*scale`，
  signed 使用完整 INT8 `[-128,127]` / INT16 `[-32768,32767]`，与 GRU 一致；
  signed symmetric 校准仍按 `max_abs/qmax` 计算 scale，不减少负端可表示值。
- `activation_encodings[模块路径]` 标记 `is_LSTM`。与 GRU 一致，`input` 和 `output`
  各为单元素编码列表，分别取正向 `input`、`output` 的 `PER_TENSOR` 标量编码；
  它们不枚举 ONNX 或模块的全部输入输出端口。
- `internal_ops[量化点].output` 是单元素编码列表，保存内部激活的量化网格；
  双向反向网格放在同构的 `internal_ops_reverse`。权重和 bias 放在 `param_encodings`。
- h0/h 使用各方向的 `internal_ops.output` 编码，c0/c 使用 `internal_ops.cell_state`
  编码；反向分别在 `internal_ops_reverse` 中读取。状态和激活在每个方向内固定为
  `PER_TENSOR`，四个数值字段均为标量；正反向可保留不同的 scale，不展开成通道数组。
- `param_encodings` 使用 `模块路径.weight_ih.weight`、`weight_hh.weight`、`bias`，
  对应归一化 ONNX 的 W/R/B。参数从 IFGO 重排为 IOFC，再按正向、反向拼接为一维数组。
  W/R 的四个数值字段长度为 `directions * 4 * hidden_size`，B 为
  `directions * 8 * hidden_size`；方向内 B 的顺序为 bias_ih、bias_hh。
  原生 per-gate 文档已按通道展开，导出时直接重排，per-tensor 标量才需要展开。
- ONNX 的 B 合并 bias_ih/bias_hh，要求两者 dtype、symmetric 一致。与 GRU 相同，
  阶段保存和 ONNX 导出使用相同布局，`for_onnx=False` 也适用这一约束。需要保存
  不同 bias 位宽时，使用算子独立的原生 `export_quant_params()` 接口。
- 每个模块仍导出为一个标准 ONNX `LSTM` 节点，双向通过 `direction="bidirectional"`
  表示。内部量化点只写入配套编码文件，不展开成 ONNX 子图。
- 导出 hook 应在常量折叠后运行；参数必须是 initializer。共享常量会复制，
  保留其他消费者的输入；重复归一化不改变结果。

例如，一个内部量化点采用以下结构（数值仅作结构示例）：

```json
{
  "internal_ops": {
    "input_gate_output": {
      "output": [{
        "dtype": "UINT8",
        "bitwidth": 8,
        "is_symmetric": "True",
        "enc_type": "PER_TENSOR",
        "scale": 0.00390625,
        "zero_point": 0,
        "real_min": 0.0,
        "real_max": 0.99609375
      }]
    }
  }
}
```

`load_quant_params_from_aimet_format()` 只读取上述公共区段：拆分参数方向和两组
bias，将 ONNX 的 IOFC 顺序还原为 IFGO，结合 `internal_ops` / `internal_ops_reverse`
重建执行参数。导出和回读均不依赖 `quant_lstm_encodings`。展开后的参数按 per-channel
恢复，量化网格及输出保持一致；不恢复原始配置曾采用的 per-tensor / per-gate 分组标签。
全部 scale 都是 2 的幂时采用 Po2 执行，否则采用 affine；cuBLAS 模式沿用目标模块配置。

算子自身的 `export_quant_params()` / `load_quant_params()` 是独立的原生检查点接口，
用于保留模型信息、执行配置和原始分组。其 `schema_version: 1`、boolean `symmetric`
及 `execution_metadata` 等字段仅属于原生格式，不写入 rx-met 的部署 encodings。
编译器应使用 `export_onnx_json()` 生成的 ONNX 和公共 encodings 配对文件。

标准 ONNX 图表达浮点计算，整数部署需要消费配套编码，并支持 LSTM 的量化点和
各方向的状态网格。与 GRU 对齐的是编码结构，不是两个算子的内部计算语义。

## 验证

从 `pytorch/` 执行 `python3 -m unittest -v tests.test_aimet_interface`，覆盖独立配置、
校准异常、锁定、Po2、单双向及三种参数粒度的部署 schema、双向门顺序和独立
h/c 网格、公共编码直接回读、非法记录拒绝及共享 ONNX initializer。Po2 回归包含
8/16 位有符号/无符号、三种校准方法、双向与深拷贝、文件加载后的舍入及 zero point
限幅；保留 signed symmetric 校准跨度和完整负端范围的回归。
该测试也已加入 `tools/run_end_to_end_test.sh`。rx-met 中的
`tests.test_native_recurrent_state` 另行覆盖 GRU/LSTM 关闭量化、混合启用状态、阶段
恢复、严格加载、ONNX 配套编码及 GRU 非对称 Po2 对照。
