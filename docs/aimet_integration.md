# 通用循环算子集成接口

QuantLSTM 的集成逻辑在本仓库维护，`pytorch/lstm_aimet.py` 通过 mixin 提供
QuantGRU 同名的公共方法，不导入 `aimet_torch` 或 `aimet_common`。工具包只需注册
类、ONNX 类型和导出 hook，遍历模块后调用公共方法，不解析 LSTM 内部量化点。

| 接口 | 职责 |
| --- | --- |
| `load_bitwidth_config(config_file, verbose=False)` | 从完整阶段 JSON 文件/字典读取 `LSTM_config`，复用原生配置校验 |
| `calibration_context()` | 开始新一轮收集，成功退出时 finalize，异常恢复标志；跳过锁定参数和未执行分支 |
| `enable_pot2(method="cover_range", tolerance=0.02)` | 转换 scale/范围，通过原生加载器重建执行参数和数值审计 |
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
h/c 网格、公共编码直接回读、非法记录拒绝及共享 ONNX initializer。
该测试也已加入 `tools/run_end_to_end_test.sh`。
