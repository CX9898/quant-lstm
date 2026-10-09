# 通用循环算子集成接口

QuantLSTM 的集成逻辑在本仓库维护，`pytorch/lstm_aimet.py` 通过 mixin 提供
QuantGRU 同名的公共方法，不导入 `aimet_torch` 或 `aimet_common`。工具包只需注册
类、ONNX 类型和导出 hook，遍历模块后调用公共方法，不解析 LSTM 内部量化点。

| 接口 | 职责 |
| --- | --- |
| `load_bitwidth_config(config_file, verbose=False)` | 从完整阶段 JSON 文件/字典读取 `LSTM_config`，复用原生配置校验 |
| `calibration_context()` | 开始新一轮收集，成功退出时 finalize，异常恢复标志；跳过锁定参数和未执行分支 |
| `enable_pot2(method="cover_range", tolerance=0.02)` | 转换 scale/范围，通过原生加载器重建执行参数和数值审计 |
| `export_quant_params_to_aimet_format(encodings_dict, module_name=None, verbose=False, for_onnx=True)` | 向总编码字典合并原生参数和部署参数 |
| `load_quant_params_from_aimet_format(encodings_dict, module_name=None, verbose=False)` | 恢复原生参数；缺少当前模块时返回 False；不修改量化开关 |
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

- 顶层 `schema_version: 3`，包含 `activation_encodings` 和 `param_encodings`。
- 每条编码包含 `dtype`、整数 `bitwidth`、字符串 `is_symmetric`（`"True"` / `"False"`）、
  `enc_type`、`scale`、`zero_point`、`real_min`、`real_max`；不输出原生 `symmetric` 字段。
- `activation_encodings[模块路径]` 标记 `is_LSTM`，`input` 和 `output` 都是编码列表。
  输入顺序为 x/h0/c0，输出顺序为 sequence/h/c，表示模块逻辑 I/O，而非 ONNX 节点
  含 W/R/B 的物理输入顺序。
- `internal_ops[量化点].output` 是单元素编码列表，保存内部激活的量化网格；
  双向反向网格放在同构的 `internal_ops_reverse`。权重和 bias 放在 `param_encodings`。
- 双向 sequence/h/c 使用 `PER_CHANNEL` 记录，四个数值字段均为长度 `2 * hidden_size`
  的一维数组：前 hidden_size 项来自正向网格，后 hidden_size 项来自反向网格。
  单向状态保留 `PER_TENSOR` 标量记录。双向各方向仍可使用不同的 scale。
- `param_encodings` 使用 `模块路径.weight_ih.weight`、`weight_hh.weight`、`bias`，
  对应归一化 ONNX 的 W/R/B。参数从 IFGO 重排为 IOFC，再按正向、反向拼接为一维数组。
  W/R 的四个数值字段长度为 `directions * 4 * hidden_size`，B 为
  `directions * 8 * hidden_size`；方向内 B 的顺序为 bias_ih、bias_hh。
  原生 per-gate 文档已按通道展开，导出时直接重排，per-tensor 标量才需要展开。
- ONNX 的 B 合并 bias_ih/bias_hh，要求两者 dtype、symmetric 一致。阶段保存使用
  `for_onnx=False`，因此不同 bias 位宽仍可保存和完整恢复。
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

`quant_lstm_encodings[模块路径]` 是附加的原生检查点，保存未经字段转换、展平或门
重排的完整参数，继续用于精确回读和旧阶段文件兼容。原生 `export_quant_params()`
的 schema_version 仍为 1，其 boolean `symmetric` 等字段保持原样；部署消费者应读取
上述标准编码区段。原生检查点回读成功不代表部署格式校验成功，两者分别测试。

标准 ONNX 图表达浮点计算，整数部署需要消费配套编码，并支持 LSTM 的量化点和
双向隐藏通道含义。与 GRU 对齐的是编码结构，不是两个算子的内部计算语义。

## 验证

从 `pytorch/` 执行 `python3 -m unittest -v tests.test_aimet_interface`，覆盖独立配置、
校准异常、锁定、Po2、单双向及三种参数粒度的部署 schema、双向门顺序和独立
h/c 网格、参数精确回读和共享 ONNX initializer。
该测试也已加入 `tools/run_end_to_end_test.sh`。
