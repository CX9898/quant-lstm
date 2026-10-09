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

- `quant_lstm_encodings[模块路径]` 保存未经重排的完整原生文档，用于精确恢复。
- `activation_encodings[模块路径]` 标记 `is_LSTM`，输入为 x/h0/c0，输出为 sequence/h/c；
  双向隐藏状态和 cell state 用 `forward` / `reverse` 保留独立的 scale。
- `param_encodings` 使用 `模块路径.weight_ih.weight`、`weight_hh.weight`、`bias`，
  与归一化 ONNX 的 W/R/B 对应。参数从 IFGO 重排为 IOFC，按方向保存每行编码。
  原生 per-gate 文档已按通道展开，导出时直接重排，per-tensor 标量才需要展开。
- ONNX 的 B 合并 bias_ih/bias_hh，要求两者 dtype、symmetric 一致。阶段保存使用
  `for_onnx=False`，保留原生文档，因此不同 bias 位宽仍可完整恢复。
- 导出 hook 应在常量折叠后运行；输入参数必须是 initializer。共享常量会复制，
  保留其他消费者的输入；重复归一化不改变结果。

原生参数文件保持原 schema，AIMET 编码是外层集成格式，两者不能混用。
标准 ONNX 图表达浮点计算，整数部署需要消费配套编码。

## 验证

从 `pytorch/` 执行 `python3 -m unittest -v tests.test_aimet_interface`，覆盖独立配置、
校准异常、锁定、Po2、双向门顺序、参数精确回读和共享 ONNX initializer。
该测试也已加入 `tools/run_end_to_end_test.sh`。
