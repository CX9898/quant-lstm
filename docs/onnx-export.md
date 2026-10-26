# QuantLSTM ONNX 导出

## 1. 导出边界

ONNX 导出表示单层浮点 LSTM 语义，不携带 q-carrier 量化参数或 QAT Clamp mask。
单向和双向模型都导出为恰好一个标准 ONNX `LSTM` 节点；输入输出仍遵循
`QuantLSTM` 的 `batch_first` 配置，图内通过 Transpose/Squeeze/Reshape 适配
ONNX 的 time-major `Y=[T,D,B,H]`。

当前使用 opset 18 和 PyTorch legacy exporter。临时 custom op 只用于触发 symbolic，
最终模型不包含 `quant_lstm_onnx` domain 节点。

## 2. 使用方式

导出要求当前 Python 环境已经构建 `_quant_lstm` extension，并安装 `torch`、`onnx`
和 `onnxruntime`。以下代码从仓库根目录执行，输出文件为
`quant_lstm.onnx`：

```python
import torch
from torch import nn

from quant_lstm import QuantLSTM, ensure_quant_lstm_onnx_registered

class ExportWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, input, h_0, c_0):
        output, (h_n, c_n) = self.module(input, (h_0, c_0))
        return output, h_n, c_n
module = QuantLSTM(16, 32, bidirectional=True).eval()
wrapper = ExportWrapper(module)
input = torch.randn(8, 4, 16)
h_0 = torch.zeros(2, 4, 32)
c_0 = torch.zeros(2, 4, 32)

ensure_quant_lstm_onnx_registered(opset=18)
try:
    module.export_mode = True
    torch.onnx.export(
        wrapper,
        (input, h_0, c_0),
        "quant_lstm.onnx",
        opset_version=18,
        dynamo=False,
        input_names=["input", "h_0", "c_0"],
        output_names=["output", "h_n", "c_n"],
    )
finally:
    module.export_mode = False
```

`ExportWrapper.forward(input, h_0, c_0)` 只需调用
`module(input, (h_0, c_0))` 并把嵌套返回值展平为三个输出。显式状态使 ONNX
接口和 PyTorch 的 `h_0/c_0` 契约完全一致；省略状态时导出路径会生成零状态。

`export_mode=True` 只允许在 `torch.onnx.export(..., dynamo=False)` 上下文中使用，
普通 eager 调用会失败，避免把导出期零值 runtime stub 当成真实前向。

检查生成文件：

```bash
python - <<'PY'
import onnx

model = onnx.load("quant_lstm.onnx")
onnx.checker.check_model(model)
lstm_nodes = [node for node in model.graph.node if node.op_type == "LSTM"]
assert len(lstm_nodes) == 1
assert not lstm_nodes[0].domain
print("QuantLSTM ONNX export passed")
PY
```

打印 `QuantLSTM ONNX export passed` 且进程退出码为 0，表示模型通过 ONNX checker
并包含一个标准 domain 的 `LSTM` 节点。数值一致性仍需要使用 ONNX Runtime 比较
`output/h_n/c_n`，由第 5 节测试完成。

## 3. 参数映射

PyTorch 参数门顺序为 `(i,f,g,o)`，ONNX LSTM 要求 `(i,o,f,c)`。导出前对每个
方向分别重排：

```text
W_onnx = concat(W_i, W_o, W_f, W_g)
R_onnx = concat(R_i, R_o, R_f, R_g)
B_onnx = concat(bW_i, bW_o, bW_f, bW_g,
                bR_i, bR_o, bR_f, bR_g)
```

最终 shape 为 `W=[D,4H,I]`、`R=[D,4H,H]`，有 bias 时 `B=[D,8H]`。双向方向顺序固定为
forward、reverse。`bias=False` 省略 ONNX 的可选 B 输入，也不生成 bias 参数编码。
`bias=True` 时即使参数值全部为零，也会保留 B 及其编码。

## 4. rx-met 的配套编码

上述独立导出示例生成浮点 ONNX。rx-met 用户使用工具包的 `export_onnx_json()`
生成 ONNX 与 `.encodings` 配对文件：每个 QuantLSTM 仍对应一个标准 `LSTM` 节点，
内部激活量化点放在 `activation_encodings`，W/R/B 对应编码放在 `param_encodings`。
层名称、参数名称和 IOFC 门顺序由算子导出接口与工具包共同对齐。

h0/h 使用各方向的 `output` 网格，c0/c 使用 `cell_state` 网格，均为 `PER_TENSOR`。
关闭量化的模块保留浮点 ONNX 节点，其 activation / parameter encoding 不写入配套
文件；即使模块此前已经校准，也按当前量化开关处理。

算子独立的 `export_quant_params()` 生成原生参数检查点，包含模型和执行信息。
编译器使用 rx-met 导出的公共 `.encodings`；具体字段、参数打包和回读规则见
[循环算子集成接口](aimet_integration.md#编码与-onnx)。

## 5. 验收

`pytorch/tests/test_onnx_export.py` 覆盖单向/双向、bias 开关和两种布局。测试要求：

- ONNX checker 通过，图中恰好一个标准 `LSTM` 且没有 custom-domain 节点。
- W/R 及存在时的 B initializer 或常量逐值符合 `(i,o,f,c)` 重排；无 bias 时 B 输入缺省。
- ONNX Runtime 的 `output/h_n/c_n` 分别满足浮点 MAE、MSE、余弦和 allclose 门禁。
- 每次运行的指标写入忽略目录
  `tests/precision/results/stage9_onnx_report.json`。
