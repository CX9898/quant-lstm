# QuantLSTM ONNX 导出

## 1. 导出边界

ONNX 导出表示单层浮点 LSTM 语义，不携带 q-carrier 量化参数或 QAT Clamp mask。
单向和双向模型都导出为恰好一个标准 ONNX `LSTM` 节点；输入输出仍遵循
`QuantLSTM` 的 `batch_first` 配置，图内通过 Transpose/Squeeze/Reshape 适配
ONNX 的 time-major `Y=[T,D,B,H]`。

当前使用 opset 18 和 PyTorch legacy exporter。临时 custom op 只用于触发 symbolic，
最终模型不包含 `quant_lstm_onnx` domain 节点。

## 2. 使用方式

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
module.export_mode = False
```

`ExportWrapper.forward(input, h_0, c_0)` 只需调用
`module(input, (h_0, c_0))` 并把嵌套返回值展平为三个输出。显式状态使 ONNX
接口和 PyTorch 的 `h_0/c_0` 契约完全一致；省略状态时导出路径会生成零状态。

`export_mode=True` 只允许在 `torch.onnx.export(..., dynamo=False)` 上下文中使用，
普通 eager 调用会失败，避免把导出期零值 runtime stub 当成真实前向。

## 3. 参数映射

PyTorch 参数门顺序为 `(i,f,g,o)`，ONNX LSTM 要求 `(i,o,f,c)`。导出前对每个
方向分别重排：

```text
W_onnx = concat(W_i, W_o, W_f, W_g)
R_onnx = concat(R_i, R_o, R_f, R_g)
B_onnx = concat(bW_i, bW_o, bW_f, bW_g,
                bR_i, bR_o, bR_f, bR_g)
```

最终 shape 为 `W=[D,4H,I]`、`R=[D,4H,H]`、`B=[D,8H]`。双向方向顺序固定为
forward、reverse；`bias=False` 使用全零 B，不改变图结构。

## 4. 验收

`pytorch/tests/test_onnx_export.py` 覆盖单向/双向、bias 开关和两种布局。测试要求：

- ONNX checker 通过，图中恰好一个标准 `LSTM` 且没有 custom-domain 节点。
- W/R/B initializer 或常量逐值符合 `(i,o,f,c)` 重排。
- ONNX Runtime 的 `output/h_n/c_n` 分别满足浮点 MAE、MSE、余弦和 allclose 门禁。
- 每次运行的指标写入忽略目录
  `tests/precision/results/stage9_onnx_report.json`。
