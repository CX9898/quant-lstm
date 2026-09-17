# Quant-LSTM 纯定点量化实现计划

> 状态：阶段 9 已完成；标准 ONNX 导出、CUDA 静态参数缓存与版本化性能门禁已验收
> 参考基线：`/home/chengxing.zou/projects/quant-gru`，commit `9c25d14`
> 目标仓库：`/home/chengxing.zou/projects/quant-lstm`

## 1. 目标与边界

本项目要实现与 PyTorch `torch.nn.LSTM` 单层算子兼容的量化版本。参考 GRU 的实际生产路径，首个可用版本以“FP32 作为量化整数载体”的 CUDA 路径为主：张量中的数值仍是经过 round/clamp 的量化整数，但存储类型为 `float`，GEMM 使用 cuBLAS SGEMM 加速，逐元素 kernel 完成 bias、rescale、门控和状态更新。

必须区分两个概念：

- FP 载体量化主路径：量化边界、zero point、位宽和融合公式真实生效，但 GEMM、rescale 和激活函数使用浮点指令。它与 `quant-gru` 默认 `quant_storage_dtype="float32"` 对齐，是本项目首个生产主路径。
- int32 载体 reference：量化值以 `int32_t` 保存，GEMM/rescale 使用整数算术；首版五个激活点先反量化并调用原始 sigmoid/tanh，再量化回目标网格。整数 LUT 与端到端纯整数部署作为后续待办；CUDA int32 载体不属于首个里程碑。

因此，若“纯定点”严格指运行时完全没有浮点算术，首版 FP 载体路径和首版 int32 载体 reference 都不满足该定义。本文只在未来同时具备 int32 载体、整数 rescale 和整数 LUT 时使用“纯整数路径”；当前两个实现分别称为“FP 载体量化路径”和“int32 载体 reference”。

首个可用版本建议冻结为：

- 单层 LSTM（`num_layers=1`）。
- 先完成单向，随后补双向。
- 支持 `bias=True/False`、`batch_first=True/False`。
- `dropout=0`；单层 LSTM 本身不需要层间 dropout。
- 输入、hidden state、cell state、权重、bias 和真实中间量化点可独立配置位宽。
- `weight_ih`、`weight_hh`、`bias_ih`、`bias_hh` 分别独立配置 `PER_TENSOR`、`PER_GATE`、`PER_CHANNEL`；finalize 后每个启用参数都物化为长度 `4H` 的 channel 参数向量，导入导出和执行期统一使用该完整向量。
- 四组 weight/bias 强制 signed symmetric，所有 channel 的 zero point 固定为 0；其余真实量化点通过 JSON 分别配置 `bitwidth`、`is_unsigned` 和 `is_symmetric`，前向必须真实消费每个字段。
- CUDA 主路径使用 FP32 保存量化整数，并用 cuBLAS SGEMM 完成两组 Linear。
- CPU int32 carrier reference 实现相同的融合计算图，用于 Golden、整数 GEMM/rescale 和融合公式验证；真实激活桥接部分不作为整数硬件语义。
- POT2 和普通 affine scale 均支持。Affine 以校准连续 scale 作为外部 standard scale，内部只把 rescale ratio 编码为 M+shift；POT2 以转换后的幂次 scale 作为 standard scale。
- MinMax 校准先落地，SQNR/Percentile 在核心正确后接入。
- FP 主路径先完成严格数值验证；代表性真实数据暂不在阶段 3 接入，因此模型级精度验证及对应声明延期。CPU int32 reference 的整数原语和已冻结 checkpoints 进行 exact 验证，完整输出同时执行 MAE/MSE/余弦门禁。

QAT backward、ONNX、AIMET 接入和 CUDA int32 载体路径不进入首个量化推理里程碑，按后续需求和阶段门禁决定是否实现。

## 2. quant-gru 实现分析

### 2.1 仓库分层

`quant-gru` 当前由以下几层组成：

| 层 | 主要目录/文件 | 职责 |
|---|---|---|
| 公共量化基础 | `include/quantize_*`、`include/scale_encoding.h` | 位宽、scale/zp、POT2 与 M+shift、量化/反量化、LUT、clamp |
| 浮点参考 | `src/gru_forward_gpu.cu`、`src/gru_backward_gpu.cu` | 浮点前向/反向，也是校准时的真实计算图 |
| CPU 定点参考 | `src/gru_forward_cpu_quant.cc` | 与 GPU 共用门控定点函数，提供可调试的整数基准 |
| FP 载体量化主路径 | `src/gru_forward_gpu_quant_fp.cu` | float 保存量化整数、cuBLAS SGEMM、浮点 rescale、真实激活，是 Python 默认路径 |
| CUDA int32 量化路径 | `src/gru_forward_gpu_quant.cu` | int32 保存量化整数、自定义整数 GEMM、整数 rescale 与 LUT，是可选 bit-exact 路径 |
| 校准 | `src/calibration_gpu.cu`、`src/histogram_calibration_utils.cu` | 收集输入、状态、权重、linear、门输入/输出和乘法中间量 |
| C++ API | `src/gru_interface.cc`、`include/gru_interface.h` | 浮点包装、int32 核心、参数计算、CPU/GPU 调度 |
| PyTorch 绑定 | `pytorch/lib/gru_interface_binding.cc` | pybind/Torch extension 暴露 C++ 数据结构与算子 |
| Python 模块 | `pytorch/quant_gru.py` | `nn.GRU` 兼容接口、配置、校准、双向、QAT、参数导入导出、ONNX/AIMET |
| 验证 | `pytorch/test_quant_gru.py`、`script/*.sh`、`quant-gru-cpu-only/` | 浮点/量化精度、训练、位宽组合和 CPU-only reference |

### 2.2 两套量化执行路径

#### 2.2.1 FP 载体主路径

`quant-gru` 的 Python 默认参数是 `quant_storage_dtype="float32"`，调用 `forward_quant_float_storage -> quantGRUForwardFP -> ForwardPassQuantFP`。这是 int32 载体路径加入之前的原始行为，也是当前默认主路径。

执行流程如下：

1. 浮点 master W/R/bias/x/h0 先量化，量化结果仍是整数值，但用 FP32 tensor 保存。
2. 所有时间步的 `W*x` 一次性调用 cuBLAS SGEMM；每个时间步的 `R*h` 调用一次 SGEMM。
3. SGEMM 不能直接表达 activation zero point，因此非对称量化时预计算 `sum(W)*zp_x` 和 `sum(R)*zp_h`，在 pointwise kernel 中从 GEMM 结果扣除。
4. bias 先 round 到 GEMM 累加尺度，与 GEMM 结果相加后再一次性 rescale 到 linear 输出尺度。
5. rescale 使用预计算的浮点比例 `src_scale/dst_scale`，执行 `round_f(value * ratio)`，不是整数 shift/M+shift。
6. 乘法融合逻辑与优化后的 int32 路径一致：乘积直接对齐到下游量化域，不生成遗留乘法量化张量。
7. sigmoid/tanh 使用 `real_sigmoid_f`/`real_tanh_f`：先按 scale/zp 反量化，调用真实浮点激活，再 round/clamp 回输出量化网格；不使用整数 PWL LUT。
8. pointwise kernel 使用 shared memory 缓存各门 bias，并同时完成 bias/rescale、门控、hidden 更新和 QAT mask。
9. 输出 h/v 在接口返回前原地反量化为 float；量化后的 W/R/bias/x 另行保留给 backward。
10. `ForwardPassQuantFP::Run` 当前会设置 `CUBLAS_TENSOR_OP_MATH`。源码注释写“禁用 TensorCore”，但 RAII 类型实际会启用 tensor-op math；LSTM 中必须修正该命名/行为并显式控制 math mode。

这条路径的性能优势来自成熟的 cuBLAS SGEMM、所有时间步输入 Linear 的批量计算，以及将 bias/rescale 融合进 pointwise kernel。它避免维护自定义通用位宽 GEMM，是 LSTM 应优先实现的生产路径。

#### 2.2.2 int32 载体路径

int32 路径通过 `forward_quant_int_storage -> quantGRUForwardInt -> quantGRUForwardInt32` 执行：

- 张量以 int32 保存，乘积/累加使用 int64。
- CUDA tiled GEMM 在加载激活时直接减 zero point。
- POT2 使用 shift，普通 scale 使用 integer multiplier+shift。
- 激活使用 16 段整数 PWL LUT。
- CPU 与 CUDA 可以围绕整数中间值做 bit-exact 验证。
- `forward_quantized` 提供 int32 输入/输出边界，但当前 GRU 仍可能在调用内量化 float master weight。

它适合硬件 reference、整数 I/O 和 AIMET fixed-eval，但需要额外维护自定义 GEMM、LUT 语义和另一套前向实现。

#### 2.2.3 两条路径的语义差异

| 项目 | FP 载体主路径 | int32 载体路径 |
|---|---|---|
| 量化值存储 | FP32，值经过 round/clamp | int32 |
| GEMM | cuBLAS SGEMM | 自定义 integer GEMM |
| 累加精度 | FP32；整数精确区间受 24 位尾数限制 | int64 |
| rescale | 浮点比例乘法 + `round_f` | shift 或 multiplier+shift |
| 激活 | 真实浮点 sigmoid/tanh 后再量化 | 整数 PWL LUT |
| 默认 Python 路径 | 是 | 否，显式选择 |
| CPU/GPU bit-exact | 不保证 | 设计目标 |
| 主要用途 | 生产推理、QAT、GPU 加速 | 硬件 reference、整数接口、bit-exact 验证 |

FP32 能精确表示的连续整数仅到 `2^24`。即使 W/x 本身是低位宽整数，GEMM 的乘积和累加绝对值超过该范围后仍可能丢失整数低位；启用 TF32/Tensor Core 后，输入尾数还会进一步缩短。POT2 下浮点 rescale 可精确表达 2 的幂比例，但普通 affine scale 的浮点比例与整数 M+shift 执行也不天然 bit-exact。

另外，当前 GRU affine 模式的边界量化使用 `toFixedScale()` 的有效 scale，而 FP pointwise 参数直接取连续 standard scale 比例，二者不完全一致。LSTM 不复制这一混合行为：量化网格统一使用 standard scale，内部只对算术 rescale ratio 做 M+shift/POT2 编码，FP 和整数载体消费同一执行参数。

#### 2.2.4 LSTM 路径取舍

本计划采用以下取舍：

1. CUDA FP 载体路径是首个生产主路径，优先获得 cuBLAS 性能收益并对齐 GRU 默认行为。
2. 保留 CPU int32 carrier reference，用于验证融合公式、整数 GEMM/rescale、舍入和 clamp；首版激活使用原始 sigmoid/tanh，LUT 作为后续待办。
3. 首版不实现 CUDA int32 载体，避免在主路径稳定前维护两套 GPU GEMM 和 pointwise kernel。
4. FP 载体与 CPU int32 reference 共享量化点、融合计算图、校准参数和真实激活数学定义，但 GEMM/rescale 的载体算术不同，跨载体只做严格误差对比，不要求 bit-exact。
5. 只有出现明确的整数 I/O、AIMET fixed-eval、硬件对齐或 FP32 精度无法通过门禁的需求时，才启动 CUDA int32 阶段。

### 2.3 GRU 量化计算图

GRU 内部顺序为 `(update, reset, new)`，PyTorch 顺序为 `(reset, update, new)`，因此 Python 层先重排并转置权重。

定点路径可概括为：

```text
q_x, q_W, q_bw ── integer GEMM + bias/rescale ──> q_weight_ih_linear
q_h, q_R, q_br ── integer GEMM + bias/rescale ──> q_weight_hh_linear

linear_u + linear_hu ──> sigmoid LUT ──> update gate
linear_r + linear_hr ──> sigmoid LUT ──> reset gate
linear_n + reset * linear_hn ──> tanh LUT ──> candidate
update * h_old + (1-update) * candidate ──> q_h_new
```

量化参数来自 MinMax、SQNR 或 Percentile 校准。权重/bias 支持三种粒度，其他激活为 per-tensor。

### 2.4 可直接复用的部分

以下代码思想和公共组件可以从 GRU 迁移并泛化，不需要重新设计：

- `QuantBitWidth`、`QuantParam`、`ChannelQuantParam` 和统一的权威 scale/zp 模型。
- scale/rescale 编码框架可以复用，但 LSTM FP 载体必须模拟由 standard scale ratio 编码出的 M+shift/POT2 参数，不能直接消费未编码的连续比例。
- `FixedPointScale`、`Pot2Rescale` 和整数 round/clamp 用于首版 CPU int32 reference；GRU 的 PWL LUT 只作为阶段 10 待办的参考，不在首版迁移。
- FP 载体的 cuBLAS SGEMM 调度：`W*x` 批量预计算、`R*h` 逐时间步计算、双 stream/event 同步。
- 非对称激活的 `sum(weight)*zero_point` 补偿。
- bias 到 GEMM 累加域、再到 linear 输出域的融合 rescale。
- shared-memory bias 缓存和融合 pointwise kernel 的组织方式。
- 乘法 contribution 直接对齐下游量化域的融合公式。
- 权重/bias 的 per-tensor/per-gate/per-channel 校准与广播模型。
- Python 的“配置 -> 校准 -> finalize -> 量化执行”状态机。
- 双向执行采用反转输入、独立量化参数、翻转输出后拼接的方式。

不能直接复用的是 GRU 的 `3H`/三门布局、GRU 候选门公式、hidden-only 状态模型，以及 float/int 路径中已经出现分叉或注释与行为不一致的代码。

### 2.5 不应原样复制的问题

LSTM 实现应在迁移时修正或明确以下问题：

1. GRU 文档主要描述 int32/LUT 流程，没有准确突出默认 FP 载体主路径。LSTM 文档必须分别描述两种载体，所有性能和精度结论标明适用路径。
2. GRU 外部 JSON 中的 `mul_reset_hidden`、`mul_old_contribution`、`mul_new_contribution` 等位宽和量化参数是未融合实现遗留的接口，不代表当前有效计算图仍存在这些量化边界。早期实现会对每次乘法和加法分别量化、clamp、再 rescale，因此每个中间算子都可以独立配置；为了改善精度，后续实现改为融合量化公式。

   源码和历史可以相互印证：commit `7c4b304` 将候选门乘积直接对齐到 `new_gate_input`，省略中间量化层；commit `4db3a3d` 将两路 hidden contribution 对齐到共同尺度，先在宽整数域相加，再统一 rescale 到 `h`。当前 `computeNewGate`、`computeHiddenState`、`computeNewGateFP` 和 `computeHiddenStateFP` 都采用融合思路。

   LSTM 必须直接采用优化后的融合公式。JSON 只暴露真实发生 requantize/clamp 的张量；纯乘法临时值不配置 bitwidth、scale 或 zero point。调试和精度分析可以观测这些临时值，但不能把观测值误建模为量化算子。
3. FP 载体不等于 bit-exact 整数计算。SGEMM 累加、浮点 rescale、真实激活和 Tensor Core math mode 都可能使其与整数 reference 不同；测试必须按载体分别建立预期。
4. GRU 的 Tensor Core 注释与代码行为不一致。LSTM 必须用语义明确的 RAII 类型或显式配置，并在验证模式关闭 TF32/Tensor Core，在性能模式单独评估。
5. FP32 仅有 24 位有效整数精度。LSTM 需要根据 W/x 位宽和 K 计算 `sum(abs(product))` 上界，标记精确区间，并对 16bit、大 K 和 cell 长序列单独验证。
6. GRU affine 模式存在“连续 standard scale、边界 `toFixedScale()` effective scale、FP raw ratio”三者不一致的风险。LSTM 已确定量化网格和导入导出使用 standard scale，内部只编码算术 rescale ratio，并用 round-trip 和逐量化点测试防止语义漂移。
7. GRU 的 Python 测试没有系统比较 float-storage 与 int-storage。LSTM 必须建立载体内正确性、跨载体误差和模型级精度三类报告。
8. 若后续实现纯整数部署接口，必须直接接收或缓存预量化权重/bias，不能把调用时量化 float master weight 计入“纯整数核心”。

## 3. LSTM 算法与内部布局

### 3.1 浮点语义

与 PyTorch `nn.LSTM` 对齐，单个时间步定义为：

```text
i_t = sigmoid(W_ii x_t + b_ii + W_hi h_(t-1) + b_hi)
f_t = sigmoid(W_if x_t + b_if + W_hf h_(t-1) + b_hf)
g_t = tanh   (W_ig x_t + b_ig + W_hg h_(t-1) + b_hg)
o_t = sigmoid(W_io x_t + b_io + W_ho h_(t-1) + b_ho)
c_t = f_t * c_(t-1) + i_t * g_t
h_t = o_t * tanh(c_t)
```

内部直接采用 PyTorch 的 `(input, forget, cell, output)`，即 `(i, f, g, o)` 顺序，避免 GRU 中额外的 PyTorch/Haste 门重排：

```text
[0, H) = i, [H, 2H) = f, [2H, 3H) = g, [3H, 4H) = o
```

权重布局沿用 GRU kernel 习惯：Python 的 `[4H, I]` / `[4H, H]` 转置为内部 `[I, 4H]` / `[H, 4H]` 连续布局。

注意：ONNX LSTM 的门顺序是 `(i, o, f, c)`，未来 ONNX 导出必须单独重排，不能改变内部顺序。

### 3.2 量化张量与算子命名

首版建议使用以下唯一命名，C++ 字段、JSON、Python `_OPERATOR_MAP` 和文档保持一致：

| 类别 | 名称 | 数学含义 |
|---|---|---|
| 输入/状态 | `input` | `x_t` |
| 输入/状态 | `output` | `h_t`，也作为下一步 `h_(t-1)` |
| 输入/状态 | `cell_state` | `c_t`，也作为下一步 `c_(t-1)` |
| 参数 | `weight_ih`, `weight_hh` | 四门输入/循环权重 |
| 参数 | `bias_ih`, `bias_hh` | 四门输入/循环 bias |
| Linear | `weight_ih_linear` | `W_ih*x + b_ih` |
| Linear | `weight_hh_linear` | `W_hh*h + b_hh` |
| 门 | `input_gate_input/output` | `i` 的 sigmoid 输入/输出 |
| 门 | `forget_gate_input/output` | `f` 的 sigmoid 输入/输出 |
| 门 | `cell_gate_input/output` | `g` 的 tanh 输入/输出 |
| 门 | `output_gate_input/output` | `o` 的 sigmoid 输入/输出 |
| cell 激活 | `cell_tanh_output` | `tanh(c_t)` |

`f_t*c_(t-1)` 和 `i_t*g_t` 是融合公式内部的宽整数乘积，不是独立量化张量，因此不进入 JSON 算子表，也不拥有独立 bitwidth、scale、zero point 或 clamp。`output` 就是 `o_t*tanh(c_t)` 的最终量化结果，同样不增加重复的 `mul_output_cell` 量化点。

基础配置采用与 GRU 一致的域感知默认策略：

| 量化点 | `is_unsigned` | `is_symmetric` |
|---|---:|---:|
| `input`、`output`、`cell_state` | `false` | `true` |
| 两路 Linear 输出、四个 gate input | `false` | `true` |
| `input_gate_output`、`forget_gate_output`、`output_gate_output` | `true` | `true` |
| `cell_gate_output`、`cell_tanh_output` | `false` | `true` |

这只是基础 profile，不是硬编码。除 weight/bias 外，每个真实量化点都像 GRU 一样在 JSON 中独立暴露 `bitwidth`、`is_unsigned` 和 `is_symmetric`；显式 JSON 配置覆盖默认值，并必须改变校准 range、scale/zp、Clamp 和前向结果。任何被 schema 接受却未进入执行参数或未被 forward 消费的字段都属于错误，配置完整性测试必须逐字段变更并观察对应执行参数。`cell_state` 始终独立于 `output` 校准，因为它不受 `[-1,1]` 限制并可能随序列长度增长。

### 3.3 定点公式

记量化值为 `q_a`，scale 为 `S_a`，zero point 为 `Z_a`：

```text
a = (q_a - Z_a) * S_a
R(v, S_src -> S_dst) = round(v * S_src / S_dst)
```

`R` 是载体无关的标准尺度转换定义。实际执行先从 standard scale 计算 ratio，再编码为 M+shift 或 POT2 shift；CUDA FP 主路径用浮点数模拟编码后的比例，CPU int32 reference 执行 integer multiplier+shift/shift。两者使用同一组 standard scale 和执行编码，但不假设跨载体数值天然 bit-exact。

#### Linear

权重和 bias 都强制 signed symmetric，`Z_W[c]=Z_bw[c]=0`。每个输出 channel 独立执行：

```text
acc_x[c] = sum_k(q_W[k,c] * (q_x[k] - Z_x))
acc_b[c] = R(q_bw[c], S_bw[c] -> S_W[c]*S_x)
q_ih[c] = clamp(R(acc_x[c] + acc_b[c], S_W[c]*S_x -> S_ih) + Z_ih)
```

循环 Linear 同理，将 `x/W/bw/ih` 替换为 `h/R/br/hh`。

#### 四门

对每个 `k in {i,f,g,o}`：

```text
q_gate_in[k] = clamp(
    R(q_ih[k] - Z_ih, S_ih -> S_gate_in[k]) +
    R(q_hh[k] - Z_hh, S_hh -> S_gate_in[k]) +
    Z_gate_in[k]
)
```

FP 载体主路径中，`i/f/o` 使用原始 sigmoid 后重新量化，`g` 使用原始 tanh 后重新量化；CPU int32 reference 首版复用同一个真实激活边界，区别是输入 q 由 int32 载体反量化且输出重新量化为 int32。门输入和门输出均在各自配置位宽处 clamp。整数 PWL LUT 延后实现。

#### Cell state 更新

cell 更新采用已经确认的双比例延迟舍入公式。两路乘积先乘各自到目标 `cell_state` 网格的比例，在目标整数域相加后只执行一次最终 round/clamp；不经过独立中间量化、zero point 或 clamp：

```text
p_forget = (q_f-Z_f) * (q_c_old-Z_c)
p_input  = (q_i-Z_i) * (q_g-Z_g)

alpha = S_f*S_c_old/S_c_new
beta  = S_i*S_g/S_c_new

cell_centered = round(p_forget*alpha + p_input*beta)

q_c_new = clamp(cell_centered + Z_c_new, bitwidth_cell_state)
```

首版新旧 Cell 共用网格，因此 `alpha=S_f`。所有 scale 都是 standard scale；FP 主路径和 CPU int32 reference 都暂定将 `alpha/beta` 编码为固定 Q31。FP 使用浮点数模拟 Q31 比例，CPU 使用 `int64_t` multiplier 和 `__int128` 形成 `p_forget*M_forget+p_input*M_input`，最后统一 `RoundShift(...,31)`。该方案允许 `i/f/g` 保持独立量化网格，同时保留两路贡献在最终舍入前的正负抵消。Q31 方案通过极值和 golden 验证后再最终冻结。

#### Hidden state 更新

```text
q_tanh_c = activation_tanh(q_c_new; S_c,Z_c -> S_cell_tanh,Z_cell_tanh)

q_h_new = clamp(
    R((q_o-Z_o)*(q_tanh_c-Z_cell_tanh),
      S_o*S_cell_tanh -> S_h) + Z_h
)
```

FP 主路径与 CPU int32 reference 首版的 `activation_tanh` 都是反量化后调用原始 tanh，再重新量化；两者必须复用同一个真实激活边界实现。最终 hidden 乘法直接对齐到 `output` 网格，不建立乘法中间量化点。整数 PWL LUT 作为纯整数阶段待办。

`h_0` 与所有 `h_t` 共用 `output` 网格，`c_0` 与所有 `c_t` 共用 `cell_state` 网格。校准范围必须包含代表性初始状态和递推状态的并集。

### 3.4 中间缓冲布局

建议首版布局：

```text
x:                  [T, B, I]
h, c:               [T+1, B, H]
W:                  [I, 4H]
R:                  [H, 4H]
bw, br:             [4H]
weight_ih_linear:   [T, B, 4H]
weight_hh_linear:   [B, 4H]       # 每个时间步复用
v (训练阶段再引入): [T, B, ...]    # 由 backward 的最小需求决定，不提前照搬 GRU
```

推理 kernel 每个线程处理一个 `(batch, hidden)` 元素，一次完成四门、cell 更新和 hidden 更新，并同时写出 `c_t`、`h_t`。

## 4. 总体架构与代码规范

### 4.1 模块拆分

LSTM 仓库保持与 GRU 相近的能力，但按“通用量化原语、LSTM 数据模型、后端实现、接口集成”拆分，避免把参数定义、定点公式、CUDA 调度和 PyTorch 绑定堆在同一文件：

```text
quant-lstm/
├── CMakeLists.txt
├── include/
│   ├── quantization/
│   │   ├── bit_width.h                # 位宽、整数范围
│   │   ├── quant_param.h              # scale/zp 与 channel 参数
│   │   ├── rounding.h                 # 统一银行家舍入与受检定点移位
│   │   ├── numeric_safety.h           # 静态上界、风险分类和诊断
│   │   ├── scale_encoding.h           # POT2、M+shift 编码
│   │   ├── fixed_point_ops.h          # 整数 rescale、clamp
│   │   ├── float_carrier_ops.h        # FP 模拟 M+shift/POT2、clamp
│   │   └── real_activation.h          # 两种载体共用的真实 sigmoid/tanh 边界
│   └── lstm/
│       ├── gate_layout.h              # 四门枚举、索引和布局
│       ├── quant_config.h             # 外部可配置量化点
│       ├── quant_config_loader.h      # override 校验与 resolved config 生成
│       ├── quant_params.h             # 独立粒度元数据与 4H 权威参数向量
│       ├── quantized_cell_fp.h        # FP 载体融合逐元素公式
│       ├── quantized_cell_int.h       # CPU int32 reference 公式
│       ├── forward_cpu.h              # CPU 定点 reference API
│       ├── forward_fp_reference_cpu.h # CPU 标量 FP 载体 reference API
│       ├── forward_fp_cuda.h          # CUDA FP 载体主路径
│       ├── calibration.h              # 校准数据模型与接口
│       └── interface.h                # 对外 C++ API
├── src/
│   ├── quantization/
│   │   ├── numeric_safety.cc
│   │   ├── scale_encoding.cc
│   │   ├── real_activation.cc
│   │   └── quantize_kernels.cu
│   └── lstm/
│       ├── forward_float_cuda.cu      # 浮点参考与校准计算图
│       ├── forward_int32_cpu.cc       # CPU int32 载体 reference
│       ├── forward_quantized_fp_cpu.cc # CPU 标量 FP 载体 reference
│       ├── forward_quantized_fp_cuda.cu # cuBLAS + FP 载体主路径
│       ├── quant_config_loader.cc     # 唯一默认合并与 canonical 序列化实现
│       ├── quant_params.cc            # range 到参数及 rescale 派生
│       ├── calibration_cuda.cu
│       └── interface.cc
├── config/
│   ├── defaults/
│   │   └── lstm_quant_default_v1.json # 入库的完整基础 profile
│   └── schema/
│       ├── lstm_quant_override.schema.json # 用户稀疏配置
│       └── lstm_quant_resolved.schema.json # 完整执行配置
├── pytorch/
│   ├── quant_lstm.py
│   ├── lib/lstm_interface_binding.cc
│   └── tests/                         # Python API/集成测试
├── tests/
│   ├── common/
│   │   ├── deterministic_rng.h        # 唯一 PCG32 与数值映射接口
│   │   ├── deterministic_rng.cc       # 版本化确定性实现
│   │   └── strict_matrix_coverage.cc  # 只检查显式矩阵，不生成 case
│   ├── cpp/                           # C++ 原语与 CPU 测试
│   │   └── golden_reference_generator.cc # 有意更新 Golden 的 CPU 工具
│   ├── cuda/                          # FP 载体 CUDA 正确性/性能测试
│   ├── precision/
│   │   ├── schema/
│   │   │   └── strict_matrix.schema.json # 显式严格矩阵 schema
│   │   ├── config/
│   │   │   ├── metric_policy.json     # 入库的指标算法与零范数规则
│   │   │   ├── synthetic_data.json    # 入库的 seed、shape 和生成配置
│   │   │   ├── strict_matrix_v1.json  # 入库的稳定 pairwise+定向 case
│   │   │   └── strict_thresholds.json # 入库的严格绝对阈值
│   │   └── results/                   # 每次运行生成且不入库的报告
│   └── golden/
│       ├── schema/
│       │   └── golden_case.schema.json # 入库的唯一版本化 schema
│       ├── spec/                      # 入库的一用例一文件权威 JSON
│       │   ├── primitive/             # 公共量化原语用例
│       │   ├── cell/                  # 单时间步融合 Cell 用例
│       │   └── recurrent/             # T=3 短递推用例
│       └── generated/                 # 测试时生成且不入库的 C++ fixture
├── tools/
│   └── generate_golden.py             # JSON 到 C++ fixture 的机械生成器
├── example/
└── docs/
```

初始迁移可以复制 GRU 的公共量化文件，但进入 LSTM 仓库后要移除 GRU 专属命名和死代码。暂不抽成跨仓库共享库，避免本项目被另一个工作区源码隐式依赖；当两个项目的公共原语稳定后再评估独立公共库。

### 4.2 依赖边界

- `quantization/` 不依赖 LSTM、PyTorch 或校准模块，分别提供载体无关参数、FP 载体原语和整数 reference 原语。
- `quantization/rounding.h` 是所有舍入的唯一入口；scale 编码、quant/dequant、FP/int rescale、激活和 LUT 禁止复制舍入实现或直接调用平台 round intrinsic。
- `quantization/numeric_safety.h` 集中计算 GEMM、乘积、multiplier、移位和累加上界；后端只消费风险分类和诊断，不重复实现边界公式。
- `lstm/quantized_cell_fp.h` 和 `lstm/quantized_cell_int.h` 共享冻结的量化点与融合顺序，但分别实现 FP 载体和整数载体原语。
- 四组 weight/bias 各自保存独立 granularity 元数据；finalize 统一展开为 `4H` channel 参数向量，所有 CPU/CUDA kernel 只消费展开后的向量，不在执行期实现 per-tensor/per-gate 广播分支。
- FP 与 int32 后端共享冻结的量化点和融合计算图，但各自使用载体专属的逐元素实现；GEMM 和调度也分别实现。
- `interface.cc` 负责参数校验、workspace 和后端调度，不放置门控数学公式。
- PyTorch binding 只做张量校验和 C++ 类型映射，不复制量化公式。
- 校准模块只生成权威 standard `scale/zp`；执行用 M+shift/POT2 rescale 参数在 finalize/set-up 阶段集中派生，不能覆盖 standard scale。
- 禁止循环依赖；公共 C++ 核心禁止包含 Torch 头文件。
- 通用位宽、round、clamp、quant/dequant、scale 编码、指标计算和布局校验分别保持单一实现；后端只能组合这些接口，不能复制等价代码。

### 4.3 命名规范

| 对象 | 规范 | 示例 |
|---|---|---|
| C++ 文件/目录 | `snake_case` | `quantized_cell_fp.h` |
| C++ 类型/枚举 | `PascalCase` | `LstmQuantParams`, `GateKind` |
| C++ 函数 | `lowerCamelCase` | `computeCellState` |
| C++ 局部变量/参数 | `snake_case` | `cell_state_scale` |
| C++ 私有成员 | `snake_case_` | `hidden_size_` |
| C++ 常量 | `kPascalCase` | `kGateCount` |
| 宏 | `UPPER_SNAKE_CASE` | `QUANT_LSTM_DEBUG` |
| Python 函数/变量/文件 | PEP 8 `snake_case` | `finalize_calibration` |
| Python 类型 | `PascalCase` | `QuantLSTM` |
| JSON 字段 | `snake_case` | `cell_gate_output` |

命名不直接复制 GRU 中已经混用的缩写风格。`input/forget/cell/output` 在公开接口中写全称，`i/f/g/o` 只用于公式或局部紧凑索引。`cell_state` 始终表示 recurrent state，`cell_gate` 始终表示候选门，避免都缩写为 `c` 造成歧义。

### 4.4 文档与注释

- 项目文档和代码注释统一使用中文；Git commit 的 subject 和 body 统一使用英文。
- 每个头文件和源文件必须有模块级注释，说明职责、依赖边界、输入输出布局以及是否允许浮点运算。
- 每个公开 API 使用中文 Doxygen 注释，写清 shape、dtype、量化网格、所有权、生命周期和错误条件。
- 融合定点公式必须在代码旁说明 scale 推导、zero-point 处理、舍入/clamp 位置及溢出前提。
- 除公共适配模块外，代码禁止直接调用 `rintf/roundf/std::round/std::llround` 或自行实现舍入；代码检查需要扫描这些违规调用。
- CUDA kernel 注释必须说明每个线程处理的逻辑元素、内存布局、stream 同步和 workspace 要求。
- 注释重点解释约束和原因，不为自解释的赋值逐行写重复说明。
- 公式、配置字段和实现命名发生变化时，同一阶段同步更新文档，禁止代码与文档长期漂移。

## 5. 分阶段实施计划

每个阶段单独提交、单独验收。前一阶段未通过，不进入后一阶段。凡阶段交付涉及数值计算或前向/反向输出比较，都必须生成阶段精度报告；报告至少包含 MAE、MSE 和余弦相似度，不能只打印 `allclose`、最大误差或最终 PASS/FAIL。测试固定采用“基础测试 -> 严格测试”的顺序：先参考 GRU 的测试方式验证典型配置和最终输出，基础测试全部通过后，才运行独立数据集、shape 矩阵、逐时间步和逐量化点的严格测试。对于完整 LSTM 前向，基础层就要分别统计 `output`、`h_n`、`c_n`；逐时间步最差值和样本索引在严格层报告。所有非退化张量的余弦相似度硬门禁统一为 `>= 0.999`；浮点语义对齐使用更严格的门禁，具体定义见第 6 节。

### 阶段 0：冻结双载体规格与精度基线

交付：

- 将本计划审核意见固化为 `docs/quantized-execution-spec.md`。
- 将已确认的载体无关数学基线维护在 `docs/lstm-quantization-formula-derivation.md`；具体执行变换只能在不改变该基线的前提下冻结。
- 冻结门顺序、张量布局、有效算子命名、融合公式、每个 round/clamp 位置和 zero-point 规则。
- 分别定义 FP 载体主路径和 CPU int32 reference 的运算语义，不使用“纯定点”笼统指代两者。
- 冻结 scale 语义：Affine 的校准连续 scale 是 standard scale；POT2 转换后的 scale 是 standard scale；边界 quant/dequant、激活和导入导出使用 standard scale，内部 rescale ratio 统一编码后执行。Golden JSON 的 standard scale 使用可唯一往返到 `float32` 的规范十进制字符串，不使用 JSON number 或冗余位模式字段。
- 冻结 zero-point 约束：四组 weight/bias 必须 signed symmetric 且完整 `4H` zp 向量全为 0；配置或导入中的非对称、unsigned 或非零 zp 直接报错。其他真实量化点按各自 schema 允许对称/非对称。
- 冻结 signed symmetric 范围：`qmax=2^(bitwidth-1)-1`、`qmin=-qmax`，非退化范围使用 `scale=max(abs(r_min),abs(r_max))/qmax` 且 `zero_point=0`。二进制补码最小负值不参与量化、Clamp、导入或 Golden；该规则适用于 weight/bias 以及其他配置为 signed symmetric 的真实量化点。
- 冻结 signed asymmetric 范围：使用完整二进制补码 `[-2^(bitwidth-1),2^(bitwidth-1)-1]` 和包含实数零的标准 MinMax affine scale/zp；最小负值在 asymmetric 模式合法，但在 signed symmetric 模式非法。
- 冻结量化点基础 profile 与覆盖规则：三个 sigmoid gate output 默认 unsigned symmetric，其余非参数量化点默认 signed symmetric；除强制 signed symmetric 的 weight/bias 外，每个真实量化点都由 JSON 独立配置 `bitwidth/is_unsigned/is_symmetric`，显式值优先且必须被 forward 实际消费。
- 冻结 unsigned 语义：范围为 `[0,2^bitwidth-1]`；unsigned symmetric 是 `zero_point=0` 的 zero-anchored 量化，使用 `scale=max(r_max,0)/qmax`，负值在真实量化边界饱和到 0；unsigned asymmetric 使用包含实数零的标准 MinMax affine scale/zp。两者都调用统一 round-to-nearest-even，POT2 在校准后转换 standard scale。
- 冻结通用 M+shift：Affine rescale 使用规范化 `uint16_t M` 和 `int8_t shift`，`M` 位于 `[32768,65535]`，编码调用统一银行家舍入；`M=65536` 时规范化为 `32768` 并增加 exponent。无效 ratio、shift 越界或载体不安全直接失败，不允许下溢为 0、饱和、raw-ratio 回退或自动切换后端；Cell 双比例仍独立使用固定 Q31。
- 冻结 POT2 转换：生产路径固定使用 `CoverRange` 和 `tolerance=0.02`，近幂次范围时用统一银行家舍入选择指数，否则向覆盖范围方向取 floor；转换后的 scale 是 standard scale，非对称 zero point 据此重算。JSON 只选择 affine/POT2，不暴露 method 或 tolerance。
- 冻结退化范围 fallback：`S_min=min(FLT_EPSILON,0.01/(qmax-qmin))`。候选 scale 小于 `S_min` 时，signed symmetric 围绕零扩展、unsigned symmetric 从零向正方向扩展、asymmetric 保留下界并向上扩展；Affine 直接使用结果，POT2 随后执行固定 CoverRange。该规则不可通过 JSON 配置。
- 冻结配置解析契约：用户 JSON 是版本化稀疏 override，唯一 C++ resolver 按字段覆盖入库的完整默认 profile，并产出严格 canonical resolved config；所有 forward 只消费 resolved config。override/resolved 使用独立 schema，未知字段、重复 key、`null` 和非法固定值均 fail fast。
- 冻结首版 bitwidth 集合：所有真实量化点只允许 8 或 16 bit，并允许混合；其他值在 resolver 阶段失败。载体、累加器、M+shift 和 Q31 的内部宽度不属于 operator bitwidth 配置。
- 冻结严格矩阵契约：基础 case 固定且独立；严格 case 以 `strict_matrix_v1.json` 显式入库，采用约束 pairwise 加强制定向 case。运行时只校验和执行，不动态生成或自动更新矩阵。
- 冻结首版性能验收：阶段 4 不设置绝对延迟或固定加速比，只设置 cuBLAS 调用和报告完整性的结构门禁；阶段 9 在稳定基线形成后，再按 GPU/profile 人工审核并版本化性能回归阈值。
- 明确 cuBLAS math mode；正确性验证关闭 TF32/Tensor Core，性能模式必须显式开启并单独出精度报告。
- 使用统一数值安全预检计算 GEMM/Cell 上界：整数真实溢出风险直接失败；FP32 超过 `2^24` 标记 `precision_risk`，仅在指标过门禁时允许；exact 模式直接失败。
- 舍入统一采用 round-to-nearest-even，并通过 `roundToNearestEven(...)` 公共接口族实现；定义安全的负数右移和受检左移。固定 Q31 Cell 融合采用“静态边界证明 -> 缩小整数域穷举 -> 固定 seed 全范围随机/对抗测试”三层验证，覆盖普通 int64 与 Cell `__int128` 累加、8/16 bit、Affine/POT2、极端比例、抵消、同号累加和长序列。
- 将 CPU int32 reference 和 CPU 标量 FP 载体 reference 定义为两套量化 reference model，并由它们分别产生对应载体的 Golden 期望值；不重复实现 Python 量化 oracle。分层小张量覆盖公共量化原语、单时间步融合 Cell 和 `T=3` 短递推，使用定向/成对用例覆盖关键边界，不做全维度笛卡尔积。`tests/golden/spec/{primitive,cell,recurrent}/*.json` 是入库并经审核的权威测试契约，`tools/generate_golden.py` 在每次构建或测试时机械生成 C++ fixture，派生文件不入库。
- 对旧式逐算子量化和候选融合公式做固定数据集精度比较，冻结最终融合形式。
- 冻结基础/严格测试的数据隔离、seed、确定性 RNG、合成分布、MAE/MSE/余弦相似度计算公式、聚合方式和阈值制定流程。测试数据统一由版本化 `pcg32-xsh-rr-v1` C++ 公共模块在 CPU 生成，普通随机数据使用 `pytorch_typical_v1`，禁止使用标准库 distribution 或在 Python/CUDA 中复制生成器。基础输入 seed 固定为 `0`；严格校准 seed 固定为 `{1001,1002,1003}`，严格验证 seed 固定为 `{2001,2002,2003,2004,2005}`，对抗用例不依赖随机 seed。七个 shape profile 的 weight/bias 参数 seed 分别固定为 `3001..3007`，且不随量化配置、载体或 split 改变。严格测试的完整目标包含合成边界数据和代表性真实数据，两层必须分别使用互斥的校准/验证 split 并独立验收；但阶段 3 暂不接入真实数据，只验证 `synthetic_numeric` 范围。余弦的 FP64 L2 norm 零范数 epsilon 固定为 `1e-12` 并写入统一指标策略。基础测试与第一轮严格测试均先采用 GRU 的非量化/INT8/INT16 MAE、MSE 和余弦门禁，并写入版本化阈值配置；完整流程跑通后再依据 LSTM 实测结果人工审核调整，禁止自动更新。没有代表性应用数据时只能声明数值严格验证通过，不能声明模型级精度达标。
- 复跑 GRU FP 载体主路径并据此版本化测试配置、seed、指标定义和审核通过的阈值；机器可读报告属于每次运行重新生成的证据，不入库。第 6.3 节记录的本机结果只用于设定初始基线。
- 冻结启动可选 CUDA int32 阶段的条件，不在首版默认实现。

验收：FP 和 int32 规格各自完整；跨载体的预期差异有明确解释；JSON schema 不包含纯乘法临时值；不存在未消费的配置项；每个 Golden 只对应一个合法 execution model，载体相关语义没有混存；`expected.checkpoints` 覆盖全部真实量化边界，`expected.diagnostics` 只包含冻结公式定义的关键宽值；Golden tensor 的 dtype/shape/data、row-major 布局和元素数量均通过严格校验；权威 golden JSON 可以稳定生成 C++ fixture，重新生成后结果一致且 Git 工作区无派生文件变化；两套 CPU reference 的反量化 `output/h_n/c_n` 均与同输入、权重和初始状态的 `torch.nn.LSTM` 完成端到端指标对比。

### 阶段 1：仓库骨架与浮点 LSTM 基准

交付：

- CMake、模块化头文件、基础测试框架和最小 C++ example。
- 测试公共模块实现 `pcg32-xsh-rr-v1`、版本化张量 stream registry、确定性 float32 映射和已知向量单测；CPU/CUDA/Python 测试只消费同一批 CPU 生成数据。
- 单层单向浮点 LSTM forward，内部门顺序 `(i,f,g,o)`。
- Python 最小绑定，验证权重布局、`h_0/c_0` 和输出形状。
- `h_0/c_0` API 分别覆盖省略和显式全零；在相同输入/参数下，两种路径的 `output/h_n/c_n` 必须一致。
- `batch_first=False/True` 分别使用 `time_major/batch_major` 布局 profile，基础和严格测试都运行；两者共享同一逻辑输入并验证布局归一化后的结果一致。

基础验收：关闭 TF32 并固定 seed，CPU 使用 `cpu_basic=(T=8,B=4,I=16,H=32)`，CUDA 使用 `cuda_gru_basic=(T=50,B=64,I=128,H=256)`，分别与 `torch.nn.LSTM` 比较 `output/h_n/c_n`；默认 hard gate 为 `atol<=1e-5, rtol<=1e-5`、MSE `<1e-5`、MAE `<0.003`、余弦相似度 `>=0.9999`。

严格验收：基础验收通过后，再覆盖多组 `T/B/I/H`、有/无 bias、有/无初始状态、`T>1` 和长序列递推；任何超限必须解释并经审核调整。

审核点：只验证 LSTM 语义和布局，不引入量化。

### 阶段 2：量化原语、参数模型与配置完整性

交付：

- 从 GRU 迁移并整理位宽、scale encoding、quant/dequant 和 clamp 原语；舍入单独收敛到 `rounding.h`，不得保留多个等价 helper。
- 提供 standard `QuantParam`、`FixedPointScale`、`Pot2Rescale` 及载体专属 apply 函数；FP 和整数 apply 共享同一 M+shift/POT2 编码参数，不提供消费 raw ratio 的生产 `FloatRescale`。
- `FixedPointScale` 固定为 `uint16_t multiplier + int8_t shift`；唯一 `encodeMShift` 使用 FP64 `frexp` 和公共 round-to-nearest-even 生成规范化 multiplier。CPU int32 和 FP carrier apply 固定执行顺序，所有乘法、负 shift 左移和类型转换均受检。
- POT2 只提供 `convertScaleToPot2CoverRange` 生产入口，固定 2% 相对容差；schema 拒绝 `pot_scale_method`、`pot_scale_tolerance` 及别名，避免遗留不可执行配置。指数和非对称 zp 均调用公共舍入/Clamp，并进行有限值和 `int8_t` 范围检查。
- 提供 `NumericSafetyReport` 和 `require_exact_accumulation` 执行选项，覆盖 int64、`__int128`、FP32 精确区间及 Inf/NaN 风险。
- `LstmOperatorQuantConfig`、`LstmQuantizationRanges`、`LstmQuantParams`。
- 四组参数独立 granularity 配置及统一 `4H` channel 容器：per-tensor 复制 1 个参数到全部 channel，per-gate 按 `(i,f,g,o)` 将 4 个参数各复制 `H` 次，per-channel 保留 `4H` 个独立参数；广播只允许发生在 finalize。
- 提供唯一 `real_activation` 模块：FP 与 int32 载体 reference 都通过它完成反量化、原始 sigmoid/tanh 和输出重新量化；首版不创建 LUT 类型、参数或配置字段。
- 提供 `lstm_quant_override.schema.json`、`lstm_quant_resolved.schema.json`、`lstm_quant_default_v1.json` 和唯一 `quant_config_loader`。Override 中 operator/字段可省略；resolved 中 `scale_mode`、全部真实量化点及适用字段必须齐全。Golden 和所有后端只消费 resolved 形式。
- 配置 schema 只枚举实际执行的量化点；对 GRU 风格遗留 `mul_*`、POT2 method/tolerance 和非参数 granularity override 直接报 unknown-field 错误。
- Weight/bias resolved config 强制 `is_symmetric=true`、`is_unsigned=false`，granularity 必须为 `per_tensor|per_gate|per_channel`；override 中错误固定值在 finalize 前 fail fast，不能静默改回。校准参数另行强制全零 `4H` zero-point 向量。
- 其余真实量化点的 resolved config 显式要求 `bitwidth`、`is_unsigned`、`is_symmetric` 和固定 `granularity=per_tensor`；override 允许逐字段稀疏覆盖。完整性测试分别翻转每个可配置字段，验证 resolved config 及 range、scale/zp、Clamp 或执行结果随之变化，禁止保留不生效的兼容字段。
- 两套 schema 将 operator `bitwidth` 限定为枚举 `[8,16]`；测试覆盖纯 INT8、纯 INT16、8/16 混合，以及 `7/9/15/17/32`、浮点数、字符串和越界整数的 fail-fast。内部执行宽度不得误走该字段校验。
- Parser 测试覆盖空 override、单字段/单 operator 覆盖、全部覆盖、默认文件版本不匹配、unknown/duplicate/null、错误类型/枚举、非法固定值，以及 canonical resolved 二次解析/序列化字节一致；Python 只调用 C++ resolver，不复制合并逻辑。
- Schema 和参数导入对所有 signed symmetric 量化点强制严格对称范围；拒绝二进制补码最小负值及任何范围外 q 值，不能因载体类型可表示而接受。
- Signed asymmetric schema、参数导入和 Golden 使用完整二进制补码范围；模式感知校验必须证明 INT8/INT16 最小负值在 asymmetric 下被接受、在 symmetric 下被拒绝，禁止仅按 C++ 载体类型统一放行或拒绝。
- Schema、参数导入和 Golden 对 unsigned q 值强制 `[0,2^bitwidth-1]`；校准测试分别覆盖 zero-anchored symmetric 和 MinMax asymmetric，验证 asymmetric 合法 `Z=0`、非零 zp、zp 两端 Clamp 及负值在 unsigned symmetric 下的饱和计数。
- 单元测试覆盖正负 half/tie、不同奇偶商、zero point、scale 转换、strict signed symmetric 与完整 unsigned 的 `qmin/qmax`、被保留的二进制补码最小负值、`INT_MIN/MAX`、非法 shift、受检左移、饱和和非法参数，并验证 CPU/CUDA 舍入结果一致。M+shift 额外覆盖 multiplier 舍入上下邻域、half-tie 奇偶、`65535/65536` 规范化进位、shift 两端、刚越界比例、NaN/Inf/零/负数，以及 FP/int 对同一编码参数的已知向量。POT2 额外覆盖 exact power、容差内、恰好 `0.02`、容差外、指数 half-tie 奇偶、int8 两端/越界、非对称 zp 重算和 unknown JSON 字段。

验收：FP 模拟/POT2/M+shift golden 全部通过；四组 granularity 配置字段与执行参数有完整性测试；无论原始粒度为何，每个启用参数的执行向量长度都严格为 `4H`。Per-tensor 展开值逐 channel 位级相同，per-gate 严格按 `(i,f,g,o)` 四段映射，per-channel 逐 channel 保留各自统计结果；不要求不同 group/channel 的数值必须互异。Standard scale 导入导出后的 `float32` 位模式不变，非规范十进制字符串和越界整数被拒绝；内部编码参数不能污染外部 scale。

### 阶段 3：CPU int32 reference 与标量 FP 载体 reference

交付：

- `lstmForwardInt32CpuReference`：预量化 int32 参数/输入/状态进入，int32 h/c 输出。
- CPU integer GEMM、整数 rescale、融合状态更新，以及通过公共 `real_activation` 模块执行的三个 sigmoid 和两个 tanh。
- `lstmForwardQuantizedFpCpuReference`：CPU 标量实现，严格复现 float 保存 q 值、编码后执行 ratio、真实激活和统一 round/clamp 顺序。
- `golden_reference_generator` 只调用上述 CPU reference 填充或更新 canonical JSON 的期望字段，不包含第三套量化公式。
- 固定 Q31 Cell 融合验证工具：静态计算 `p/M/product/sum/shift` 上界；在缩小整数域穷举输入与舍入边界；在完整 8/16-bit 范围运行固定 seed 随机和定向对抗用例。测试侧只实现局部高精度数学公式，不实现第三套完整 LSTM reference。
- 调试输出按 Golden 的 checkpoints/diagnostics 划分：记录全部真实量化边界，以及 Cell 两路原始乘积、按执行比例缩放后的宽 contribution、最终舍入前 Cell 宽和与 Hidden 原始乘积；这些诊断值没有量化配置。
- 统一测试向量驱动两种 reference，并生成逐时间步对比报告。
- `bias_profile=enabled_random|disabled` 均进入 CPU reference 基础和严格测试；disabled 路径不得创建、量化或读取 bias 参数。

验收：

- CPU int32 reference 的整数原语、融合 diagnostics 和同 execution model 的已冻结 Golden checkpoints 逐值一致；该结论不等价于端到端纯整数或硬件 LUT bit-exact。
- CPU 标量 FP reference 与已冻结的 FP golden 一致。
- 两套 reference 的反量化 `output/h_n/c_n` 分别与相同输入、权重、bias、`h_0/c_0` 的 `torch.nn.LSTM` 比较，完整报告 MAE、MSE 和余弦相似度，非退化张量余弦相似度必须 `>=0.999`。
- Q31 静态分析证明所有获准配置的 `p/M/product/sum/shift` 均在声明载体范围内，不安全配置在 setup 阶段 fail fast；缩小域穷举相对已冻结 Q31 整数公式 mismatch 为 0。
- 全范围随机/对抗测试比较 Q31 可执行公式与未编码的双比例数学公式，报告逐 Cell 和长序列的 max abs、MAE、MSE、余弦相似度、饱和率、contribution 编码为 0 的次数及相对 Cell LSB 误差；非退化结果余弦相似度必须 `>=0.999`。
- 在固定且互斥的合成校准集、验证集和 shape 矩阵上，使用 GRU 初始阈值分别验收两套 CPU reference 相对 `torch.nn.LSTM` 的 `output/h_n/c_n`。完整流程跑通后汇总最差值、逐时间步结果和失败样本，提出 LSTM 阈值调整建议，经人工审核后再修改 `tests/precision/config/strict_thresholds.json`。阶段 3 不接入代表性真实数据，也不生成模型级阈值。
- 首轮 8/16-bit 混合位宽按所有真实量化点中的最小有效位宽选择阈值档：存在 8-bit 量化点时使用 INT8 profile，最小位宽不低于 16 bit 时使用 INT16 profile。执行参数、累加器和 diagnostics 不参与档位判断。
- 融合公式相对旧式逐算子量化 baseline 达到阶段 0 的精度门槛。
- 8/16 bit、POT2/affine、对称/非对称激活和三种权重粒度均覆盖。
- 基础测试四组参数均固定使用 `per_channel`；严格测试用成对组合覆盖四组参数的独立 `per_tensor`、`per_gate`、`per_channel` 配置，并验证原始粒度元数据与展开后的 `4H` channel 参数。
- 跨载体报告差异；首版两者使用同一真实激活定义，但不把 FP 累加/rescale 与整数累加/rescale 强行要求 bit-exact。
- CPU reference 基础测试使用 `cpu_basic`；严格正确性覆盖 `minimal`、`short_recurrent`、`non_aligned` 和 `long_sequence`，shape 定义见第 6.2 节。
- 状态 profile 覆盖 `omitted`、`explicit_zero`、`typical_random`、`near_quant_boundary` 和 `mixed_h_random_c_boundary`；按定向/成对组合执行，不做 `h_0/c_0` 笛卡尔积。

审核点：CPU FP reference 是后续 CUDA FP 主路径的数值规范；CPU int32 reference 是整数算术和融合公式规范。未来整数 LUT 必须建立新的 `cpu_int32_lut` 语义，不能静默改变首版 reference。

### 阶段 4：CUDA FP 载体量化主路径

交付：

- `lstmForwardQuantizedFpCuda`：float master 输入在边界量化为 FP32 保存的 q 值，输出反量化为 float。
- 所有时间步的 `W*x` 使用一次 cuBLAS SGEMM；每个时间步的 `R*h` 使用一次 SGEMM。
- 非对称激活使用 `sum(W)*zp_x` / `sum(R)*zp_h` 补偿。
- bias 先 round 到 GEMM 累加域，再与 GEMM 一起 rescale 到 linear 输出。
- 一个融合 pointwise kernel 完成四门 bias/rescale、真实 sigmoid/tanh 重新量化、cell 更新和 hidden 更新。
- 四门 bias 使用 shared memory；cell/hidden contribution 不落地为可配置量化张量。
- 工作区预分配和复用，输入 Linear 与 recurrent Linear 使用双 stream/event 调度。
- cuBLAS math mode 使用语义明确的配置，验证模式默认关闭 TF32/Tensor Core。
- 首版不包含 CUDA int32 GEMM 或 int32 载体 pointwise kernel。

验收：

- 在阶段 0 证明的 FP32 精确整数区间内，CUDA 各量化点与标量 FP reference 的 q 值逐值相等。
- 超出精确区间的 16bit/大 K 场景按固定阈值验证 max abs、MAE、MSE、SQNR、cosine similarity 和饱和率。
- 每个配置保存 `NumericSafetyReport`；`precision_risk` 必须在控制台和机器可读报告中可见，不能静默降级或自动切换后端。
- `compute-sanitizer` 的 memcheck/racecheck 基础用例通过。
- 覆盖非 tile 对齐维度、小 batch、长 T 和非零 h0/c0。
- profiler 必须看到 cuBLAS SGEMM；报告相对朴素浮点/标量量化实现的延迟和吞吐。
- TF32/Tensor Core 开启与关闭分别报告性能和精度，不允许只报告加速结果。
- 首版不因某个延迟或加速比数值单独阻塞阶段 4，但缺少 cuBLAS 调用证据、环境/计时方法、预热同步、P50/P95、吞吐、workspace、量化开销或对应精度指标时验收失败。
- CUDA 基础测试使用 `cuda_gru_basic`；严格正确性复用固定 shape profile，压力/性能额外覆盖 `large_batch`。

### 阶段 5：校准与量化参数生成

交付：

- 浮点校准 forward 输出所有真实量化点。
- MinMax：输入、h/c 状态、linear、四门输入/输出和 cell tanh；cell contribution 只进入精度报告。
- W/R/bias 的 `4H` per-channel、四组 per-gate 和 per-tensor 统计。
- Weight/bias 校准统一使用 signed symmetric range，生成的完整 `4H` zero-point 向量恒为 0；激活和状态仍按各自对称性配置计算 zp。
- FP 主路径的真实激活使用 standard scale/zp；所有算术 rescale 生成并模拟 M+shift/POT2 执行参数。
- CPU int32 reference 生成 FixedPointScale/Pot2Rescale 参数；首版不生成 LUT，五个 LUT 留待条件性纯整数阶段。
- 多 batch 范围累积、reset、dirty/locked 状态处理。
- 常量/全零范围按公共 minimum-scale 规则逐校准组处理，报告原始/调整范围、`S_min` 和 fallback 次数；禁止 scale=0/1、借用相邻 channel 或跳过量化。
- 审计 standard scale、zero point、执行 ratio 编码和导入导出，确保 forward 不使用 raw ratio，也不把内部 ratio_exec 当作 tensor scale。

验收：校准张量与浮点计算图逐项核对；每个 scale/zp 有限且 scale>0；常量/全零测试覆盖 signed/unsigned、symmetric/asymmetric、Affine/POT2、per-tensor/per-gate/per-channel、恰好等于及刚低于 `S_min`，并核对 fallback 诊断；FP 主路径导出再导入后的输出满足精度门禁。

随后单独补 SQNR 和 Percentile，复用 GRU 直方图实现，但数组维度和 gate 数必须从 3 改为 4，不能留下硬编码。

### 阶段 6：PyTorch FP 载体量化接口

交付：

- `QuantLSTM` 对齐单层 `nn.LSTM` 构造和 forward 签名。
- 返回 `output, (h_n, c_n)`，正确处理 `batch_first`、bias、初始状态与设备/类型校验。
- 默认量化执行调用 CUDA FP 载体主路径，不暴露尚未实现的 int32 backend 选项。
- 配置加载、`set_all_bitwidth`、`adjust/get_quant_config`、calibration finalize 和参数导入导出只枚举真实量化点；Python 的稀疏配置必须先经 C++ resolver，`get_quant_config` 返回完整 canonical resolved config。
- QAT 前向保存 FP32 载体的 W/R/bias/x、状态、中间值和真实 clamp mask。
- 参数导出中标记执行载体、activation mode、cuBLAS math mode 和 standard scale 模式；内部 M+shift/POT2 参数只进入调试/编译报告，不成为第二套外部 scale。
- 四组 weight/bias 的外部量化参数始终使用完整 `4H` standard scale/zp 向量并携带各自 granularity；不导出 1/4 元素 compact 副本。导入时验证长度、粒度重复模式和 float32 位模式后直接使用，不再次广播。

验收：

- 浮点模式与 `nn.LSTM` 对齐。
- Python FP 载体路径与直接 C++ CUDA 接口结果一致。
- `batch_first=False/True` 在相同逻辑输入下生成相同量化参数和内部 checkpoints；公开 output shape 分别为 `[T,B,H]` 与 `[B,T,H]`，转置到同一布局后结果一致，`h_n/c_n` 均为 `[1,B,H]`。
- 参数导出再导入后通过量化点与最终 output/h/c 精度门禁。
- 未校准、shape 错误、配置越界和不安全 FP32 累加场景给出明确警告或错误。

首个 CUDA FP 载体量化 LSTM 实现里程碑到此完成。若代表性真实数据仍未接入，只能声明生产路径实现和数值严格验证完成，不能声明模型级精度或生产精度已经验证。

### 阶段 7：双向、CPU-only reference 与工程完善

交付：

- 双向 LSTM，两方向独立校准参数；输入共享量化网格，h/c 状态分别管理。
- CPU-only int32 reference 包、C++ example、安装和打包检查。
- CI 测试矩阵、clang-format、README、配置说明和双载体计算流程文档。
- 所有项目文档与代码注释使用中文。

验收：双向 FP 主路径的 `output/h_n/c_n` 顺序与 PyTorch 一致；CPU-only reference 不链接 CUDA；文档不混淆 FP 载体和纯整数语义。

### 阶段 8：QAT backward

交付：

- 浮点 LSTM backward。
- FP 载体量化前向只为真实 clamp 点保存 mask，融合乘法临时值不创建遗留 mask。
- 保存量化后的 FP32 W/R/bias/x、h/c 和最少中间张量，反量化后复用经过验证的浮点 backward。
- h0/c0 梯度、bias=False 和双向梯度支持。

验收：float backward 对齐 PyTorch；FP 载体 QAT 完成梯度对比、单步优化、多步 loss 下降和被 clamp/未 clamp 行为测试。

### 阶段 9：ONNX 与 FP 主路径性能优化

交付：

- 标准 ONNX `LSTM` 单节点导出，执行 `(i,f,g,o) -> (i,o,f,c)` 权重重排。
- 根据 profile 评估 packed/cached quantized weights、cuBLASLt、CUDA Graph、kernel fusion 和 math mode。
- 性能基准覆盖不同 T/B/I/H，同时报告延迟、吞吐、workspace 和精度。
- 基于阶段 4 的稳定结果，为目标 GPU 和具名 profile 人工制定版本化性能回归阈值；阈值不得由当前运行自动更新，跨 GPU 不复用绝对数值。
- 所有优化保持冻结的量化点和融合公式，不恢复遗留乘法配置。

验收：ONNX Runtime 浮点语义一致；优化前后满足 FP 主路径精度门禁；性能结论包含完整环境和可复现命令。性能阈值经单独审核入库后，后续回归必须通过对应 GPU/profile 门禁。

完成证据：

- 单向/双向、bias 开关与两种布局均导出为恰好一个标准 ONNX `LSTM` 节点，ONNX Runtime 的 `output/h_n/c_n` 通过门禁。
- CUDA context 以显式非零 generation key 缓存量化 W/R/bias 与 weight sums；同 key 命中、key 变化失效及 Golden 逐值一致均有测试。
- RTX 6000D 的四个 Pedantic/TF32 profile 相对优化前基线，P50 降低 11.6%–19.6%，P95 降低 11.6%–18.3%；memcheck、racecheck 和 Nsight 136 次 SGEMM 交叉计数通过。
- `cuda_performance_thresholds_v1.json` 按 GPU/CUDA/cuBLAS/profile 冻结 P50/P95/吞吐门禁，只读检查器拒绝环境错配和优化前基线。
- cuBLASLt、CUDA Graph 与额外 pointwise fusion 经评估未采用；具体理由、workspace 代价、命令和指标见 `docs/cuda-performance.md`。

### 阶段 10：条件性 CUDA int32 载体与整数集成

只有满足以下任一条件并经审核确认后才启动：

- 需要 int32/int16 输入输出边界或 AIMET fixed-eval。
- 需要与目标硬件逐中间值 bit-exact。
- FP32/TF32 在目标位宽和维度上无法达到精度门禁。
- 目标 GPU 上 CUDA int32/cublasLt integer 路径有明确性能收益。

交付：

- 先冻结并实现三个 sigmoid、两个 tanh 的整数 PWL LUT，新增独立 `cpu_int32_lut` execution model、Golden 和真实激活对比报告；LUT 未通过精度门禁前不启动 CUDA int32 pointwise。
- 先评估 cuBLASLt `int8 x int8 -> int32` 是否覆盖目标配置；不覆盖的任意位宽场景再实现自定义 integer GEMM。
- CUDA int32 载体 pointwise kernel，复用 `cpu_int32_lut` reference 的整数 rescale、LUT 和融合顺序。
- 预量化权重/bias 缓存和真正的 integer input/output API。
- AIMET encodings 与 integer I/O 元数据契约。

验收：`cpu_int32_lut` 先相对首版真实激活 reference 报告逐激活点和端到端 MAE、MSE、余弦相似度、最大误差及饱和率；通过单独审核的 LUT 门禁后，CUDA 与 `cpu_int32_lut` 所有有效量化点及融合临时值 bit-exact，mismatch 为 0。最终提交相对 FP 主路径的性能、精度、显存和维护成本报告。

## 6. 测试策略

### 6.1 正确性层级

测试按以下顺序建立，失败时能定位到具体层：

1. 载体无关量化原语 golden test。
2. CPU int32 reference 的 gate、融合 cell/hidden update 和完整递推测试。
3. 标量 FP 载体 reference 的量化点与完整递推测试。
4. CUDA FP 载体与标量 FP reference 的逐量化点对比。
5. 未融合公式与融合公式的离线精度对比。
6. 浮点实现与 PyTorch `output/h_n/c_n` 对比。
7. 校准、配置、参数 round-trip 和 Python API 集成。
8. 代表性模型/数据集端到端精度回归。
9. 阶段 10 启动后，先增加 `cpu_int32_lut` 相对真实激活的精度矩阵，再增加 CUDA/CPU LUT 路径 bit-exact 矩阵。

每一层实现内部也遵循相同顺序：先运行基础精度测试；基础测试失败时立即停止，不启动耗时更长的严格矩阵。基础测试通过只代表主流程可继续调试，不代表该阶段已经满足最终精度要求；需要严格测试的阶段仍须通过对应严格门禁后才能提交。

#### 6.1.1 Golden vector 分层覆盖

Golden vector 用于锁定公式、舍入、饱和和递推语义，不承担完整精度矩阵职责。Canonical JSON 按以下三层组织：

1. 公共量化原语：覆盖 signed/unsigned、对称/非对称 zero point、量化/反量化、M+shift/POT2 rescale、正负 half-tie 的 round-to-nearest-even、上下界 clamp，以及合法编码的边界值。
2. 单时间步融合 Cell：使用 `B=1` 和便于人工复核的小 `I/H`，分别给出 Linear、四门输入/输出、Cell 双比例融合和 Hidden 融合的中间期望值；定向覆盖相同/悬殊 scale、正负抵消、同号累加、饱和以及非零 `h_0/c_0`。
3. 短递推：使用 `T=3, B=1` 的小尺寸用例，保存每个时间步的 `q_h/q_c` 及最终 `output/h_n/c_n`；用最小成对组合覆盖 8/16 bit、Affine/POT2 和关键 signed/zero-point 语义，验证误差传播和状态网格复用。

共享整数算术部分只保留一份权威期望；首版 CPU FP 和 CPU int32 载体 reference 都使用真实 sigmoid/tanh，并分别产生载体相关预期。未来 LUT 由新增的 `cpu_int32_lut` 产生独立预期。Golden 集合不遍历完整 shape、粒度、math mode 和校准方法组合，这些维度由基础测试及严格测试矩阵覆盖。

文件组织采用“一用例一个自包含 JSON”。文件分别放入 `tests/golden/spec/primitive/`、`cell/` 和 `recurrent/`；每个文件独立携带 schema 版本、稳定用例 ID、适用载体、完整 resolved 量化配置、输入、量化参数和期望值，不引用稀疏 override、其他用例或外部张量数据。一个用例的审查、复现和失败定位不得依赖读取同目录的其他 JSON。

`tests/golden/schema/golden_case.schema.json` 是唯一版本化 schema。根对象使用 `kind=primitive|cell|recurrent` 作为判别字段，并通过 `$defs` 与 `oneOf` 选择对应的严格 payload；公共元数据只定义一次。schema 及各层对象默认设置 `additionalProperties: false`，未知字段、缺失必填字段、目录与 `kind` 不一致或不支持的 `schema_version` 都必须在生成 fixture 前失败。`tools/generate_golden.py` 直接加载该 schema 完成校验，不得再维护一套等价的手写字段白名单。

每个用例必须且只能声明一个 `execution_model`：首版为 `common`、`cpu_int32` 或 `cpu_fp32`，并只包含该语义下的一份 `expected`。`common` 仅用于舍入、clamp、编码等已经证明结果与载体无关的公共原语；只要涉及载体专属累加/rescale 或激活边界，就必须拆成 `cpu_int32` 与 `cpu_fp32` 两个自包含文件。`execution_model` 表示 reference 语义而不是执行设备；CUDA FP 主路径与 `cpu_fp32` 比较。未来实现 LUT 时升级 schema 并新增 `cpu_int32_lut`，CUDA int32 LUT 路径只与该模型比较，禁止覆盖原 `cpu_int32` 语义或通过目录/dtype 推断 execution model。

`expected` 明确拆为两组：

- `checkpoints`：保存所有真实 Round/Clamp 后的量化边界，包括两路 Linear 输出、四门输入/输出、`cell_state`、`cell_tanh_output`、`output`，以及递推用例中每个时间步的对应 q 值和最终 `output/h_n/c_n`。
- `diagnostics`：只保存冻结数学公式中的关键未量化宽值，包括 `p_forget`、`p_input`、两路按执行比例缩放后的宽 contribution、最终舍入前的 Cell 宽和以及 Hidden 原始乘积。它们没有独立 bitwidth、scale、zero point 或 clamp，不能出现在算子配置中。

不保存 GEMM tile 部分和、shared memory 内容、线程局部变量、cuBLAS workspace 或其他后端实现临时值。CPU reference 回归按字段语义逐值检查 checkpoints/diagnostics；CUDA 后端只比较其可观测且数学语义对应的 checkpoints，并按 FP32 精确区间规则决定逐值相等或指标验收，不要求复现 CPU 调试缓冲。

Golden 中的 standard scale 及其他必须持久化的 `float32` 非整数量使用“最短可往返规范十进制字符串”，用例显式声明目标类型为 `float32`。加载器采用与 locale 无关、完整消费字符串且正确舍入的转换，随后将该 `float32` 重新格式化为最短十进制；结果与原字符串不完全一致时拒绝文件。量化张量、zero point、M+shift multiplier/shift、shape 和位宽使用 JSON integer，并由 schema/加载器检查目标 C++ 类型范围。Canonical JSON 不同时保存 `scale_bits`；生成的 C++ fixture 可以机械派生精确位模式或十六进制浮点字面量。

所有张量统一表示为包含 `dtype`、`shape` 和一维 `data` 的自包含对象，内存顺序固定为 row-major，不允许嵌套数组或从数组层级推断 shape。`shape` 中的维度必须是正整数，加载器检查 `data.size()==product(shape)`；标量使用空 `shape` 和恰好一个元素的 `data`。`dtype` 决定元素编码和目标范围：逻辑量化值使用整数 dtype 和 JSON integer，即使 FP 载体运行时用 `float` 存储 q 值也不改变 Golden 的整数表示；真正的 `float32` 张量元素使用上一段定义的规范十进制字符串。用例根部的 `execution_model` 独立描述 reference 载体语义，不能用 tensor `dtype` 隐式推断。

### 6.2 必测维度

首轮固定以下具名 shape profile，tuple 顺序统一为 `(T,B,I,H)`：

| Profile | `(T,B,I,H)` | 用途 |
|---|---|---|
| `cpu_basic` | `(8,4,16,32)` | CPU reference 快速基础测试 |
| `cuda_gru_basic` | `(50,64,128,256)` | CUDA 基础测试及与 GRU 结果横向参考 |
| `minimal` | `(1,1,1,1)` | 标量边界、shape 和初始状态 |
| `short_recurrent` | `(3,1,2,2)` | Golden 短递推和逐时间步检查 |
| `non_aligned` | `(7,3,31,33)` | 非 tile/warp 对齐维度 |
| `long_sequence` | `(256,1,64,128)` | Cell 累积、漂移与饱和 |
| `large_batch` | `(16,128,64,128)` | CUDA batch 压力和性能 |

基础测试只运行对应 backend 的 basic profile。严格正确性运行 `minimal`、`short_recurrent`、`non_aligned`、`long_sequence`；CUDA 压力与性能再运行 `cuda_gru_basic` 和 `large_batch`。shape 与 bitwidth、scale mode、granularity、math mode 使用定向/最小成对组合，不做全量笛卡尔积。每个报告必须记录 profile 名和完整 `(T,B,I,H)`，未知 profile 或 tuple 不匹配时 fail fast。

- `h_0/c_0`: 使用下文冻结的五个具名状态 profile；省略与显式零分别走 API 并验证结果一致。
- 参数：`bias=True` 使用 `enabled_random`，`bias=False` 使用 `disabled`，两者都进入基础和严格测试；常量权重、全零权重和极端值仍由显式原语/对抗用例覆盖。
- 布局：`batch_first=False/True` 均进入基础和严格正确性测试；同一 case 不重新生成随机数据，只改变外部布局。
- 量化：8/16 bit、混合位宽、POT2/affine、signed/unsigned、对称/非对称。
- 粒度：基础将四组 weight/bias 都设为 `per_channel`；严格使用成对覆盖验证四组参数独立选择 `per_tensor`、`per_gate`、`per_channel`。粒度仅适用于 weight 和启用的 bias，激活/状态保持 per-tensor。
- 状态：cell 持续累加、遗忘门接近 0/1、输出门接近 0/1、饱和与负值。
- 融合：两路贡献尺度相同/悬殊、正负抵消、同号累加、接近载体安全边界。
- FP 载体：`sum(abs(product))` 低于、接近和超过 `2^24`。
- cuBLAS：TF32/Tensor Core 关闭和开启两种 math mode。

#### 6.2.1 显式严格测试矩阵

基础测试使用少量固定 case，不从严格矩阵抽取。严格正确性使用入库的 `tests/precision/config/strict_matrix_v1.json`；该文件通过 `tests/precision/schema/strict_matrix.schema.json` 校验，是测试配置而不是运行输出。每个 case 必须包含稳定 `case_id`、适用阶段/backend/execution model、shape/state/bias/layout profile、math mode 和完整 canonical resolved quant config，不允许引用稀疏 override 或依赖运行时补默认值。

矩阵采用约束 pairwise：对所有合法组合，保证每个维度取值至少出现一次、每两个独立维度的合法取值对至少出现一次。覆盖维度包括 shape、载体/backend、8/16 与混合位宽 profile、Affine/POT2、四组 weight/bias granularity、signedness/symmetry profile、state、bias、layout，以及仅 CUDA 适用的 math mode。强制 signed symmetric 的 weight/bias、`bias=False` 下的 `not_applicable`、backend 与 execution model 映射等约束先排除非法 pair，不能为了满足覆盖率构造无效配置。

Pairwise 之外追加不可删除的定向 case：全 INT8/全 INT16、至少一组 8/16 混合、Affine/POT2、signed symmetric/asymmetric 最小负值、unsigned symmetric/asymmetric、`bias=True/False`、两种 layout、五种 state profile、四组参数跨 Linear/weight-bias 的不同粒度、FP32 `2^24` 精确边界以及 Cell contribution 抵消/同号累加。Half-tie、非法配置、fallback、溢出和饱和等局部边界由 primitive/adversarial 用例负责，不强行塞入递推 pairwise case。

`strict_matrix_coverage` 只读取并验证显式 case：检查 schema、case ID 唯一性、完整 resolved config、所有声明取值、合法 pair 覆盖和定向 case 标签；它不得新增、删除或重排 case。运行时也不得根据当前失败或耗时自动裁剪矩阵。矩阵变更必须显式修改 JSON、审查覆盖差异并作为测试契约单独提交。

固定 calibration/evaluation seed 集不是 pairwise 维度；每个随机矩阵 case 均运行对应的固定 seed 角色，定向 case 使用显式数据。报告记录 matrix schema/version、文件摘要和 case ID，使相同 commit/config/seed 可精确复现。

#### 6.2.2 严格测试数据组成与隔离

严格测试的完整目标包含以下两层数据，任何一层失败都不能由另一层的平均值抵消。当前阶段 3 只实施第一层，第二层延期到代表性真实数据可用后补充：

1. 合成边界数据：由入库的 `tests/precision/config/synthetic_data.json`、固定 seed 和确定性生成器在每次运行时重新产生，不提交生成数据。覆盖普通随机分布，以及饱和、门值接近 0/1、Cell 持续增长、悬殊 scale、正负 contribution 抵消、同号累加、非零 `h_0/c_0`、长序列和 FP32 `2^24` 精确边界。
2. 代表性真实数据（延期）：使用目标应用的代表性模型权重、输入序列和初始状态，比较量化 reference/后端与相同参数的 `torch.nn.LSTM`。除 `output/h_n/c_n` 的 MAE、MSE、余弦相似度外，有任务级指标时同时报告允许下降值。

任何启用的数据层都必须先固定 split 再执行校准，calibration 与 evaluation 的样本 ID 不得相交；测试入口在运行前执行泄漏检查并将 split 标识、样本数和 `calibration_equals_evaluation=false` 写入报告。阶段 3 报告必须记录 `validation_scope=synthetic_numeric` 和 `real_data_status=not_configured`，只能标记 `numeric_validation_passed`，不能标记 `model_accuracy_validated`。未来接入真实数据后，两层分别产生逐用例指标和汇总，不允许拼接后只报告总体指标。

Seed 角色固定如下：基础测试使用 `seed=0`；严格校准使用 `{1001,1002,1003}`；严格验证使用 `{2001,2002,2003,2004,2005}`。这些 calibration/evaluation seed 只生成每个适用 shape profile/config 的输入和 `h_0/c_0`；同一 case 的 weight/bias 在两个 split 间必须完全一致，不能因数据 seed 改变。case ID 必须包含 seed 角色与数值。基础测试允许沿用 GRU 的同源行为并标记 `calibration_equals_evaluation=true`；严格校准和验证 seed 集不得重叠。饱和、half-tie、抵消等定向对抗用例使用显式数据，不消耗随机 seed，也不计入随机样本数量。

Weight/bias 使用独立于数据 split 的 parameter seed，并按 shape profile 固定：`cpu_basic=3001`、`cuda_gru_basic=3002`、`minimal=3003`、`short_recurrent=3004`、`non_aligned=3005`、`long_sequence=3006`、`large_batch=3007`。同一 profile 的所有 bitwidth、scale mode、granularity、execution model 和 calibration/evaluation split 必须复用完全相同的浮点 master weight/bias；报告记录 `parameter_seed`。定向权重边界用例使用显式参数并标记 `parameter_source=explicit`，不覆盖随机 profile 的 seed 规则。

唯一随机数实现位于 `tests/common/deterministic_rng.*`，版本固定为 `pcg32-xsh-rr-v1`，采用 PCG-XSH-RR 64/32 状态转换和参考初始化流程。`uniform_float32` 固定取一个 `uint32` 的高 24 bit 并乘以精确的 `2^-24` 得到 `[0,1)`；需要类正态数据时固定累加 12 个 uniform 后减 6，不调用 `std::uniform_real_distribution`、`std::normal_distribution`、Python/PyTorch RNG 或平台 `log/sqrt`。所有随机张量在 CPU 一次生成，随后原样传给 CPU reference、CUDA 和 Python/PyTorch 对照。

RNG 单元测试必须包含 PCG32 官方已知输出序列、seed 初始化、零/最大随机字、float32 映射端点和固定消费次数检查。`synthetic_data.json` 和每份报告记录 `rng_version`；修改算法、初始化、数值映射或每元素消费规则必须升级版本并作为测试数据契约变更单独审核，旧版本基线不能静默重算。

PCG32 使用 `(seed,stream_id)` 参考初始化，每个张量角色拥有独立 stream：`input=1`、`h0=2`、`c0=3`、`weight_ih=10`、`weight_hh=11`、`bias_ih=12`、`bias_hh=13`。这些编号构成 `rng_stream_registry=v1`，未分配编号保留给未来方向、层和新增角色，禁止复用或从张量名称动态哈希。每个 stream 只按该张量的 row-major 元素顺序推进；`bias=False`、新增其他张量或改变不同张量之间的生成顺序不得改变现有张量。

Stream registry 由一个稳定枚举和一份公共映射实现，禁止各测试入口复制数字。单元测试除逐 stream 已知向量外，还必须验证调整张量生成调用顺序、跳过 bias 和插入未知角色不会改变现有 stream 的输出。`synthetic_data.json` 与报告同时记录 `rng_stream_registry`；新增或改变 ID 属于测试数据契约变更，必须升级 registry 版本并单独审核。

普通随机张量统一使用 `distribution_profile=pytorch_typical_v1`，所有 master 数据在生成时即舍入为 float32：

- `weight_ih`、`weight_hh` 使用 `k=1/sqrt(H)`，由 `uniform_float32` 按固定 float32 运算顺序映射到 `[-k,k)`，对齐 PyTorch LSTM 默认初始化范围。`bias_profile=enabled_random` 时，`bias_ih/bias_hh` 使用相同范围；`bias_profile=disabled` 时不创建、不量化也不读取 bias，且不推进 bias stream。
- `input`、`h_0`、`c_0` 每个元素固定消费 12 个 `uniform_float32`，求和后减 6，得到均值约 0、方差约 1、范围 `[-6,6)` 的类正态数据。
- 同一 profile/config 下不得根据 bitwidth、scale mode 或 execution model 改变随机分布或幅值；饱和、极端 scale、贡献抵消、门接近 0/1 等场景由显式对抗用例覆盖，不混入普通随机 profile。

`synthetic_data.json` 和报告记录 `distribution_profile`。单元测试固定检查每类张量的首批输出位模式、每元素随机字消费数量、区间端点和映射顺序；修改区间、求和次数、float32 运算顺序或参数初始化公式必须升级 profile 版本并单独审核。

初始状态固定为以下五个 `state_profile`：

| Profile | `h_0` | `c_0` | 目的 |
|---|---|---|---|
| `omitted` | 不传入 | 不传入 | 验证 API 默认零状态路径 |
| `explicit_zero` | 显式全零 | 显式全零 | 验证张量校验和显式状态路径 |
| `typical_random` | `pytorch_typical_v1` | `pytorch_typical_v1` | 验证普通非零递推 |
| `near_quant_boundary` | 显式边界值 | 显式边界值 | 验证 round/clamp 和状态饱和 |
| `mixed_h_random_c_boundary` | `pytorch_typical_v1` | 显式边界值 | 验证 Cell 独立网格和混合状态幅值 |

`near_quant_boundary` 使用审核过的显式 q-domain 模式构造，包含 qmin/qmax 内侧值和 clamp 阈值邻域，按对应 `output`/`cell_state` standard scale 反量化为 master float32；它不消耗 `h0/c0` 随机 stream。`omitted` 与 `explicit_zero` 在相同输入、参数、量化配置和 execution model 下必须得到一致 checkpoints 和最终输出。状态 profile 与 shape/bitwidth/scale mode 使用定向/最小成对组合，禁止展开为 `h_0` 与 `c_0` 模式的全量笛卡尔积。

Bias 固定为两个 profile：`enabled_random` 对应 `bias=True`，使用两个独立 bias stream 生成 `bias_ih/bias_hh`；`disabled` 对应 `bias=False`，整个调用链都不应存在 bias tensor、量化参数、rescale 或读取。CPU/CUDA 基础测试分别运行两种 profile；严格矩阵使用成对组合，但必须确保 CPU int32、CPU FP、CUDA FP、INT8、INT16、Affine 和 POT2 中两种 profile 都至少有覆盖。两种 profile 分别与相同 bias 设置的 `torch.nn.LSTM` 比较，不把 disabled 与随机 bias 的结果互相比较。

`weight_ih`、`weight_hh`、`bias_ih`、`bias_hh` 分别拥有独立的 `granularity=per_tensor|per_gate|per_channel`。基础测试四者均为 `per_channel`；严格测试使用版本化成对组合，而不是遍历 `3^4=81` 个组合，保证每个参数都出现三种粒度，并覆盖两个 Linear 分支粒度不同、weight 与对应 bias 粒度不同的情况。

无论原始 granularity 为何，finalize 后每个启用参数都只保留一个长度为 `4H` 的执行向量：per-tensor 的同一 scale/zp 复制到全部 channel，per-gate 的四组 scale/zp 按 `(i,f,g,o)` 各复制 `H` 次，per-channel 逐 channel 填充。执行后端不得读取 granularity 做动态广播，只按输出 channel 索引该向量。不同 gate/channel 的校准结果允许数值相同，测试验证来源和映射规则，不通过“数值必须不同”判断粒度是否生效。激活和 recurrent state 始终为 per-tensor。`bias_profile=disabled` 时两组 bias granularity 均为 `not_applicable`，且不创建 `4H` bias 参数向量。

导入导出同样只使用完整 `4H` 向量，该向量是 standard scale/zp 的唯一外部数值来源：per-tensor 文件中 `4H` 个 scale 元素必须位级相同；per-gate 文件中每个长度 `H` 的 scale 门段必须位级相同；per-channel scale 只校验长度和逐元素合法性，不要求数值互异。四组 weight/bias 的 `4H` zero-point 元素无条件全部为 0。Granularity 元数据保留用于审计来源和执行导入校验，不允许出现 `compact_values`、1/4 元素数组或同时保存 compact/expanded 两份数值。校准可以内部统计 1/4/`4H` 组 range，但必须在 finalize 时一次性展开并只导出完整向量。`bias_profile=disabled` 时 bias 字段必须缺失，空数组或占位参数均视为 schema 错误。

布局固定为两个 `layout_profile`：`time_major` 对应 `batch_first=False` 和外部 `[T,B,I]`，`batch_major` 对应 `batch_first=True` 和外部 `[B,T,I]`。数据生成器只产生一次规范 `[T,B,I]` 逻辑输入，batch-major case 由确定性转置得到，不能重新消费 RNG。C++ 核心内部统一使用 time-major；两种 profile 必须生成相同的校准统计、standard scale/zp 和归一化 checkpoints。公开 output 分别为 `[T,B,H]` 与 `[B,T,H]`，统一转置为 time-major 后逐值/指标比较；`h_n/c_n` 的 `[1,B,H]` 布局不受 `batch_first` 影响。

CPU/CUDA 基础测试和所有严格正确性 shape profile 都运行两种布局。报告记录 `layout_profile`、`batch_first`、外部 input/output shape 和内部规范 shape；性能报告必须说明是否包含 Python/binding 转置开销。

### 6.3 指标定义与 GRU 参考基线

对参考张量 `y_ref` 和待测张量 `y`，统一展平后使用 FP64 累加计算：

```text
MAE    = mean(abs(y - y_ref))
MSE    = mean((y - y_ref)^2)
Cosine = dot(y, y_ref) / (norm(y) * norm(y_ref))
```

计算和报告规则：

- 基础层和严格层都要对 `output`、`h_n`、`c_n` 分别计算，不允许拼成一个张量后用大张量掩盖局部误差。
- 严格层对 `output` 逐时间步计算，并报告最大 MAE、最大 MSE、最小余弦相似度及对应时间步；双向时两个方向还要分别报告。
- 严格层为每个 shape/bitwidth/scale mode/granularity/math mode 测试用例保留独立指标，汇总报告提供平均值、P50、P95、最差值和失败样本，不允许只报告所有样本的总体平均。
- 非退化张量的余弦相似度必须 `>=0.999`。统一使用 FP64 计算 L2 norm，零范数判定固定为 `norm<=1e-12`：若参考和待测张量都满足该条件，余弦相似度记为 `N/A` 并依靠 exact/MAE/MSE 验收；若仅一方满足，则该用例直接失败，不能用 epsilon 将余弦结果伪装为通过。
- 报告必须记录 `cosine_status=valid|both_near_zero|one_near_zero`。聚合最小余弦时排除 `both_near_zero`，但单独统计其数量；`one_near_zero` 始终计为失败。如果一个汇总组内全部为 `both_near_zero`，该组不产生余弦数值结论，只能依据 exact/MAE/MSE 验收。
- bit-exact 用例仍然要求 mismatch 为 0；即使 MAE/MSE/余弦相似度通过，也不能替代逐值相等检查。

GRU 当前测试代码提供的参考门禁如下：

| GRU 测试类别 | MSE | MAE | 余弦相似度 |
|---|---:|---:|---:|
| 非量化前向 | `<1e-5` | `<0.003` | `>0.9999` |
| INT8 量化前向 | `<1e-3` | `<0.05` | `>0.999` |
| INT16 量化前向 | `<1e-4` | `<0.015` | `>0.999` |
| 量化训练前向 | `<1e-4` | `<0.015` | `>0.999` |

阶段 0 将以下数值直接固定为 LSTM 基础测试和第一轮 `synthetic_numeric` 严格测试的初始绝对门禁，并写入 `tests/precision/config/strict_thresholds.json`：

| LSTM 初始测试类别 | MSE | MAE | 余弦相似度 |
|---|---:|---:|---:|
| 非量化前向 | `<1e-5` | `<0.003` | `>=0.9999` |
| INT8 量化前向 | `<1e-3` | `<0.05` | `>=0.999` |
| INT16 量化前向 | `<1e-4` | `<0.015` | `>=0.999` |

上述门禁对 `output`、`h_n`、`c_n` 分别检查，Affine/POT2 和 CPU int32/FP 载体 reference 第一轮共用同一组初始数值；逐值相等或 mismatch=0 等 exact 门禁仍需额外满足。阈值只在完整 LSTM 流程跑通并取得独立合成数据报告后调整，测试不得根据当前结果自动改写。调整可以收紧；若需要放宽，必须先排除实现、公式、校准和数据泄漏问题，再单独审核。

首轮混合位宽只覆盖由 8/16 bit 组成的目标矩阵，并按真实量化点的最小有效位宽选择 profile：任一真实量化点为 8 bit 时使用 `int8`，最小有效位宽不低于 16 bit 时使用 `int16`。`int64/__int128` 累加器、Q31 multiplier/shift 和 `expected.diagnostics` 不是量化点，不参与选择。报告必须记录 `minimum_effective_bitwidth`、`threshold_profile` 和触发该档位的量化点名称。未来出现 8/16 之外的位宽时，必须先增加显式映射和审核阈值；禁止静默插值或套用最近档位。

2026-09-14 在本机直接编译 commit `9c25d14` 后，固定 seed `0` 复跑默认 FP32 载体主路径，得到以下参考结果：

- 环境：NVIDIA RTX 6000D、driver 610.43.02、PyTorch 2.13.0+cu130、PyTorch CUDA 13.0、nvcc 13.2.78。
- 配置：`T=50, B=64, I=128, H=256`，单层单向、`batch_first=False`、`bias=True`、MinMax、Affine/M+shift。
- INT8 使用测试脚本原有行为，即校准数据与测试数据相同；INT16 使用独立随机测试输入。因此 INT8 结果只能作为乐观参考，不能作为最终泛化精度结论。

| GRU 路径 | 张量 | MSE | MAE | 余弦相似度 |
|---|---|---:|---:|---:|
| 非量化 | `output` | `3.9999400e-13` | `2.3572656e-7` | `1.0000000000` |
| 非量化 | `h_n` | `5.0078870e-13` | `3.4750298e-7` | `0.9999999171` |
| INT8 FP 载体 | `output` | `7.5595931e-6` | `2.2036054e-3` | `0.9997737409` |
| INT8 FP 载体 | `h_n` | `7.6824190e-6` | `2.2230886e-3` | `0.9997053950` |
| INT16 FP 载体 | `output` | `2.1107400e-9` | `3.3961267e-5` | `1.0000000000` |
| INT16 FP 载体 | `h_n` | `1.8721531e-9` | `3.4584671e-5` | `0.9999998031` |

`output` 的逐时间步最差结果如下：

| GRU 路径 | 最大单步 MSE | 最大单步 MAE | 最小单步余弦相似度 |
|---|---:|---:|---:|
| 非量化 | `8.3623169e-13` | `4.3242673e-7` | `0.9999995370` |
| INT8 FP 载体 | `7.8772237e-6` | `2.2472877e-3` | `0.9996509545` |
| INT16 FP 载体 | `1.3106255e-8` | `3.5185854e-5` | `0.9999994646` |

GRU 的位宽遍历 shell 测试另外采用 `MSE<=1e-4`、余弦相似度 `>=0.999`，但没有检查 MAE。LSTM 不沿用这个缺口，三个指标始终同时报告。GRU 仓库没有版本化的 JSON/CSV/Markdown 实测报告，而且现有指标函数使用 FP32 reduction，整体余弦相似度还会 clamp 到 `[-1, 1]`，所以表中 INT16 `output` 的 `1.0` 不应解释为数学上完全相等。LSTM 先用上表的固定初始值跑通基础和严格流程；阶段 3 使用统一的 FP64 指标实现，在审核确认的固定环境中用独立合成校准集、验证集和 shape 矩阵实测，再人工审核是否调整 MAE/MSE 门禁。LSTM 的余弦相似度下限不得低于 `0.999`。

### 6.4 基础精度测试与门禁

基础测试直接参考 GRU 当前测试结构，目标是尽快发现接口、布局、量化参数、前向调度和明显精度问题：

1. 固定 `seed=0`，使用对应 backend 的 basic shape profile，分别运行非量化、INT8 和 INT16 前向。
2. 先使用默认 FP32 载体、MinMax、Affine/M+shift 和 `per_channel`；POT2、混合位宽、per-tensor/per-gate 粒度和不同 math mode 放到严格测试矩阵。
3. 为便于在开发早期复现 GRU 结果，量化基础用例允许暂时复用同一批校准和测试数据，但报告必须明确标记 `calibration_equals_evaluation=true`，不得把结果解释为泛化精度。
4. 比较浮点 baseline 与量化结果的 `output`、`h_n`、`c_n`，分别报告 MAE、MSE 和余弦相似度；基础层不要求逐中间量化点报告。
5. 初始门禁沿用 GRU：非量化 MSE `<1e-5`、MAE `<0.003`、余弦相似度 `>=0.9999`；INT8 MSE `<1e-3`、MAE `<0.05`、余弦相似度 `>=0.999`；INT16 MSE `<1e-4`、MAE `<0.015`、余弦相似度 `>=0.999`。
6. 任一张量或任一指标未通过时，阶段保持失败并先修复基础路径，不启动严格测试。

### 6.5 严格精度门禁

1. 启动条件：对应基础测试必须全部通过；阶段 3 严格测试使用校准 seed `{1001,1002,1003}` 和验证 seed `{2001,2002,2003,2004,2005}` 形成互斥 split，不允许 `calibration_equals_evaluation=true`。代表性真实数据暂不接入，本阶段只允许完成数值严格验证。
2. 初始阈值：第一轮严格测试直接使用第 6.3 节固定的 GRU 数值，按 `output/h_n/c_n` 分别验收，并写入 scope 为 `synthetic_numeric` 的 `tests/precision/config/strict_thresholds.json`。8/16-bit 混合配置按最小真实量化点位宽选择 INT8/INT16 profile；未配置映射的其他位宽直接失败。完整流程跑通后才能依据固定独立合成数据和 shape 矩阵报告提出调整；未来真实数据阈值必须使用独立 scope 另行审核，不能复用或覆盖合成阈值。报告文件不入库，阈值配置入库。
3. 阈值稳定性：常规测试只读阈值，禁止根据当前结果自动更新。任何放宽都必须单独提交，提供更新前后报告路径、最差样本和原因，并与实现修复分开审核。
4. 全阶段报告门禁：凡存在参考张量的数值阶段，都必须输出 MAE、MSE 和余弦相似度；每个用例和每个 `output/h_n/c_n` 必须分别通过适用的绝对阈值及余弦硬门禁，不能用总体平均掩盖失败。缺少任一适用指标、缺少原始机器可读报告或余弦相似度 `<0.999`，该阶段不得验收。
5. 浮点语义门禁：自研浮点 LSTM 与 `torch.nn.LSTM` 比较全部 `output/h_n/c_n`，默认 `atol<=1e-5, rtol<=1e-5`、MSE `<1e-5`、MAE `<0.003`、余弦相似度 `>=0.9999`。关闭 TF32、固定算法和 seed，超限即阶段失败。
6. CPU reference 门禁：int32 reference 的整数原语/diagnostics 及同 execution model 的 checkpoints 与已冻结 Golden 逐值相等，CPU 标量 FP reference 与已冻结 FP Golden 按字段语义一致；要求 exact 的字段 mismatch 必须为 0，不允许用 `allclose` 替代。两套 reference 的反量化 `output/h_n/c_n` 都必须相对 `torch.nn.LSTM` 浮点 baseline 报告 MAE、MSE 和余弦相似度；int32 reference 的真实激活桥接不得被描述成整数硬件 bit-exact。
7. FP 载体精确区间门禁：验证模式关闭 TF32/Tensor Core；当 `sum(abs(product)) < 2^24` 且所有中间整数可被 FP32 精确表示时，CUDA 与标量 FP reference 的 round/clamp 后 q 值必须逐值相等，同时报告最终反量化输出的三个指标。
8. FP 载体非精确区间门禁：16bit、大 K 或长序列超出精确区间时，不宣称 bit-exact，必须按冻结阈值检查 max abs、MAE、MSE、SQNR、余弦相似度、饱和率和逐时间步漂移，其中余弦相似度不得低于 `0.999`。
9. 跨载体门禁：FP 主路径与 CPU int32 reference 首版使用同一个真实激活边界，分别以浮点和整数算术执行同一组 M+shift/POT2 编码比例；只比较误差和趋势，不要求二者整数相等。任何差异必须归因到 GEMM、rescale、FP 精度或激活边界重新量化之一，三个指标仍需完整报告。
10. 融合公式门禁：旧式逐算子量化和候选融合公式使用相同输入、scale/zp 和 bitwidth，最终公式必须达到阶段 0 冻结阈值，不能通过删除困难样本或改变 seed 放宽。
11. 模型级精度门禁：固定代表性模型和数据集，记录 float baseline、8-bit、16-bit、POT2、affine 和 math mode。任务指标及允许下降值必须预先版本化并审核；无应用数据时只能声明数值验证通过。
12. 性能门禁：阶段 4 只执行结构与报告完整性门禁，不预设绝对延迟或加速比。报告 cuBLAS 主路径相对朴素 CUDA/标量 reference 的延迟、P50/P95、吞吐、workspace 和量化开销；记录 GPU/驱动/CUDA/cuBLAS、时钟与功耗模式、shape、math mode、计时范围、预热/迭代次数和同步方法，性能比较同时报告三个精度指标。阶段 9 取得稳定基线后，再按 GPU/profile 人工冻结版本化回归阈值，禁止自动更新或跨设备套用。
13. 回归门禁：公式、scale、LUT、rounding、math mode 或 fusion 改动必须重跑完整精度矩阵；相同 commit/config/seed 的 q 值和报告必须可复现。

### 6.6 精度与性能报告产物

- 脚本放在 `tests/precision/` 和 `tests/benchmarks/`，`synthetic_data.json` 中的 seed/shape/生成配置、`metric_policy.json` 和审核通过的严格绝对阈值纳入 Git；所有测试入口调用同一个指标实现，不复制 epsilon 或聚合逻辑。合成张量仍在每次运行时重新生成，不入库。
- 基础测试先生成轻量机器可读 JSON 和控制台摘要，至少包含环境、commit、seed、shape、bitwidth、carrier、calibration/evaluation 是否同源、MAE、MSE、余弦相似度和 PASS/FAIL。
- 严格测试生成完整 JSON/CSV 和中文 Markdown 摘要，另外包含 validation scope、real-data status、数据层、RNG/stream-registry/distribution-profile version、state profile、bias profile、layout profile、`batch_first`、外部/内部 shape、dataset/split 标识、数据 seed/角色、parameter seed/source、样本数、shape profile 与完整 `(T,B,I,H)`、泄漏检查结果、GPU、CUDA/cuBLAS、math mode、granularity、scale mode、逐时间步最差值、失败样本和误差归因。
- 报告区分“浮点语义误差”“FP 载体 GEMM 误差”“rescale 误差”“真实激活边界误差”“量化误差”“整数一致性”，禁止合并成单一 MSE；阶段 10 实现 LUT 后再增加独立“LUT 近似误差”。
- JSON/CSV/Markdown 报告、控制台日志和性能结果统一写入 `.gitignore` 覆盖的结果目录，每次运行重新生成，不作为版本化资产提交。阶段验收时向审核者提供本次报告路径和关键指标。
- `tests/golden/spec/{primitive,cell,recurrent}/*.json` 是唯一权威 golden 源，采用一用例一个自包含文件，必须入库并接受代码审查；`float32` scale/非整数量使用最短可往返规范十进制字符串，整数域字段使用带目标范围校验的 JSON integer。期望值由对应的 CPU int32/FP 载体 reference model 产生，不实现 Python 量化 oracle。Golden 用于锁定 CPU reference 的已审核输出并验证后续后端，不能被解释为对 CPU reference 自身正确性的独立证明。
- Canonical JSON 只允许通过显式维护流程更新：运行 `golden_reference_generator` 后审查全部期望值 diff，并重新执行两套 CPU reference 相对 `torch.nn.LSTM` 的端到端基础和严格精度测试。常规测试只读 Golden，禁止测试失败时自动刷新期望值。
- `tools/generate_golden.py` 入库，直接使用 `tests/golden/schema/golden_case.schema.json` 校验统一根结构和按 `kind` 区分的 payload，再机械转换为 C++ fixture，不计算新的期望值。生成文件写入构建目录或 `tests/golden/generated/`，由构建/测试流程每次重新生成并通过 `.gitignore` 排除。
- 阈值修改必须单独提交，英文 commit body 说明原因和对比数据；不得与实现修复混在同一提交中。

## 7. 主要风险与控制措施

| 风险 | 影响 | 控制措施 |
|---|---|---|
| FP32 量化整数超过 24 位精确区间 | SGEMM 或 contribution 丢失整数低位 | 计算 `sum(abs(product))` 上界；8/16bit 分开验证；超界走误差门禁或启动整数路径 |
| int64/int128 乘加或左移溢出 | 未定义行为或静默错误 | setup 静态上界预检并 fail fast；禁止中间 saturate/wrap；仅正式量化边界 Clamp |
| TF32/Tensor Core 隐式启用 | 尤其 16bit q 值在乘法前被截断 | math mode 显式配置；验证默认关闭；开启时单独出报告 |
| affine scale 表示不一致 | 边界量化、执行 ratio 和导出结果不一致 | 外部统一 standard scale；内部只编码 ratio；round-trip 与逐量化点测试 |
| 首版 int32 reference 的激活仍使用浮点函数 | 不能宣称端到端纯整数或完整硬件 bit-exact | 明确命名为 int32 载体 reference；真实激活使用公共边界；LUT 在阶段 10 独立实现和验收 |
| cell state 无界增长 | FP 精度、量化饱和和累计误差恶化 | 独立 cell 网格；长序列校准；逐时间步漂移/饱和报告 |
| 融合公式或舍入顺序漂移 | 精度回退或 reference 不一致 | 冻结融合公式；两套 reference；逐量化点回归 |
| Golden 与 CPU reference 同源 | CPU reference 的公式错误可能被一起冻结 | 常规测试只读 Golden；更新期望值单独审核；两套量化 reference 都与 `torch.nn.LSTM` 做端到端精度门禁 |
| 严格阈值随当前结果漂移 | 回归被自动放宽掩盖 | 先固定 GRU 初始绝对值；常规测试只读；LSTM 实测后人工调整；阈值放宽单独提交和审核 |
| 校准集与验证集泄漏 | 指标虚高，无法反映泛化误差 | 先固定互斥 split；运行前按样本 ID 检查交集；报告明确记录检查结果 |
| 只使用合成或真实数据 | 分别遗漏实际分布或数值边界 | 严格测试采用双数据层并独立验收；缺少真实数据时禁止声明模型级精度达标 |
| 阶段 3 延期真实数据 | 无法验证实际分布和任务指标 | 阈值标记 `synthetic_numeric`；报告标记真实数据未配置；实现可推进但禁止模型级/生产精度声明 |
| seed 或数据 split 漂移 | 指标无法复现或发生校准泄漏 | seed 角色和集合入库；case ID/报告记录 seed；严格校准与验证集合检查无交集 |
| 不同配置使用不同权重 | 横向精度差异无法归因于量化设置 | parameter seed 只由 shape profile 决定；跨配置/载体/split 复用相同 master weight/bias |
| 标准库/Python RNG 漂移 | 相同 seed 在不同环境生成不同输入 | 项目内唯一版本化 PCG32；固定整数到 float 映射；已知向量 bit-exact 测试；CPU 集中生成 |
| 单一随机 stream 顺序耦合 | 新增/跳过张量导致后续全部基线变化 | 每个张量角色固定独立 stream ID；公共 registry；生成顺序不变性测试 |
| 合成分布或映射漂移 | 相同 seed 的数值范围改变，精度基线失效 | 版本化 `pytorch_typical_v1`；固定 float32 运算顺序和消费次数；输出位模式单测 |
| 初始状态模式组合爆炸 | 重复用例增加耗时却没有新增覆盖 | 五个具名 state profile；定向/成对组合；省略与显式零建立等价性断言 |
| bias=False 误用残留 bias | 无 bias 路径仍读取或量化无效参数 | enabled/disabled 均进入基础和严格测试；disabled 不创建 bias；分别对齐 PyTorch |
| batch_first 布局处理错误 | 时间和 batch 维互换或状态输出布局错误 | 两种布局共享逻辑数据；内部统一 time-major；归一化 checkpoints/参数等价性断言 |
| 粒度广播或门顺序错误 | scale/zp 应用于错误 channel 或 gate | 四组粒度独立配置；finalize 统一物化 `4H`；成对组合检查 `(i,f,g,o)` 分段和来源映射 |
| kernel 运行时重复广播 | 后端分支重复、CPU/CUDA 语义漂移 | 广播只在 finalize；后端统一消费 `4H` channel 向量；配置到执行向量做完整性测试 |
| 外部 compact/expanded 双表示 | 两份 scale/zp 不一致且难以确定权威来源 | 导入导出只允许完整 `4H` 向量；拒绝 compact 字段；round-trip 验证位模式和 granularity |
| Weight/bias 非零 zp | 引入额外 GEMM/bias 补偿并偏离硬件 reference | 强制 signed symmetric；完整 `4H` zp 向量全零；非法配置/导入 fail fast |
| 遗留配置项未被执行 | 用户以为位宽生效但结果不变 | JSON 仅含真实量化点；unknown field 报错；配置到执行参数完整性测试 |
| 混合位宽阈值档位误选 | 使用过松门禁掩盖低位宽量化误差 | 只统计真实量化点的最小位宽；报告触发点；未映射位宽 fail fast |
| shape 矩阵笛卡尔爆炸 | 测试时间失控，阶段反馈变慢 | 固定具名 profile；按风险做定向/成对组合；基础层失败时不启动严格层 |
| gate 顺序混淆 | 与 PyTorch/ONNX 结果错误 | 内部固定 `(i,f,g,o)`；ONNX 边界显式重排并单测 |
| 模块边界模糊 | 两套载体公式散落、难以审查 | 共享规格，FP/int 原语和后端分文件；binding 不复制公式 |
| 同时实现两套 CUDA 后端 | 延迟主路径且增加维护成本 | 首版只实现 FP CUDA；int32 CUDA 设条件性阶段 |

## 8. 审核时需要确认的决策

建议按以下默认项批准；有不同要求时，应在阶段 0 开始前修改：

1. CUDA FP32 载体 + cuBLAS SGEMM 是首个生产主路径。
2. CPU int32 路径首版是 int32 载体 reference：整数 GEMM/rescale 加原始 sigmoid/tanh，不作为首个生产后端，也不宣称端到端纯整数。
3. 首版不实现 CUDA int32 载体；只有满足阶段 10 的触发条件才启动。
4. FP 主路径和 CPU int32 reference 首版都通过同一公共边界使用原始 sigmoid/tanh 后重新量化；整数 PWL LUT 延后到条件性纯整数阶段。
5. `f*c_old`、`i*g` 和 `o*tanh(c)` 使用融合公式，不暴露独立乘法位宽或量化参数。
6. Cell 更新使用双比例合并后一次最终舍入；`i/f/g` 保持独立量化网格，不采用 `g -> cell_state` 预对齐。
7. CPU int32 reference 的 Cell 融合暂定使用固定 Q31、`int64_t` multiplier 和 `__int128` 累加，通过边界验证后最终冻结。
8. 所有舍入统一使用 `roundToNearestEven(...)` 公共接口族；后端不得直接调用平台舍入函数或复制通用实现。
9. 数值安全在 setup 统一预检：整数溢出风险 fail fast；FP32 超过 `2^24` 标记风险并走指标门禁，exact 模式 fail fast；禁止隐藏饱和和自动后端切换。
10. 内部门顺序使用 PyTorch `(i,f,g,o)`；cell state 使用独立量化网格。
11. Affine 使用校准连续 standard scale，POT2 使用转换后的 standard scale；FP/int 内部都模拟从 standard scale ratio 派生的执行编码，外部不暴露第二套量化网格。
12. 正确性验证关闭 TF32/Tensor Core；性能模式是否启用由严格精度报告决定。
13. FP 主路径、CPU int32 reference 和跨载体对比分别验收；缺少应用数据时不宣称模型精度达标。
14. 每个数值阶段先通过 GRU 风格基础测试，再运行严格测试；两层都必须报告 MAE、MSE 和余弦相似度。
15. 通用能力通过单一模块/API 复用，减少重复代码；项目文档和代码注释使用中文，每阶段提交使用英文规范 commit。
16. 若后续实现 integer I/O，权重/bias 必须预量化或缓存，调用期间不临时量化 float master weight。
17. Golden 采用“权威 JSON + 机械生成 C++ fixture”：只提交规范 JSON、生成脚本、测试配置和阈值；生成 fixture 及所有测试/精度/性能报告均被 `.gitignore` 排除并在每次运行时重新生成。
18. Golden 使用分层核心覆盖：公共量化原语、单时间步融合 Cell、`T=3` 短递推；定向覆盖边界并对关键维度做最小成对组合，不做全量笛卡尔积。
19. Golden 规范按 `primitive/cell/recurrent` 分类，每个用例使用一个自包含 JSON；用例不得引用共享数据文件，确保单文件即可审查、复现和定位失败。
20. Golden 使用单一版本化 JSON schema，以 `kind` 判别三类严格 payload；拒绝未知字段，生成器直接消费该 schema，不复制字段校验规则。
21. 不实现 Python 量化 oracle；CPU int32 和 CPU 标量 FP 载体实现分别作为 reference model 生成 Golden，并将反量化结果与 `torch.nn.LSTM` 做端到端 MAE、MSE 和余弦相似度验证。Golden 更新必须显式执行、单独审核，常规测试不得自动刷新。
22. Golden 中的 standard scale 和其他 `float32` 非整数量使用最短可往返规范十进制字符串并进行位级 round-trip 校验；整数域字段使用 JSON integer。Canonical JSON 不保存冗余的浮点位模式。
23. Golden tensor 统一使用显式 `dtype`、`shape` 和 row-major 一维 `data`；禁止嵌套数组，严格校验元素数量和类型范围。量化 q 值保持逻辑整数表示，运行时 carrier 由独立元数据声明。
24. 每个首版 Golden 文件只声明一个 `execution_model=common|cpu_int32|cpu_fp32` 并只保存一份 expected；载体相关语义必须拆分用例，CUDA 后端映射到对应 CPU reference model 验证。未来 LUT 使用 schema 新版本增加 `cpu_int32_lut`，不得改变原模型语义。
25. Golden expected 分为 `checkpoints` 与 `diagnostics`：前者覆盖所有真实量化边界，后者只保存公式定义的关键融合宽值；diagnostics 不获得量化配置，也不包含后端实现临时值。
26. 固定 Q31 Cell 融合使用三层验证：静态边界证明、缩小整数域穷举、固定 seed 的全范围随机/对抗测试。实际通过全部安全和精度门禁后，Q31 才从暂定方案升级为最终执行规格。
27. 基础测试与第一轮 `synthetic_numeric` 严格测试直接使用 GRU 初始门禁：非量化 `MSE<1e-5, MAE<0.003, cosine>=0.9999`；INT8 `MSE<1e-3, MAE<0.05, cosine>=0.999`；INT16 `MSE<1e-4, MAE<0.015, cosine>=0.999`。完整 LSTM 流程跑通后再依据报告人工审核调整，阈值不得自动更新。
28. 余弦相似度统一使用 FP64 L2 norm 和 `epsilon=1e-12`：双边近零记为 `N/A` 并依靠 exact/MAE/MSE，单边近零直接失败；规则由唯一指标模块和版本化 `metric_policy.json` 共享。
29. 严格测试的完整目标包含合成边界数据和代表性真实数据，两者分别使用互斥校准/验证 split 并独立验收；缺少真实数据时只能声明数值严格验证通过，不能声明模型级精度达标。
30. 阶段 3 暂不接入代表性真实数据，只冻结 scope 为 `synthetic_numeric` 的数值阈值；真实数据与模型级阈值后续单独接入，不阻塞实现阶段，但阻塞模型级和生产精度声明。
31. 首轮 8/16-bit 混合配置按真实量化点的最小有效位宽选择阈值 profile：存在 8 bit 使用 INT8，否则最小位宽不低于 16 bit 使用 INT16；执行参数、累加器和 diagnostics 不参与判断，其他位宽没有显式映射时 fail fast。
32. 测试采用七个固定具名 shape profile：CPU/CUDA 各自基础 profile，四个严格正确性 profile 和一个 CUDA large-batch profile；配置维度使用定向/最小成对组合，不做全量笛卡尔积。
33. 基础测试固定 `seed=0`；严格校准固定 `{1001,1002,1003}`，严格验证固定 `{2001,2002,2003,2004,2005}`。这些 seed 只生成输入和 `h_0/c_0`，同一 case 的 weight/bias 跨 split 保持一致；定向对抗用例不依赖随机 seed。
34. Weight/bias parameter seed 按七个 shape profile 固定为 `3001..3007`；同一 profile 跨 bitwidth、scale mode、granularity、execution model 和 split 复用同一浮点 master 参数。显式对抗参数单独标记，不覆盖该规则。
35. 合成数据使用唯一 C++ `pcg32-xsh-rr-v1` 模块和固定整数到 float32/类正态映射，统一在 CPU 生成；禁止标准库 distribution、Python/PyTorch RNG 和 CUDA 独立生成。RNG 变更必须升级版本并单独审核。
36. PCG32 按 `(seed,stream_id)` 初始化，固定 `input/h0/c0=1/2/3`、`weight_ih/weight_hh=10/11`、`bias_ih/bias_hh=12/13`；公共 `rng_stream_registry=v1` 是唯一映射，张量生成顺序变化不得影响其他 stream。
37. 普通合成数据使用 `pytorch_typical_v1`：weight/bias 为 `U(-1/sqrt(H),1/sqrt(H))`，input/`h_0/c_0` 为 12-uniform 求和减 6 的类正态 float32；显式对抗数据独立覆盖数值边界。
38. `h_0/c_0` 使用五个 state profile：省略、显式零、普通随机、双状态量化边界、`h_0` 随机且 `c_0` 边界；采用定向/成对覆盖，省略与显式零必须结果一致。
39. `bias=True` 与 `bias=False` 都必须测试：分别使用 `bias_profile=enabled_random|disabled`，进入 CPU/CUDA 基础测试并在严格矩阵中覆盖两种位宽和 scale 模式；disabled 路径不得创建、量化或读取 bias。
40. `batch_first=False/True` 都进入基础和严格正确性测试；使用同一逻辑数据，内部统一 time-major，归一化后的量化参数/checkpoints 和 `h_n/c_n` 必须一致，公开 output shape 按 PyTorch 语义变化。
41. 四组 weight/bias granularity 独立配置；基础测试四者均为 `per_channel`，严格测试用成对组合覆盖三种粒度及分支/weight-bias 混合配置。每个启用参数在 finalize 后都物化为 `4H` channel 向量，kernel 不做运行时广播；激活/状态保持 per-tensor，bias disabled 时粒度为 `not_applicable`。
42. Weight/bias 量化参数导入导出始终使用完整 `4H` standard scale/zp 向量和 granularity 元数据；禁止 compact 或双表示。导入验证 per-tensor/per-gate 重复模式，bias disabled 时字段必须缺失。
43. 四组 weight/bias 强制 signed symmetric，所有 `4H` zero-point 元素为 0；`is_symmetric=false`、`is_unsigned=true` 或非零 zp 均直接报错。激活、Linear 和状态的对称性仍独立配置。
44. 所有 signed symmetric 量化点采用严格对称范围：`qmax=2^(bitwidth-1)-1`、`qmin=-qmax`，非退化校准范围使用 `scale=max(abs(r_min),abs(r_max))/qmax` 和 `zero_point=0`。二进制补码最小负值保持未使用，schema、参数导入、Clamp 和边界测试均执行该约束。
45. 基础 profile 使用域感知默认值：`i/f/o` 三个 sigmoid gate output 为 unsigned symmetric，其他非参数量化点为 signed symmetric。除强制规则固定的 weight/bias 外，所有真实量化点仍像 GRU 一样通过 JSON 独立配置 `bitwidth/is_unsigned/is_symmetric`；JSON 显式值覆盖默认值并必须被 forward 真实消费。
46. unsigned 使用完整 `[0,2^bitwidth-1]` 范围。unsigned symmetric 表示 zero-anchored：`scale=max(r_max,0)/qmax,zp=0`，负值正常 Clamp 到 0；unsigned asymmetric 使用包含实数零的标准 MinMax affine scale/zp，zero point 由统一银行家舍入后 Clamp 到 unsigned 范围，不强制非零。
47. 通用 Affine rescale 采用 GRU 风格的 16-bit 规范化 M+shift：`M` 为 `[32768,65535]` 的 `uint16_t`，`shift` 为 `int8_t`，编码使用统一银行家舍入；`M_raw=65536` 时进位规范化。不可表示或载体不安全的 ratio 在 setup 失败，不下溢、不饱和、不使用 raw ratio 回退。Cell 双比例继续使用独立 Q31。
48. POT2 standard scale 固定使用 `CoverRange` 和 `tolerance=0.02`：校准范围接近 2 的幂时对指数执行统一银行家舍入，否则 floor 以覆盖范围；严格比较 `<0.02`。策略和容差不允许 JSON 配置，相关字段按 unknown field 拒绝；转换后重算 asymmetric zp。
49. 五个整数 PWL LUT 作为条件性纯整数阶段待办。首版 CPU int32 carrier reference 在激活点反量化后调用公共原始 sigmoid/tanh，再量化回 int32 网格，因此不宣称端到端纯整数。未来 LUT 必须使用独立 `cpu_int32_lut` execution model、Golden 和精度门禁接入，不能静默替换首版语义。
50. 常量和全零范围采用 GRU/AIMET minimum-scale fallback：`S_min=min(FLT_EPSILON,0.01/(qmax-qmin))`。候选 scale 小于该值时按量化模式确定性扩展范围；该规则固定、不可通过 JSON 配置，并在校准报告中逐组记录。
51. Signed asymmetric 使用完整二进制补码范围：INT8 `[-128,127]`、INT16 `[-32768,32767]`，并使用包含实数零的标准 MinMax affine scale/zp。最小负值只在 asymmetric 模式合法；signed symmetric 仍使用严格对称范围并拒绝该值。
52. Operator JSON 分为版本化稀疏 override 和完整 canonical resolved 两种严格 schema。唯一 C++ resolver 按字段应用入库默认 profile；所有 forward、Golden 和报告只消费 resolved config。未知/重复/null/非法字段失败，resolved round-trip 必须字节稳定。
53. 首版所有真实量化点只支持 8/16 bit，允许混合配置；其他 bitwidth 在 schema/resolver 阶段直接失败。内部 carrier、累加器、M+shift 和 Q31 宽度不是 operator bitwidth。未来扩展必须升级 schema 并先补齐安全证明、Golden、严格测试和阈值。
54. 严格测试使用入库的显式 `strict_matrix_v1.json`：约束 pairwise 覆盖主要维度并追加不可删除的高风险定向 case。覆盖检查器只验证、不生成或改写 case；矩阵变更作为测试契约单独审核。基础测试不经过该矩阵。
55. 阶段 4 不设置硬编码延迟或加速比，并强制证明 cuBLAS SGEMM 生效；阶段 9 已基于两次稳定实测为 RTX 6000D/CUDA 13.2/cuBLAS 13.4 的四个具名 profile 人工冻结 P50/P95/吞吐阈值。检查器只读，环境不匹配直接失败。

## 9. Git 提交规范与预期顺序

### 9.1 提交规则

- 我会在每个阶段完成实现和验证后执行 Git commit，不把多个阶段堆在一个提交中。
- commit subject 和 body 全部使用英文，采用 Conventional Commits：`docs`、`feat`、`test`、`fix`、`refactor`、`perf`、`build`、`chore`。
- subject 使用祈使语气，建议不超过 72 个字符；scope 使用稳定模块名，例如 `quantization`、`reference`、`cuda`、`calibration`、`pytorch`。
- 一个 commit 只表达一个逻辑变更。实现、测试和直接相关文档可以同提交；阈值调整、无关格式化和生成产物不得混入。
- commit 前固定执行：检查 diff、格式化、相关单元测试、阶段精度矩阵、性能基准和 `git diff --check`。
- 不提交编译产物、临时数据、生成 fixture 或任何运行报告。只版本化权威 golden JSON、生成脚本、测试配置、seed、指标定义和审核通过的阈值；这些测试契约的变更必须与实现变更分开审查。
- 每次提交后向审核者提供 commit hash、英文 subject、测试命令、关键结果和未完成项。

### 9.2 预期提交顺序

```text
docs: freeze dual-carrier LSTM execution semantics
build: add modular LSTM project skeleton
feat: add floating-point LSTM reference
feat(quantization): add carrier-specific quantization primitives
test(reference): add integer and float-carrier reference models
feat(cuda): add cuBLAS float-carrier quantized forward
feat(calibration): add LSTM calibration and parameter generation
feat(pytorch): add QuantLSTM float-carrier inference
feat: add bidirectional and CPU-only reference support
feat(qat): add float-carrier quantization-aware backward
feat(onnx): add standard LSTM export
perf(cuda): optimize the float-carrier execution path
feat(quantization): add optional integer activation LUTs
feat(integer): add optional CUDA integer execution
```
test(cuda): freeze device-specific performance thresholds

最后两条 LUT/CUDA integer 提交都是条件性的，不属于首个生产里程碑；LUT 提交及其独立精度门禁必须先于 CUDA integer 提交。阶段内部按依赖顺序拆分，最后一个 commit 前必须完成该阶段全部验收，不能用“后续补测试”结束阶段。
