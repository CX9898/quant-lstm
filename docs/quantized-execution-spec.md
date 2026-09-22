# LSTM 量化执行规格

> 状态：阶段 9 已完成标准 ONNX 导出、CUDA 静态参数缓存和版本化性能门禁；Speech Commands v0.02 全量真实网络精度门禁已接入
> 参考基线：`/home/chengxing.zou/projects/quant-gru`，commit `9c25d14`
> 公式推导：`docs/lstm-quantization-formula-derivation.md`
> 分阶段计划：`docs/implementation-plan.md`

## 1. 适用范围

首个生产路径是 CUDA FP32 载体量化前向：量化值经过 Round/Clamp 并位于整数网格，但用 `float` 保存，Linear 使用 cuBLAS SGEMM。

同时实现两套 CPU reference：

- CPU 标量 FP 载体 reference：作为 CUDA FP 主路径的数值规范。
- CPU int32 载体 reference：GEMM、rescale 和融合状态更新使用整数算术；首版激活边界反量化后调用原始 sigmoid/tanh，再量化回 int32 网格。

首版 CPU int32 reference 不是端到端纯整数实现。三个 sigmoid 和两个 tanh 的整数 PWL LUT 是阶段 10 待办，未来必须使用独立 `cpu_int32_lut` execution model、Golden 和精度门禁接入。

当前支持单层单向/双向 LSTM、`bias=True/False` 和 `batch_first=True/False`。内部统一使用 time-major 和 PyTorch 门顺序 `(i,f,g,o)`。

## 2. LSTM 数学语义

单时间步定义为：

```text
i_t = sigmoid(W_ii*x_t + b_ii + W_hi*h_(t-1) + b_hi)
f_t = sigmoid(W_if*x_t + b_if + W_hf*h_(t-1) + b_hf)
g_t = tanh   (W_ig*x_t + b_ig + W_hg*h_(t-1) + b_hg)
o_t = sigmoid(W_io*x_t + b_io + W_ho*h_(t-1) + b_ho)
c_t = f_t*c_(t-1) + i_t*g_t
h_t = o_t*tanh(c_t)
```

ONNX 边界若使用 `(i,o,f,c)`，必须显式重排，不能改变内部顺序。

## 3. 真实量化点

JSON、C++、Python 和报告只允许以下真实量化点：

| 类别 | 名称 |
|---|---|
| 输入/状态 | `input`、`output`、`cell_state` |
| 参数 | `weight_ih`、`weight_hh`、`bias_ih`、`bias_hh` |
| Linear | `weight_ih_linear`、`weight_hh_linear` |
| 门输入/输出 | `input_gate_input/output`、`forget_gate_input/output`、`cell_gate_input/output`、`output_gate_input/output` |
| Cell 激活 | `cell_tanh_output` |

`f*c_old`、`i*g`、`o*tanh(c)` 及其 contribution 是融合公式内部宽值，不拥有独立 bitwidth、scale、zero point 或 Clamp，也不得出现在 operator JSON 中。

## 4. 配置契约

### 4.1 Override 与 resolved config

配置采用两种版本化 JSON：

- `lstm_quant_override.schema.json`：用户稀疏覆盖配置。
- `lstm_quant_resolved.schema.json`：完整 canonical 执行配置。

唯一 C++ resolver 按字段执行：

```text
lstm_quant_default_v1.json <- user override -> resolved config
```

Python、CPU、CUDA、Golden 和报告只消费 resolved config，不得各自补默认值。Unknown operator/字段、重复 key、`null`、错误类型、非法枚举、遗留 `mul_*`、`pot_scale_method` 和 `pot_scale_tolerance` 均 fail fast。Resolved config 二次解析和序列化必须字节一致。

首版 operator `bitwidth` 只允许 `8` 或 `16`，允许不同量化点混合使用。其他位宽在 resolver 阶段失败。载体、累加器、M+shift 和 Q31 的内部宽度不属于 operator bitwidth。

### 4.2 默认 profile

默认 `scale_mode=affine`，所有真实量化点默认 8 bit。四组 weight/bias 默认 `per_channel`。

| 量化点 | `is_unsigned` | `is_symmetric` |
|---|---:|---:|
| 三个 sigmoid gate output | `true` | `true` |
| 其他非参数量化点 | `false` | `true` |
| 四组 weight/bias | `false` | `true`，不可覆盖为 false |

除 weight/bias 固定约束外，非参数量化点可通过 override 独立配置 `bitwidth/is_unsigned/is_symmetric`。激活和状态 granularity 固定为 `per_tensor`。

### 4.3 参数粒度

`weight_ih`、`weight_hh`、`bias_ih`、`bias_hh` 分别配置 `per_tensor|per_gate|per_channel`。Finalization 后每个启用参数都只保留长度 `4H` 的 scale/zp 向量：

- per-tensor：同一值复制到全部 `4H` channel。
- per-gate：按 `(i,f,g,o)` 将四个值各复制 `H` 次。
- per-channel：保留 `4H` 个独立值。

Kernel 不得根据 granularity 动态广播。参数导入导出也只使用完整 `4H` 向量；compact/expanded 双表示非法。`bias=False` 时 bias 参数、量化参数和导入导出字段都必须缺失。

### 4.4 公共参数交换格式

PyTorch 公共导入导出使用 GRU-compatible v3 文档。与 GRU 共有的模型字段放在
`model_info`，量化点放在 `operators`，双向反向量化点放在
`operators_reverse`。LSTM 可额外携带 `schema_version` 和
`execution_metadata`，但共有信息不得改名或改变 JSON 类型。

每个 operator 固定使用 `dtype`、`symmetric`、`scale`、`zero_point`、
`enc_type`、`real_min` 和 `real_max`。单组值使用 JSON number/integer，多组值使用
同类型数组；参数算子的多组值仍按 4.3 节完整展开为 `4H`。`dtype` 为
`INT8|UINT8|INT16|UINT16`，`enc_type` 为
`PER_TENSOR|PER_GATE|PER_CHANNEL`。

公共 `scale/real_min/real_max` 是 JSON number。Python 边界 adapter 仅负责将公共
表示转换为私有 canonical C++ bundle；bundle 中 scale 的最短可往返 FP32 字符串
继续用于位模式稳定审计。执行配置解析、参数合法性判断和执行编码派生仍由 C++
完成。私有 bundle 不构成另一种公共交换格式。

## 5. 量化范围与校准

统一表示：

```text
a_hat = (q_a - Z_a) * S_a
d_a   = q_a - Z_a
```

所有 scale 必须是有限正数，zero point 必须位于对应 q 范围。

### 5.1 Signed symmetric

```text
qmax = 2^(b-1) - 1
qmin = -qmax
S = max(abs(r_min), abs(r_max)) / qmax
Z = 0
```

INT8 为 `[-127,127]`，INT16 为 `[-32767,32767]`。二进制补码最小负值在该模式非法。四组 weight/bias 强制使用此模式，所有 `4H` zero point 均为 0。

### 5.2 Signed asymmetric

```text
qmin = -2^(b-1)
qmax =  2^(b-1) - 1
N = qmax - qmin
r_lo = min(r_min, 0)
r_hi = max(r_max, 0)
S = (r_hi - r_lo) / N
Z = Clamp(RoundToNearestEven(qmin - r_lo/S), qmin, qmax)
```

INT8 的 `-128` 和 INT16 的 `-32768` 在该模式合法。

### 5.3 Unsigned

```text
qmin = 0
qmax = 2^b - 1
```

Unsigned symmetric 使用 zero-anchored 语义：

```text
S = max(r_max, 0) / qmax
Z = 0
```

负值在真实量化边界 Clamp 到 0。Unsigned asymmetric 使用包含实数零的标准 MinMax affine 公式；计算得到的 zero point 可以合法地等于 0。

### 5.4 退化范围 fallback

```text
N = qmax - qmin
S_min = min(2^-23, 0.01/N)
```

候选 scale 小于 `S_min` 时：

- signed symmetric 围绕零扩展到 `[-qmax*S_min, qmax*S_min]`。
- unsigned symmetric 扩展到 `[0, qmax*S_min]`。
- asymmetric 保留包含零后的下界，并将上界设为 `lower+N*S_min`。

Fallback 固定且不可配置。报告必须逐校准组记录原始/调整范围、`S_min` 和是否触发。

## 6. Standard Scale 与执行编码

### 6.1 Affine

Affine 校准得到的连续 FP32 scale 是唯一 standard scale。外部导入导出、quant/dequant 和真实激活均使用该值。只有算术 rescale ratio 转换为内部 M+shift。

### 6.2 POT2

POT2 固定使用 `CoverRange` 和 `tolerance=0.02`，不提供 JSON 策略字段：

```text
real_range = abs(r_max-r_min)
near_pot = relative_distance(real_range, nearest_power_of_two) < 0.02
n = -log2(S_calibrated)
n_pot = near_pot ? RoundToNearestEven(n) : floor(n)
S_std = 2^(-n_pot)
```

相对误差恰好为 `0.02` 时进入 floor 分支。`n_pot` 使用受检 `int8_t`。非对称 zero point 按 POT2 standard scale 重新计算。转换后的 scale 才是外部 standard scale。

### 6.3 通用 M+shift

普通 Affine rescale 固定使用 16-bit 规范化编码：

```text
ratio = mantissa * 2^exponent, mantissa in [0.5,1)
M_raw = RoundToNearestEven(mantissa * 2^16)

M_raw == 65536 时：M=32768, exponent+=1
否则：M=M_raw

shift = 16-exponent
ratio_exec = M * 2^(-shift)
```

`M` 为 `[32768,65535]` 的 `uint16_t`，`shift` 为 `int8_t`。编码使用 FP64 中间计算。零、负数、NaN、Inf、shift 越界或载体不安全均在 setup 失败；禁止下溢为 0、饱和、raw-ratio 回退或自动切换后端。

## 7. 统一舍入与数值安全

所有量化相关舍入必须调用 `roundToNearestEven` 公共接口，包括 quant、zero point、M+shift/Q31 编码、整数移位、FP rescale 和真实激活输出量化。业务代码禁止直接调用平台 round 函数或复制“加半后移位”。

负数右移通过无符号幅值或更宽类型实现。负 shift 使用受检乘法表达左移，禁止左移负有符号整数、对最小负值取负或产生有符号溢出。

Setup 阶段统一生成 `NumericSafetyReport`：

- int64/`__int128` 乘加、左移或累加无法证明安全时 fail fast。
- FP32 整数运算上界小于 `2^24` 时标记 `exact_integer_range`。
- 超过 `2^24` 但仍有限时标记 `precision_risk`，必须通过严格指标。
- `require_exact_accumulation=true` 时，超过 `2^24` 直接失败。
- 可能产生 Inf/NaN 时无条件失败。

除真实量化边界外，不允许中间隐藏 Clamp、饱和或 wrap。

## 8. 融合前向公式

定义：

```text
R(v, S_src -> S_dst) = RoundToNearestEven(v*S_src/S_dst)
Q_y(v, S_src) = Clamp(R(v, S_src -> S_y)+Z_y, qmin_y, qmax_y)
```

实际执行时 `R` 使用已编码 M+shift/POT2 ratio。

### 8.1 Linear 与门输入

Weight/bias 的 zero point 固定为 0。对输出 channel `c`：

```text
acc_x[c] = sum_k(q_W[k,c] * (q_x[k]-Z_x))
acc_b[c] = R(q_bias[c], S_bias[c] -> S_W[c]*S_x)
q_ih[c] = Q_ih(acc_x[c]+acc_b[c], S_W[c]*S_x)
```

Recurrent Linear 同理。四个门输入分别将两路 Linear 的中心化 q 值 rescale 到各自门输入网格，在宽域相加后只在门输入边界 Clamp。

### 8.2 真实激活

三个 sigmoid 和两个 tanh 首版统一执行：

```text
x_real = (q_in-Z_in)*S_in
y_real = sigmoid(x_real) 或 tanh(x_real)
q_out = Clamp(RoundToNearestEven(y_real/S_out)+Z_out)
```

FP 和 int32 载体 reference 必须复用同一个真实激活边界实现。

### 8.3 Cell 双比例延迟舍入

```text
p_forget = (q_f-Z_f)*(q_c_old-Z_c)
p_input  = (q_i-Z_i)*(q_g-Z_g)

alpha = S_f*S_c_old/S_c_new
beta  = S_i*S_g/S_c_new

q_c_new = Clamp(
    RoundToNearestEven(p_forget*alpha + p_input*beta) + Z_c_new)
```

两路乘积没有独立量化边界。CPU int32 reference 将 `alpha/beta` 编码为固定 Q31 multiplier，使用 `int64_t` multiplier 和 `__int128` 合并累加，最后只 RoundShift 一次。阶段 3 的静态 fail-fast、缩小域穷举和 8/16-bit Affine/POT2 随机/对抗验证均已通过，因此该编码现已冻结为最终 CPU reference 规格。

### 8.4 Hidden 融合

```text
q_tanh_c = QuantizedRealTanh(q_c_new)

q_h_new = Clamp(
    R((q_o-Z_o)*(q_tanh_c-Z_tanh_c),
      S_o*S_tanh_c -> S_output) + Z_output)
```

最终乘法不创建 `mul_output_cell` 量化点。`h_0` 与所有 `h_t` 共享 output 网格；`c_0` 与所有 `c_t` 共享 cell-state 网格。

## 9. 载体执行契约

### 9.1 CPU FP 载体 reference

量化 q 值用 FP32 保存；普通 rescale 和 Cell Q31 ratio 都以浮点算术模拟已编码参数。它是 CUDA FP 路径的 reference。

### 9.2 CUDA FP 载体主路径

- 全部时间步的 `W*x` 合并为一次 cuBLAS SGEMM。
- 每个时间步的 `R*h` 使用一次 SGEMM。
- 非对称输入通过 `sum(W)*Z_x`、`sum(R)*Z_h` 修正。
- Bias/rescale、四门真实激活、Cell 和 Hidden 更新在融合 pointwise kernel 中完成。
- 正确性模式关闭 TF32/Tensor Core；性能模式显式开启并单独报告精度。

- 调用方可提供持久 buffer 和非零 generation key，缓存量化 W/R/bias 与 weight sums；master 参数内容变化时必须更换 key，key 为 0 时保持逐调用量化。
- 缓存命中不得改变量化点、融合公式、checkpoint 或输出；持久缓存字节数与临时 workspace 分别报告。
- host 签名与 hit 判定属于 setup，不计入 CUDA event 的 quantize/core/dequantize 设备计时范围。
### 9.3 CPU int32 载体 reference

GEMM/普通乘积使用 int64，Cell Q31 合并使用 `__int128`。普通 rescale 执行整数 multiplier+shift/POT2 shift。首版真实激活桥接包含浮点函数，因此只冻结整数算术与融合公式，不代表完整硬件 LUT 语义。

CPU FP32 与 int32 实现只作为显式 C++ reference model。PyTorch `QuantLSTM` 的
完全浮点、量化和 QAT 执行路径均只接受 CUDA tensor，不得按设备自动回退到 CPU
reference。

### 9.4 FP32 q-carrier QAT backward

训练态保存量化后的 input、W/R、可选 bias、h0/c0，以及 7 类真实 checkpoint：
两路 Linear、四门 input/output、Cell、`tanh(Cell)` 和 Hidden。Backward 将 q 值按
standard scale/zp 反量化到 real domain，并复用已验证的浮点 LSTM backward。

每个真实量化边界的 Clamp mask 中 `1` 表示发生 Clamp；STE 只在这些位置把梯度置零，
未 Clamp 的 Round 使用恒等梯度。Cell 两路 contribution、Hidden 原始乘积及其他融合
临时值没有独立 mask。双向模块分别执行两个单向 backward，再由时间翻转和拼接算子
恢复 PyTorch 的 input、h0/c0 与参数梯度顺序。本节不改变第 8 节冻结的前向公式。

## 10. Golden 与验证

Canonical Golden JSON 按 `primitive/cell/recurrent` 分类，一个文件一个自包含用例。每个文件只包含一个 `execution_model=common|cpu_int32|cpu_fp32` 和一份 expected；未来 LUT 通过 schema 新版本增加 `cpu_int32_lut`。

`expected` 分为：

- `checkpoints`：全部真实 Round/Clamp 边界。
- `diagnostics`：冻结公式中的宽值，例如两路 Cell 乘积、缩放 contribution、最终舍入前宽和及 Hidden 原始乘积。

Golden 使用显式 `dtype/shape/data`、row-major 一维 data。Standard scale 使用最短可往返 FP32 的规范十进制字符串；逻辑整数使用 JSON integer。权威 JSON、schema、生成器、阈值和测试配置入库；生成 fixture、测试数据、日志及精度/性能报告不入库。

基础测试先运行，随后才运行严格测试。严格测试使用入库的 `strict_matrix_v1.json`，采用约束 pairwise 和强制定向 case；覆盖检查器只验证，不生成或改写矩阵。

所有阶段至少报告 `output/h_n/c_n` 的 MAE、MSE 和余弦相似度。余弦使用 FP64 L2 norm 和 `epsilon=1e-12`，非退化张量必须 `>=0.999`。初始门禁为：

| 类别 | MSE | MAE | 余弦相似度 |
|---|---:|---:|---:|
| 非量化 | `<1e-5` | `<0.003` | `>=0.9999` |
| INT8 | `<1e-3` | `<0.05` | `>=0.999` |
| INT16 | `<1e-4` | `<0.015` | `>=0.999` |

混合位宽只要存在 8-bit 真实量化点就使用 INT8 profile，否则使用 INT16 profile。阈值只能依据固定数据报告人工审核修改，测试不得自动更新。

## 11. 性能验收

阶段 4 不设置绝对延迟或固定加速比，但必须：

- Profiler 证明 cuBLAS SGEMM 实际生效。
- 报告 P50/P95 延迟、吞吐、workspace 和量化开销。
- 记录 GPU、驱动、CUDA/cuBLAS、时钟/功耗模式、shape、math mode、计时范围、预热/迭代次数和同步方法。
- 性能比较同时报告 MAE、MSE 和余弦相似度。

缺少上述任一项时阶段 4 验收失败。阶段 9 已在两次独立稳定测量后，将 RTX 6000D/CUDA 13.2/cuBLAS 13.4 的四个具名 profile 冻结到 `tests/benchmarks/config/cuda_performance_thresholds_v1.json`。阈值由 `tools/check_stage9_cuda_performance.py` 只读检查；禁止自动更新，环境不匹配时禁止复用绝对数值。完整证据与命令见 `docs/cuda-performance.md`。

## 12. 实现证据状态

以下不是未决设计，而是分阶段验收证据：

1. 已完成：Q31 Cell 静态上界证明、缩小域穷举和全范围随机/对抗报告。
2. 待后续真实数据阶段完成：两套 CPU reference 相对 `torch.nn.LSTM` 的实测 MAE、MSE、余弦相似度及逐时间步最差值。
3. 已完成：CUDA FP32 `exact_integer_range/precision_risk` 逐量化点结果及每 operator/channel `NumericSafetyReport`。
4. 已完成：compute-sanitizer memcheck/racecheck、Nsight GEMM kernel 计数和可复现 Pedantic/TF32 结构化性能基线。
5. 已完成：正式 FP32 checkpoint 的 18 点 MinMax/直方图收集、Empty/Dirty/Locked 生命周期、逐组 fallback 报告，以及完整 `4H` 参数包 canonical round-trip。
6. 已完成：SQNR/Percentile 独立候选范围搜索复用统一 MinMax、minimum-scale、POT2 CoverRange 和执行参数派生链；参数包导入后 CUDA FP 主路径结果逐值一致。
7. 已完成：PyTorch 接口只通过 C++ resolver 消费 canonical resolved config；校准、完整 `4H` 参数包导入导出、CUDA 直接调用、两种布局和真实 Clamp mask 已通过阶段 6 验收。
8. 已完成：双向 forward/reverse 分别校准并强制共享 input 网格，输出与 `h_n/c_n` 顺序对齐 PyTorch；CPU-only 构建、测试、安装、外部消费和无 CUDA 链接门禁通过。
9. 已完成：CUDA 浮点 backward 对齐 PyTorch；训练态 CUDA forward 原生输出实际
   消费的 master q-carrier/checkpoint/Clamp mask，QAT 通过 CUDA 反量化与
   mask-aware backward 完成 gradient、h0/c0、bias disabled、双向、Clamp STE、
   单步优化和多步 loss 下降验收。生产 Python 仅负责扩展调度并拒绝 CPU 执行；
   CPU FP/int32 实现仅作为 C++ reference model。
10. 已完成：标准 ONNX `LSTM` 单节点导出；量化静态参数缓存保持 Golden 与精度指标不变，P50/P95 获得稳定收益，memcheck/racecheck、Nsight SGEMM 计数和版本化设备阈值通过。
11. 已完成：Google Research `kws_streaming` LSTM 拓扑的 Speech Commands
    v0.02 全量真实网络门禁。官方 split 的 105,829 条带标签语音全部纳入
    35 类训练/验证/测试，分别比较 `torch.nn.LSTM`、native CUDA FP32、INT8
    QAT 和 INT16 QAT，并冻结任务准确率、macro/per-class F1、logit 误差、预测
    一致率、真实 batch CUDA backward 和校准安全门禁。该结果验证 CUDA 生产
    路径的模型级精度，不替代第 2 项尚待补充的 CPU reference 真实张量严格矩阵。

上述证据文件按次生成且不提交仓库；审核通过的 schema、配置、阈值和规则变更必须入库并单独审查。
