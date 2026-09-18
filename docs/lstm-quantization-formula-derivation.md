# LSTM 量化融合公式推导

> 状态：阶段 9 已通过标准 ONNX 导出与 CUDA 静态参数缓存性能验收；前向量化公式和 Cell 固定 Q31 整数编码保持冻结
> 参考实现：`/mnt/data2/chengxing.zou/projects/quant-gru`，commit `9c25d14`
> 待完成证据：真实数据 LSTM 精度与模型级门禁；当前没有待审核的数学设计项

## 1. 文档目的与边界

本文从浮点 LSTM 定义出发，推导量化 Linear、四门、Cell 更新和 Hidden 更新的完整公式，并与 `quant-gru` 当前优化后的融合公式逐项对照。

本文冻结的是“应该逼近哪个量化数学结果”，不提前冻结 FP32 载体和 int32 载体的具体指令实现。两种载体必须共享量化点、zero-point 语义和融合计算图，但允许使用不同的 rescale 和激活实现：

- FP32 载体：量化整数值保存在 `float` 中，GEMM 使用 cuBLAS SGEMM；rescale ratio 先编码为 M+shift/POT2，再用浮点算术模拟该执行比例；激活使用真实 sigmoid/tanh 后重新量化。
- int32 carrier reference：量化整数值保存在 `int32_t` 中，乘积和累加使用宽整数，rescale 使用 POT2 或 multiplier+shift；首版在激活边界反量化后调用原始 sigmoid/tanh，再按输出网格重新量化。整数 LUT 延后实现，因此首版不是端到端纯整数路径。

本文中的乘法临时值和宽累加值不是量化张量，不拥有独立 bitwidth、scale、zero point 或 clamp，也不进入 JSON 配置。

## 2. 统一量化记号

### 2.1 仿射量化

对任意实数张量 `a`，定义：

```text
a_hat = (q_a - Z_a) * S_a
d_a   = q_a - Z_a
```

其中：

- `q_a`：量化整数值；FP32 载体中类型为 `float`，但数值应位于整数网格。
- `S_a`：正数 scale。
- `Z_a`：zero point。
- `d_a`：中心化后的量化整数。
- `a_hat`：由量化值表示的实数近似。

### 2.2 Rescale 与最终量化

定义不带 zero point 和 clamp 的尺度转换：

```text
R(v, S_src -> S_dst) = Round(v * S_src / S_dst)
```

定义到目标张量 `y` 的完整量化边界：

```text
Q_y(v, S_src) = Clamp(R(v, S_src -> S_y) + Z_y, BW_y)
```

必须区分：

- `R` 是融合公式内部的数值对齐，不自动创建量化张量，也不自动 clamp。
- `Q` 才表示真实量化边界，执行 Round、加目标 zero point 和按目标位宽 Clamp。
- `Round` 统一采用第 2.5 节定义的 round-to-nearest-even，任何模块不得自行选择其他舍入函数。

### 2.3 Per-channel 参数

`weight_ih`、`weight_hh`、`bias_ih`、`bias_hh` 分别独立配置 per-tensor、per-gate 或 per-channel。无论配置粒度为何，finalize 后每个启用参数都物化为长度 `4H` 的 channel 参数向量，公式中的 `S_W[c]`、`S_b[c]` 始终直接表示输出 channel `c` 对应的 scale：per-tensor 将同一参数复制到全部 `4H` channel，per-gate 按 `(i,f,g,o)` 将四个参数各复制 `H` 次，per-channel 逐 channel 填充。执行公式不再根据 granularity 分支。

Granularity 元数据仍随参数导入导出，用于审计 `4H` 向量的来源，但不是运行时广播开关。不同 gate/channel 的 scale/zp 允许恰好相等，正确性由统计来源、索引映射和展开规则验证，不能用“数值是否不同”推断粒度。

外部参数文件同样只保存完整 `4H` standard scale/zp 向量：per-tensor 的全部元素位级相同，per-gate 的每个长度 `H` 门段位级相同，per-channel 保留逐 channel 值。Granularity 元数据与完整向量共同导入导出，但不存在 1/4 元素 compact 数值副本；导入只验证映射不再广播。Bias disabled 时对应字段缺失。

权重和 bias 强制使用 signed symmetric 量化：

```text
qmax(b) = 2^(b-1) - 1
qmin(b) = -qmax(b)

Z_W[c] = 0
Z_b[c] = 0
W_hat[k,c] = q_W[k,c] * S_W[c]
b_hat[c] = q_b[c] * S_b[c]
```

所有配置为 signed symmetric 的真实量化点都使用严格对称整数范围 `[-qmax(b), qmax(b)]`，例如 INT8 为 `[-127,127]`、INT16 为 `[-32767,32767]`。二进制补码载体可表示的最小负值（INT8 的 `-128`、INT16 的 `-32768`）不属于有效量化范围，量化 Clamp、参数导入和 Golden/schema 校验均不得接受该值。对非退化校准范围 `[r_min,r_max]`：

```text
S_symmetric = max(abs(r_min), abs(r_max)) / qmax(b)
Z_symmetric = 0
```

常量或全零范围执行第 2.3.1 节的固定 minimum-scale fallback，不能通过使用最小负值改变上述严格对称范围。四组 weight/bias 的外部完整 `4H` zero-point 向量因此全部为 0；非对称、unsigned 或非零 zp 配置无效。输入、状态、Linear 输出和门激活仍可按各自配置使用非零 zero point；其中任何量化点一旦配置为 signed symmetric，也必须遵循相同的严格对称范围和 scale 公式。

signed asymmetric 使用完整二进制补码范围，最小负值在该模式下是合法量化值：

```text
qmin_signed_asymmetric(b) = -2^(b-1)
qmax_signed_asymmetric(b) =  2^(b-1) - 1
N = qmax - qmin = 2^b - 1

r_lo = min(r_min, 0)
r_hi = max(r_max, 0)
S_signed_asymmetric = (r_hi - r_lo) / N
Z_signed_asymmetric = Clamp(
    RoundToNearestEven(qmin - r_lo / S_signed_asymmetric),
    qmin,
    qmax)
```

因此 INT8 asymmetric 为 `[-128,127]`，INT16 asymmetric 为 `[-32768,32767]`；只有 signed symmetric 使用严格对称范围并禁用最小负值。退化范围先执行第 2.3.1 节，POT2 模式随后以转换后的 standard scale 重算 zero point。

基础 profile 沿用 GRU 的域感知设置：三个 sigmoid gate output（`input_gate_output`、`forget_gate_output`、`output_gate_output`）默认 `is_unsigned=true,is_symmetric=true`；`input`、`output`、`cell_state`、两路 Linear 输出、四个 gate input、`cell_gate_output` 和 `cell_tanh_output` 默认 `is_unsigned=false,is_symmetric=true`。

上述设置是 JSON 缺省值，不是执行公式中的硬编码。除 weight/bias 的强制 signed symmetric 约束外，每个真实量化点都独立读取 JSON 中的 `bitwidth`、`is_unsigned` 和 `is_symmetric`，并据此确定整数范围、校准 scale/zp 和最终 Clamp。显式配置必须覆盖基础 profile；配置字段不能仅被解析或导出而不参与 forward。

unsigned 量化统一使用完整整数范围：

```text
qmin_unsigned(b) = 0
qmax_unsigned(b) = 2^b - 1
```

`is_unsigned=true,is_symmetric=true` 沿用 GRU 的 zero-anchored 语义，而不是围绕零点两侧对称：

```text
r_hi = max(r_max, 0)
S_unsigned_symmetric = r_hi / qmax_unsigned(b)
Z_unsigned_symmetric = 0
```

常量/全零范围执行第 2.3.1 节的统一 fallback。负校准值不参与该 scale 的计算；实际量化时按 `Round(a/S)+Z` 计算并 Clamp 到 `[0,qmax]`，因此所有负值饱和为 0，不触发隐式改配或报错。精度报告必须包含饱和率，使不合适的 JSON unsigned 配置可被观察。

`is_unsigned=true,is_symmetric=false` 使用标准 MinMax affine 语义，并先保证实数零包含在校准范围内：

```text
r_lo = min(r_min, 0)
r_hi = max(r_max, 0)
S_unsigned_asymmetric = (r_hi - r_lo) / qmax_unsigned(b)
Z_unsigned_asymmetric = Clamp(
    RoundToNearestEven(-r_lo / S_unsigned_asymmetric),
    0,
    qmax_unsigned(b))
```

`is_symmetric=false` 表示 zero point 由 affine 公式计算，不要求最终值一定非零；当校准范围从零开始时，合法结果仍可为 `Z=0`。POT2 模式先按上述规则得到校准 scale，再执行第 2.4 节的 POT2 转换和相应 zero-point 重算。

#### 2.3.1 常量与全零范围 fallback

MinMax 对每个独立校准组（per-tensor、per-gate 或展开前的 per-channel）使用固定 minimum-scale 规则：

```text
N = qmax - qmin
epsilon_f32 = 2^-23
S_min = min(epsilon_f32, 0.01 / N)
```

`N` 必须大于 0，`S_min` 以 FP32 保存并要求有限正数。各模式先按正常公式得到候选 scale；仅当 `S_candidate < S_min` 时执行 fallback，恰好相等时不标记 fallback：

```text
signed symmetric:
    adjusted_min = -qmax_symmetric * S_min
    adjusted_max =  qmax_symmetric * S_min
    S_calibrated = S_min
    Z = 0

unsigned symmetric:
    adjusted_min = 0
    adjusted_max = qmax_unsigned * S_min
    S_calibrated = S_min
    Z = 0

signed/unsigned asymmetric:
    r_lo = min(r_min, 0)
    r_hi = max(r_max, 0)
    adjusted_min = r_lo
    adjusted_max = r_lo + N * S_min
    S_calibrated = S_min
    Z = Clamp(RoundToNearestEven(qmin - adjusted_min / S_min),
              qmin, qmax)
```

asymmetric fallback 的触发条件保证扩展后的 `adjusted_max` 仍覆盖原 `r_hi` 和实数零。不得用 `scale=0/1`、复制相邻 channel scale、跳过量化或要求外部补参等隐式替代策略。Affine 直接将 `S_calibrated` 作为 standard scale；POT2 再对该非退化范围执行固定 CoverRange 转换。

校准诊断必须逐量化点/组记录 `fallback_used`、原始 min/max、调整后 min/max、`N` 和 `S_min`。这属于每次运行生成的校准报告，不进入外部量化参数或 JSON 配置。

### 2.4 Standard scale 与内部执行比例

Affine 模式下，校准得到的连续 scale 是唯一 standard scale：

```text
S_std = S_calibrated
```

它用于定义量化网格、计算 zero point、权重/输入量化、输出反量化、真实激活和外部参数导入导出。运行时只对算术 rescale ratio 做 M+shift 编码：

```text
ratio_std = S_src / S_dst
            或 S_a * S_b / S_dst

(M, shift) = EncodeMShift(ratio_std)
ratio_exec = M * 2^(-shift)
```

通用 Affine rescale 固定使用 16-bit 规范化 M+shift 编码。`ratio_std` 必须是有限正数；编码计算使用 FP64，输入 standard scale 本身仍保持外部契约规定的 IEEE-754 `float32` 值：

```text
ratio_std = mantissa * 2^exponent,  mantissa in [0.5, 1)
M_raw     = RoundToNearestEven(mantissa * 2^16)

if M_raw == 65536:
    M        = 32768
    exponent = exponent + 1
else:
    M = M_raw

shift      = 16 - exponent
ratio_exec = M * 2^(-shift)
```

编码结果类型固定为：

```text
M     : uint16_t, [32768, 65535]
shift : int8_t,   [-128, 127]
```

`ratio_std<=0`、NaN、Inf、规范化后 `shift` 越界，以及执行时乘法/左移无法通过载体安全预检，均在 finalize/setup 阶段直接失败。禁止把过小比例下溢为 `M=0`、把过大比例或 shift 饱和到边界，也禁止自动改用 raw float ratio 或切换后端。负 shift 表示受检左移，必须先提升到普通整数再取相反数，不能在 `int8_t` 上对 `-128` 直接求负。

该 16-bit 编码用于 Linear、bias 和普通单比例/乘积比例 rescale；Cell 双比例延迟舍入仍使用第 7.5 节独立的固定 Q31 公共分母，不得用本编码分别舍入两路 contribution。

FP32 载体和 int32 reference 都消费同一组 `(M,shift)`：

```text
FP32 carrier: RoundToNearestEven(ldexp(float(value) * float(M), -shift))
int32 ref:    RoundToNearestEven(wide(integer(value) * M), shift)
```

FP32 与整数执行都必须调用统一舍入接口。整数路径先完成受检宽乘法，再执行安全的右移舍入或左移；FP32 路径固定上述运算顺序，并由 `NumericSafetyReport` 标记精确整数区间或 `precision_risk`。

`ratio_exec` 是内部执行近似，不是新的 tensor scale，不能覆盖 `S_std`。外部只把 standard scale 视为量化参数契约；M+shift 可以出现在调试/编译报告中，但不作为第二套外部量化网格。

Golden JSON 对 `S_std` 使用最短可往返到 IEEE-754 `float32` 的规范十进制字符串，并显式声明目标类型为 `float32`。加载后必须重新格式化并与原字符串完全一致，同时验证 `float32` 位模式 round-trip 不变。Canonical JSON 不再保存冗余的 `scale_bits`；M+shift 的 multiplier/shift、zero point 和所有量化值仍使用带目标类型范围检查的 JSON integer。

POT2 模式先将校准 scale 转换为幂次 scale，转换后的值才是 standard scale：

```text
tolerance = 0.02
real_range = abs(r_max - r_min)
range_exp = RoundToNearestEven(log2(real_range))
nearest_range_pot = 2^range_exp
is_near_pot = abs(real_range - nearest_range_pot) /
              nearest_range_pot < tolerance

n = -log2(S_calibrated)
if is_near_pot:
    n_pot = RoundToNearestEven(n)
else:
    n_pot = floor(n)

S_std = 2^(-n_pot)
```

这就是唯一生产策略 `CoverRange`。`tolerance` 固定为 `0.02`，严格使用 `<`；相对误差恰好等于 `0.02` 时进入 `floor` 分支。`n_pot` 必须可表示为 `int8_t`，无效/非有限 scale、range 或越界指数均在校准 finalize 阶段失败。常量/全零 range 仍由统一 fallback 先产生合法的非退化范围，再进入本转换。

POT2 策略和 tolerance 不进入 JSON，也不存在运行时 `Round/Floor/CoverRange` 分支。JSON 只选择 `scale_mode=pot2|affine`；`pot_scale_method`、`pot_scale_tolerance` 或等价字段均作为 unknown field 拒绝。内部只能保留一个 `convertScaleToPot2CoverRange` 公共入口，测试辅助函数不得变成生产配置接口。

转换后对称量化继续使用 `Z=0`；非对称量化以转换后的 `S_std` 重新计算。这里的
`r_lo` 是包含实数零并完成第 2.3.1 节 fallback 后的调整下界，不是原始观测
`r_min`：

```text
Z = Clamp(
    RoundToNearestEven(qmin - r_lo / S_std),
    qmin,
    qmax)
```

此后边界 quant/dequant、激活、导入导出均使用该 POT2 standard scale 和重算后的 zero point；算术 rescale ratio 通过 shift 执行。接近幂次范围时的 round 分支允许在固定 2% 容差内略微缩小覆盖范围，因此严格报告必须包含饱和率。

因此 LSTM 不复制 GRU 当前的混合行为：Affine 边界不能使用单个 tensor scale 经 `toFixedScale()` 后的 effective scale，同时又在 FP pointwise 中使用 continuous ratio。

### 2.5 统一银行家舍入接口

所有量化相关舍入统一采用 round-to-nearest-even：

```text
 1.5 ->  2
 2.5 ->  2
-1.5 -> -2
-2.5 -> -2
```

舍入能力集中在 `include/quantization/rounding.h`，通过同一语义接口族提供：

```text
roundToNearestEven(float value)
roundToNearestEven(double value)
roundToNearestEven(Integer value, int fractional_bits)
roundToInteger<Result>(Float value)
```

第三个重载表示整数定点值除以 `2^fractional_bits` 后的 ties-to-even 舍入，是 Q31、M+shift 和 POT2 右移的唯一入口。`roundToInteger` 复用浮点重载并执行受检类型转换。

以下位置必须调用该公共接口：

- 浮点输入、权重、bias 和状态量化。
- zero point 计算。
- M+shift/Q31 multiplier 编码。
- FP32 载体的 rescale。
- int32 reference 的 RoundShift。
- 真实 sigmoid/tanh 的输出重新量化。
- 后续整数 LUT 的参数、阈值和截距量化；首版不实现该模块。

除 `rounding.h` 的平台适配实现外，项目代码禁止直接调用 `rintf`、`rint`、`roundf`、`std::round`、`std::llround`，也禁止自行实现“加半后移位”。

负数定点右移必须使用无符号幅值或更宽类型处理，不能对 `INT_MIN` 直接取负。`fractional_bits<=0` 时使用受检乘法实现左移语义，不能左移负的有符号整数；非法 shift 和溢出不能产生未定义行为。失败策略遵循第 2.6 节。

### 2.6 数值安全预检与失败策略

对量化张量 `a`，定义中心化整数的最大绝对值：

```text
D_a = max(abs(qmin_a - Z_a), abs(qmax_a - Z_a))
```

finalize/setup 阶段必须在 kernel 启动前计算各路径的保守上界，例如：

```text
B_gemm = K * D_weight * D_input

B_forget = D_f * D_cell
B_input  = D_i * D_g

B_cell_q31 =
    B_forget * abs(M_forget)
  + B_input  * abs(M_input)
```

整数执行规则：

- 逐项证明 GEMM、bias 对齐、普通乘积、multiplier 乘法、左移和累加能被选定载体表示。
- int64 或 `__int128` 上界不安全时，finalize/setup 直接失败并报告算子、位宽、`K`、scale、上界和载体极限。
- 禁止有符号溢出、静默 wrap、中间自动 saturate 或自动切换计算载体。
- 只有本文定义的真实量化边界允许按目标位宽 Clamp。

FP32 载体按精度而不是整数溢出分类：

- 所有相关整数操作数、乘积和绝对累加上界 `<2^24` 时，标记为 `exact_integer_range`。
- 超过 `2^24` 但仍处于有限 FP32 范围时，标记为 `precision_risk`；普通模式允许执行，但必须通过 MAE、MSE、余弦相似度和逐时间步漂移门禁。
- `require_exact_accumulation=true` 时，任何相关上界超过 `2^24` 都在 setup 阶段失败。
- 可能产生 Inf/NaN 或超出 FP32 有限范围时，无论模式如何都直接失败。

安全分析集中在公共 `numeric_safety` 模块，生成可序列化的 `NumericSafetyReport`。正常 kernel 不执行逐元素安全分支，避免增加主路径开销。

## 3. LSTM 浮点语义

内部固定使用 PyTorch 的 `(input, forget, cell, output)`，即 `(i,f,g,o)` 门顺序：

```text
Lx_t = W_ih * x_t       + b_ih
Lh_t = W_hh * h_(t-1)  + b_hh

i_t = sigmoid(Lx_i + Lh_i)
f_t = sigmoid(Lx_f + Lh_f)
g_t = tanh   (Lx_g + Lh_g)
o_t = sigmoid(Lx_o + Lh_o)

c_t = f_t * c_(t-1) + i_t * g_t
h_t = o_t * tanh(c_t)
```

内部张量布局为：

```text
x:      [T, B, I]
W_ih:   [I, 4H]
W_hh:   [H, 4H]
Lx:     [T, B, 4H]
Lh:     [B, 4H]
h, c:   [T+1, B, H]
```

## 4. 有效量化点

以下 18 个张量发生真实 quantize/requantize 和 clamp，因此拥有独立量化参数：

| 类别 | 量化点 | 含义 |
|---|---|---|
| 输入与状态 | `input` | `x_t` |
| 输入与状态 | `output` | `h_t`，同时作为下一时间步的 `h_(t-1)` |
| 输入与状态 | `cell_state` | `c_t`，同时作为下一时间步的 `c_(t-1)` |
| 参数 | `weight_ih`, `weight_hh` | 四门输入权重和循环权重 |
| 参数 | `bias_ih`, `bias_hh` | 四门输入 bias 和循环 bias |
| Linear | `weight_ih_linear` | `W_ih*x+b_ih` |
| Linear | `weight_hh_linear` | `W_hh*h+b_hh` |
| 门输入 | `input_gate_input` | input gate 激活前输入 |
| 门输入 | `forget_gate_input` | forget gate 激活前输入 |
| 门输入 | `cell_gate_input` | cell gate 激活前输入 |
| 门输入 | `output_gate_input` | output gate 激活前输入 |
| 门输出 | `input_gate_output` | `i_t` |
| 门输出 | `forget_gate_output` | `f_t` |
| 门输出 | `cell_gate_output` | `g_t` |
| 门输出 | `output_gate_output` | `o_t` |
| Cell 激活 | `cell_tanh_output` | `tanh(c_t)` |

以下值只是融合内部临时值，不是量化点：

```text
f_t * c_(t-1)
i_t * g_t
o_t * tanh(c_t)
Cell 两路 contribution 的公共累加值
```

四个门输出保持独立量化点。已确认的 Cell 双比例延迟舍入方案不要求 `input_gate_output` 与 `forget_gate_output` 共享量化网格。

## 5. Linear 融合推导

### 5.1 输入 Linear

对输出 channel `c`：

```text
Lx[c] = sum_k(W_ih[k,c] * x[k]) + b_ih[c]
```

代入量化表示：

```text
W_ih[k,c] ~= q_W[k,c] * S_W[c]
x[k]      ~= d_x[k] * S_x
b_ih[c]   ~= q_b[c] * S_b[c]   # Z_b[c]=0，因此 d_b[c]=q_b[c]
```

GEMM 的中心化整数累加为：

```text
acc_x[c] = sum_k(q_W[k,c] * d_x[k])
scale(acc_x[c]) = S_W[c] * S_x
```

若 cuBLAS 直接计算 `sum(q_W*q_x)`，则通过权重列和修正输入 zero point：

```text
raw_gemm[c] = sum_k(q_W[k,c] * q_x[k])
acc_x[c]    = raw_gemm[c] - Z_x * sum_k(q_W[k,c])
```

先把 bias 对齐到 GEMM 累加域：

```text
bias_acc[c] = R(q_b[c], S_b[c] -> S_W[c] * S_x)
```

再将 GEMM 与 bias 的和统一量化到 Linear 输出：

```text
q_Lx[c] = Clamp(
    R(acc_x[c] + bias_acc[c],
      S_W[c] * S_x -> S_Lx)
    + Z_Lx,
    BW_Lx
)
```

这个顺序对应 GRU FP 载体主路径：bias 先进入 GEMM 域，GEMM 与 bias 相加后只做一次到 Linear 输出域的最终 rescale。

### 5.2 循环 Linear

循环 Linear 完全同构：

```text
acc_h[c] = sum_k(q_R[k,c] * d_h[k])
bias_acc_h[c] = R(q_br[c], S_br[c] -> S_R[c] * S_h)  # Z_br[c]=0

q_Lh[c] = Clamp(
    R(acc_h[c] + bias_acc_h[c],
      S_R[c] * S_h -> S_Lh)
    + Z_Lh,
    BW_Lh
)
```

CUDA FP 载体主路径中，所有时间步的输入 Linear 使用一次 SGEMM，循环 Linear 每个时间步使用一次 SGEMM。

## 6. 四门输入与激活推导

对 `k in {i,f,g,o}`，浮点门输入为：

```text
a_k = Lx_k + Lh_k
```

两个 Linear 可能使用不同 scale，因此分别对齐到门输入网格，然后相加：

```text
d_a_k =
    R(d_Lx_k, S_Lx -> S_a_k)
  + R(d_Lh_k, S_Lh -> S_a_k)

q_a_k = Clamp(d_a_k + Z_a_k, BW_a_k)
```

门输出为：

```text
q_i = Q_input_gate_output(
    sigmoid(d_input_gate_input * S_input_gate_input), 1.0)

q_f = Q_forget_gate_output(
    sigmoid(d_forget_gate_input * S_forget_gate_input), 1.0)

q_g = Q_cell_gate_output(
    tanh(d_cell_gate_input * S_cell_gate_input), 1.0)

q_o = Q_output_gate_output(
    sigmoid(d_output_gate_input * S_output_gate_input), 1.0)
```

上式中的 `1.0` 表示激活函数结果已经处于真实数值域，随后除以目标 scale 并量化。

载体实现差异：

- FP32 载体调用真实 sigmoid/tanh，再执行 `Round(real/S_out)+Z_out` 和 Clamp。
- int32 carrier reference 首版将输入 q 按 standard scale/zp 反量化，调用与 FP reference 相同的原始 sigmoid/tanh，再通过统一 round/clamp 量化到输出网格。

两条路径共享输入/输出量化点和真实激活数学定义，但 GEMM 累加与 rescale 的载体算术不同，因此仍不要求跨载体 bit-exact。真实激活边界必须使用一个公共实现，不能在两个 reference 中复制公式。

## 7. Cell 更新的量化数学基线

### 7.1 浮点目标

```text
c_t = f_t * c_(t-1) + i_t * g_t
```

用各量化张量所表示的实数近似代入：

```text
f_hat      = d_f      * S_f
c_old_hat  = d_c_old  * S_c_old
i_hat      = d_i      * S_i
g_hat      = d_g      * S_g

c_sum_hat =
    d_f * d_c_old * S_f * S_c_old
  + d_i * d_g     * S_i * S_g
```

### 7.2 一般形式

若新旧 Cell 可以使用不同量化网格，则理想目标为：

```text
d_c_new = Round(
    d_f * d_c_old * S_f * S_c_old / S_c_new
  + d_i * d_g     * S_i * S_g     / S_c_new
)

q_c_new = Clamp(d_c_new + Z_c_new, BW_c_new)
```

### 7.3 递推共享网格形式

首版要求 `c_0` 和所有 `c_t` 共用 `cell_state` 网格：

```text
S_c_old = S_c_new = S_c
Z_c_old = Z_c_new = Z_c
```

因此公式化简为：

```text
d_c_new = Round(
    d_f * d_c_old * S_f
  + d_i * d_g     * S_i * S_g / S_c
)

q_c_new = Clamp(d_c_new + Z_c, BW_c)
```

这是已经确认的 Cell 更新量化数学基线。关键约束是：

1. 两路乘积先在数学意义上求和，整个和只经过一次最终 Round。
2. `f*c_old` 和 `i*g` 不单独量化、不单独加 zero point、不单独 Clamp。
3. 最终只在 `cell_state` 量化边界 Clamp。
4. 具体实现采用双比例合并和延迟舍入，必须报告相对该数学基线的误差。

### 7.4 已确认的双比例延迟舍入方案

定义两路宽乘积及其到目标 Cell 网格的比例：

```text
p_forget = d_f * d_c_old
p_input  = d_i * d_g

alpha = S_f * S_c_old / S_c_new
beta  = S_i * S_g     / S_c_new
```

这里所有 `S` 都是第 2.4 节定义的 standard scale。FP32 载体的 `alpha/beta` 同样需要模拟内部编码后的执行比例；Cell 当前暂定使用第 7.5 节的公共 Q31 编码，而不是直接使用未经编码的连续比例。

FP32 载体不对两路 contribution 分别执行量化 Round，而是先计算目标 Cell 整数域中的和：

```text
cell_acc_fp = p_forget * alpha + p_input * beta

q_c_new = Clamp(
    Round(cell_acc_fp) + Z_c_new,
    BW_c_new
)
```

当新旧 Cell 共用网格时，`alpha=S_f`。这里的 FP32 乘法和加法仍有 IEEE 754 算术误差，但量化意义上只有最外层的一次 `Round`。

int32 reference 将两个比例编码到同一个二进制分母：

```text
alpha ~= M_forget / 2^N
beta  ~= M_input  / 2^N

wide_acc = p_forget * M_forget + p_input * M_input

q_c_new = Clamp(
    RoundShift(wide_acc, N) + Z_c_new,
    BW_c_new
)
```

该形式同样只有一次最终量化舍入，并在舍入前保留两路 contribution 的正负抵消。

该方案明确排除：

- 不先把 `g` 量化对齐到可能更粗的 `cell_state` 网格。
- 不强制 `input_gate_output` 和 `forget_gate_output` 共用 scale 或 bitwidth。
- 不分别 Round 两路 contribution 后再相加。
- 不为 contribution 增加独立的 zero point、Clamp 或 JSON 配置。

GRU 风格的操作数预对齐方案和两路分别 rescale 方案仅保留为离线精度对照，不作为生产执行公式。

### 7.5 最终冻结的 CPU 整数编码

CPU int32 reference 使用固定 Q31 公共分母和 `__int128` 累加：

```text
N = 31

M_forget = Round(alpha * 2^31)
M_input  = Round(beta  * 2^31)

wide_acc =
    int128(p_forget) * M_forget
  + int128(p_input)  * M_input

d_c_new = RoundShift(wide_acc, 31)
```

参数和执行约束：

- `M_forget`、`M_input` 使用有符号 `int64_t` 保存，避免比例大于 1 时受 Q31 小数类型范围限制。
- `p_forget`、`p_input` 至少使用 `int64_t` 保存，乘 multiplier 和求和提升到 `__int128`。
- `__int128` 只用于 CPU reference 的 Cell 融合累加，不进入公共 JSON，也不约束未来可选 CUDA int32 后端。
- `RoundShift` 是唯一最终量化舍入，并调用第 2.5 节的 `roundToNearestEven(wide_acc, 31)`。
- 当比例绝对值小于 `0.5*2^-31` 而编码为 0 时，报告 contribution 消失次数及其相对目标 Cell LSB 的误差。
- 阶段 3 已完成以下三层验证，本方案现已最终冻结。

三层验证方法已经冻结：

1. 静态边界证明：根据 bitwidth、zero point 和 scale ratio 推导 `p_forget/p_input`、`M_forget/M_input`、两路乘积、求和及移位的最坏上界；证明 `int64_t` 输入/multiplier 和 `__int128` 累加安全。任何无法证明安全的配置必须在 setup 阶段 fail fast，并输出完整 `NumericSafetyReport`。
2. 缩小整数域穷举：穷举缩小后的 signed/unsigned q 域，以及 multiplier/shift 的 0、1、最大安全值、正负 half-tie 邻域和左右移边界。Q31 实现与本节冻结的整数公式逐值比较，mismatch 必须为 0；测试侧只实现局部高精度数学计算，不复制完整 LSTM reference。
3. 全范围随机/对抗测试：固定 seed，覆盖 8/16 bit、Affine/POT2、比例接近 0、接近 `0.5*2^-31`、大于 1、接近最大安全值、两路正负抵消、同号累加、量化饱和和长序列递推。Q31 可执行结果与未编码双比例数学公式比较，必须报告 max abs、MAE、MSE、余弦相似度、饱和率、contribution 消失次数及其相对 Cell LSB 误差；非退化结果余弦相似度不得低于 `0.999`。

## 8. Hidden 更新推导

先计算 Cell 激活：

```text
tanh_c_hat = tanh(d_c_new * S_c)

q_tanh_c = Clamp(
    Round(tanh_c_hat / S_tanh_c) + Z_tanh_c,
    BW_tanh_c
)
```

Hidden 浮点目标为：

```text
h_t = o_t * tanh(c_t)
```

代入量化表示：

```text
h_hat = d_o * d_tanh_c * S_o * S_tanh_c
```

直接量化到 `output` 网格：

```text
q_h_new = Clamp(
    R(d_o * d_tanh_c,
      S_o * S_tanh_c -> S_h)
    + Z_h,
    BW_h
)
```

`o*tanh(c)` 是单路乘积，直接对齐到 `output`，不需要也不允许额外的 `mul_output_cell` 量化点。

`h_0`、所有 `h_t` 和最终 `h_n` 共用 `output` 网格。

## 9. LSTM 单时间步完整量化流程

```text
q_x, q_W_ih, q_b_ih
  -> centered GEMM + zero-point compensation
  -> bias 对齐到 GEMM 域
  -> q_weight_ih_linear

q_h_old, q_W_hh, q_b_hh
  -> centered GEMM + zero-point compensation
  -> bias 对齐到 GEMM 域
  -> q_weight_hh_linear

q_weight_ih_linear + q_weight_hh_linear
  -> 分别对齐并相加
  -> q_input/forget/cell/output_gate_input

q_gate_input
  -> sigmoid/tanh
  -> q_input/forget/cell/output_gate_output

q_forget_gate_output * q_c_old
  + q_input_gate_output * q_cell_gate_output
  -> 宽域融合求和
  -> 一次最终 Round/Clamp
  -> q_c_new

q_c_new
  -> tanh
  -> q_cell_tanh_output

q_output_gate_output * q_cell_tanh_output
  -> 直接对齐到 output
  -> q_h_new
```

## 10. GRU 优化后融合公式

### 10.1 GRU 浮点公式

```text
u = sigmoid(Lx_u + Lh_u)
r = sigmoid(Lx_r + Lh_r)
n = tanh(Lx_n + r * Lh_n)
h_new = u * h_old + (1-u) * n
```

### 10.2 Update/Reset gate

Update 和 Reset gate 与 LSTM 普通门输入同构：

```text
q_gate_input = Clamp(
    R(d_Lx, S_Lx -> S_gate_input)
  + R(d_Lh, S_Lh -> S_gate_input)
  + Z_gate_input,
    BW_gate_input
)

q_gate_output = Quantize(sigmoid(Dequantize(q_gate_input)))
```

### 10.3 Candidate 中的乘法融合

旧实现曾对 `r*Lh_n` 单独量化和 Clamp。commit `7c4b304` 删除了这个中间量化边界，改为从乘积尺度直接对齐到 candidate 输入：

```text
d_n_input =
    R(d_Lx_n, S_Lx -> S_n_input)
  + R(d_r * d_Lh_n,
      S_r * S_Lh -> S_n_input)

q_n_input = Clamp(d_n_input + Z_n_input, BW_n_input)
q_n       = Quantize(tanh(Dequantize(q_n_input)))
```

这里的 `r*Lh_n` 与 LSTM 的 `o*tanh(c)` 都属于单路乘积直接对齐目标张量，可以直接复用相同的融合原则。

### 10.4 Hidden contribution 的共同尺度融合

GRU hidden 更新为：

```text
h_new = u*h_old + (1-u)*n
```

commit `4db3a3d` 先将 `n` 对齐到 `h` 网格：

```text
d_n_h = R(d_n, S_n -> S_h)
```

令常数 1 在 update gate 中心化网格中的表示为：

```text
d_one_u = Round(1 / S_u)
d_one_minus_u = d_one_u - d_u
```

此时两路乘积具有相同尺度 `S_u*S_h`：

```text
old_contribution = d_u             * d_h_old
new_contribution = d_one_minus_u   * d_n_h
combined         = old_contribution + new_contribution

q_h_new = Clamp(
    R(combined, S_u * S_h -> S_h) + Z_h,
    BW_h
)
```

优化前会分别量化、Clamp 两路 contribution，再对齐到 `h`；优化后先在宽域相加，最后只做一次到 `h` 的 rescale 和 Clamp。

非对称量化时必须直接使用上面的中心化定义。`quant_one` 若保存为完整量化值，则应先转换为中心化值；`quant_one-q_u` 本身已经消除了 zero point，不能再次无条件减 `Z_u`。GRU 当前源码的非对称条件分支存在重复处理 zero point 的风险，LSTM 不复制该分支写法。

## 11. LSTM 与 GRU 融合关系

| 项目 | GRU | LSTM | 结论 |
|---|---|---|---|
| Linear | `3H` GEMM + bias | `4H` GEMM + bias | 公式和 cuBLAS 调度可直接泛化 |
| 普通门输入 | `Lx+Lh` | 四个门均为 `Lx+Lh` | 可直接复用对齐、相加、Clamp 顺序 |
| Candidate | `Lx_n+r*Lh_n` | `Lx_g+Lh_g` | LSTM 更简单，不含 reset 乘法 |
| 单路乘法 | `r*Lh_n` 直达 candidate 输入 | `o*tanh(c)` 直达 output | 直接使用乘法融合原则 |
| 两路状态更新 | `u*h+(1-u)*n` | `f*c+i*g` | 都应删除 contribution 量化点 |
| 两路系数 | `u` 与 `1-u` 天然同网格 | `f` 与 `i` 相互独立 | GRU 共同尺度变换不能原样复制 |
| 状态范围 | `h`、`n` 通常均在 `[-1,1]` | `c` 可无界，`g` 在 `[-1,1]` | `g -> c` 预对齐可能损失明显精度 |
| 输出状态 | hidden 直接是最终输出 | Cell 后还有 `tanh(c)` 与 output gate | LSTM 多一个激活和乘法误差来源 |

由此得到两条结论：

1. “删除乘法中间量化点、宽域融合、最终量化一次”可以从 GRU 直接继承。
2. “先把一个状态操作数对齐到另一个状态网格”不能从 GRU 直接照搬到 LSTM，必须通过 Cell 专项实验选择执行变换。

## 12. 两种载体如何实现同一数学图

### 12.1 FP32 载体主路径

```text
量化值存储：float，数值为 Round/Clamp 后的整数
GEMM：cuBLAS SGEMM
Rescale ratio：Affine 使用 M+shift 编码，POT2 使用 shift
执行：round_f(value * decoded_execution_ratio)
激活：真实 sigmoid/tanh 后重新量化
乘积/累加：FP32
```

FP32 只能连续精确表示绝对值小于 `2^24` 的整数。即使输入和权重是 8/16 bit，GEMM 累加或 Cell contribution 超过该范围后也可能丢失低位。所有配置先按第 2.6 节分类；正确性模式必须显式关闭 TF32/Tensor Core，性能模式单独评估。

阶段 9 的 generation-key 缓存只复用已经按本节公式量化的 W/R/bias 和 weight sums，不新增量化点、舍入或 Clamp。key 变化后重新执行相同量化边界；cache hit 与 miss 的 checkpoint 必须逐值一致。host 签名和缓存判定属于 setup，不改变 CUDA 设备计算图。

### 12.2 CPU int32 reference

```text
量化值存储：int32_t
GEMM/普通乘积：int64_t；Cell Q31 融合累加使用 __int128
Rescale：POT2 shift 或 integer multiplier+shift
激活：反量化 -> 原始 sigmoid/tanh -> 按输出网格重新量化
```

它用于冻结整数 GEMM、rounding、shift、Clamp 和融合公式，但由于激活阶段仍使用浮点函数，首版只能称为“int32 载体 reference”，不能称为端到端纯整数 reference 或完整硬件 bit-exact reference。CPU int32 reference 与 FP32 载体仍因 GEMM/rescale 的载体差异而不要求跨载体 bit-exact。

整数 PWL LUT 是明确待办项，只在启动可选纯整数部署阶段后实现。未来 LUT 版本必须增加独立的 execution model、Golden 和激活近似误差报告，不能原地改变首版 `cpu_int32` Golden 的语义。

### 12.3 共享与分离边界

必须共享：

- 有效量化点及其命名。
- zero point 的中心化语义。
- 融合计算图和最终 Clamp 位置。
- 外部 standard scale/zp 的权威定义。
- 由 standard scale ratio 派生出的 M+shift/POT2 执行参数。

允许分离：

- 浮点模拟 M+shift 与整数 multiplier+shift 的载体舍入误差。
- GEMM 的载体精度和性能实现。
- 后续 `cpu_int32_lut` execution model 相对真实激活的近似误差；首版不包含该分支。

### 12.4 QAT backward 的浮点代理公式

浮点 backward 对每个时间步从后向前计算。将 output、最终 h/c 和下一时间步的梯度
合并为 `dh`、`dc` 后，固定使用：

```text
do = dh * tanh(c_t)
dc = dc + dh * o_t * (1 - tanh(c_t)^2)

df = dc * c_(t-1)
di = dc * g_t
dg = dc * i_t
dc_(t-1) = dc * f_t

da_i = di * i_t * (1 - i_t)
da_f = df * f_t * (1 - f_t)
da_g = dg * (1 - g_t^2)
da_o = do * o_t * (1 - o_t)
```

随后 `da=(da_i,da_f,da_g,da_o)` 分别用于 input/recurrent Linear 的 input、weight
和 bias 梯度。CPU 浮点 reference 从 master input/parameter 重建最少 trace；CUDA
完全浮点训练由 native forward 保存 gate/cell trace，再由 CUDA pointwise kernel
和 cuBLAS 直接执行同一组公式。量化模式把前向保存的 q-carrier master 和
gate/cell/hidden checkpoint 按各自 standard scale/zp 反量化后代入同一组公式。

QAT 对 Round 使用恒等 STE。对任意真实量化边界 `y=Clamp(Round(x))`，只有前向
记录为 Clamp 的位置使用 `dy/dx=0`，其余位置使用 `dy/dx=1`。Mask 按计算图逆序
应用到 Hidden、`tanh(Cell)`、Cell、gate output、gate input、两路 Linear 和
master input/parameter/state；融合乘法临时值没有量化边界，因此不产生 mask。

CUDA QAT backward 固定执行以下 mask 顺序，其中 mask 值 `1` 表示发生 Clamp，
保留因子均为 `1-mask`：

1. `hidden_outputs` 作用于合并 output、最终 h 和后续时间步所得的 `dh`。
2. `cell_tanh_outputs` 只作用于 `dh*o` 到 Cell 的分支；`cell_states` 作用于合并
   最终 c、后续时间步和 Hidden 分支后的 `dc`。
3. `gate_outputs` 在激活导数之前截断 `di/df/dg/do`，`gate_inputs` 在激活导数
   之后截断 `da`。
4. `weight_ih_linear` 与 `weight_hh_linear` 必须把 `da` 分为独立 `dp/dq`；前者
   生成 input/W/bias_ih 梯度，后者生成 recurrent h/R/bias_hh 梯度，不能提前
   合并为同一个 buffer。
5. 完成 GEMM/reduction 后，再分别应用 input、W/R、bias_ih/bias_hh、h0/c0 的
   master mask。h0/c0 mask 不能错误复用最后一个时间步的 state mask。

q-carrier master 和四个最少 trace 的反量化在 C++ binding 内完成；Python
autograd 只传递 saved tensors、mask 和上游梯度。上述顺序由 mask-aware CUDA
pointwise、cuBLAS GEMM、bias reduction 和最终 master-mask kernel 实现。

## 13. 配置约束

JSON 只允许配置第 4 节列出的真实量化点。以下 GRU 风格字段不得出现在 LSTM schema 中：

```text
mul_forget_cell
mul_input_cell
mul_output_cell
forget_contribution
input_contribution
```

每个配置字段必须能追踪到一次真实的 Round/Clamp；反向也必须保证每个执行量化边界都能追踪到配置或明确的固定规则。未知字段直接报错，不能静默忽略。

算子配置采用两种严格、版本化表示：

1. `config/schema/lstm_quant_override.schema.json`：用户输入的稀疏覆盖配置。根对象必须包含 `schema_version=1`；`scale_mode` 和 `operators` 可省略并继承默认 profile。某个 operator 出现时，允许只填写要覆盖的合法字段。
2. `config/schema/lstm_quant_resolved.schema.json`：解析后的完整执行配置。`scale_mode`、全部真实量化点及每个适用字段都必须存在，任何 forward 只接受该表示。

版本化默认值保存在 `config/defaults/lstm_quant_default_v1.json`：基础位宽为 8 bit、`scale_mode=affine`、四组 weight/bias 为 `per_channel`，signedness/symmetry 使用第 2.3 节冻结的基础 profile。Weight/bias 的 override 可修改 `bitwidth` 和 `granularity`；`is_unsigned/is_symmetric` 若出现只能分别为 `false/true`。其他真实量化点可逐字段覆盖 `bitwidth/is_unsigned/is_symmetric`，其 granularity 在 resolved config 中固定为 `per_tensor`。

首版所有真实量化点的 `bitwidth` 枚举严格限制为 `{8,16}`，允许不同量化点混合使用这两个值。`int32_t` carrier、`int64_t/__int128` 累加器、16-bit M+shift multiplier 和 Q31 Cell multiplier 是执行类型或编码参数，不属于该枚举。任何其他 bitwidth 在 override 校验/resolution 阶段直接失败；未来扩展必须升级 schema 版本，并先补充数值安全证明、Golden、严格矩阵和独立精度阈值。

解析器按字段执行一次确定性合并：`versioned defaults <- user override`。不支持全局/类别/多级继承，不把 JSON Schema 的 `default` 注解当作运行时赋值机制。解析完成后立即产生 canonical resolved config；Python binding、CPU reference、CUDA 和报告模块共享同一个 C++ resolver，不得分别补默认值。Golden 用例只嵌入 resolved config，不能嵌入稀疏 override。

两套 schema 的所有对象都使用 `additionalProperties:false`；未知 operator、遗留 `mul_*`、不适用字段、`null`、重复 key、错误类型、越界位宽和非法枚举在 forward 前失败。Canonical writer 固定字段顺序和枚举拼写，resolved config 再次解析和序列化必须字节一致。精度报告记录完整 resolved config 及其内容摘要，保证稀疏输入最终执行了什么可以审计。

Affine JSON 和参数导入导出只暴露校准得到的 standard scale；POT2 模式暴露转换后的 standard scale。M+shift/POT2 执行参数在 finalize/编译阶段从 standard scale ratio 集中派生，不能反向覆盖外部 scale。

Golden vector 已确认采用分层核心覆盖：第一层验证公共量化原语，第二层验证单时间步 Linear、四门和 Cell/Hidden 融合，第三层以 `T=3, B=1` 小尺寸验证短递推。用例必须定向覆盖 8/16 bit、Affine/POT2、正负 half-tie、饱和、非零 zero point、悬殊 scale、贡献抵消/同号累加和非零 `h_0/c_0`，关键维度使用最小成对组合而不是全量笛卡尔积。首版 CPU FP 与 CPU int32 载体 reference 都使用原始 sigmoid/tanh，并分别产生对应载体的预期；差异来自载体累加和 rescale，而不是 LUT 近似。

权威规范按 `primitive/cell/recurrent` 分类存放在入库的 canonical JSON 中，采用“一用例一个自包含文件”；每个文件独立携带 schema 版本、稳定用例 ID、适用载体、配置、输入、量化参数和期望值，不依赖共享数据文件。C++ fixture 由入库脚本机械生成且不入库；测试输出每次运行重新生成并由 `.gitignore` 排除。项目不重复实现 Python 量化 oracle，期望值由对应的 CPU int32/FP 载体 reference model 生成。

Golden 中所有张量都使用显式 `dtype`、`shape` 和 row-major 一维 `data`，禁止嵌套数组；加载器必须验证元素数量等于 shape 各维乘积，并按 dtype 检查编码和范围。标量使用空 shape 和一个 data 元素。量化 q 值始终按逻辑整数和 JSON integer 保存，即使 FP 载体在运行时使用 `float` 存储；carrier 由用例元数据独立声明。真正的 `float32` 张量元素使用最短可往返规范十进制字符串。

每个 Golden 文件只允许一个 `execution_model=common|cpu_int32|cpu_fp32` 和一份 `expected`。`common` 仅用于结果与载体无关的公共原语；包含载体专属累加/rescale 或激活边界时，必须分别建立 `cpu_fp32` 与 `cpu_int32` 用例。首版两者都使用真实激活，但 carrier 语义仍不同。未来实现 LUT 时必须升级 schema 并新增 `cpu_int32_lut`，不得覆盖 `cpu_int32`。该字段描述 reference 语义而不是设备，不能从路径或 dtype 推断；CUDA 后端应与对应的 CPU execution model 比较。

`expected.checkpoints` 保存所有真实 Round/Clamp 后的量化边界：两路 Linear、四门输入/输出、`cell_state`、`cell_tanh_output`、`output`，以及短递推中每个时间步的对应 q 值和最终 `output/h_n/c_n`。`expected.diagnostics` 只保存公式定义的关键未量化宽值：`p_forget`、`p_input`、两路缩放后的宽 contribution、最终舍入前 Cell 宽和与 Hidden 原始乘积。Diagnostics 没有独立 bitwidth/scale/zero point/clamp，不进入配置；GEMM 分块、线程局部值和 workspace 等实现细节不得写入 Golden。

由于 Golden 与 CPU reference 同源，它用于冻结已审核语义和验证后续后端，不独立证明 CPU reference 正确。CPU reference 的正确性通过代码/公式审核、公共原语边界测试，以及反量化 `output/h_n/c_n` 相对相同输入、权重和初始状态的 `torch.nn.LSTM` 端到端 MAE、MSE、余弦相似度共同验证。常规测试只读 Golden；更新期望值必须显式运行 reference generator、单独审查 diff 并重跑完整精度门禁，禁止失败时自动刷新。

Golden 只使用一个入库的版本化 JSON schema。根对象以 `kind=primitive|cell|recurrent` 作为判别字段，通过 `$defs` 与 `oneOf` 约束各类 payload，并统一拒绝未知字段。生成器必须直接加载该 schema，在任何 C++ 代码生成前拒绝无效版本、缺失字段以及目录与 `kind` 不一致的用例；不能在生成器中复制另一套字段白名单。

## 14. 验证状态

固定 Q31 已在阶段 3 完成以下实现期证据：

1. 静态边界证明和不安全配置 fail fast 已通过。
2. 缩小整数域穷举相对独立整数公式 mismatch 为 0。
3. 固定 seed 的 8/16 bit、极端 scale、Affine/POT2、抵消、同号、饱和及长递推随机/对抗测试已通过；报告由测试写入构建目录且不入库。

阶段 5 校准参数生成已完成以下实现期证据：

1. 正式 FP32 reference 输出两路 Linear、四门输入/输出、Cell、`tanh(Cell)` 与 Hidden checkpoint，18 个真实量化点跨 batch/time 取并集；`h_0/c_0` 分别并入 Output/CellState。
2. MinMax、SQNR 和 Percentile 仅产生候选连续范围，统一经 minimum-scale 与 POT2 CoverRange 生成 standard scale/zp；恰等于 `S_min` 不 fallback、刚低于时 fallback。
3. 外部参数包只保存完整 `4H` standard scale/zp 与 granularity/config 元数据；canonical FP32 字符串导出导入后重新派生执行编码，CUDA FP `output/h_n/c_n` 保持逐值一致。

阶段 8 backward 已完成以下实现期证据：

1. 浮点 input、h0/c0、W/R 和可选 bias 梯度在单向/双向、两种布局上对齐 `torch.nn.LSTM`。
2. INT16 QAT 梯度相对浮点代理满足 MAE、MSE 和余弦门禁；双向 `bias=False` 同时覆盖。
3. 单步参数更新、12 步 loss 下降以及 master input 被 Clamp/未 Clamp 的梯度行为通过。
4. 证据写入忽略目录 `tests/precision/results/stage8_backward_report.json`，验证范围仍为 `synthetic_numeric`。

阶段 9 已完成以下不改变公式的后端与交换格式证据：

1. ONNX 边界显式执行 `(i,f,g,o) -> (i,o,f,c)`，单向/双向图均只有一个标准 `LSTM` 节点，ONNX Runtime 的 `output/h_n/c_n` 与浮点语义一致。
2. 静态参数 cache miss、hit 和 generation key 失效路径的所有 Golden checkpoint 逐值一致；优化前后精度指标完全一致。
3. 两次稳定 CUDA benchmark、memcheck/racecheck 和 Nsight SGEMM 计数通过；版本化阈值按环境和 profile 隔离。
4. 缓存、ONNX 重排和性能门禁均未引入新的量化点、乘法配置或执行公式分支。

阶段 9 后续的全浮点与 QAT CUDA backward 补充证据：

1. CUDA forward 保存 gate/cell 最少 trace；native backward 的逐时间步点算子、
   循环梯度、input/weight 梯度和 bias reduction 均位于 CUDA/cuBLAS。
2. 单向/双向、bias 开关、两种布局、显式和省略 h0/c0 的全部梯度对齐
   `torch.nn.LSTM`，且路径测试禁止 CUDA 浮点训练回退到 Python backward。
3. QAT 的 mask-aware CUDA 结果与迁移前 Python STE oracle 逐梯度一致；路径测试
   禁止 CUDA QAT 回退到 Python 时间步循环。直接 CUDA mask 公式测试同时覆盖
   checkpoint 与 master 边界。
4. 全量 C++/Python 回归、memcheck、racecheck 和 Nsight kernel trace 通过。
   `T=4` 的 QAT trace 包含 4 次 backward pointwise、1 次 bias reduction、7 次
   master clamp mask kernel 及对应 cuBLAS GEMM；未修改本节公式和 STE 语义。

最终冻结继续受以下回归门禁保护：

- 不恢复乘法临时值的独立 bitwidth/scale/zero point。
- 基础测试先通过 GRU 风格的 MAE、MSE、余弦相似度门禁。
- 严格测试再比较相对本文数学基线的逐量化点误差、逐时间步误差和饱和率。
- 非退化输出的余弦相似度不得低于 `0.999`。L2 norm 使用 FP64 计算并固定 `epsilon=1e-12`；双边 norm 均不超过 epsilon 时 cosine 记为 `N/A` 并依靠 exact/MAE/MSE 验收，仅一边不超过时直接失败。

## 15. GRU 源码与历史依据

- `src/gru_forward_gpu_quant_fp.cu`：FP32 载体门控、真实激活、融合 contribution 和 cuBLAS 前向。
- `include/quantize_ops_helper.h`：GRU int32 reference 的融合门控、整数 rescale 和 LUT；LSTM 首版只迁移整数原语，LUT 作为后续待办。
- commit `7c4b304`：删除 `r*Lh_n` 的独立中间量化层，乘积直接对齐 candidate 输入。
- commit `4db3a3d`：先统一 GRU 两路 hidden contribution 的尺度，在宽域相加后统一 rescale。
