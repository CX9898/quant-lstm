# CUDA FP32 q-carrier 性能验收

## 1. 验收范围

本性能验收覆盖 CUDA FP32 q-carrier 的稳态推理，不改变
[量化执行规格](quantized-execution-spec.md)定义的量化点、门顺序、Cell Q31 融合或
Round/Clamp 边界。benchmark 直接调用 C++ CUDA 核心，不包含 Python/binding
转置；host 参数展开、签名和 cache hit 判定属于 setup，也不计入 CUDA event
设备计时。

优化以调用方拥有的持久 buffer 缓存量化 W/R/可选 bias 和两组 weight sums。
非零 generation key 显式启用缓存；master 参数内容变化时调用方必须更换 key。
input、h0/c0 仍逐调用量化。key 为 0 时保持原有逐调用静态参数量化语义。

## 2. 环境与测量方法

版本化阈值的审核环境：

| 项目 | 值 |
|---|---|
| GPU | NVIDIA RTX 6000D，compute capability 12.0 |
| GPU UUID | `GPU-a5c0eb76-1971-cf70-55d2-c6699ce558f0` |
| Driver | 610.43.02，CUDA driver API 13030 |
| CUDA Runtime | 13020（13.2） |
| cuBLAS | 130400（13.4） |
| 功耗上限 | 600 W |
| 时钟策略 | 动态；环境采样为 P8、graphics/SM 180 MHz、memory 405 MHz |
| 同步 | 每次测量同步 `timing.complete` CUDA event |
| 主测量 | 20 次预热、200 次测量 |
| 独立复测 | 10 次预热、100 次测量 |
| 计时范围 | quantize + quantized core + final-state dequantize |
| 精度范围 | `synthetic_numeric`；真实数据仍为 `not_configured` |

动态时钟没有锁频，因此冻结阈值相对两次稳定结果保留约 10% 人工余量。阈值仍
精确匹配 GPU 型号、compute capability、CUDA driver/runtime 和 cuBLAS 版本；
环境不匹配时检查器直接失败，不跨设备套用绝对数值。

## 3. 优化结果

主测量的优化前报告基于 commit
`ef412d1073684691339f403e49c1f8323c57c985`，优化后报告基于 commit
`ebbbc64ee1f878f59e7dc4b10e557bb0276c3da1`。报告位于
`tests/benchmarks/results/` 忽略目录，不提交仓库。

| Profile / math mode | P50 before -> after (ms) | P50 降低 | P95 before -> after (ms) | P95 降低 | after seq/s |
|---|---:|---:|---:|---:|---:|
| `cuda_gru_basic` / Pedantic | 1.450848 -> 1.283040 | 11.57% | 1.458848 -> 1.290176 | 11.56% | 49,882 |
| `cuda_gru_basic` / TF32 | 1.203328 -> 1.036768 | 13.84% | 1.209376 -> 1.039744 | 14.03% | 61,730 |
| `large_batch` / Pedantic | 0.501184 -> 0.416640 | 16.87% | 0.507200 -> 0.419936 | 17.21% | 307,220 |
| `large_batch` / TF32 | 0.403584 -> 0.324608 | 19.57% | 0.408736 -> 0.333952 | 18.30% | 394,322 |

量化开销 P50 分别从
`0.191776/0.191328/0.102464/0.096640 ms` 降至
`0.024704/0.026432/0.014720/0.014656 ms`。quantized core P50 保持在
`1.255104/1.008416/0.400224/0.308064 ms`，说明收益来自消除重复静态参数
量化，而不是改写 SGEMM 或 pointwise 数学。

内存代价：

| Profile / bias | 临时 workspace bytes | 持久 cache bytes |
|---|---:|---:|
| `cuda_gru_basic` / bias | 16,850,944 | 1,589,248 |
| `cuda_gru_basic` / no bias | 16,842,752 | 1,581,056 |
| `large_batch` / bias | 5,574,656 | 401,408 |
| `large_batch` / no bias | 5,570,560 | 397,312 |

临时 workspace 为兼容 key=0 的逐调用路径仍保留原静态参数区域；报告将持久 cache
单独列出，不能把两者相互抵消或隐藏。

## 4. 精度与诊断

cache miss、相同 key hit 和 key 变化后的 miss 均通过 Golden checkpoint 逐值
比较。优化前后四个 profile 的 `output/h_n/c_n` 指标完全一致。最差 INT8
Pedantic case 仍满足既有门禁：

| Tensor | MAE | MSE | Cosine |
|---|---:|---:|---:|
| output | 0.001553756 | 3.784136e-6 | 0.999176331 |
| h_n | 0.001561180 | 3.811524e-6 | 0.999188919 |
| c_n | 0.002822434 | 1.240258e-5 | 0.999352626 |

完整验证同时通过：

- compute-sanitizer memcheck。
- compute-sanitizer racecheck。
- Nsight Systems 中 expected、benchmark 和 trace 的 SGEMM 次数均为 136。
- benchmark result v3 schema、首次 miss/后续 hit 计数和持久 cache 容量契约。
- ONNX Runtime 浮点语义验收，详见[ONNX 导出](onnx-export.md)。

## 5. 候选方案结论

| 方案 | 结论 | 原因 |
|---|---|---|
| packed/cached quantized weights | 采用 | 四个 profile 均有稳定收益；generation key 提供显式失效，额外显存已报告 |
| cuBLASLt | 不采用 | 当前 SGEMM 已由 Nsight 证明生效；冻结的 zp 修正、rescale、真实激活和 Cell/Hidden 融合不能直接映射为单一 Lt epilogue，当前 profile 没有足以抵消维护成本的证据 |
| CUDA Graph | 延后 | 公共 API 允许调用方更换 input/output/state 指针和 generation；在 graph 所有权、更新与失效契约冻结前不引入隐式持久状态 |
| kernel fusion | 保持现状 | 四门、Cell、Hidden 和输出反量化已在逐时间步 pointwise 中融合；静态 quantize + sum 只发生在 cache miss，继续融合不改善稳态路径 |
| math mode | 保留双模式 | Pedantic 用于正确性，TF32 独立报告精度并在两个 profile 上提供额外吞吐 |

cuBLASLt、CUDA Graph 或新的 fusion 只有在新的固定 profile 上提供独立性能、精度、
workspace 和生命周期证据后才能重新评估，不能为了降低延迟恢复遗留乘法量化点。

## 6. 回归门禁

版本化阈值：

`tests/benchmarks/config/cuda_performance_thresholds_v1.json`

阈值 schema：

`tests/benchmarks/schema/cuda_performance_thresholds.schema.json`

可复现命令：

```bash
cmake --build build --target quant_lstm_cuda_benchmark -j

build/tests/benchmarks/quant_lstm_cuda_benchmark \
  --device 0 \
  --warmup-iterations 20 \
  --iterations 200 \
  --json-report tests/benchmarks/results/stage9_after.json

python3 tools/run_stage4_cuda_validation.py \
  --build-dir build \
  --device 0 \
  --artifacts-root results

python3 tools/check_stage9_cuda_performance.py \
  --report tests/benchmarks/results/stage9_after.json
```

阈值只由人工审核后的配置提交修改。benchmark、检查器和测试均不会生成、学习或
回写阈值；优化前四个 profile 由单元测试固定验证为全部拒绝。
