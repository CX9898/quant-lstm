# LSTM 双载体计算流程

## 1. 共同数学边界

CUDA FP32 q-carrier 与 CPU int32 reference 共享同一组 18 个真实量化点、standard
scale/zp、门顺序 `(i,f,g,o)`、M+shift/POT2 执行编码、Cell Q31 双比例融合以及
Round/Clamp 位置。乘法临时值没有独立量化参数。

共同流程为：

1. input、W/R、可选 bias 和初始 h/c 进入各自真实量化边界。
2. input Linear 与 recurrent Linear 独立累加并量化。
3. 两路 Linear 对齐到四个 gate input 网格，只在 gate input 边界 Clamp。
4. 三个 sigmoid 和两个 tanh 使用真实浮点激活后重新量化。
5. Cell 两路乘积以固定 Q31 比例合并，最终只舍入一次。
6. output gate 与 `tanh(Cell)` 融合到 output 网格。
7. output 与最终 h/c 按各自 standard scale 反量化。

## 2. CUDA FP32 q-carrier

量化 q 值以 FP32 保存，数值仍是整数网格值。全部时间步的 input Linear 使用一次
cuBLAS SGEMM，每个时间步的 recurrent Linear 使用一次 SGEMM，门、Cell 和 Hidden
由融合 pointwise kernel 完成。

推理调用方可分配持久静态参数缓存，并以非零 generation key 显式启用。首次调用
量化 W/R/可选 bias 并计算 weight sums，后续相同 key 调用直接复用；master 参数
内容变化时调用方必须更换 key。key 为 0 时保持逐调用量化。缓存不覆盖 input、
h0/c0，不改变任何 checkpoint 或 Clamp 边界。持久缓存容量通过
`lstmQuantizedFpCudaStaticParameterBytes()` 独立查询，不混入临时 workspace
统计；性能证据和适用阈值见 `docs/cuda-performance.md`。

Pedantic 模式关闭 TF32/Tensor Core，用于正确性验收；TF32 只能显式选择并独立报告
精度。每次执行都会消费导入参数重新派生出的编码，不接受 raw ratio。训练态前向会
由同一次 CUDA 量化 kernel 直接保存实际消费的 master q-carrier、最少 backward
checkpoint 和对应真实 Clamp mask，不在 Python 中重新量化。QAT backward 通过
CUDA kernel 将这些 q 值按 standard scale/zp 反量化后复用同一浮点 LSTM backward；
Round 使用 STE 恒等梯度，只有 mask 标记为 Clamp 的真实边界会把梯度置零。两路
Linear、四门 input/output、Cell、`tanh(Cell)` 和 Hidden 拥有 mask，融合乘法临时值
没有 mask。Python autograd 只保存张量并调用 C++/CUDA 扩展。

PyTorch `QuantLSTM` 的完全浮点、量化和 QAT 执行边界均为 CUDA-only。CPU FP32
与 int32 实现是显式 reference model，只供 Golden、校准和数值验证使用，不参与
Python binding 的设备分发，也不作为 CUDA 不可用时的 fallback。

双向模块调用同一个单向 CUDA 核心两次。reverse 方向只在输入和输出的时间维做翻转；
两个方向的 output 在最后一维拼接，`h_n/c_n` 按 forward、reverse 顺序堆叠。

## 3. CPU int32 reference

量化值使用 `int32_t`，GEMM 与普通乘积使用 `int64_t`，Cell Q31 合并使用
`__int128`。普通 rescale 执行整数 M+shift 或 POT2 shift。它冻结整数累加、舍入、
Clamp 和融合公式。

首版 CPU int32 reference 的 sigmoid/tanh 仍会反量化后调用真实浮点函数，再量化回
目标网格，因此不是端到端纯整数实现。整数 PWL LUT 与 CUDA integer 后端只有满足
阶段 10 条件并经审核后才能启动。

## 4. CPU-only 构建与安装

```bash
cmake -S . -B build-cpu \
  -DQUANT_LSTM_ENABLE_CUDA=OFF \
  -DQUANT_LSTM_BUILD_TESTS=ON \
  -DQUANT_LSTM_BUILD_EXAMPLES=ON
cmake --build build-cpu -j
ctest --test-dir build-cpu --output-on-failure
cmake --install build-cpu --prefix build-cpu/install
```

`lstm_int32_example` 展示校准、输入/参数量化、int32 reference 前向和输出反量化。
安装包导出 `quant_lstm::quant_lstm` 与 `quant_lstm::headers`。CPU-only 配置
不会定义 `QUANT_LSTM_WITH_CUDA`，也不会链接 CUDA/cuBLAS。

完整打包验收可直接运行：

```bash
tools/run_cpu_only_package_check.sh
```

该脚本会在独立构建树中执行 CPU 测试、安装、外部 `find_package` 消费、示例运行
和动态链接检查。
