# quant-lstm 系统架构

## 1. 目标与范围

quant-lstm 将单层 LSTM 的完全浮点训练、FP32 q-carrier 量化执行、CUDA 校准和
QAT backward 放在 C++/CUDA 核心中。Python 层提供与 `torch.nn.LSTM` 相近的模块
接口，并负责配置、张量布局和 autograd 调度。

量化数学、Round/Clamp 边界和载体契约由
[量化执行规格](quantized-execution-spec.md)定义。本文只描述当前代码结构、模块接口、
运行数据流和发布制品。

## 2. 系统上下文

```text
PyTorch model
    |
    | QuantLSTM / autograd Function
    v
_quant_lstm extension
    |
    +-- CUDA FP32 forward/backward
    +-- CUDA FP32 q-carrier forward/QAT backward
    +-- CUDA calibration collectors
    |
    v
quant_lstm C++ static library
    |
    +-- configuration and parameter validation
    +-- CPU FP32 q-carrier reference
    +-- CPU int32 carrier reference
    +-- Golden and numeric-safety support
```

PyTorch runtime 不包含 CPU dispatch。CPU 实现由 C++ 测试、示例和外部 C++ consumer
显式调用。

## 3. 模块与接口

| 目录 | 职责 | 稳定入口 |
| --- | --- | --- |
| `include/lstm/` | LSTM shape、配置、参数、CPU/CUDA forward/backward 接口 | `lstm/interface.h` 及各功能头文件 |
| `include/quantization/` | 舍入、scale 编码、量化参数和数值安全原语 | 对应公共头文件 |
| `src/lstm/` | LSTM CPU/CUDA 实现、校准和参数 I/O | 由 `quant_lstm` target 编译 |
| `src/quantization/` | 量化公共实现 | 由 `quant_lstm` target 编译 |
| `pytorch/` | `QuantLSTM`、autograd、ONNX 和 C++ binding | `quant_lstm.QuantLSTM` |
| `config/` | 默认配置与 JSON Schema | `lstm_quant_default_v1.json` |
| `tests/` | C++、CUDA、Python、Golden、性能和真实网络验证 | CTest 与测试脚本 |
| `tools/` | 端到端测试、Golden 生成、格式化和性能检查 | 各脚本的 `--help` 或目录 README |

## 4. 运行数据流

### 4.1 完全浮点训练

1. `QuantLSTM.forward()` 验证 CUDA FP32 input 和状态。
2. Python autograd Function 调用 `_quant_lstm` binding。
3. CUDA forward 使用 cuBLAS 和 pointwise kernel 计算输出，并保存最少 gate/cell
   checkpoint。
4. CUDA backward 逆序遍历时间步，使用 pointwise kernel、cuBLAS GEMM 和 bias
   reduction 产生 input、参数和初始状态梯度。

`use_quantization=False` 不执行校准、Round/Clamp 或 STE。

### 4.2 校准与量化执行

1. C++ resolver 将默认配置与用户 override 合并为 canonical resolved config。
2. `calibrating=True` 时，CUDA 浮点 forward 同时产生输出和 18 个真实量化点的
   checkpoint。
3. CUDA collector 计算 range、直方图和 cell contribution 诊断，只将紧凑统计量
   返回 C++ finalization。
4. `finalize_calibration()` 生成 standard scale/zp、执行比例编码和
   `NumericSafetyReport`。
5. `use_quantization=True` 后，CUDA q-carrier forward 在每个真实边界执行
   Round/Clamp，并使用真实 sigmoid/tanh。

校准数据应来自训练分布的代表性样本。随机 tensor 只适用于接口冒烟测试。

### 4.3 QAT backward

量化训练 forward 保存实际消费的 q-carrier master、最少 checkpoint 和 Clamp mask。
Backward 在 CUDA 中反量化这些张量并复用浮点 LSTM 导数。Round 使用恒等 STE；
Clamp mask 只在前向发生饱和的位置将梯度置零。Python autograd 不重新实现 LSTM
公式，也不重建 STE mask。

### 4.4 双向执行

双向模块调用同一个单向核心两次。reverse 方向在时间维翻转输入和输出；两个方向
独立校准 h/c、Linear、门和参数，共享 input 量化网格。最终 output 在 hidden 维
拼接，`h_n/c_n` 按 forward、reverse 顺序堆叠。

## 5. 配置和参数来源

配置链路为：

```text
config/defaults/lstm_quant_default_v1.json
    + user override
    -> C++ resolver
    -> canonical resolved config
    -> calibration/finalization
    -> standard scale/zp parameter document
    -> CUDA execution parameters
```

默认值、字段、状态约束和公共参数格式见[配置与校准](configuration.md)。外部参数
文档只保存 standard scale/zp；M+shift、POT2 shift 和 Q31 multiplier 属于内部派生
执行参数。

## 6. 构建与发布制品

CMake 构建 `quant_lstm` 静态库，并根据选项加入 CUDA 源码。安装流程导出：

- `libquant_lstm.a` 和公共头文件；
- `quant-lstm-config.cmake`、version file 和 exported targets；
- 默认配置、JSON Schema 和公开文档；
- 可选 C++ 示例。

Python extension 通过 `pytorch/setup.py` 链接 `build/libquant_lstm.a`。当前 Python
交付方式是源码可编辑安装；独立 wheel 发布尚未接入。完整命令见
[安装指南](installation.md)。

## 7. 验证策略

| 层级 | 验证内容 | 入口 |
| --- | --- | --- |
| C++/CPU | 配置、量化原语、Golden、两套 reference、数值安全 | `ctest --test-dir build` |
| CUDA | rounding、FP32、q-carrier、校准 round-trip、严格精度 | CTest CUDA targets |
| PyTorch | 接口、布局、双向、FP32 backward、QAT STE、ONNX | `tools/run_end_to_end_test.sh` |
| 性能 | cuBLAS 调用、sanitizer、P50/P95、设备专用阈值 | `tools/run_end_to_end_test.sh --with-cuda-validation` |
| 真实网络 | Speech Commands FP32、INT8 QAT、INT16 QAT | `tests/real_network/run_speech_commands_lstm_test.sh` |

Golden schema、测试矩阵和阈值入库。构建产物、运行日志和逐次生成的报告保存在忽略
目录中。性能和精度结论必须同时记录环境、数据、shape、seed 和测量范围。

## 8. 约束、风险与扩展

- FP32 q-carrier 只能连续精确表示绝对值小于 `2^24` 的整数。超过该范围但仍有限的
  配置会产生 `precision_risk`，并由严格精度门禁约束。
- CPU int32 reference 在激活边界使用浮点 sigmoid/tanh，不表示完整硬件整数语义。
- CUDA int32 backend 和整数 PWL LUT 只有在独立 execution model、Golden、误差预算
  和性能收益明确后才会启动。
- 静态参数缓存由 generation key 显式失效。调用方修改 master 参数后必须更新 key。
- ONNX 导出只表达浮点 LSTM，不序列化量化执行细节。
- Python wheel 尚未形成可独立分发的制品，源码可编辑安装依赖仓库布局。

CUDA-only Python runtime 的设计理由记录在
[ADR-0001](adr/0001-cuda-only-python-runtime.md)。新的跨模块设计决策应新增 ADR；
当前行为变化则直接更新本文和对应专题文档。
