# quant-lstm

基于 CUDA FP32 载体的量化 LSTM 实现。当前已完成阶段 9：支持单层单向/双向
`QuantLSTM`、双方向独立校准参数、共享 input 网格、完整 `4H` standard scale/zp
参数导入导出，以及 native CUDA FP32 q-carrier QAT backward。训练态只为真实
Round/Clamp 边界保存 STE mask，并支持 h0/c0、bias=False 和双向梯度。标准 ONNX
`LSTM` 导出、显式 generation-key 量化静态参数缓存，以及 RTX 6000D
设备/profile/version 性能门禁均已验收。CPU int32 与标量 FP32 q-carrier
reference 可通过无 CUDA 的安装包独立消费，统一 Golden、NumericSafety 和
synthetic numeric 精度门禁保持生效。量化执行语义以
`docs/quantized-execution-spec.md` 为准；配置和双载体流程分别见
`docs/configuration.md` 与 `docs/dual-carrier-execution.md`；标准 ONNX LSTM 导出见
`docs/onnx-export.md`，CUDA 性能证据见 `docs/cuda-performance.md`。

`use_quantization=False` 是完全 FP32 训练模式。在 CUDA 上，训练前向由原生
CUDA/cuBLAS 执行并保存最少的 gate/cell checkpoint，反向的逐时间步链式计算、
循环状态梯度、input/weight GEMM 和 bias reduction 均由 CUDA kernel/cuBLAS
完成。CPU 浮点训练由 C++ 标量 forward/backward 执行。量化训练前向由同一次
CUDA 执行直接保存实际使用的 q-carrier master、checkpoint 和 Clamp mask；QAT
backward 由 CUDA kernel 反量化这些张量，再调用同一 CUDA backward 核心的
mask-aware 模式。checkpoint STE 在逐时间步 kernel 内按计算图逆序执行，master
mask 在最终梯度上执行。Python autograd 只负责张量保存、布局整理和扩展调用；
PyTorch 公式仅存在于测试专用 oracle。

## 构建

```bash
cmake -S . -B build -DQUANT_LSTM_ENABLE_CUDA=ON
cmake --build build -j
ctest --test-dir build --output-on-failure

python3 tools/run_stage4_cuda_validation.py \
  --build-dir build --device 0 --artifacts-root build

python3 tools/check_stage9_cuda_performance.py \
  --report build/stage4-validation-results/<run>/benchmark_report.json

cd pytorch
python3 setup.py build_ext --inplace
cd tests
python3 setup_test_extension.py build_ext --inplace
```

Docker CUDA/PyTorch 构建环境：

```bash
docker build -f docker/Dockerfile -t quant-lstm:cuda .
docker run --rm -it --gpus all -v "$PWD:/workspace" quant-lstm:cuda
```

CPU-only 构建、安装和外部消费验收：

```bash
tools/run_cpu_only_package_check.sh
```

C++ 配置 resolver 使用系统提供的 `nlohmann_json 3.11.2`；schema、CUDA benchmark
契约与 Golden 生成测试的 Python 依赖通过
`python -m pip install --requirement requirements-test.txt` 安装。构建测试目标时会从
`tests/golden/spec/` 机械生成统一的临时 C++ fixture，生成文件位于 build
目录且不入库。

## 精度与接口测试

完整 CUDA/C++、PyTorch extension、FP32、量化、双向、QAT backward 与 ONNX
端到端回归：

```bash
tools/run_end_to_end_test.sh

# 额外运行 benchmark、compute-sanitizer、Nsight 与 Stage 9 性能阈值
tools/run_end_to_end_test.sh --with-cuda-validation --device 0
```

单独运行各 Python 测试：

```bash
cd pytorch
QUANT_LSTM_TEST_SUITE=basic PYTHONPATH=. python3 tests/test_float_reference.py
QUANT_LSTM_TEST_SUITE=strict PYTHONPATH=. python3 tests/test_float_reference.py
PYTHONPATH=. python3 -m unittest -v tests.test_quantized_interface
PYTHONPATH=. python3 -m unittest -v tests.test_bidirectional_interface
PYTHONPATH=. python3 -m unittest -v tests.test_backward
PYTHONPATH=. python3 -m unittest -v tests.test_onnx_export
```

典型流程是先以 `calibrating=True` 运行一个或多个校准 batch，随后调用
`finalize_calibration()`，再设置 `use_quantization=True`。量化模式只接受 CUDA
FP32 tensor，并固定调用 CUDA FP32 q-carrier 主路径；未暴露未实现的 int32 后端。
双向模块会分别导出 forward/reverse 参数，并拒绝没有共享 input 网格的参数文档。

`get_quant_config()` 返回 C++ resolver 产生的完整 canonical resolved config。
`set_all_bitwidth()` 和 `adjust_quant_config()` 修改配置后会使旧校准参数失效。
`export_quant_params()` 产生带 carrier、真实激活模式、cuBLAS math mode 和 standard
scale mode 元数据的文档；外部参数仍只包含完整 standard scale/zp，不包含
M+shift、POT2 shift 或 raw ratio。

正确性模式固定使用 Pedantic math；TF32 仅作为显式性能模式并独立报告精度。
当前精度范围仍为 `synthetic_numeric`，真实数据状态为 `not_configured`。
阶段 9 已完成标准 ONNX 单节点导出和 CUDA 静态参数缓存优化；版本化绝对性能阈值
只适用于配置中精确匹配的 GPU/CUDA/cuBLAS 环境，不跨设备复用。
CUDA int32 与整数 LUT 仍受阶段 10 的条件性启动规则约束。
