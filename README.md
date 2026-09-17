# quant-lstm

基于 CUDA FP32 载体的量化 LSTM 实现。当前已完成阶段 8：支持单层单向/双向
`QuantLSTM`、双方向独立校准参数、共享 input 网格、完整 `4H` standard scale/zp
参数导入导出，以及复用浮点 backward 的 FP32 q-carrier QAT。训练态只为真实
Round/Clamp 边界保存 mask，并支持 h0/c0、bias=False 和双向梯度。CPU int32 与
标量 FP32 q-carrier reference 可通过无 CUDA 的安装包独立消费，统一 Golden、
NumericSafety 和 synthetic numeric 精度门禁保持生效。量化执行语义以
`docs/quantized-execution-spec.md` 为准；配置和双载体流程分别见
`docs/configuration.md` 与 `docs/dual-carrier-execution.md`；标准 ONNX LSTM 导出见
`docs/onnx-export.md`。

## 构建

```bash
cmake -S . -B build -DQUANT_LSTM_ENABLE_CUDA=ON
cmake --build build -j
ctest --test-dir build --output-on-failure

python3 tools/run_stage4_cuda_validation.py \
  --build-dir build --device 0 --artifacts-root build

cd pytorch
python3 setup.py build_ext --inplace
cd tests
python3 setup_test_extension.py build_ext --inplace
```

CPU-only 构建、安装和外部消费验收：

```bash
tools/run_cpu_only_package_check.sh
```


C++ 配置 resolver 使用系统提供的 `nlohmann_json 3.11.2`；schema 与 Golden
生成测试需要 Python `jsonschema`。构建测试目标时会从
`tests/golden/spec/` 机械生成统一的临时 C++ fixture，生成文件位于 build
目录且不入库。

## 精度与接口测试

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
阶段 9 ONNX/性能优化尚未实现；
CUDA int32 与整数 LUT 仍受阶段 10 的条件性启动规则约束。
