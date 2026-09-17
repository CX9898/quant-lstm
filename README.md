# quant-lstm

基于 CUDA FP32 载体的量化 LSTM 实现。当前已完成阶段 5：除 CPU int32 与
标量 FP32 q-carrier reference、CUDA FP32 q-carrier 主路径外，已提供正式 FP32
checkpoint 驱动的 18 点多 batch 校准、MinMax/SQNR/Percentile、完整 `4H`
参数物化，以及 canonical standard scale/zp 参数包导入导出。统一 Golden、
NumericSafety 和 synthetic numeric 精度门禁保持生效。
量化执行语义以 `docs/quantized-execution-spec.md` 为准。

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

C++ 配置 resolver 使用系统提供的 `nlohmann_json 3.11.2`；schema 与 Golden
生成测试需要 Python `jsonschema`。构建测试目标时会从
`tests/golden/spec/` 机械生成统一的临时 C++ fixture，生成文件位于 build
目录且不入库。

## 阶段 1 精度测试

```bash
cd pytorch
QUANT_LSTM_TEST_SUITE=basic PYTHONPATH=. python3 tests/test_float_reference.py
QUANT_LSTM_TEST_SUITE=strict PYTHONPATH=. python3 tests/test_float_reference.py
```

正确性模式固定使用 Pedantic math；TF32 仅作为显式性能模式并独立报告精度。
CUDA 报告包含 `output`、`h_n`、`c_n` 的 MAE/MSE/FP64 cosine/SQNR/饱和率、
P50/P95、吞吐、workspace 和量化/core 分段计时，并以 compute-sanitizer 与
Nsight 的 `sum_cases(1+T)` GEMM kernel 计数验收。当前精度范围仍为
`synthetic_numeric`，真实数据状态为 `not_configured`。校准参数包仅包含 standard
scale/zp，不包含 M+shift/POT2 shift 或 raw ratio；PyTorch 量化接口、CUDA int32、
LUT、反向传播、多层和双向接口尚未实现。