# quant-lstm

基于 CUDA FP32 载体的量化 LSTM 实现。当前已完成阶段 2：除阶段 1 的单层、
单向 FP32 基准外，已冻结公共量化原语、18 个真实量化点、参数 finalize 和严格
配置 resolver。量化执行语义以 `docs/quantized-execution-spec.md` 为准。

## 构建

```bash
cmake -S . -B build -DQUANT_LSTM_ENABLE_CUDA=ON
cmake --build build -j
ctest --test-dir build --output-on-failure

cd pytorch
python3 setup.py build_ext --inplace
cd tests
python3 setup_test_extension.py build_ext --inplace
```

C++ 配置 resolver 使用系统提供的 `nlohmann_json 3.11.2`；schema 与 Golden
生成测试需要 Python `jsonschema`。构建测试目标时会从
`tests/golden/primitive/` 机械生成临时 C++ fixture，生成文件位于 build
目录且不入库。

## 阶段 1 精度测试

```bash
cd pytorch
QUANT_LSTM_TEST_SUITE=basic PYTHONPATH=. python3 tests/test_float_reference.py
QUANT_LSTM_TEST_SUITE=strict PYTHONPATH=. python3 tests/test_float_reference.py
```

测试固定关闭 TF32，并分别报告 `output`、`h_n`、`c_n` 的 MAE、MSE 和余弦相似度。
当前 forward 仍只提供 FP32 inference；阶段 2 的原语和配置将在阶段 3 的双 CPU
quantized reference 中接入。反向传播、多层和双向接口尚未实现。