# quant-lstm

基于 CUDA FP32 载体的量化 LSTM 实现。当前已完成阶段 3：除单层、单向 FP32
基准外，已提供 CPU int32 与标量 FP32 q-carrier reference，冻结 Cell Q31
融合编码，并建立统一 Golden 和 synthetic numeric 精度门禁。量化执行语义以
`docs/quantized-execution-spec.md` 为准。

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
`tests/golden/spec/` 机械生成统一的临时 C++ fixture，生成文件位于 build
目录且不入库。

## 阶段 1 精度测试

```bash
cd pytorch
QUANT_LSTM_TEST_SUITE=basic PYTHONPATH=. python3 tests/test_float_reference.py
QUANT_LSTM_TEST_SUITE=strict PYTHONPATH=. python3 tests/test_float_reference.py
```

测试固定关闭 TF32，并分别报告 `output`、`h_n`、`c_n` 的 MAE、MSE 和余弦相似度。
公共 C++ 接口已提供两套接收预量化 q 的 CPU quantized reference；CUDA 量化
forward 尚未实现。当前精度报告范围为 `synthetic_numeric`，真实数据状态为
`not_configured`。反向传播、多层和双向接口尚未实现。