# quant-lstm

基于 CUDA FP32 载体的量化 LSTM 实现。当前已完成阶段 1：单层、单向 FP32
CPU/CUDA 前向、确定性测试数据和最小 PyTorch 推理接口。量化执行语义以
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

## 阶段 1 精度测试

```bash
cd pytorch
QUANT_LSTM_TEST_SUITE=basic PYTHONPATH=. python3 tests/test_float_reference.py
QUANT_LSTM_TEST_SUITE=strict PYTHONPATH=. python3 tests/test_float_reference.py
```

测试固定关闭 TF32，并分别报告 `output`、`h_n`、`c_n` 的 MAE、MSE 和余弦相似度。
阶段 1 只提供 FP32 inference forward；量化、反向传播、多层和双向接口尚未实现。