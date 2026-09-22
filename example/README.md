# C++ 示例

本目录包含两个最小 C++ 示例：

- `lstm_float_example.cc` 调用 CPU 浮点 reference forward。
- `lstm_int32_example.cc` 演示校准、参数量化、CPU int32 carrier forward 和输出反量化。

构建并运行：

```bash
cmake -S . -B build-example \
  -DQUANT_LSTM_ENABLE_CUDA=OFF \
  -DQUANT_LSTM_BUILD_TESTS=OFF \
  -DQUANT_LSTM_BUILD_EXAMPLES=ON
cmake --build build-example --parallel
build-example/lstm_float_example
build-example/lstm_int32_example
```

两个程序均退出码为 0 表示示例运行成功。示例输入用于接口演示，不代表模型校准数据。
