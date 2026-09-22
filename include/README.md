# 公共 C++ 头文件

本目录是 `quant_lstm::headers` target 对外暴露的 include root。

- `lstm/` 定义 LSTM shape、配置、参数、校准和 CPU/CUDA 执行接口。
- `quantization/` 定义舍入、scale 编码、量化参数、直方图和数值安全原语。

下游项目应链接 `quant_lstm::quant_lstm` 或 `quant_lstm::headers`，不要依赖源码树的
绝对 include 路径。安装和 `find_package()` 示例见
[安装指南](../docs/installation.md#3-安装-cuda-c-package)。
