# CMake package 配置

本目录保存安装后供下游 CMake 项目使用的 package 配置模板。

- `quant-lstm-config.cmake.in` 由根 `CMakeLists.txt` 的
  `configure_package_config_file()` 处理。
- 生成文件查找 `nlohmann_json`，并仅在 CUDA build 中查找 `CUDAToolkit`。
- 生成文件加载安装时导出的 `quant_lstm::quant_lstm` 和
  `quant_lstm::headers` targets。

该模板不参与 quant-lstm 自身 target 的定义。完整安装和下游消费方式见
[安装指南](../docs/installation.md#3-安装-cuda-c-package)。
