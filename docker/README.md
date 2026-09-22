# Docker 构建环境

`Dockerfile` 提供 CUDA、PyTorch、CMake、编译器和测试依赖的开发环境。镜像不复制
项目源码，运行时将仓库挂载到 `/workspace`：

```bash
docker build -f docker/Dockerfile -t quant-lstm:cuda .
docker run --rm -it --gpus all -v "$PWD:/workspace" quant-lstm:cuda
```

宿主机需要 NVIDIA Container Toolkit。可配置的 base image、PyTorch index、镜像源
和目标 CUDA architecture 由 `Dockerfile` 顶部的 build arguments 定义。安装流程见
[安装指南](../docs/installation.md#5-使用-docker-构建环境)。
