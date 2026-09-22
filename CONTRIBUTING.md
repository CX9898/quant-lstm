# 贡献指南

## 开发环境

CPU-only 开发要求 CMake 3.24、C++17 compiler、Python 3 和
`nlohmann_json` 3.11.2。CUDA 与 PyTorch 开发还要求 CUDA Toolkit、可访问的 NVIDIA
GPU，以及与 CUDA 环境匹配的 PyTorch。完整安装方式见
[安装指南](docs/installation.md)。

仓库也提供统一开发镜像：

```bash
docker build -f docker/Dockerfile -t quant-lstm:cuda .
docker run --rm -it --gpus all -v "$PWD:/workspace" quant-lstm:cuda
```

## 修改与测试

修改应保持 C++/CUDA 核心拥有运算逻辑。Python 层负责公共模块接口、配置传递、布局
整理和扩展调度；测试 oracle 可以使用独立 PyTorch 公式。

提交前至少运行与改动范围对应的验证：

```bash
# C++ 格式
tools/format_cpp.sh --check

# CPU-only 构建、测试、安装和外部消费
tools/run_cpu_only_package_check.sh

# CUDA、PyTorch、QAT 和 ONNX 端到端测试
tools/run_end_to_end_test.sh

# 影响 CUDA kernel、workspace 或缓存时追加
tools/run_end_to_end_test.sh --with-cuda-validation --device 0
```

Speech Commands 测试使用外部数据集，不进入默认 CI。影响校准、QAT、位宽或模型级
精度的改动应按[真实网络测试说明](tests/real_network/README.md)运行相应 profile。

## 代码与提交

- C++ 和 CUDA 使用仓库根目录的 `.clang-format`。
- 修改保持在所属模块内，公共量化语义应集中在已有公共原语中。
- 不提交构建目录、Python cache、扩展 `.so`、测试报告或数据集 cache。
- commit message 使用英文 Conventional Commits，例如
  `fix(calibration): handle degenerate POT2 ranges`。
- 功能实现和测试可以拆分提交，但每个提交都应保持可构建，并明确对应行为。

## 文档更新

用户可见行为、配置、依赖、命令、制品、量化公式或测试阈值变化时，需要同步更新
对应权威文档和 [CHANGELOG](CHANGELOG.md)。根 README 只保留摘要和入口；完整内容
写入专题文档。开发过程和临时调试记录由 Git 历史保存。

提交文档前执行：

```bash
git diff --check
```

同时检查 Markdown 相对链接、代码围栏、JSON/Python/Shell 示例、公开路径和敏感信息。
性能或精度结论必须包含可复现条件，不得把单次本地结果写成通用保证。
