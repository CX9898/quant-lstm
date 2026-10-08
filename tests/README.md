# 测试目录

测试按验证边界组织：

| 目录 | 内容 |
| --- | --- |
| `cpp/` | CPU reference、配置、量化原语和参数 I/O |
| `cuda/` | CUDA rounding、forward、校准和严格精度 |
| `python/` | Schema、报告和阈值契约 |
| `golden/` | 入库的 Golden schema 与规范 JSON |
| `precision/` | 精度矩阵、阈值和报告 schema |
| `benchmarks/` | CUDA benchmark、设备 profile 和性能阈值 |
| `package/` | 独立 CMake consumer 和源码无关的 Python wheel 安装 |
| `real_network/` | Speech Commands 浮点预训练、PTQ 与固定参数 QAT 质量门禁 |

构建产物和运行报告写入 Git 忽略目录。

## 按改动范围验证

以下命令从仓库根目录执行。开发环境与 Docker 用法见
[安装指南](../docs/installation.md)。提交前至少运行与改动范围对应的验证：

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
精度的改动应按[真实网络测试说明](real_network/README.md)运行相应 profile。
