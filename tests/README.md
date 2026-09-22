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
| `package/` | 独立 CMake consumer |
| `real_network/` | Speech Commands v0.02 训练比较 |

构建产物和运行报告写入 Git 忽略目录。默认验证入口为：

```bash
tools/run_cpu_only_package_check.sh
tools/run_end_to_end_test.sh
```

测试层级和改动对应的必跑范围见[贡献指南](../CONTRIBUTING.md)。
