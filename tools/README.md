# 维护脚本

| 脚本 | 用途 |
| --- | --- |
| `format_cpp.sh` | 检查或修复 C++/CUDA 格式 |
| `run_cpu_only_package_check.sh` | CPU 构建、CTest、安装和外部 consumer 验收 |
| `run_end_to_end_test.sh` | CUDA/C++、PyTorch、QAT 和 ONNX 端到端验证 |
| `run_stage4_cuda_validation.py` | benchmark、sanitizer 和 profiler 验证 |
| `check_stage9_cuda_performance.py` | 只读检查设备专用性能阈值 |
| `generate_golden.py` | 从规范 JSON 生成或检查 C++ fixture |
| `golden_reference_generator.cc` | 生成 reference Golden 数据 |
| `strict_jsonschema.py` | 严格 JSON Schema 校验辅助模块 |

脚本均从仓库根目录调用。环境依赖见[安装指南](../docs/installation.md)，验证范围和
命令见[测试说明](../tests/README.md#按改动范围验证)及
[CUDA 性能验收](../docs/cuda-performance.md)。
