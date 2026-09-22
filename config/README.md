# 配置与 Schema

本目录维护量化配置、内部参数 bundle 和公共 Python 参数文档的机器可读契约。

- `defaults/lstm_quant_default_v1.json` 是运行时默认值的唯一来源。
- `schema/lstm_quant_override.schema.json` 校验稀疏用户 override。
- `schema/lstm_quant_resolved.schema.json` 校验完整 resolved config。
- `schema/lstm_quant_params_bundle.schema.json` 校验 C++ 内部 canonical bundle。
- `schema/lstm_pytorch*_quant_params.schema.json` 校验公共参数交换文档。

Schema 和默认文件发生变化时，需要同步更新
[配置参考](../docs/configuration.md)、对应测试和 [CHANGELOG](../CHANGELOG.md)。
