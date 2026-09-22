# ADR-0001：PyTorch runtime 使用 CUDA-only 执行边界

状态：Accepted

日期：2026-09-22

## 背景

quant-lstm 同时包含 CUDA 生产实现和 CPU reference。若 Python 模块根据 tensor
device 自动选择两套实现，CUDA 缺失、输入误放 CPU 或 backend 能力不一致都可能
静默改变训练和推理语义。CPU int32 reference 的激活边界还包含浮点函数，不能替代
CUDA q-carrier 生产路径。

## 决策

`QuantLSTM` 的完全浮点、量化和 QAT 路径只接受 CUDA FP32 tensor。Python binding
不注册 CPU fallback。CPU FP32 q-carrier 和 CPU int32 carrier 由 C++ API、测试和
示例显式调用。

## 理由

- 设备错误会在模块边界直接失败，调用方可以观察实际执行 backend。
- 浮点训练、量化执行、校准和 STE 均使用相同的 CUDA/C++ 所有权边界。
- CPU reference 可以保持面向 Golden 和整数语义验证的实现，不承担生产 fallback
  的兼容性承诺。

## 考虑过的方案

自动 CPU fallback 会降低无 CUDA 环境中的运行门槛，但会让同一 Python 模型随环境
改变 backend 和数值语义，因此未采用。

在 Python 中使用 PyTorch 公式实现 CPU fallback 可以提供功能覆盖，但会形成第二套
生产公式和 STE 实现，因此未采用。PyTorch 公式仅保留在测试 oracle 中。

## 影响与限制

- 用户必须把模块、输入和初始状态放到 CUDA。
- CPU-only 构建只提供 C++ package，不提供可运行的 `QuantLSTM` Python backend。
- 文档、测试和错误消息必须明确 CUDA-only 边界。
- 新增 CPU 生产 backend 需要新的 ADR、完整行为契约和独立验证，不得作为隐式回退
  接入。
