# 历史从零训练 QAT 实验记录

以下为原文保存的历史结果，使用旧训练/校准协议，部分记录早于数学修复；
它们不是当前 HEAD 的验收结果，也不代表当前承诺支持从零训练 QAT。
旧固定参数、从零训练 10 轮门禁的最后一次运行另有 3 个质量断言失败：
INT8 验证准确率 48.4375% < 50%，INT16 验证准确率 53.90625% < 60%，
INT8 同权重预测一致率 61.71875% < 90%。该门禁因目标协议不匹配而被替换，
不能把替换视为这项旧训练能力已经修复。

快速 profile 使用 NVIDIA RTX 6000D、PyTorch 2.13.0+cu130、10 epochs 和 primary
seed `20260921`：

| Metric | `torch.nn.LSTM` | Native FP32 | INT8 QAT | INT16 QAT |
| --- | ---: | ---: | ---: | ---: |
| Initial train loss | 1.38985 | 1.38985 | 1.38983 | 1.38985 |
| Final train loss | 0.90363 | 0.93811 | 0.91254 | 0.89466 |
| Best validation accuracy | 59.38% | 59.38% | 57.03% | 67.19% |
| Final test accuracy | 64.06% | 67.97% | 65.63% | 67.19% |
| Parameter update norm | 7.82258 | 7.93028 | 9.01713 | 8.09350 |
| Logit MAE vs own native path | N/A | N/A | 0.017390 | 0.000098 |
| Prediction agreement | N/A | N/A | 99.22% | 100% |

三个 seed 的最低 test accuracy 为 PyTorch 59.38%、INT8 54.69%、INT16 60.94%；
INT8/INT16 mean test accuracy 为 60.68% 和 65.10%。真实 batch CUDA backward 的
最大绝对误差为 `1.31e-8`。

完整 profile 使用相同 GPU、seed `20260921`、10 epochs、hidden size 64 和 batch
size 256。官方 split 包含 84,843 条 training、9,981 条 validation 和 11,005 条
testing 音频，split overlap 为 0：

| Metric | `torch.nn.LSTM` | Native FP32 | INT8 QAT | INT16 QAT |
| --- | ---: | ---: | ---: | ---: |
| Best validation accuracy | 90.18% | 89.94% | 86.90% | 87.86% |
| Final test accuracy | 89.09% | 89.13% | 85.16% | 86.96% |
| Test macro F1 | 0.8815 | 0.8813 | 0.8390 | 0.8628 |
| Logit MAE vs own native path | N/A | N/A | 0.562870 | 0.003233 |
| Prediction agreement | N/A | N/A | 92.24% | 99.94% |

完整 profile 的真实 batch backward 最大绝对误差为 `3.73e-9`。INT16 logit MAE
为 INT8 的 0.58%，因此完整数据结果可以区分两个位宽路径。上述结果只适用于列出的
数据、训练参数、软件和 GPU 环境，不构成其他模型或硬件上的精度保证。

INT16 q-carrier 的部分乘积和累加可能超过 FP32 精确整数范围 `2^24`。报告保留
`precision_risk`，但没有 non-finite unsafe entry，也不会选择 CPU 或浮点 fallback。
