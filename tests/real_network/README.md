# Speech Commands：浮点预训练、PTQ 和固定参数 QAT

真实数据质量门禁采用 `MFCC [49,20] -> 单层单向 LSTM(H64) -> 浮点分类头`。
只替换 LSTM；没有 peephole/projection，不是 Google 原模型或其论文精度的复现。
本项目的质量验收流程是 **浮点预训练 → PTQ 校准 → 固定参数 QAT → 推理**。
不再把随机初始化后从零训练 QAT 的收敛精度作为支持能力。

## 1. 第三方来源

Google Research 引用固定在 commit
`4700efb9afa54286b0e04473ba80a13e8461e25f`：

- [LSTM 模型拓扑](https://github.com/google-research/google-research/blob/4700efb9afa54286b0e04473ba80a13e8461e25f/kws_streaming/models/lstm.py#L80-L127)
- [无 peephole/projection 的 toy 参数](https://github.com/google-research/google-research/blob/4700efb9afa54286b0e04473ba80a13e8461e25f/kws_streaming/models/model_params.py#L227-L239)
- [Speech Commands URL 与特征参数](https://github.com/google-research/google-research/blob/4700efb9afa54286b0e04473ba80a13e8461e25f/kws_streaming/train/base_parser.py#L33-L39)
- [Google Research Apache-2.0 许可证](https://github.com/google-research/google-research/blob/4700efb9afa54286b0e04473ba80a13e8461e25f/LICENSE)

数据集统计和官方 split 语义来自
[Speech Commands v2 论文](https://arxiv.org/abs/1804.03209)。下载文件为
`speech_commands_v0.02.tar.gz`，大小 `2,428,923,189` bytes，SHA-256 为：

```text
af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58
```

校验值来源于
[TensorFlow Datasets checksums](https://github.com/tensorflow/datasets/blob/1401448b0c6c7aaf12bb5ee666a73fd6898650d1/tensorflow_datasets/datasets/speech_commands/checksums.tsv)。
仓库没有复制 Google Research 源码或提交 Speech Commands 音频；测试代码只根据公开
拓扑和数据契约实现独立 PyTorch 版本。

## 2. 训练与选模协议

- 每个种子先用 `torch.nn.LSTM` 预训练 50 轮，Adam lr=0.003，第 25/40 轮后乘 0.3。
- 按验证集准确率最大、同分时验证 loss 最低选择浮点检查点。浮点验证基线不达标时停止，不能继续把结果解释为 QAT 质量。
- 从同一个所选浮点检查点克隆原生浮点续训、INT8 QAT、INT16 QAT；记录完整权重 SHA256 并核对完全一致。
- 仅使用 training 样本进行一次 PTQ 校准：INT8 使用 SQNR，INT16 使用 MinMax。
- 三个微调分支使用相同 batch 顺序、全新 Adam、lr=0.0003、20 轮、梯度裁剪 5；TF32 关闭。
- PTQ 的 scale、zero point、位宽和范围全程固定；master 权重仍更新，每次前向仍执行量化。
- 每个阶段完成训练和验证选模后才评估测试集，测试集不参与选模或调参。
- 微调阶段第 0 轮可被选中。分别报告 `starting_test`（QAT 分支即 PTQ）、`selected_test`、`best_trained_test`（epoch>0）和 `last_test`。训练函数结束时恢复验证集所选权重。
- `selected_quantization_error` 和扩展校准/位宽诊断对应所选检查点，不能当成末轮结果。

快速 profile：4 类（yes/no/up/down），512/128/128 条训练/验证/测试，batch 32；
PTQ 使用全部 512 条训练数据。完整 profile：35 类，84,843/9,981/11,005 条，batch 256；
PTQ 使用类别平衡的 1,024 条训练数据。两者均使用种子 20260921、20260922、20260923、20261008。
MFCC 仅以训练集均值和标准差归一化。全量验收默认重新提取特征，不复用旧缓存。

## 3. 运行

先构建 CUDA 扩展，确保 Python 可导入 torch、torchaudio 和 `_quant_lstm`，并准备官方数据集。
从仓库根目录运行：

```bash
# 快速 profile（四种子，预训练 50 轮 + 微调 20 轮）
tests/real_network/run_speech_commands_lstm_test.sh --dataset-root /datasets/speech_commands_v0.02

# 全部官方样本
tests/real_network/run_speech_commands_lstm_test.sh --dataset-root /datasets/speech_commands_v0.02 --full-dataset

# 无需音频数据的编排、固定参数和选模回归
PYTHONPATH=pytorch:tests/real_network python -m unittest test_pretrained_qat_protocol test_qat_fixed_quant_params

# 独立前向、掩码和梯度 oracle
PYTHONPATH=pytorch python -m unittest tests.test_qat_independent
```

使用 `PYTHON=/path/to/python` 选择解释器；仅显式 `--download` 才下载官方 2.3 GiB 数据。
CUDA 不可用时，正式 runner 以非零状态退出；不能将 unittest 的可选 skip 视为通过。
报告输出到 `tests/results/speech_commands_lstm_training.json` 和
`tests/results/speech_commands_lstm_full_training.json`（均被 Git 忽略）。

直接运行 `speech_commands_lstm_training.py` 可配置 `--pretrain-epochs`、
`--pretrain-learning-rate`、`--minimum-pretrain-validation-accuracy`；
`--epochs` 和 `--learning-rate` 仅控制预训练后的微调。正式门禁使用测试中固定的协议。

## 4. 分层验收

数值正确性继续由独立前向/掩码/反向 oracle、真实 batch 梯度、固定参数、
权重更新和保存/重新加载回归严格验收。随机小模型只用于这些性质检查，不要求分类收敛。

质量策略固定在 `pretrained_quality_assertions.py`，实跑前确定；本次增加种子
20261008 验证，不能根据测试结果事后移动阈值：

| 要求 | 快速 profile | 全量 profile |
|---|---:|---:|
| 浮点所选验证/测试准确率 | ≥80% | ≥90% |
| 浮点测试 Macro-F1 | ≥78% | ≥89% |
| INT8 相对各自浮点的准确率和 Macro-F1 最大下降 | 5 个百分点 | 2 个百分点 |
| INT16 相对各自浮点的准确率和 Macro-F1 最大下降 | 3 个百分点 | 1 个百分点 |
| 浮点及 QAT 最差类别 F1 | ≥60% | ≥65% |

这些条件分别对每个种子验证，对 QAT 的 `selected_test` 和 `best_trained_test` 都要求通过，
防止仅选中 PTQ 第 0 轮掩盖训练退化。末轮结果必须报告，但不要求每轮单调改善。
浮点续训作为额外训练的对照，不作为 QAT 初始浮点基线。

以下仍记录在报告中，但不再作为分类质量必然规律：INT8/INT16 之间的排序、
不同校准方法的 MAE 排名/比例、同权重预测一致率、逐时间步误差及 clamp rate。
这些诊断不替代独立 oracle 对数学正确性的阻断检查。

报告 schema 为 10；多种子结果保留各自完整训练历史、校准来源、参数指纹、
PTQ/所选/实际训练后最佳/末轮分类指标和梯度证据。旧字段不应解释为新协议的结果。

## 5. 历史记录

旧从零训练 QAT 的结果见 [HISTORICAL_RESULTS.md](HISTORICAL_RESULTS.md)。
旧质量门禁因目标协议不匹配被替换，没有声称旧训练能力已修复或通过。
INT16 的 FP32 q-carrier 累加精度风险仍存在；当前验收不等于纯整数硬件逐位一致性验证。
