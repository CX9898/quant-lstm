# Speech Commands LSTM 真实网络测试

本测试将 Google Research `kws_streaming` 的关键词识别 LSTM 拓扑适配到本仓库支持
的标准 LSTM 接口，并比较 `torch.nn.LSTM` 与三种 `QuantLSTM` 执行模式。

```text
1 s / 16 kHz waveform
  -> 49 x 20 MFCC sequence
  -> one unidirectional LSTM (hidden size 64)
  -> final hidden state
  -> dropout(0)
  -> profile-sized linear classifier
```

Google 的完整模型使用 peephole 和 projection。本测试禁用这两项，因为
`torch.nn.LSTM` 与 `QuantLSTM` 的替换边界都是标准 LSTM。测试属于 Google 拓扑的
标准-LSTM 适配，不是其模型参数或 97.3% 结果的复现。

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

## 2. 对照契约

四个分支共享样本、预计算 MFCC、初始化参数、batch 顺序、optimizer、学习率、epoch、
loss 和 gradient clipping。唯一变化是 recurrent module：

| 分支 | Recurrent module | 校准 |
| --- | --- | --- |
| baseline | CUDA `torch.nn.LSTM` | 无 |
| native FP32 | `QuantLSTM(use_quantization=False)` | 无 |
| INT8 QAT | `QuantLSTM` 8-bit q-carrier | SQNR |
| INT16 QAT | `QuantLSTM` 16-bit q-carrier | MinMax |

QAT 初始校准使用类别平衡的 training subset。每个 epoch 训练结束后刷新 range，再执行
validation，并把结果用于下一 epoch。Validation 和 testing 样本不参与校准。

快速 profile 使用 `yes`、`no`、`up` 和 `down`，每类选择 128 条 training、32 条
validation 和 32 条 testing 音频。完整 profile 对 35 个词目录分类，并按照官方
split 使用全部 105,829 条带标签音频；`_background_noise_` 只单独审计，不作为词类。

MFCC 参数为 16 kHz、1 s、40 ms window、20 ms hop、40 mel bins 和 20 coefficients，
输出 shape 为 `[batch,49,20]`。特征在 CPU 分块提取和缓存，然后送入 CUDA；该预处理
不属于 LSTM CPU fallback。

## 3. 运行

前置条件：

- 已按[安装指南](../../docs/installation.md#2-安装-pytorch-cuda-模块)构建 CUDA 核心
  和 `_quant_lstm` extension；
- 当前 Python 环境可以导入 `torch`、匹配版本的 `torchaudio` 和 `_quant_lstm`；
- 运行环境可以访问 CUDA GPU；
- 数据集目录包含 `validation_list.txt` 和 `testing_list.txt`。

以下命令从仓库根目录执行。

快速 profile：

```bash
tests/real_network/run_speech_commands_lstm_test.sh \
  --dataset-root /datasets/speech_commands_v0.02
```

完整 35 类 profile：

```bash
tests/real_network/run_speech_commands_lstm_test.sh \
  --dataset-root /datasets/speech_commands_v0.02 \
  --full-dataset
```

只有显式传入 `--download` 时，脚本才会下载官方 2.3 GiB archive：

```bash
tests/real_network/run_speech_commands_lstm_test.sh --download
```

也可以通过 `QUANT_LSTM_SPEECH_COMMANDS_ROOT` 指定已解压数据集，或通过
`QUANT_LSTM_SPEECH_COMMANDS_CACHE` 指定下载和解压 cache。

快速 profile 输出：

```text
tests/results/speech_commands_lstm_training.json
```

完整 profile 输出：

```text
tests/results/speech_commands_v0.02_full_mfcc.pt
tests/results/speech_commands_lstm_full_training.json
```

完整 profile 的 feature cache 只有在 split path digest 和特征契约都匹配时才会复用。
JSON report 包含训练曲线、validation/test 指标、35x35 confusion matrix、逐类指标、
参数更新、校准诊断和 native QAT checkpoint 证据。测试进程退出码为 0 且全部断言通过
表示验收成功。数据、cache 和报告位于 Git 忽略目录。

## 4. 快速 Profile 门禁

测试要求四个分支都降低训练 loss 并更新参数，两个 QAT 分支必须产生 native
checkpoint。核心阈值为：

| 指标 | 门禁 |
| --- | --- |
| baseline 与 native FP32 best validation | `>=55%` |
| baseline 与 native FP32 final test | `>=60%` |
| native FP32 相对 baseline validation 降幅 | `<=5` percentage points |
| INT8 best validation 与 final test | `>=50%`，相对 baseline 降幅 `<=10` points |
| INT16 best validation 与 final test | `>=60%`，相对 baseline 降幅 `<=5` points |
| INT16 相对 INT8 | validation 至少高 `5` points，test 不低于 INT8，三 seed mean test 至少高 `2` points |
| INT16 logit MAE | 小于 INT8 的 `10%`，cosine `>=0.9999` |
| INT8 prediction agreement | primary `>=90%`，每个 quality seed `>=95%` |
| weight Clamp rate | 每个 epoch 的 `weight_ih/weight_hh <10%` |
| CUDA QAT backward | 七组梯度相对独立 oracle 最大绝对误差 `<=5e-6` |

扩展诊断还要求：

- 报告 49 个时间步、末四分之一、P99/max 和 normalized error；
- 用固定 INT8 权重逐一把 18 个量化点提升为 INT16，并按 logit MAE 收益排序；
- 使用 128/512 个平衡样本比较 MinMax、Percentile、SQNR 的 8/16-bit 校准矩阵；
- 配对 INT16 case 的 cell-state resolution 至少细 166 倍，logit MAE 小于 INT8 的
  `1%`，且小于一个 INT16 cell-state step；
- 每轮校准每类使用 32 个样本，safety report 不含 non-finite unsafe entry；
- seed `20260921`、`20260922` 和 `20260923` 均降低训练 loss。

## 5. 可复现实测结果

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
