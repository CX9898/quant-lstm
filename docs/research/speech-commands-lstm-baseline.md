# Speech Commands LSTM 真实训练基线调研

## 1. 结论

真实网络测试建议以 Google Research `kws_streaming` 的 LSTM keyword spotting
模型为来源，保留其核心拓扑：

```text
1 s / 16 kHz waveform
  -> MFCC time sequence
  -> one unidirectional LSTM
  -> last time-step output
  -> dropout
  -> linear classifier
```

来源模型直接面向 Speech Commands v0.02，结构短，且 LSTM 是网络中唯一的时序
算子，适合做“只把 `torch.nn.LSTM` 替换为 `QuantLSTM`”的受控训练比较。Google
源码中的模型定义依次构造 speech feature、一个或多个 LSTM、flatten、dropout、可选
全连接层和最终分类层，见固定 commit 的
[`kws_streaming/models/lstm.py`](https://github.com/google-research/google-research/blob/4700efb9afa54286b0e04473ba80a13e8461e25f/kws_streaming/models/lstm.py#L80-L127)。

不能原样复制 Google 的默认 recurrent cell：官方配置使用 500 个 cell、200 维
projection 和 peephole，而当前 `QuantLSTM` 对齐的是标准单层 `nn.LSTM`，不支持
peephole/projection。测试应明确称为 **Google `kws_streaming` LSTM 拓扑的标准-LSTM
适配版**，而不是相同模型的逐参数复现。这个适配仍在 Google 官方配置的覆盖范围内：
其 toy LSTM 参数明确设置 `use_peepholes=0` 和 `num_proj=-1`，见
[`model_params.py`](https://github.com/google-research/google-research/blob/4700efb9afa54286b0e04473ba80a13e8461e25f/kws_streaming/models/model_params.py#L227-L239)。

## 2. 权威来源

本记录检索日期为 2026-09-21。Google Research 源码链接固定在 commit
`4700efb9afa54286b0e04473ba80a13e8461e25f`，避免 `master` 后续变化影响结论。

| 结论 | 一手来源 |
|---|---|
| Google 提供独立 LSTM keyword spotting 模型，拓扑为 feature -> LSTM -> flatten -> dropout -> dense | [Google Research LSTM model](https://github.com/google-research/google-research/blob/4700efb9afa54286b0e04473ba80a13e8461e25f/kws_streaming/models/lstm.py#L80-L127) |
| 官方 v0.02 LSTM 实验使用 40 ms window、20 ms stride、40 mel bins、20 MFCC、500 cell、200 projection、peephole、dropout 0.3 | [Google Research 12-label experiment, `lstm_peep`](https://github.com/google-research/google-research/blob/4700efb9afa54286b0e04473ba80a13e8461e25f/kws_streaming/experiments/kws_experiments_paper_12_labels.md#L306-L338) |
| 官方另有无 peephole/projection、非 stateful 的 toy LSTM 配置，可作为标准 LSTM 适配依据 | [Google Research toy LSTM parameters](https://github.com/google-research/google-research/blob/4700efb9afa54286b0e04473ba80a13e8461e25f/kws_streaming/models/model_params.py#L227-L239) |
| 完整官方 LSTM 实验报告 float/quant accuracy 均为 97.3%，但这是 Google/TensorFlow 的完整训练配方结果 | [Google Research quantized 12-label experiment, `lstm_peep`](https://github.com/google-research/google-research/blob/4700efb9afa54286b0e04473ba80a13e8461e25f/kws_streaming/experiments/kws_experiments_quantized_12_labels.md#lstm_peep) |
| 官方默认数据 URL 是 `speech_commands_v0.02.tar.gz` | [Google Research base parser](https://github.com/google-research/google-research/blob/4700efb9afa54286b0e04473ba80a13e8461e25f/kws_streaming/train/base_parser.py#L33-L39) |
| 官方默认特征参数为 16 kHz、1 s、40 ms window、20 ms stride、40 mel bins、20 DCT/MFCC coefficients | [Google Research speech feature flags](https://github.com/google-research/google-research/blob/4700efb9afa54286b0e04473ba80a13e8461e25f/kws_streaming/train/base_parser.py#L247-L286) 和 [MFCC flags](https://github.com/google-research/google-research/blob/4700efb9afa54286b0e04473ba80a13e8461e25f/kws_streaming/train/base_parser.py#L354-L385) |
| v0.02 有 105,829 条、35 个词、2,618 位说话者；音频最长 1 s、16-bit mono PCM、16 kHz | [Speech Commands v2 原始论文，第 6 节](https://arxiv.org/pdf/1804.03209v1) |
| 标准 12 类为 10 个关键词加 unknown 和 silence；论文要求测试时各类等权 | [Speech Commands v2 原始论文，第 7.1 节](https://arxiv.org/pdf/1804.03209v1) |
| 官方列表定义 validation/testing，其他音频属于 training；划分以文件名 hash 保持跨版本稳定 | [Speech Commands v2 原始论文，第 7 节](https://arxiv.org/pdf/1804.03209v1) |
| Google 参考实现先去掉文件名 `_nohash_` 后缀，再用 SHA-1 分桶，保证同一说话者的相关录音不跨集合 | [Google Research `which_set`](https://github.com/google-research/google-research/blob/4700efb9afa54286b0e04473ba80a13e8461e25f/kws_streaming/data/input_data_utils.py#L33-L78) |
| TensorFlow Datasets 也读取 archive 内的列表，使用 16 kHz，并把 10 个目标词之外的词映射为 unknown、背景噪声切成 silence | [TFDS Speech Commands builder](https://github.com/tensorflow/datasets/blob/1401448b0c6c7aaf12bb5ee666a73fd6898650d1/tensorflow_datasets/datasets/speech_commands/speech_commands_dataset_builder.py#L150-L169) |

## 3. 数据契约

### 3.1 下载与校验

- URL：`https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz`
- archive size：`2,428,923,189` bytes
- SHA-256：`af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58`

大小和 checksum 来自 TensorFlow Datasets 仓库的
[`checksums.tsv`](https://github.com/tensorflow/datasets/blob/1401448b0c6c7aaf12bb5ee666a73fd6898650d1/tensorflow_datasets/datasets/speech_commands/checksums.tsv)，
下载 URL 同时由 Google Research 的
[`base_parser.py`](https://github.com/google-research/google-research/blob/4700efb9afa54286b0e04473ba80a13e8461e25f/kws_streaming/train/base_parser.py#L33-L39)
使用。测试脚本应缓存 archive 和解压目录，并在已有 archive 上检查 SHA-256；不能每次
测试重新下载 2.4 GB。

### 3.2 标签

保持官方 12 类协议：

```text
yes, no, up, down, left, right, on, off, stop, go, _unknown_, _silence_
```

10 个词列表以及额外 silence/unknown 的规则来自 Google Research 默认参数和
`prepare_words_list()`，见
[`base_parser.py`](https://github.com/google-research/google-research/blob/4700efb9afa54286b0e04473ba80a13e8461e25f/kws_streaming/train/base_parser.py#L97-L110)
与
[`input_data_utils.py`](https://github.com/google-research/google-research/blob/4700efb9afa54286b0e04473ba80a13e8461e25f/kws_streaming/data/input_data_utils.py#L81-L94)。
非目标词应确定性抽样为 unknown；silence 应从 `_background_noise_` 中确定性裁出 1 秒
片段。缩小测试仍应保持每类样本数一致，否则短训练结果会主要反映类别不平衡。

### 3.3 划分

测试直接消费 archive 内的 `validation_list.txt` 和 `testing_list.txt`：

- validation：`validation_list.txt` 中的音频；
- testing：`testing_list.txt` 中的音频；
- training：排除上述两个列表及 `_background_noise_` 后的其余音频；
- 同一 speaker/hash group 不得跨 split；
- 限制样本数时必须在完成官方 split 之后，按固定 seed 在每个 label 内抽样，不能先
  截断全量文件再划分。

这是原始论文规定的可比协议。PyTorch 官方 `torchaudio.datasets.SPEECHCOMMANDS`
也按相同列表定义三个 subset，代码可作为 PyTorch 侧数据读取参考，见
[`speechcommands.py`](https://github.com/pytorch/audio/blob/ec13a815b13ec6be3eeb8c3eb9ccb725dc322233/src/torchaudio/datasets/speechcommands.py)。

## 4. 特征和模型适配

### 4.1 特征

建议使用固定、无可训练参数的 MFCC，并让两个训练分支共享同一份预处理结果：

| 参数 | 值 | 依据 |
|---|---:|---|
| sample rate | 16,000 Hz | 数据集与 Google parser |
| clip length | 1,000 ms | 数据集与 Google parser |
| analysis window | 40 ms / 640 samples | Google LSTM experiment |
| hop | 20 ms / 320 samples | Google LSTM experiment |
| mel bins | 40 | Google LSTM experiment |
| MFCC coefficients | 20 | Google LSTM experiment |
| sequence shape | `[batch, 49, 20]` | `1 + floor((16000 - 640) / 320) = 49` |

实现可用 `torchaudio.transforms.MFCC` 或纯 PyTorch 等价变换，但测试结果必须记录所用
版本和全部 feature 参数。49-frame 计算与官方实现一致，见
[`model_flags.py`](https://github.com/google-research/google-research/blob/4700efb9afa54286b0e04473ba80a13e8461e25f/kws_streaming/models/model_flags.py#L37-L56)；
官方外置 MFCC 流程依次执行 spectrogram、Mel filterbank、log 和 DCT，见
[`input_data.py`](https://github.com/google-research/google-research/blob/4700efb9afa54286b0e04473ba80a13e8461e25f/kws_streaming/data/input_data.py#L478-L502)。
特征应在 CPU 预计算/缓存，然后 batch 一次性送到 CUDA；
这不是 LSTM 的 CPU fallback。`nn.LSTM` 和 `QuantLSTM` 的 forward/backward 必须都在
CUDA 上执行。

### 4.2 适配模型

建议的默认测试网络为：

```text
MFCC[batch, 49, 20]
  -> single-layer, unidirectional standard LSTM(input=20, hidden=64)
  -> output[:, -1, :]
  -> dropout(p=0.0 for controlled comparison)
  -> Linear(64, 12)
```

与 Google 原模型相比有三项有意缩小：

1. hidden size 从 500 降为 64，控制真实数据测试时间和 CUDA memory；
2. 禁用 peephole 和 200 维 projection，使 cell 语义严格落在当前 `QuantLSTM` 支持面；
3. 对照测试将 dropout 设为 0，避免不同 kernel 的轻微数值差异因随机 mask 被放大。

这些修改保留了要验证的主路径（时间特征 -> LSTM -> 最终状态分类），但缩小后的绝对
精度不能与 Google 报告的 97.3% 横向比较。

## 5. 公平的训练比较

测试报告以下四个训练分支：

| 分支 | recurrent layer | 用途 |
|---|---|---|
| baseline | `torch.nn.LSTM` | 原始 PyTorch 训练基线 |
| replacement-8 | `QuantLSTM(use_quantization=True)`, 8-bit | 8-bit QAT 替换效果 |
| replacement-16 | `QuantLSTM(use_quantization=True)`, 16-bit | 16-bit QAT 替换效果及 FP32 integer-carrier 风险 |
| native-float diagnostic | `QuantLSTM(use_quantization=False)` | 区分 native CUDA 算子差异与量化/STE 差异 |

公平性约束：

1. 同一个固定 seed 生成模型初始参数；将 baseline 的 LSTM 与 classifier 权重复制给
   replacement，不能分别随机初始化。
2. 四个分支使用完全相同的 example IDs、预计算 feature、batch 顺序、loss、optimizer、
   learning rate、epoch 数和 gradient clipping。
3. QAT 分支在训练前用固定的 training calibration subset 完成 calibration/finalize；
   validation/testing 数据绝不能参与 calibration。
4. 不允许 PyTorch runtime 的 CPU LSTM 或 CPU reference fallback。CPU 只负责文件读取和
   feature preprocessing；模型 tensor 进入 recurrent layer 前必须已经是 CUDA FP32。
5. 分支逐个训练，训练结束后清理 CUDA cache；不要并行训练导致显存或调度差异。
6. 每个分支至少记录首个 epoch 和最后一个 epoch 的 train loss、validation accuracy、
   test accuracy、总训练时间以及最终参数更新范数。

## 6. 测试规模与门禁建议

2.4 GB 下载和真实训练不适合并入每个 PR 的普通单元测试。建议同一脚本提供两档：

| 档位 | 每类上限 | epoch | 作用 |
|---|---:|---:|---|
| smoke | train 64 / validation 16 / test 16 | 2 | 手动或带缓存的 GPU CI，验证真实 I/O、forward、backward、STE 全链路 |
| report | 不截断，或由显式参数指定较大上限 | 配置化 | 产生有意义的精度比较报告，不作为常规 CI 默认项 |

`smoke` 的硬门禁只应检查：

- 数据 split 无交集、12 类齐全且抽样平衡；
- 两个必需分支的 loss/gradient 均为 finite，LSTM 与 classifier 参数确实更新；
- QAT 确实进入 quantized native CUDA forward/backward，并产生/消费 STE mask；
- 输出 JSON 中包含相同 example IDs/seed/config 和各分支指标；
- 进程返回成功且没有 CPU LSTM fallback。

不要在获得本仓库机器上的重复实验数据前硬编码诸如“QAT 精度至少 90%”或“相对 baseline
下降不超过 1%”的门禁。2 epoch、每类 64 条的结果方差远大于完整 Google 配方；Google
的 97.3% 来自 80,000 steps、500-cell peephole/projection 模型和 SpecAugment，不能推导
缩小测试阈值。第一版应固定 seed 并保存 baseline/QAT 报告；积累至少 3 次相同配置结果
后，再用观测到的最差值加容差冻结版本化阈值。

## 7. 实现验收清单

- 脚本位于 `tests/`，默认不隐式下载；显式 `--download` 才联网。
- 支持 `--dataset-root`、`--seed`、`--epochs`、每 split 每类上限和 JSON report 路径。
- 下载后验证 archive SHA-256，复用缓存。
- 先执行官方 split，再执行固定 seed 的 per-class 抽样。
- 同一模型类只通过构造参数切换 `nn.LSTM` / `QuantLSTM`，其余代码路径相同。
- 对 baseline 和 replacement 复制相同初始化，并验证 copy 后张量逐值相等。
- QuantLSTM 训练前只用 training subset 校准；forward/backward tensor 均为 CUDA FP32。
- 报告训练曲线、validation/test top-1、相对 baseline 差值、参数更新范数和时间。
- 失败时保留 report，便于判断是数据、native backward、STE 还是数值精度问题。
