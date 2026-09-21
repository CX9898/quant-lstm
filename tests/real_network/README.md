# Speech Commands LSTM real-network test

This opt-in test adapts Google Research `kws_streaming`'s LSTM keyword-spotting
topology to the standard LSTM interface supported by this repository:

```text
1 s / 16 kHz waveform
  -> 49 x 20 MFCC sequence
  -> one unidirectional LSTM (hidden size 64)
  -> final hidden state
  -> dropout(0)
  -> four-class linear classifier
```

The source selection and exact upstream citations are recorded in
[`docs/research/speech-commands-lstm-baseline.md`](../../docs/research/speech-commands-lstm-baseline.md).
Peepholes and projection are disabled because neither `torch.nn.LSTM` nor the
current `QuantLSTM` replacement exposes Google's projected peephole cell.

## Comparison contract

The three branches share selected examples, precomputed MFCC tensors, initial
parameters, batch order, optimizer, learning rate, epochs, loss, and gradient
clipping. Only the recurrent module changes:

- baseline: `torch.nn.LSTM` on CUDA;
- replacements: 8-bit and 16-bit `QuantLSTM` QAT on CUDA, each calibrated only
  with the same training examples.

The deterministic subset contains `yes`, `no`, `up`, and `down`, with 128
training, 32 validation, and 32 testing examples per label. Selection happens
after applying v0.02's official `validation_list.txt` and `testing_list.txt`.
This is a short real-data regression, not Google's complete 12-label recipe,
and its accuracy must not be compared with Google's reported 97.3% result.

## Run

Build the CUDA library and PyTorch extension first, then pass an existing
extracted v0.02 directory:

```bash
tests/real_network/run_speech_commands_lstm_test.sh \
  --dataset-root /path/to/speech_commands_v0.02
```

To use the script-managed cache and explicitly download the official 2.3 GiB
archive:

```bash
tests/real_network/run_speech_commands_lstm_test.sh --download
```

The test requires `torch`, matching `torchaudio`, CUDA, and the built
`_quant_lstm` extension. It is intentionally excluded from default CI because
the external dataset is large. The complete JSON report is written to
`tests/results/speech_commands_lstm_training.json`.

## Acceptance thresholds

The test requires all three branches to reduce training loss and update
parameters, requires both replacements to expose native QAT checkpoints, and
checks:

- baseline best validation accuracy >= 40%;
- each QAT best validation accuracy >= 35%;
- each QAT result no more than 20 percentage points below baseline;
- neither calibration safety report contains a non-finite unsafe entry.

These conservative thresholds were frozen after three identical runs on
2026-09-21 with an NVIDIA RTX 6000D, PyTorch 2.13.0+cu130, and seed 20260921:

| Metric | `torch.nn.LSTM` | 8-bit QAT | 16-bit QAT |
|---|---:|---:|---:|
| Initial train loss | 1.38985 | 1.38986 | 1.38984 |
| Final train loss | 0.90363 | 0.98312 | 0.97134 |
| Best validation accuracy | 59.38% | 51.56% | 51.56% |
| Final test accuracy | 64.06% | 51.56% | 50.00% |
| Parameter update norm | 7.82258 | 8.22504 | 8.23538 |

The thresholds deliberately leave room for library and GPU variation while
still rejecting chance-level training or a broken QAT backward path.

The 8-bit calibration has 518 `exact_integer_range` entries and no precision
risk. The 16-bit calibration has 5 exact entries, 513 `precision_risk` entries,
and no non-finite unsafe entries. This is expected for the FP32 integer carrier:
16-bit products and accumulations can exceed FP32's exact integer range of
`2^24`. The warning is retained in the JSON safety report as required by the
quantized execution specification; it does not select a CPU or floating-point
LSTM fallback.
