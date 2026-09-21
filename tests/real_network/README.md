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

The four branches share selected examples, precomputed MFCC tensors, initial
parameters, batch order, optimizer, learning rate, epochs, loss, and gradient
clipping. Only the recurrent module changes:

- baseline: `torch.nn.LSTM` on CUDA;
- native-float diagnostic: `QuantLSTM(use_quantization=False)` on CUDA, with no
  calibration, quantization, or STE;
- replacements: 8-bit and 16-bit `QuantLSTM` QAT on CUDA. Calibration uses a
  deterministic class-balanced training subset. MinMax ranges are refreshed
  after every training epoch, before validation, and then reused by the next
  epoch. Validation and testing examples never participate in calibration.

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

The test requires all four branches to reduce training loss and update
parameters, requires both QAT replacements to expose native checkpoints, and
checks:

- baseline and native-float best validation accuracy >= 55% and final test
  accuracy >= 60%;
- native-float best validation accuracy no more than 5 percentage points below
  baseline;
- 8-bit QAT best validation and final test accuracy >= 50%, each no more than
  10 percentage points below baseline;
- 16-bit QAT best validation and final test accuracy >= 60%, each no more than
  5 percentage points below baseline;
- 16-bit QAT validation and test accuracy exceed 8-bit by at least 5 percentage
  points;
- final 16-bit logit MAE against the same trained model's native-float path is
  less than 10% of the corresponding 8-bit MAE, with cosine >= 0.9999;
- per-epoch `weight_ih` and `weight_hh` Clamp rates remain below 10%;
- each QAT epoch reports bias Clamp rates before and after optimizer updates,
  plus per-channel parameter values, representable ranges, quantization steps,
  and Clamp decisions before and after calibration refresh;
- final quantization error includes all 49 recurrent time steps, tail-quarter
  metrics, P99/max error, normalized errors, and prediction agreement;
- the trained INT8 model is evaluated with each of the 18 quantization points
  promoted to INT16 in isolation, using fresh balanced calibration and the same
  fixed model weights, then ranked by logit MAE improvement;
- every initial and refreshed calibration uses 32 samples from each label;
- neither calibration safety report contains a non-finite unsafe entry.

These thresholds were frozen after the STE Clamp fix and repeated deterministic
runs on 2026-09-21 with an NVIDIA RTX 6000D, PyTorch 2.13.0+cu130, and seed
20260921:

| Metric | `torch.nn.LSTM` | Native FP32 | 8-bit QAT | 16-bit QAT |
|---|---:|---:|---:|---:|
| Initial train loss | 1.38985 | 1.38985 | 1.38986 | 1.38984 |
| Final train loss | 0.90363 | 0.93811 | 0.97098 | 0.83593 |
| Best validation accuracy | 59.38% | 59.38% | 53.13% | 67.19% |
| Final test accuracy | 64.06% | 67.97% | 55.47% | 64.84% |
| Parameter update norm | 7.82258 | 7.93028 | 7.85768 | 8.38498 |
| Logit MAE vs own native-float path | N/A | N/A | 0.012899 | 0.000338 |

The fixed-weight INT8-to-INT16 ablation identifies `cell_state`, `input`, and
`weight_ih_linear` as the three largest individual logit-MAE contributors.
Promoting all points reduces MAE to `0.000043`. The recurrent sequence MAE
peaks at time step 16 rather than at the tail; the first, last, and final-quarter
MAEs are `0.009251`, `0.009131`, and `0.009800`, respectively. These diagnostics
separate the high bias Clamp rate observed during training from the dominant
forward quantization-error sources.

The thresholds leave several samples of accuracy headroom while rejecting
stale calibration, a broken QAT backward path, and an INT16 path that does not
provide a measurable advantage over INT8.

The 8-bit calibration has 518 `exact_integer_range` entries and no precision
risk. The 16-bit calibration has 5 exact entries, 513 `precision_risk` entries,
and no non-finite unsafe entries. This is expected for the FP32 integer carrier:
16-bit products and accumulations can exceed FP32's exact integer range of
`2^24`. The warning is retained in the JSON safety report as required by the
quantized execution specification; it does not select a CPU or floating-point
LSTM fallback.
