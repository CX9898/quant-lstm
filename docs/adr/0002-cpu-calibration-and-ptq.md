# ADR-0002: Explicit CPU Calibration and PTQ

Status: Accepted

Date: 2026-10-26

## Context

Standalone CPU environments need LSTM calibration and PTQ without CUDA. The
existing C++ float and FP32 q-carrier references already own the required math.
ADR-0001's training and QAT constraints remain appropriate.

## Decision

Build a dedicated CPU extension with `RX_MET_ENABLE_CUDA=0` against a CPU-only
native library and compatible PyTorch. It accepts CPU float32 parameters,
inputs and states. Public execution requires `eval()` and disabled gradients;
training, backward and QAT are rejected. CUDA wheels retain their CUDA binding.

CPU calibration reuses a single native float forward and its trace, collects
batch histograms before session merging, and finalizes the existing parameter
format. CPU PTQ converts real tensors to the FP32 q-carrier reference grids and
returns dequantized values. Python owns no production LSTM formulas.

## Validation and Consequences

Native CTest covers histogram merge parity and existing numerical contracts.
`pytorch/tests/test_cpu_ptq.py` compares float/calibration outputs with PyTorch,
checks PTQ parameter round trips, layouts, directions, bias absence, calibration
methods, training rejection and execution through ONNX Runtime's CPU provider.
The CPU wheel must also import with its packaged config outside the source tree.

CPU and CUDA floating point activations may round differently at quantization
boundaries; no bitwise equality between devices is promised. This decision does
not introduce an automatic device fallback or a CPU QAT implementation.

## Revision in 1.1.1

CPU execution still requires `eval()`, but forward and calibration also accept
normal grad-enabled calls. `torch.no_grad()` and `torch.inference_mode()` remain
recommended for lightweight inference. Any backward traversal through the CPU
operator raises an explicit error rather than silently detaching gradients.
CPU training and QAT remain unsupported; the native forward and calibration
references are unchanged.
