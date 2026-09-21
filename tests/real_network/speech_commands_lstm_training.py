"""Speech Commands v0.02 training comparison for torch LSTM and QuantLSTM QAT.

The model follows the small Google Research kws_streaming LSTM topology:
MFCC features, one LSTM, dropout, and a dense classifier. Peepholes and
projection are intentionally disabled because QuantLSTM does not expose them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tarfile
import time
import urllib.request
import warnings
import wave
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torchaudio
from torch import Tensor, nn
from torch.nn import functional as F


DATASET_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/"
    "speech_commands_v0.02.tar.gz"
)
DATASET_SHA256 = "af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58"
SAMPLE_RATE = 16_000
CLIP_SAMPLES = SAMPLE_RATE
MEL_BINS = 40
MFCC_BINS = 20
QAT_CALIBRATION_METHODS = {8: "sqnr", 16: "minmax"}


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_root: Path
    labels: tuple[str, ...] = ("yes", "no", "up", "down")
    train_samples_per_label: int = 128
    validation_samples_per_label: int = 32
    test_samples_per_label: int = 32
    hidden_size: int = 64
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 3.0e-3
    calibration_batches: int = 4
    calibration_refresh_epochs: int = 1
    quant_bitwidths: tuple[int, ...] = (8, 16)
    seed: int = 20260921
    quality_gate_seeds: tuple[int, ...] = ()
    device: str = "cuda"

    def validate(self) -> None:
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be non-empty and unique")
        for name in (
            "train_samples_per_label",
            "validation_samples_per_label",
            "test_samples_per_label",
            "hidden_size",
            "batch_size",
            "epochs",
            "calibration_batches",
            "calibration_refresh_epochs",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not self.quant_bitwidths:
            raise ValueError("quant_bitwidths must be non-empty")
        if len(set(self.quant_bitwidths)) != len(self.quant_bitwidths):
            raise ValueError("quant_bitwidths must be unique")
        if any(bitwidth not in (8, 16) for bitwidth in self.quant_bitwidths):
            raise ValueError("quant_bitwidths may only contain 8 or 16")
        if len(set(self.quality_gate_seeds)) != len(self.quality_gate_seeds):
            raise ValueError("quality_gate_seeds must be unique")
        if self.quality_gate_seeds and self.seed not in self.quality_gate_seeds:
            raise ValueError("quality_gate_seeds must include the primary seed")


class SpeechCommandsLstmClassifier(nn.Module):
    """MFCC-LSTM-Dropout-Dense keyword classifier."""

    def __init__(
        self,
        hidden_size: int,
        label_count: int,
        *,
        use_quant_lstm: bool,
        device: torch.device,
    ) -> None:
        super().__init__()
        if use_quant_lstm:
            from quant_lstm import QuantLSTM

            self.lstm = QuantLSTM(
                MFCC_BINS,
                hidden_size,
                batch_first=True,
                device=device,
                use_quantization=False,
            )
        else:
            self.lstm = nn.LSTM(
                MFCC_BINS,
                hidden_size,
                batch_first=True,
                device=device,
            )
        self.dropout = nn.Dropout(p=0.0)
        self.classifier = nn.Linear(hidden_size, label_count, device=device)

    def forward(self, features: Tensor) -> Tensor:
        _, (hidden, _) = self.lstm(features)
        return self.classifier(self.dropout(hidden[-1]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as source:
        for member in source.getmembers():
            member_path = (destination / member.name).resolve()
            try:
                member_path.relative_to(destination)
            except ValueError as error:
                raise RuntimeError(f"unsafe archive member: {member.name}") from error
        source.extractall(destination)


def prepare_dataset(cache_root: Path, *, download: bool) -> Path:
    """Return an extracted Speech Commands v0.02 root, optionally downloading it."""
    cache_root = Path(cache_root)
    dataset_root = cache_root / "SpeechCommands" / "speech_commands_v0.02"
    if (dataset_root / "validation_list.txt").is_file():
        return dataset_root
    if not download:
        raise RuntimeError(
            f"Speech Commands v0.02 is absent at {dataset_root}; pass --download"
        )

    cache_root.mkdir(parents=True, exist_ok=True)
    archive = cache_root / "speech_commands_v0.02.tar.gz"
    if not archive.is_file() or _sha256(archive) != DATASET_SHA256:
        temporary = archive.with_suffix(archive.suffix + ".part")
        print(f"Downloading {DATASET_URL} to {archive}", file=sys.stderr, flush=True)
        urllib.request.urlretrieve(DATASET_URL, temporary)
        if _sha256(temporary) != DATASET_SHA256:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("Speech Commands v0.02 SHA-256 mismatch")
        temporary.replace(archive)

    dataset_root.mkdir(parents=True, exist_ok=True)
    _safe_extract(archive, dataset_root)
    if not (dataset_root / "validation_list.txt").is_file():
        raise RuntimeError("extracted dataset is missing validation_list.txt")
    return dataset_root


def _listed_paths(dataset_root: Path, filename: str) -> set[str]:
    return {
        line.strip().replace("\\", "/")
        for line in (dataset_root / filename).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _select_audio_paths(
    config: ExperimentConfig, split: str
) -> list[tuple[Path, int]]:
    validation = _listed_paths(config.dataset_root, "validation_list.txt")
    testing = _listed_paths(config.dataset_root, "testing_list.txt")
    if split == "training":
        count = config.train_samples_per_label
        excluded = validation | testing
    elif split == "validation":
        count = config.validation_samples_per_label
        excluded = set()
    elif split == "testing":
        count = config.test_samples_per_label
        excluded = set()
    else:
        raise ValueError(f"unsupported split: {split}")

    selected: list[tuple[Path, int]] = []
    split_offsets = {"training": 0, "validation": 100_000, "testing": 200_000}
    split_offset = split_offsets[split]
    for label_index, label in enumerate(config.labels):
        label_dir = config.dataset_root / label
        if not label_dir.is_dir():
            raise RuntimeError(f"dataset label directory not found: {label_dir}")
        relative_paths = [f"{label}/{path.name}" for path in label_dir.glob("*.wav")]
        if split == "training":
            candidates = [path for path in relative_paths if path not in excluded]
        elif split == "validation":
            candidates = [path for path in relative_paths if path in validation]
        else:
            candidates = [path for path in relative_paths if path in testing]
        candidates.sort()
        random.Random(config.seed + split_offset + 1009 * label_index).shuffle(candidates)
        if len(candidates) < count:
            raise RuntimeError(
                f"{split} label {label!r} has {len(candidates)} samples, needs {count}"
            )
        selected.extend(
            (config.dataset_root / relative_path, label_index)
            for relative_path in candidates[:count]
        )
    return selected


def _load_waveforms(samples: Sequence[tuple[Path, int]]) -> tuple[Tensor, Tensor]:
    waveforms = []
    labels = []
    for path, label in samples:
        with wave.open(str(path), "rb") as source:
            if source.getsampwidth() != 2 or source.getcomptype() != "NONE":
                raise RuntimeError(f"expected uncompressed 16-bit PCM WAV: {path}")
            if source.getframerate() != SAMPLE_RATE:
                raise RuntimeError(
                    f"unexpected sample rate {source.getframerate()} in {path}"
                )
            channels = source.getnchannels()
            pcm = source.readframes(source.getnframes())
        waveform = torch.frombuffer(bytearray(pcm), dtype=torch.int16).float()
        waveform = waveform.reshape(-1, channels).mean(dim=1) / 32768.0
        if waveform.numel() < CLIP_SAMPLES:
            waveform = F.pad(waveform, (0, CLIP_SAMPLES - waveform.numel()))
        else:
            waveform = waveform[:CLIP_SAMPLES]
        waveforms.append(waveform)
        labels.append(label)
    return torch.stack(waveforms), torch.tensor(labels, dtype=torch.long)


def _mfcc(waveforms: Tensor) -> Tensor:
    transform = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=MFCC_BINS,
        log_mels=True,
        melkwargs={
            "n_fft": 640,
            "win_length": 640,
            "hop_length": 320,
            "f_min": 20.0,
            "f_max": 7600.0,
            "n_mels": MEL_BINS,
            "power": 2.0,
            "center": False,
            "mel_scale": "htk",
        },
    )
    with torch.no_grad():
        return transform(waveforms).transpose(1, 2).contiguous()


def _load_feature_sets(
    config: ExperimentConfig,
) -> tuple[dict[str, tuple[Tensor, Tensor]], dict[str, str]]:
    train_samples = _select_audio_paths(config, "training")
    validation_samples = _select_audio_paths(config, "validation")
    test_samples = _select_audio_paths(config, "testing")
    train_waveforms, train_labels = _load_waveforms(train_samples)
    validation_waveforms, validation_labels = _load_waveforms(validation_samples)
    test_waveforms, test_labels = _load_waveforms(test_samples)
    train_features = _mfcc(train_waveforms)
    validation_features = _mfcc(validation_waveforms)
    test_features = _mfcc(test_waveforms)
    mean = train_features.mean()
    standard_deviation = train_features.std().clamp_min(1.0e-6)
    feature_sets = {
        "training": ((train_features - mean) / standard_deviation, train_labels),
        "validation": (
            (validation_features - mean) / standard_deviation,
            validation_labels,
        ),
        "testing": ((test_features - mean) / standard_deviation, test_labels),
    }
    example_digests = {}
    for split, samples in (
        ("training", train_samples),
        ("validation", validation_samples),
        ("testing", test_samples),
    ):
        relative_paths = "\n".join(
            str(path.relative_to(config.dataset_root)) for path, _ in samples
        )
        example_digests[split] = hashlib.sha256(relative_paths.encode()).hexdigest()
    return feature_sets, example_digests


def _copy_shared_initial_state(source: nn.Module, target: nn.Module) -> float:
    source_parameters = dict(source.named_parameters())
    target_parameters = dict(target.named_parameters())
    if source_parameters.keys() != target_parameters.keys():
        raise RuntimeError("model parameter names differ beyond the LSTM implementation")
    with torch.no_grad():
        for name, parameter in target_parameters.items():
            if parameter.shape != source_parameters[name].shape:
                raise RuntimeError(f"parameter shape differs for {name}")
            parameter.copy_(source_parameters[name])
    return max(
        (target_parameters[name] - source_parameters[name]).abs().max().item()
        for name in source_parameters
    )


def _batches(
    features: Tensor,
    labels: Tensor,
    batch_size: int,
    *,
    indices: Iterable[int] | Tensor | None = None,
) -> Iterable[tuple[Tensor, Tensor]]:
    order = torch.arange(features.size(0)) if indices is None else torch.as_tensor(indices)
    for start in range(0, order.numel(), batch_size):
        batch_indices = order[start : start + batch_size]
        yield features[batch_indices], labels[batch_indices]


def _balanced_calibration_indices(labels: Tensor, sample_count: int) -> Tensor:
    if sample_count <= 0 or sample_count > labels.numel():
        raise ValueError("calibration sample count is out of range")
    label_values = torch.unique(labels, sorted=True)
    positions = [
        torch.nonzero(labels == value, as_tuple=False).flatten()
        for value in label_values
    ]
    selected = []
    offset = 0
    while len(selected) < sample_count:
        added = False
        for group in positions:
            if offset < group.numel():
                selected.append(group[offset])
                added = True
                if len(selected) == sample_count:
                    break
        if not added:
            raise ValueError("not enough labeled samples for calibration")
        offset += 1
    return torch.stack(selected)


def _evaluate(
    model: nn.Module,
    data: tuple[Tensor, Tensor],
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for features, labels in _batches(*data, batch_size):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(features)
            total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.numel()
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def _calibrate_quant_lstm(
    model: SpeechCommandsLstmClassifier,
    training: tuple[Tensor, Tensor],
    config: ExperimentConfig,
    device: torch.device,
    bitwidth: int,
    *,
    refresh: bool = False,
    operator_bitwidths: dict[str, int] | None = None,
    calibration_sample_count: int | None = None,
) -> dict:
    model.eval()
    if refresh:
        model.lstm.reset_calibration()
    else:
        model.lstm.set_all_bitwidth(bitwidth)
        for operator, operator_bitwidth in (operator_bitwidths or {}).items():
            model.lstm.adjust_quant_config(
                operator, bitwidth=operator_bitwidth
            )
    sample_count = (
        config.calibration_batches * config.batch_size
        if calibration_sample_count is None
        else calibration_sample_count
    )
    calibration_indices = _balanced_calibration_indices(training[1], sample_count)
    model.lstm.calibrating = True
    with torch.no_grad():
        for features, _ in _batches(
            *training, config.batch_size, indices=calibration_indices
        ):
            model(features.to(device, non_blocking=True))
    model.lstm.calibrating = False
    calibration = model.lstm.finalize_calibration()
    model.lstm.use_quantization = True
    calibration["selection"] = "balanced_round_robin"
    calibration["sample_count"] = calibration_indices.numel()
    calibration["label_counts"] = torch.bincount(
        training[1][calibration_indices], minlength=len(config.labels)
    ).tolist()
    return calibration


def _calibration_summary(calibration: dict, epoch: int) -> dict:
    safety = calibration["safety"]
    return {
        "epoch": epoch,
        "batch_count": calibration["batch_count"],
        "method": calibration["method"],
        "selection": calibration["selection"],
        "sample_count": calibration["sample_count"],
        "label_counts": calibration["label_counts"],
        "safety": {
            name: safety[name]
            for name in (
                "exact_integer_range_count",
                "precision_risk_count",
                "unsafe_non_finite_count",
            )
        },
    }


def _quant_range(operator: dict) -> tuple[int, int]:
    bitwidth = operator["bitwidth"]
    if operator["is_unsigned"]:
        return 0, (1 << bitwidth) - 1
    if operator["is_symmetric"]:
        maximum = (1 << (bitwidth - 1)) - 1
        return -maximum, maximum
    return -(1 << (bitwidth - 1)), (1 << (bitwidth - 1)) - 1


def _quant_param_range_summary(bundle: dict) -> dict[str, dict]:
    summaries = {}
    for name, operator in bundle["operators"].items():
        qmin, qmax = _quant_range(operator)
        scales = torch.tensor(
            [float(value) for value in operator["scales"]], dtype=torch.float64
        )
        zero_points = torch.tensor(operator["zero_points"], dtype=torch.float64)
        lower = (qmin - zero_points) * scales
        upper = (qmax - zero_points) * scales
        spans = upper - lower
        summaries[name] = {
            "bitwidth": operator["bitwidth"],
            "granularity": operator["granularity"],
            "group_count": scales.numel(),
            "quantized_levels": qmax - qmin,
            "quantization_step_min": scales.min().item(),
            "quantization_step_median": scales.median().item(),
            "quantization_step_max": scales.max().item(),
            "representable_min": lower.min().item(),
            "representable_max": upper.max().item(),
            "representable_span_min": spans.min().item(),
            "representable_span_median": spans.median().item(),
            "representable_span_max": spans.max().item(),
        }
    return summaries


def _bias_range_diagnostics(lstm: nn.Module) -> dict[str, dict]:
    bundle = json.loads(lstm._quant_params_bundle_json)
    diagnostics = {}
    for name in ("bias_ih", "bias_hh"):
        parameter = getattr(lstm, f"{name}_l0").detach().double().cpu()
        operator = bundle["operators"][name]
        scales = torch.tensor(
            [float(value) for value in operator["scales"]], dtype=torch.float64
        )
        zero_points = torch.tensor(operator["zero_points"], dtype=torch.float64)
        qmin, qmax = _quant_range(operator)
        lower = (qmin - zero_points) * scales
        upper = (qmax - zero_points) * scales
        rounded = torch.round(parameter / scales + zero_points)
        below = rounded < qmin
        above = rounded > qmax
        clamped = below | above
        diagnostics[name] = {
            "parameter_min": parameter.min().item(),
            "parameter_max": parameter.max().item(),
            "representable_min": lower.min().item(),
            "representable_max": upper.max().item(),
            "quantization_step_min": scales.min().item(),
            "quantization_step_max": scales.max().item(),
            "below_count": int(below.sum().item()),
            "above_count": int(above.sum().item()),
            "clamp_rate": clamped.double().mean().item(),
            "channels": [
                {
                    "index": index,
                    "parameter": parameter[index].item(),
                    "representable_min": lower[index].item(),
                    "representable_max": upper[index].item(),
                    "quantization_step": scales[index].item(),
                    "clamped": bool(clamped[index].item()),
                }
                for index in range(parameter.numel())
            ],
        }
    return diagnostics


def _tensor_error_metrics(actual: Tensor, reference: Tensor) -> dict[str, float]:
    actual = actual.detach().double().reshape(-1)
    reference = reference.detach().double().reshape(-1)
    difference = actual - reference
    absolute = difference.abs()
    mse = difference.square().mean()
    rmse = mse.sqrt()
    reference_mae = reference.abs().mean().clamp_min(1.0e-12)
    reference_rms = reference.square().mean().sqrt().clamp_min(1.0e-12)
    denominator = torch.linalg.vector_norm(
        actual
    ) * torch.linalg.vector_norm(reference)
    cosine = 1.0 if denominator.item() <= 1.0e-12 else (
        torch.dot(actual, reference) / denominator
    ).item()
    return {
        "mae": absolute.mean().item(),
        "mse": mse.item(),
        "rmse": rmse.item(),
        "normalized_mae": (absolute.mean() / reference_mae).item(),
        "normalized_rmse": (rmse / reference_rms).item(),
        "cosine": cosine,
        "p99_absolute_error": torch.quantile(absolute, 0.99).item(),
        "max_absolute_error": absolute.max().item(),
    }


def _real_batch_backward_oracle(
    model: SpeechCommandsLstmClassifier,
    training: tuple[Tensor, Tensor],
    config: ExperimentConfig,
    device: torch.device,
) -> dict:
    from tests import lstm_backward_oracle as backward_oracle

    indices = _balanced_calibration_indices(training[1], config.batch_size)
    input_value = (
        training[0][indices].to(device, non_blocking=True).requires_grad_()
    )
    labels = training[1][indices].to(device, non_blocking=True)
    batch = input_value.size(0)
    hidden_size = model.lstm.hidden_size
    initial_hidden = torch.zeros(
        (1, batch, hidden_size), device=device, requires_grad=True
    )
    initial_cell = torch.zeros(
        (1, batch, hidden_size), device=device, requires_grad=True
    )
    model.zero_grad(set_to_none=True)
    model.train()
    model.lstm.use_quantization = True
    output, (hidden, cell) = model.lstm(
        input_value, (initial_hidden, initial_cell)
    )
    logits = model.classifier(model.dropout(hidden[-1]))
    loss = F.cross_entropy(logits, labels)

    grad_logits = torch.softmax(logits.detach(), dim=1)
    grad_logits[torch.arange(batch, device=device), labels] -= 1.0
    grad_logits /= batch
    grad_hidden = torch.zeros_like(hidden)
    grad_hidden[-1] = grad_logits.matmul(model.classifier.weight.detach())
    grad_output = torch.zeros_like(output)
    grad_cell = torch.zeros_like(cell)
    expected = backward_oracle.qat_backward_reference(
        model.lstm, grad_output, grad_hidden, grad_cell
    )
    loss.backward()
    actual = (
        input_value.grad,
        model.lstm.weight_ih_l0.grad,
        model.lstm.weight_hh_l0.grad,
        model.lstm.bias_ih_l0.grad,
        model.lstm.bias_hh_l0.grad,
        initial_hidden.grad[0],
        initial_cell.grad[0],
    )
    names = (
        "input",
        "weight_ih",
        "weight_hh",
        "bias_ih",
        "bias_hh",
        "h_0",
        "c_0",
    )
    result = {
        "sample_count": batch,
        "label_counts": torch.bincount(
            labels.detach().cpu(), minlength=len(config.labels)
        ).tolist(),
        "loss": loss.item(),
        "gradients": {
            name: _tensor_error_metrics(value, reference)
            for name, value, reference in zip(names, actual, expected)
        },
    }
    model.zero_grad(set_to_none=True)
    return result


def _quantization_error(
    model: SpeechCommandsLstmClassifier,
    data: tuple[Tensor, Tensor],
    batch_size: int,
    device: torch.device,
    *,
    include_time_step_trace: bool = True,
) -> dict:
    model.eval()
    quantized_logits = []
    float_logits = []
    quantized_sequences = []
    float_sequences = []
    with torch.no_grad():
        for features, _ in _batches(*data, batch_size):
            features = features.to(device, non_blocking=True)
            model.lstm.use_quantization = True
            quantized_sequence, (quantized_hidden, _) = model.lstm(features)
            quantized_logits.append(
                model.classifier(model.dropout(quantized_hidden[-1]))
            )
            model.lstm.use_quantization = False
            float_sequence, (float_hidden, _) = model.lstm(features)
            float_logits.append(model.classifier(model.dropout(float_hidden[-1])))
            if include_time_step_trace:
                quantized_sequences.append(quantized_sequence)
                float_sequences.append(float_sequence)
    model.lstm.use_quantization = True
    quantized = torch.cat(quantized_logits)
    reference = torch.cat(float_logits)
    result = {
        **_tensor_error_metrics(quantized, reference),
        "sample_count": quantized.size(0),
        "prediction_agreement": (
            quantized.argmax(dim=1) == reference.argmax(dim=1)
        )
        .double()
        .mean()
        .item(),
        "prediction_mismatch_count": int(
            (quantized.argmax(dim=1) != reference.argmax(dim=1)).sum().item()
        ),
    }
    if include_time_step_trace:
        quantized_sequence = torch.cat(quantized_sequences)
        float_sequence = torch.cat(float_sequences)
        steps = [
            {
                "time_step": time_step,
                **_tensor_error_metrics(
                    quantized_sequence[:, time_step],
                    float_sequence[:, time_step],
                ),
            }
            for time_step in range(quantized_sequence.size(1))
        ]
        tail_start = (3 * quantized_sequence.size(1)) // 4
        result["time_step_trace"] = {
            "steps": steps,
            "peak_mae_time_step": max(
                range(len(steps)), key=lambda index: steps[index]["mae"]
            ),
            "tail": {
                "start_time_step": tail_start,
                **_tensor_error_metrics(
                    quantized_sequence[:, tail_start:],
                    float_sequence[:, tail_start:],
                ),
            },
        }
    return result


def _operator_bitwidth_ablation(
    model: SpeechCommandsLstmClassifier,
    feature_sets: dict[str, tuple[Tensor, Tensor]],
    config: ExperimentConfig,
    device: torch.device,
    baseline: dict,
) -> dict:
    diagnostic_model = SpeechCommandsLstmClassifier(
        config.hidden_size,
        len(config.labels),
        use_quant_lstm=True,
        device=device,
    )
    _copy_shared_initial_state(model, diagnostic_model)
    diagnostic_model.lstm.calibration_method = model.lstm.calibration_method
    operators = tuple(diagnostic_model.lstm.get_quant_config()["operators"])

    def evaluate(
        bitwidth: int, operator_bitwidths: dict[str, int] | None = None
    ) -> dict:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            calibration = _calibrate_quant_lstm(
                diagnostic_model,
                feature_sets["training"],
                config,
                device,
                bitwidth,
                operator_bitwidths=operator_bitwidths,
            )
        error = _quantization_error(
            diagnostic_model,
            feature_sets["testing"],
            config.batch_size,
            device,
            include_time_step_trace=False,
        )
        error["mae_delta"] = error["mae"] - baseline["mae"]
        error["calibration_unsafe_non_finite_count"] = calibration["safety"][
            "unsafe_non_finite_count"
        ]
        return error

    promoted = {
        operator: evaluate(8, {operator: 16}) for operator in operators
    }
    baseline_metrics = {
        key: value for key, value in baseline.items() if key != "time_step_trace"
    }
    return {
        "base_bitwidth": 8,
        "promoted_bitwidth": 16,
        "baseline": baseline_metrics,
        "operators": promoted,
        "all_promoted": evaluate(16),
        "ranked_by_mae_improvement": sorted(
            operators, key=lambda name: promoted[name]["mae_delta"]
        ),
    }


def _quantized_clamp_rates(
    model: SpeechCommandsLstmClassifier,
    data: tuple[Tensor, Tensor],
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    was_training = model.training
    was_quantized = model.lstm.use_quantization
    model.train()
    model.lstm.use_quantization = True
    clamp_counts: dict[str, int] = {}
    clamp_elements: dict[str, int] = {}
    with torch.no_grad():
        for features, _ in _batches(*data, batch_size):
            model(features.to(device, non_blocking=True))
            state = model.lstm.qat_saved_state()
            for family in ("master_clamp_masks", "checkpoint_clamp_masks"):
                for name, mask in state[family].items():
                    key = f"{family}.{name}"
                    clamp_counts[key] = clamp_counts.get(key, 0) + int(
                        mask.sum().item()
                    )
                    clamp_elements[key] = clamp_elements.get(key, 0) + mask.numel()
    model.train(was_training)
    model.lstm.use_quantization = was_quantized
    return {
        name: clamp_counts[name] / clamp_elements[name]
        for name in sorted(clamp_counts)
    }


def _calibration_strategy_matrix(
    model: SpeechCommandsLstmClassifier,
    source_model: str,
    feature_sets: dict[str, tuple[Tensor, Tensor]],
    config: ExperimentConfig,
    device: torch.device,
) -> dict:
    diagnostic_model = SpeechCommandsLstmClassifier(
        config.hidden_size,
        len(config.labels),
        use_quant_lstm=True,
        device=device,
    )
    _copy_shared_initial_state(model, diagnostic_model)
    methods = ("minmax", "percentile", "sqnr")
    sample_counts = tuple(
        dict.fromkeys(
            (
                config.calibration_batches * config.batch_size,
                feature_sets["training"][0].size(0),
            )
        )
    )
    bitwidths = (8, 16)
    entries = []
    for method in methods:
        diagnostic_model.lstm.calibration_method = method
        for sample_count in sample_counts:
            for bitwidth in bitwidths:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    calibration = _calibrate_quant_lstm(
                        diagnostic_model,
                        feature_sets["training"],
                        config,
                        device,
                        bitwidth,
                        calibration_sample_count=sample_count,
                    )
                bundle = json.loads(
                    diagnostic_model.lstm._quant_params_bundle_json
                )
                entries.append(
                    {
                        "method": method,
                        "sample_count": sample_count,
                        "label_counts": calibration["label_counts"],
                        "bitwidth": bitwidth,
                        "unsafe_non_finite_count": calibration["safety"][
                            "unsafe_non_finite_count"
                        ],
                        "operators": _quant_param_range_summary(bundle),
                        "clamp_rates": _quantized_clamp_rates(
                            diagnostic_model,
                            feature_sets["testing"],
                            config.batch_size,
                            device,
                        ),
                        "error": _quantization_error(
                            diagnostic_model,
                            feature_sets["testing"],
                            config.batch_size,
                            device,
                            include_time_step_trace=False,
                        ),
                    }
                )
    return {
        "source_model": source_model,
        "fixed_weights": True,
        "methods": list(methods),
        "sample_counts": list(sample_counts),
        "bitwidths": list(bitwidths),
        "entries": entries,
    }


def _train(
    name: str,
    model: nn.Module,
    feature_sets: dict[str, tuple[Tensor, Tensor]],
    config: ExperimentConfig,
    device: torch.device,
    qat_bitwidth: int | None = None,
    *,
    extended_diagnostics: bool = True,
) -> dict:
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    initial_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    initial_training = _evaluate(
        model, feature_sets["training"], config.batch_size, device
    )
    initial_validation = _evaluate(
        model, feature_sets["validation"], config.batch_size, device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = []
    calibration_refreshes = []
    native_qat_checkpoint_observed = False
    started = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        generator = torch.Generator().manual_seed(config.seed + epoch)
        order = torch.randperm(
            feature_sets["training"][0].size(0), generator=generator
        )
        loss_sum = 0.0
        sample_count = 0
        clamp_counts: dict[str, int] = {}
        clamp_elements: dict[str, int] = {}
        post_step_bias_clamp_counts = {"bias_ih": 0, "bias_hh": 0}
        post_step_bias_elements = {"bias_ih": 0, "bias_hh": 0}
        for features, labels in _batches(
            *feature_sets["training"], config.batch_size, indices=order
        ):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            qat_state = (
                model.lstm.qat_saved_state()
                if hasattr(model.lstm, "qat_saved_state")
                else None
            )
            native_qat_checkpoint_observed |= bool(
                qat_state
                and qat_state.get("quantized_master")
                and qat_state.get("checkpoint_clamp_masks")
            )
            if qat_state:
                for family in (
                    "master_clamp_masks",
                    "checkpoint_clamp_masks",
                ):
                    for tensor_name, mask in qat_state[family].items():
                        mask_name = f"{family}.{tensor_name}"
                        clamp_counts[mask_name] = clamp_counts.get(
                            mask_name, 0
                        ) + int(
                            mask.sum().item()
                        )
                        clamp_elements[mask_name] = clamp_elements.get(
                            mask_name, 0
                        ) + mask.numel()
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            if qat_bitwidth is not None:
                post_step = _bias_range_diagnostics(model.lstm)
                for bias_name, diagnostic in post_step.items():
                    post_step_bias_clamp_counts[bias_name] += (
                        diagnostic["below_count"] + diagnostic["above_count"]
                    )
                    post_step_bias_elements[bias_name] += len(
                        diagnostic["channels"]
                    )
            loss_sum += loss.item() * labels.numel()
            sample_count += labels.numel()
        bias_ranges_before_refresh = (
            _bias_range_diagnostics(model.lstm)
            if qat_bitwidth is not None
            else None
        )
        if (
            qat_bitwidth is not None
            and (epoch + 1) % config.calibration_refresh_epochs == 0
        ):
            refreshed = _calibrate_quant_lstm(
                model,
                feature_sets["training"],
                config,
                device,
                qat_bitwidth,
                refresh=True,
            )
            calibration_refreshes.append(
                _calibration_summary(refreshed, epoch + 1)
            )
        validation = _evaluate(
            model, feature_sets["validation"], config.batch_size, device
        )
        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / sample_count,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
        }
        if clamp_counts:
            epoch_result["qat_clamp_rates"] = {
                name: clamp_counts[name] / clamp_elements[name]
                for name in sorted(clamp_counts)
            }
            epoch_result["qat_bias_clamp_rates"] = {
                "pre_step": {
                    name: clamp_counts[f"master_clamp_masks.{name}"]
                    / clamp_elements[f"master_clamp_masks.{name}"]
                    for name in post_step_bias_clamp_counts
                },
                "post_step": {
                    name: post_step_bias_clamp_counts[name]
                    / post_step_bias_elements[name]
                    for name in post_step_bias_clamp_counts
                },
            }
            epoch_result["qat_bias_ranges"] = {
                "before_refresh": bias_ranges_before_refresh,
                "after_refresh": _bias_range_diagnostics(model.lstm),
            }
        history.append(epoch_result)
        print(
            json.dumps(
                {
                    "model": name,
                    "epoch": epoch_result["epoch"],
                    "train_loss": epoch_result["train_loss"],
                    "validation_loss": epoch_result["validation_loss"],
                    "validation_accuracy": epoch_result[
                        "validation_accuracy"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    torch.cuda.synchronize(device)
    final_testing = _evaluate(
        model, feature_sets["testing"], config.batch_size, device
    )
    quantization_error = (
        _quantization_error(
            model, feature_sets["testing"], config.batch_size, device
        )
        if qat_bitwidth is not None
        else None
    )
    operator_bitwidth_ablation = (
        _operator_bitwidth_ablation(
            model,
            feature_sets,
            config,
            device,
            quantization_error,
        )
        if qat_bitwidth == 8 and extended_diagnostics
        else None
    )
    update_squared_norm = sum(
        (parameter.detach() - initial_parameters[name]).square().sum().item()
        for name, parameter in model.named_parameters()
    )
    return {
        "initial_train_loss": initial_training["loss"],
        "initial_validation_accuracy": initial_validation["accuracy"],
        "final_train_loss": history[-1]["train_loss"],
        "best_validation_accuracy": max(
            epoch["validation_accuracy"] for epoch in history
        ),
        "final_test_loss": final_testing["loss"],
        "final_test_accuracy": final_testing["accuracy"],
        "parameter_update_norm": update_squared_norm**0.5,
        "native_qat_checkpoint_observed": native_qat_checkpoint_observed,
        "calibration_refreshes": calibration_refreshes,
        "final_quantization_error": quantization_error,
        "operator_bitwidth_ablation": operator_bitwidth_ablation,
        "duration_seconds": time.perf_counter() - started,
        "epochs": history,
    }


def _quality_result(result: dict, *, quantized: bool) -> dict:
    summary = {
        name: result[name]
        for name in (
            "initial_train_loss",
            "final_train_loss",
            "best_validation_accuracy",
            "final_test_accuracy",
        )
    }
    if quantized:
        summary["logit_mae"] = result["final_quantization_error"]["mae"]
        summary["prediction_agreement"] = result["final_quantization_error"][
            "prediction_agreement"
        ]
    return summary


def _run_additional_quality_seed(
    feature_sets: dict[str, tuple[Tensor, Tensor]],
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, dict]:
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    baseline = SpeechCommandsLstmClassifier(
        config.hidden_size,
        len(config.labels),
        use_quant_lstm=False,
        device=device,
    )
    quantized_variants = {}
    for bitwidth in config.quant_bitwidths:
        name = f"quant_lstm_qat_{bitwidth}bit"
        model = SpeechCommandsLstmClassifier(
            config.hidden_size,
            len(config.labels),
            use_quant_lstm=True,
            device=device,
        )
        _copy_shared_initial_state(baseline, model)
        model.lstm.calibration_method = QAT_CALIBRATION_METHODS[bitwidth]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _calibrate_quant_lstm(
                model, feature_sets["training"], config, device, bitwidth
            )
        quantized_variants[name] = model

    baseline_result = _train(
        f"seed_{config.seed}_torch_lstm",
        baseline,
        feature_sets,
        config,
        device,
        extended_diagnostics=False,
    )
    results = {
        "torch_lstm": _quality_result(baseline_result, quantized=False)
    }
    for name, model in quantized_variants.items():
        bitwidth = int(name.removesuffix("bit").rsplit("_", 1)[1])
        result = _train(
            f"seed_{config.seed}_{name}",
            model,
            feature_sets,
            config,
            device,
            qat_bitwidth=bitwidth,
            extended_diagnostics=False,
        )
        results[name] = _quality_result(result, quantized=True)
    return results


def _multi_seed_quality(
    primary_results: dict[str, dict],
    feature_sets: dict[str, tuple[Tensor, Tensor]],
    config: ExperimentConfig,
    device: torch.device,
) -> dict:
    seeds = config.quality_gate_seeds or (config.seed,)
    model_names = (
        "torch_lstm",
        *(f"quant_lstm_qat_{bitwidth}bit" for bitwidth in config.quant_bitwidths),
    )
    runs = {
        str(config.seed): {
            name: _quality_result(
                primary_results[name], quantized=name != "torch_lstm"
            )
            for name in model_names
        }
    }
    for seed in seeds:
        if seed == config.seed:
            continue
        seed_config = replace(config, seed=seed, quality_gate_seeds=())
        runs[str(seed)] = _run_additional_quality_seed(
            feature_sets, seed_config, device
        )
    aggregate = {}
    for name in model_names:
        values = [run[name] for run in runs.values()]
        aggregate[name] = {
            "minimum_validation_accuracy": min(
                value["best_validation_accuracy"] for value in values
            ),
            "mean_validation_accuracy": sum(
                value["best_validation_accuracy"] for value in values
            )
            / len(values),
            "minimum_test_accuracy": min(
                value["final_test_accuracy"] for value in values
            ),
            "mean_test_accuracy": sum(
                value["final_test_accuracy"] for value in values
            )
            / len(values),
        }
    return {
        "seeds": list(seeds),
        "runs": runs,
        "aggregate": aggregate,
    }


def run_training_comparison(config: ExperimentConfig) -> dict:
    """Train matched torch-LSTM and QuantLSTM-QAT keyword spotters."""
    config.validate()
    config = ExperimentConfig(**{**asdict(config), "dataset_root": Path(config.dataset_root)})
    if not torch.cuda.is_available():
        raise RuntimeError("QuantLSTM real-network training requires CUDA")
    device = torch.device(config.device)
    if device.type != "cuda":
        raise ValueError("device must select CUDA")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    feature_sets, example_digests = _load_feature_sets(config)
    baseline = SpeechCommandsLstmClassifier(
        config.hidden_size,
        len(config.labels),
        use_quant_lstm=False,
        device=device,
    )
    native_float = SpeechCommandsLstmClassifier(
        config.hidden_size,
        len(config.labels),
        use_quant_lstm=True,
        device=device,
    )
    quantized_variants = {}
    initial_differences = {
        "quant_lstm_float": _copy_shared_initial_state(baseline, native_float)
    }
    calibrations = {}
    for bitwidth in config.quant_bitwidths:
        name = f"quant_lstm_qat_{bitwidth}bit"
        quantized = SpeechCommandsLstmClassifier(
            config.hidden_size,
            len(config.labels),
            use_quant_lstm=True,
            device=device,
        )
        initial_differences[name] = _copy_shared_initial_state(baseline, quantized)
        quantized.lstm.calibration_method = QAT_CALIBRATION_METHODS[bitwidth]
        calibrations[name] = _calibrate_quant_lstm(
            quantized, feature_sets["training"], config, device, bitwidth
        )
        quantized_variants[name] = quantized

    real_batch_backward_oracle = {
        str(bitwidth): _real_batch_backward_oracle(
            quantized_variants[f"quant_lstm_qat_{bitwidth}bit"],
            feature_sets["training"],
            config,
            device,
        )
        for bitwidth in config.quant_bitwidths
    }

    baseline_result = _train(
        "torch_lstm", baseline, feature_sets, config, device
    )
    native_float_result = _train(
        "quant_lstm_float", native_float, feature_sets, config, device
    )
    quantized_results = {
        name: _train(
            name,
            model,
            feature_sets,
            config,
            device,
            qat_bitwidth=int(name.removesuffix("bit").rsplit("_", 1)[1]),
        )
        for name, model in quantized_variants.items()
    }
    matrix_source = (
        "quant_lstm_qat_8bit"
        if "quant_lstm_qat_8bit" in quantized_variants
        else next(iter(quantized_variants))
    )
    calibration_strategy_matrix = _calibration_strategy_matrix(
        quantized_variants[matrix_source],
        matrix_source,
        feature_sets,
        config,
        device,
    )
    primary_results = {
        "torch_lstm": baseline_result,
        **quantized_results,
    }
    multi_seed_quality = _multi_seed_quality(
        primary_results, feature_sets, config, device
    )
    return {
        "schema_version": 8,
        "validation_scope": "real_network_training",
        "dataset": {
            "name": "speech_commands_v0.02",
            "root": str(config.dataset_root),
            "split_source": "validation_list.txt and testing_list.txt",
            "labels": list(config.labels),
            "training_samples": feature_sets["training"][0].size(0),
            "validation_samples": feature_sets["validation"][0].size(0),
            "test_samples": feature_sets["testing"][0].size(0),
            "feature_shape": list(feature_sets["training"][0].shape[1:]),
            "example_id_sha256": example_digests,
        },
        "model": {
            "source": "google-research/kws_streaming/models/lstm.py",
            "topology": "20-coefficient MFCC -> LSTM -> dropout(0) -> dense",
            "hidden_size": config.hidden_size,
            "peepholes": False,
            "projection": False,
        },
        "replacement": {
            "changed_module": "lstm",
            "baseline": "torch.nn.LSTM",
            "candidate": "quant_lstm.QuantLSTM",
            "initial_shared_state_max_abs_diff": initial_differences,
        },
        "quantization": {
            "mode": "qat",
            "bitwidths": list(config.quant_bitwidths),
            "calibration_batches": config.calibration_batches,
            "calibration_strategy": {
                "selection": "balanced_round_robin",
                "refresh_interval_epochs": config.calibration_refresh_epochs,
                "refresh_timing": "after_training_before_validation",
                "methods_by_bitwidth": {
                    str(bitwidth): QAT_CALIBRATION_METHODS[bitwidth]
                    for bitwidth in config.quant_bitwidths
                },
            },
            "calibration": calibrations,
            "calibration_strategy_matrix": calibration_strategy_matrix,
            "real_batch_backward_oracle": real_batch_backward_oracle,
        },
        "environment": {
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        "config": {
            **asdict(config),
            "dataset_root": str(config.dataset_root),
            "labels": list(config.labels),
        },
        "training": {
            "torch_lstm": baseline_result,
            "quant_lstm_float": native_float_result,
            **quantized_results,
        },
        "multi_seed_quality": multi_seed_quality,
    }


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path.home() / ".cache" / "quant-lstm" / "speech_commands_v0.02",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--labels", default="yes,no,up,down")
    parser.add_argument("--train-samples-per-label", type=int, default=128)
    parser.add_argument("--validation-samples-per-label", type=int, default=32)
    parser.add_argument("--test-samples-per-label", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--calibration-batches", type=int, default=4)
    parser.add_argument("--calibration-refresh-epochs", type=int, default=1)
    parser.add_argument(
        "--quality-gate-seed",
        type=int,
        action="append",
        dest="quality_gate_seeds",
        help=(
            "training seed for aggregate quality gates; repeat for multiple "
            "seeds and include --seed"
        ),
    )
    parser.add_argument(
        "--quant-bitwidth",
        type=int,
        choices=(8, 16),
        action="append",
        dest="quant_bitwidths",
        help="QAT bitwidth to test; repeat to test both (default: 8 and 16)",
    )
    parser.add_argument("--seed", type=int, default=20260921)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    dataset_root = arguments.dataset_root or prepare_dataset(
        arguments.cache_root, download=arguments.download
    )
    if arguments.prepare_only:
        print(dataset_root)
        return
    report = run_training_comparison(
        ExperimentConfig(
            dataset_root=dataset_root,
            labels=tuple(label.strip() for label in arguments.labels.split(",")),
            train_samples_per_label=arguments.train_samples_per_label,
            validation_samples_per_label=arguments.validation_samples_per_label,
            test_samples_per_label=arguments.test_samples_per_label,
            hidden_size=arguments.hidden_size,
            batch_size=arguments.batch_size,
            epochs=arguments.epochs,
            learning_rate=arguments.learning_rate,
            calibration_batches=arguments.calibration_batches,
            calibration_refresh_epochs=arguments.calibration_refresh_epochs,
            quant_bitwidths=tuple(arguments.quant_bitwidths or (8, 16)),
            seed=arguments.seed,
            quality_gate_seeds=tuple(arguments.quality_gate_seeds or ()),
        )
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
