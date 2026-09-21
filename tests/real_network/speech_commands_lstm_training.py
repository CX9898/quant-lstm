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
import wave
from dataclasses import asdict, dataclass
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
) -> dict:
    model.eval()
    if refresh:
        model.lstm.reset_calibration()
    else:
        model.lstm.set_all_bitwidth(bitwidth)
    calibration_indices = _balanced_calibration_indices(
        training[1], config.calibration_batches * config.batch_size
    )
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


def _quantization_error(
    model: SpeechCommandsLstmClassifier,
    data: tuple[Tensor, Tensor],
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    quantized_logits = []
    float_logits = []
    with torch.no_grad():
        for features, _ in _batches(*data, batch_size):
            features = features.to(device, non_blocking=True)
            model.lstm.use_quantization = True
            quantized_logits.append(model(features))
            model.lstm.use_quantization = False
            float_logits.append(model(features))
    model.lstm.use_quantization = True
    quantized = torch.cat(quantized_logits).double()
    reference = torch.cat(float_logits).double()
    difference = quantized - reference
    denominator = torch.linalg.vector_norm(
        quantized
    ) * torch.linalg.vector_norm(reference)
    cosine = 1.0 if denominator.item() <= 1.0e-12 else (
        torch.dot(quantized.flatten(), reference.flatten()) / denominator
    ).item()
    return {
        "mae": difference.abs().mean().item(),
        "mse": difference.square().mean().item(),
        "cosine": cosine,
    }


def _train(
    name: str,
    model: nn.Module,
    feature_sets: dict[str, tuple[Tensor, Tensor]],
    config: ExperimentConfig,
    device: torch.device,
    qat_bitwidth: int | None = None,
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
        "duration_seconds": time.perf_counter() - started,
        "epochs": history,
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
        calibrations[name] = _calibrate_quant_lstm(
            quantized, feature_sets["training"], config, device, bitwidth
        )
        quantized_variants[name] = quantized

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
    return {
        "schema_version": 5,
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
            },
            "calibration": calibrations,
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
        )
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
