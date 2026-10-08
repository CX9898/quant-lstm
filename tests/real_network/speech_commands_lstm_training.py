"""Speech Commands v0.02 training comparison for torch LSTM and QuantLSTM QAT.

The model follows the small Google Research kws_streaming LSTM topology:
MFCC features, one LSTM, dropout, and a dense classifier. Peepholes and
projection are intentionally disabled because QuantLSTM does not expose them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
FULL_SPEECH_COMMAND_LABELS = (
    "backward",
    "bed",
    "bird",
    "cat",
    "dog",
    "down",
    "eight",
    "five",
    "follow",
    "forward",
    "four",
    "go",
    "happy",
    "house",
    "learn",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "sheila",
    "six",
    "stop",
    "three",
    "tree",
    "two",
    "up",
    "visual",
    "wow",
    "yes",
    "zero",
)


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_root: Path
    labels: tuple[str, ...] = ("yes", "no", "up", "down")
    train_samples_per_label: int | None = 128
    validation_samples_per_label: int | None = 32
    test_samples_per_label: int | None = 32
    dataset_profile: str = "subset"
    feature_chunk_size: int = 512
    feature_cache: Path | None = None
    hidden_size: int = 64
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 3.0e-4
    pretrain_epochs: int = 50
    pretrain_learning_rate: float = 3.0e-3
    minimum_pretrain_validation_accuracy: float = 0.80
    calibration_batches: int = 4
    quant_bitwidths: tuple[int, ...] = (8, 16)
    seed: int = 20260921
    quality_gate_seeds: tuple[int, ...] = ()
    device: str = "cuda"
    extended_diagnostics: bool = True

    def validate(self) -> None:
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be non-empty and unique")
        if self.dataset_profile not in ("subset", "full"):
            raise ValueError("dataset_profile must be subset or full")
        for name in (
            "train_samples_per_label",
            "validation_samples_per_label",
            "test_samples_per_label",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or None")
        if self.dataset_profile == "full" and any(
            getattr(self, name) is not None
            for name in (
                "train_samples_per_label",
                "validation_samples_per_label",
                "test_samples_per_label",
            )
        ):
            raise ValueError("full dataset_profile requires all sample limits to be None")
        for name in (
            "hidden_size",
            "batch_size",
            "epochs",
            "pretrain_epochs",
            "calibration_batches",
            "feature_chunk_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("learning_rate", "pretrain_learning_rate"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 <= self.minimum_pretrain_validation_accuracy <= 1.0:
            raise ValueError("minimum_pretrain_validation_accuracy must be in [0, 1]")
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


@dataclass(frozen=True)
class SpeechCommandsSample:
    path: Path
    label: int
    source_label: str


@dataclass(frozen=True)
class SpeechCommandsDatasetManifest:
    labels: tuple[str, ...]
    samples: dict[str, tuple[SpeechCommandsSample, ...]]
    audit: dict[str, object]


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


def build_dataset_manifest(config: ExperimentConfig) -> SpeechCommandsDatasetManifest:
    """Resolve configured labels against the official, disjoint dataset splits."""
    config.validate()
    dataset_root = Path(config.dataset_root)
    validation = _listed_paths(dataset_root, "validation_list.txt")
    testing = _listed_paths(dataset_root, "testing_list.txt")
    official_overlap = validation & testing
    if official_overlap:
        raise RuntimeError("validation_list.txt and testing_list.txt overlap")

    discovered_labels = tuple(
        sorted(
            path.name
            for path in dataset_root.iterdir()
            if path.is_dir() and not path.name.startswith("_")
        )
    )
    missing_labels = sorted(set(config.labels) - set(discovered_labels))
    if missing_labels:
        raise RuntimeError(f"dataset label directories not found: {missing_labels}")
    if config.dataset_profile == "full" and set(config.labels) != set(discovered_labels):
        raise RuntimeError(
            "full dataset_profile labels must exactly match all word directories"
        )

    limits = {
        "training": config.train_samples_per_label,
        "validation": config.validation_samples_per_label,
        "testing": config.test_samples_per_label,
    }
    split_offsets = {"training": 0, "validation": 100_000, "testing": 200_000}
    split_samples: dict[str, list[SpeechCommandsSample]] = {
        split: [] for split in limits
    }
    all_configured_paths: set[str] = set()
    selected_paths: set[str] = set()
    per_label_counts = {
        split: {label: 0 for label in config.labels} for split in limits
    }

    for label_index, label in enumerate(config.labels):
        relative_paths = sorted(
            f"{label}/{path.name}" for path in (dataset_root / label).glob("*.wav")
        )
        all_configured_paths.update(relative_paths)
        candidates_by_split = {
            "training": [
                path for path in relative_paths if path not in validation and path not in testing
            ],
            "validation": [path for path in relative_paths if path in validation],
            "testing": [path for path in relative_paths if path in testing],
        }
        for split, candidates in candidates_by_split.items():
            limit = limits[split]
            if limit is not None:
                random.Random(
                    config.seed + split_offsets[split] + 1009 * label_index
                ).shuffle(candidates)
                if len(candidates) < limit:
                    raise RuntimeError(
                        f"{split} label {label!r} has {len(candidates)} samples, "
                        f"needs {limit}"
                    )
                candidates = candidates[:limit]
            per_label_counts[split][label] = len(candidates)
            selected_paths.update(candidates)
            split_samples[split].extend(
                SpeechCommandsSample(dataset_root / relative_path, label_index, label)
                for relative_path in candidates
            )

    split_path_sets = {
        split: {sample.path for sample in samples}
        for split, samples in split_samples.items()
    }
    split_overlap_count = sum(
        len(split_path_sets[left] & split_path_sets[right])
        for left, right in (
            ("training", "validation"),
            ("training", "testing"),
            ("validation", "testing"),
        )
    )
    background_noise = dataset_root / "_background_noise_"
    background_noise_file_count = (
        len(tuple(background_noise.glob("*.wav"))) if background_noise.is_dir() else 0
    )
    audit = {
        "dataset_profile": config.dataset_profile,
        "discovered_word_labels": list(discovered_labels),
        "selected_labels": list(config.labels),
        "available_word_sample_count": len(all_configured_paths),
        "selected_word_sample_count": len(selected_paths),
        "omitted_word_sample_count": len(all_configured_paths - selected_paths),
        "split_sample_counts": {
            split: len(samples) for split, samples in split_samples.items()
        },
        "per_label_split_counts": per_label_counts,
        "split_overlap_count": split_overlap_count,
        "official_validation_count": len(validation),
        "official_testing_count": len(testing),
        "background_noise_file_count": background_noise_file_count,
    }
    return SpeechCommandsDatasetManifest(
        labels=config.labels,
        samples={split: tuple(samples) for split, samples in split_samples.items()},
        audit=audit,
    )


def _select_audio_paths(
    config: ExperimentConfig, split: str
) -> list[tuple[Path, int]]:
    if split not in ("training", "validation", "testing"):
        raise ValueError(f"unsupported split: {split}")
    return [
        (sample.path, sample.label)
        for sample in build_dataset_manifest(config).samples[split]
    ]


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


def _load_feature_split(
    samples: Sequence[SpeechCommandsSample], chunk_size: int
) -> tuple[Tensor, Tensor]:
    feature_chunks = []
    label_chunks = []
    for offset in range(0, len(samples), chunk_size):
        chunk = samples[offset : offset + chunk_size]
        waveforms, labels = _load_waveforms(
            [(sample.path, sample.label) for sample in chunk]
        )
        feature_chunks.append(_mfcc(waveforms))
        label_chunks.append(labels)
    if not feature_chunks:
        raise RuntimeError("dataset split contains no samples")
    return torch.cat(feature_chunks), torch.cat(label_chunks)


def load_feature_sets(
    config: ExperimentConfig,
) -> tuple[
    dict[str, tuple[Tensor, Tensor]], SpeechCommandsDatasetManifest, dict[str, str]
]:
    """Load manifest samples and compute MFCCs without materializing all waveforms."""
    manifest = build_dataset_manifest(config)
    example_digests = {}
    for split, samples in manifest.samples.items():
        relative_paths = "\n".join(
            sample.path.relative_to(config.dataset_root).as_posix() for sample in samples
        )
        example_digests[split] = hashlib.sha256(relative_paths.encode()).hexdigest()
    cache_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "contract": "speech_commands_mfcc_v1",
                "sample_rate": SAMPLE_RATE,
                "clip_samples": CLIP_SAMPLES,
                "mel_bins": MEL_BINS,
                "mfcc_bins": MFCC_BINS,
                "example_id_sha256": example_digests,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if config.feature_cache is not None and Path(config.feature_cache).is_file():
        cached = torch.load(
            Path(config.feature_cache), map_location="cpu", weights_only=True
        )
        if cached.get("fingerprint") == cache_fingerprint:
            return cached["feature_sets"], manifest, example_digests

    train_features, train_labels = _load_feature_split(
        manifest.samples["training"], config.feature_chunk_size
    )
    validation_features, validation_labels = _load_feature_split(
        manifest.samples["validation"], config.feature_chunk_size
    )
    test_features, test_labels = _load_feature_split(
        manifest.samples["testing"], config.feature_chunk_size
    )
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
    if config.feature_cache is not None:
        cache_path = Path(config.feature_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        torch.save(
            {"fingerprint": cache_fingerprint, "feature_sets": feature_sets}, temporary
        )
        temporary.replace(cache_path)
    return feature_sets, manifest, example_digests


def _load_feature_sets(
    config: ExperimentConfig,
) -> tuple[dict[str, tuple[Tensor, Tensor]], dict[str, str]]:
    feature_sets, _, example_digests = load_feature_sets(config)
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


def classification_metrics(confusion: Tensor, labels: Sequence[str]) -> dict[str, object]:
    """Compute complete multiclass metrics from a row=true, column=predicted matrix."""
    confusion = confusion.detach().to(dtype=torch.int64, device="cpu")
    if confusion.shape != (len(labels), len(labels)):
        raise ValueError("confusion matrix shape does not match labels")
    sample_count = int(confusion.sum().item())
    if sample_count == 0:
        raise ValueError("confusion matrix must contain at least one sample")
    per_class = {}
    precisions = []
    recalls = []
    f1_scores = []
    for index, label in enumerate(labels):
        true_positive = int(confusion[index, index].item())
        support = int(confusion[index].sum().item())
        predicted = int(confusion[:, index].sum().item())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[label] = {
            "support": support,
            "correct": true_positive,
            "accuracy": recall,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    correct_count = int(confusion.diagonal().sum().item())
    return {
        "sample_count": sample_count,
        "correct_count": correct_count,
        "accuracy": correct_count / sample_count,
        "macro_precision": sum(precisions) / len(precisions),
        "macro_recall": sum(recalls) / len(recalls),
        "macro_f1": sum(f1_scores) / len(f1_scores),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def _evaluate(
    model: nn.Module,
    data: tuple[Tensor, Tensor],
    batch_size: int,
    device: torch.device,
    label_names: Sequence[str] | None = None,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    confusion = None
    with torch.no_grad():
        for features, labels in _batches(*data, batch_size):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(features)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite evaluation logits")
            total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
            predictions = logits.argmax(dim=1)
            if confusion is None:
                confusion = torch.zeros(
                    (logits.size(1), logits.size(1)), dtype=torch.int64
                )
            confusion += torch.bincount(
                (labels * logits.size(1) + predictions).detach().cpu(),
                minlength=logits.size(1) * logits.size(1),
            ).reshape(logits.size(1), logits.size(1))
            total_samples += labels.numel()
    names = (
        tuple(label_names)
        if label_names is not None
        else tuple(str(index) for index in range(confusion.size(0)))
    )
    result = classification_metrics(confusion, names)
    result["loss"] = total_loss / total_samples
    return result


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


def _state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _validation_key(metrics: dict) -> tuple[float, float]:
    return metrics["accuracy"], -metrics["loss"]


def _train(
    name: str,
    model: nn.Module,
    feature_sets: dict[str, tuple[Tensor, Tensor]],
    config: ExperimentConfig,
    device: torch.device,
    qat_bitwidth: int | None = None,
    *,
    extended_diagnostics: bool = True,
    pretraining: bool = False,
) -> dict:
    """Select only on validation; leave model at the selected checkpoint."""
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    fixed_quant_params = (
        model.lstm.export_quant_params() if qat_bitwidth is not None else None
    )
    quant_params_sha256 = (
        hashlib.sha256(
            json.dumps(fixed_quant_params, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if fixed_quant_params is not None
        else None
    )
    initial_state_sha256 = _state_sha256(model)
    initial_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    initial_training = _evaluate(
        model, feature_sets["training"], config.batch_size, device, config.labels
    )
    initial_validation = _evaluate(
        model, feature_sets["validation"], config.batch_size, device, config.labels
    )
    epochs = config.pretrain_epochs if pretraining else config.epochs
    learning_rate = config.pretrain_learning_rate if pretraining else config.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = (
        torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 40], gamma=0.3)
        if pretraining else None
    )

    def snapshot(epoch, validation):
        return {
            "epoch": epoch, "validation": validation,
            "weights": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        }

    start = snapshot(0, initial_validation)
    best, best_trained = start, None
    history = []
    native_qat_checkpoint_observed = False
    started = time.perf_counter()
    for epoch in range(epochs):
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
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss: {name}, epoch {epoch + 1}")
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5.0, error_if_nonfinite=True
            )
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
        if fixed_quant_params is not None:
            if model.lstm.export_quant_params() != fixed_quant_params:
                raise RuntimeError(
                    "QAT must preserve the initial PTQ quantization parameters"
                )
        validation = _evaluate(
            model, feature_sets["validation"], config.batch_size, device, config.labels
        )
        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / sample_count,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
        }
        if fixed_quant_params is not None:
            epoch_result["quant_params_sha256"] = quant_params_sha256
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
                "post_step": _bias_range_diagnostics(model.lstm),
            }
        last = snapshot(epoch + 1, validation)
        if _validation_key(validation) > _validation_key(best["validation"]):
            best = last
        if best_trained is None or _validation_key(validation) > _validation_key(best_trained["validation"]):
            best_trained = last
        epoch_result["learning_rate"] = optimizer.param_groups[0]["lr"]
        history.append(epoch_result)
        if scheduler is not None:
            scheduler.step()
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
    update_squared_norm = sum(
        (parameter.detach() - initial_parameters[name]).square().sum().item()
        for name, parameter in model.named_parameters()
    )
    # All selection is finished before any test-set evaluation.
    testing = {}
    for checkpoint in (start, best, best_trained, last):
        epoch = checkpoint["epoch"]
        if epoch not in testing:
            model.load_state_dict(checkpoint["weights"])
            testing[epoch] = _evaluate(
                model, feature_sets["testing"], config.batch_size, device, config.labels
            )
    model.load_state_dict(best["weights"])
    restored_validation = _evaluate(
        model, feature_sets["validation"], config.batch_size, device, config.labels
    )
    if restored_validation != best["validation"]:
        raise RuntimeError("Selected checkpoint did not reproduce validation metrics")
    quantization_error = (
        _quantization_error(model, feature_sets["testing"], config.batch_size, device)
        if qat_bitwidth is not None else None
    )
    operator_bitwidth_ablation = (
        _operator_bitwidth_ablation(model, feature_sets, config, device, quantization_error)
        if qat_bitwidth == 8 and extended_diagnostics else None
    )
    if fixed_quant_params is not None:
        if model.lstm.export_quant_params() != fixed_quant_params:
            raise RuntimeError("Evaluation changed the initial PTQ quantization parameters")
    return {
        "initial_state_sha256": initial_state_sha256,
        "selected_state_sha256": _state_sha256(model),
        "initial_train_loss": initial_training["loss"],
        "final_train_loss": history[-1]["train_loss"],
        "initial_validation": initial_validation,
        "selected_epoch": best["epoch"],
        "selected_validation": best["validation"],
        "best_trained_epoch": best_trained["epoch"],
        "best_trained_validation": best_trained["validation"],
        "last_validation": last["validation"],
        "starting_test": testing[0],
        "selected_test": testing[best["epoch"]],
        "best_trained_test": testing[best_trained["epoch"]],
        "last_test": testing[last["epoch"]],
        "parameter_update_norm": update_squared_norm**0.5,
        "native_qat_checkpoint_observed": native_qat_checkpoint_observed,
        "quant_params_sha256": quant_params_sha256,
        "quant_params_policy": "fixed_after_ptq" if qat_bitwidth is not None else None,
        "selected_quantization_error": quantization_error,
        "operator_bitwidth_ablation": operator_bitwidth_ablation,
        "duration_seconds": time.perf_counter() - started,
        "epochs": history,
    }


def _run_quality_seed(feature_sets, config, device, *, extended_diagnostics=False):
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    baseline = SpeechCommandsLstmClassifier(
        config.hidden_size, len(config.labels), use_quant_lstm=False, device=device
    )
    baseline_result = _train(
        f"seed_{config.seed}_torch_lstm", baseline, feature_sets, config, device,
        pretraining=True, extended_diagnostics=False,
    )
    accuracy = baseline_result["selected_validation"]["accuracy"]
    if accuracy < config.minimum_pretrain_validation_accuracy:
        raise RuntimeError(
            f"Float baseline validation accuracy {accuracy:.6f} is below "
            f"{config.minimum_pretrain_validation_accuracy:.6f}; QAT quality cannot be assessed"
        )
    # Every branch is cloned from this same validation-selected FLOAT checkpoint.
    finetune_config = replace(config, seed=config.seed + 10000)
    models, differences, calibrations, backward = {}, {}, {}, {}
    for bits in (None, *config.quant_bitwidths):
        name = "quant_lstm_float" if bits is None else f"quant_lstm_qat_{bits}bit"
        model = SpeechCommandsLstmClassifier(
            config.hidden_size, len(config.labels), use_quant_lstm=True, device=device
        )
        differences[name] = _copy_shared_initial_state(baseline, model)
        if _state_sha256(model) != baseline_result["selected_state_sha256"]:
            raise RuntimeError("Finetuning did not start from the selected float checkpoint")
        if bits is not None:
            model.lstm.calibration_method = QAT_CALIBRATION_METHODS[bits]
            calibrations[name] = _calibrate_quant_lstm(
                model, feature_sets["training"], config, device, bits
            )
            backward[str(bits)] = _real_batch_backward_oracle(
                model, feature_sets["training"], config, device
            )
        models[name] = model
    training = {"torch_lstm": baseline_result}
    for name, model in models.items():
        bits = None if name == "quant_lstm_float" else int(name.removesuffix("bit").rsplit("_", 1)[1])
        training[name] = _train(
            f"seed_{config.seed}_{name}", model, feature_sets, finetune_config, device,
            qat_bitwidth=bits, extended_diagnostics=extended_diagnostics,
        )
    matrix_source = next(name for name in models if name != "quant_lstm_float")
    matrix = (
        _calibration_strategy_matrix(models[matrix_source], matrix_source, feature_sets, config, device)
        if extended_diagnostics else {"enabled": False, "reason": "extended diagnostics disabled"}
    )
    return {
        "training": training,
        "initial_shared_state_max_abs_diff": differences,
        "calibration": calibrations,
        "real_batch_backward_oracle": backward,
        "calibration_strategy_matrix": matrix,
    }


def run_training_comparison(config: ExperimentConfig) -> dict:
    """Pretrain float, calibrate PTQ once, then finetune with fixed quantizers."""
    config.validate()
    if not torch.cuda.is_available():
        raise RuntimeError("QuantLSTM real-network training requires CUDA")
    device = torch.device(config.device)
    if device.type != "cuda":
        raise ValueError("device must select CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    feature_sets, manifest, example_digests = load_feature_sets(config)
    seeds = config.quality_gate_seeds or (config.seed,)
    runs = {}
    for seed in seeds:
        runs[str(seed)] = _run_quality_seed(
            feature_sets, replace(config, seed=seed, quality_gate_seeds=()), device,
            extended_diagnostics=config.extended_diagnostics and seed == config.seed,
        )
    primary = runs[str(config.seed)]
    return {
        "schema_version": 10,
        "validation_scope": "pretrained_fixed_quantizer_qat",
        "selection": {
            "source": "validation_only",
            "criterion": "highest_accuracy_then_lowest_loss",
            "epoch_zero_eligible": True,
            "best_trained_and_last_reported_separately": True,
        },
        "dataset": {
            "name": "speech_commands_v0.02", "root": str(config.dataset_root),
            "profile": config.dataset_profile,
            "split_source": "validation_list.txt and testing_list.txt",
            "labels": list(config.labels),
            "training_samples": len(feature_sets["training"][1]),
            "validation_samples": len(feature_sets["validation"][1]),
            "test_samples": len(feature_sets["testing"][1]),
            "feature_shape": list(feature_sets["training"][0].shape[1:]),
            "example_id_sha256": example_digests, "audit": manifest.audit,
        },
        "model": {
            "source": "google-research/kws_streaming/models/lstm.py",
            "topology": "20-coefficient MFCC -> LSTM -> dropout(0) -> dense",
            "hidden_size": config.hidden_size, "peepholes": False, "projection": False,
        },
        "replacement": {
            "changed_module": "lstm", "baseline": "torch.nn.LSTM",
            "candidate": "quant_lstm.QuantLSTM",
            "source_checkpoint": "validation_selected_pretrained_float",
            "initial_shared_state_max_abs_diff": primary["initial_shared_state_max_abs_diff"],
        },
        "quantization": {
            "mode": "pretrained_qat", "bitwidths": list(config.quant_bitwidths),
            "calibration_batches": config.calibration_batches,
            "calibration_strategy": {
                "selection": "balanced_round_robin",
                "quant_params_policy": "fixed_after_ptq",
                "calibration_timing": "after_float_pretraining_before_qat",
                "methods_by_bitwidth": {str(b): QAT_CALIBRATION_METHODS[b] for b in config.quant_bitwidths},
            },
            "calibration": primary["calibration"],
            "calibration_strategy_matrix": primary["calibration_strategy_matrix"],
            "real_batch_backward_oracle": primary["real_batch_backward_oracle"],
        },
        "environment": {"torch": torch.__version__, "torchaudio": torchaudio.__version__,
                        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device)},
        "config": {**asdict(config), "dataset_root": str(config.dataset_root),
                   "feature_cache": None if config.feature_cache is None else str(config.feature_cache)},
        "training": primary["training"],
        "multi_seed_quality": {"seeds": list(seeds), "runs": runs},
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
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="train all 35 word classes using every official split sample",
    )
    parser.add_argument("--train-samples-per-label", type=int, default=128)
    parser.add_argument("--validation-samples-per-label", type=int, default=32)
    parser.add_argument("--test-samples-per-label", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20, help="QAT/float continuation epochs")
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--pretrain-learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--minimum-pretrain-validation-accuracy", type=float, default=0.80)
    parser.add_argument("--calibration-batches", type=int, default=4)
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
    parser.add_argument("--feature-chunk-size", type=int, default=512)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--no-extended-diagnostics", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    dataset_root = arguments.dataset_root or prepare_dataset(
        arguments.cache_root, download=arguments.download
    )
    if arguments.prepare_only:
        print(dataset_root)
        return
    labels = (
        FULL_SPEECH_COMMAND_LABELS
        if arguments.full_dataset
        else tuple(label.strip() for label in arguments.labels.split(","))
    )
    report = run_training_comparison(
        ExperimentConfig(
            dataset_root=dataset_root,
            labels=labels,
            train_samples_per_label=(
                None if arguments.full_dataset else arguments.train_samples_per_label
            ),
            validation_samples_per_label=(
                None
                if arguments.full_dataset
                else arguments.validation_samples_per_label
            ),
            test_samples_per_label=(
                None if arguments.full_dataset else arguments.test_samples_per_label
            ),
            dataset_profile="full" if arguments.full_dataset else "subset",
            feature_chunk_size=arguments.feature_chunk_size,
            feature_cache=arguments.feature_cache,
            hidden_size=arguments.hidden_size,
            batch_size=arguments.batch_size,
            epochs=arguments.epochs,
            learning_rate=arguments.learning_rate,
            pretrain_epochs=arguments.pretrain_epochs,
            pretrain_learning_rate=arguments.pretrain_learning_rate,
            minimum_pretrain_validation_accuracy=arguments.minimum_pretrain_validation_accuracy,
            calibration_batches=arguments.calibration_batches,
            quant_bitwidths=tuple(arguments.quant_bitwidths or (8, 16)),
            seed=arguments.seed,
            quality_gate_seeds=tuple(arguments.quality_gate_seeds or ()),
            extended_diagnostics=not arguments.no_extended_diagnostics,
        )
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
