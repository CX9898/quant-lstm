"""Contracts for full Speech Commands v0.02 dataset coverage."""

from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import torch

from speech_commands_lstm_training import (
    ExperimentConfig,
    build_dataset_manifest,
    classification_metrics,
    load_feature_sets,
)


class SpeechCommandsFullDatasetContractTest(unittest.TestCase):
    @staticmethod
    def _write_silence(path: Path) -> None:
        with wave.open(str(path), "wb") as destination:
            destination.setnchannels(1)
            destination.setsampwidth(2)
            destination.setframerate(16_000)
            destination.writeframes(b"\0\0" * 16_000)

    def test_full_profile_consumes_every_word_sample_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label, names in {
                "no": ("no_train.wav", "no_test.wav"),
                "yes": ("yes_train.wav", "yes_validation.wav"),
                "_background_noise_": ("noise.wav",),
            }.items():
                directory = root / label
                directory.mkdir()
                for name in names:
                    (directory / name).touch()
            (root / "validation_list.txt").write_text(
                "yes/yes_validation.wav\n", encoding="utf-8"
            )
            (root / "testing_list.txt").write_text(
                "no/no_test.wav\n", encoding="utf-8"
            )

            manifest = build_dataset_manifest(
                ExperimentConfig(
                    dataset_root=root,
                    labels=("no", "yes"),
                    train_samples_per_label=None,
                    validation_samples_per_label=None,
                    test_samples_per_label=None,
                    dataset_profile="full",
                )
            )

            self.assertEqual(
                manifest.audit["split_sample_counts"],
                {"training": 2, "validation": 1, "testing": 1},
            )
            self.assertEqual(manifest.audit["selected_word_sample_count"], 4)
            self.assertEqual(manifest.audit["omitted_word_sample_count"], 0)
            self.assertEqual(manifest.audit["split_overlap_count"], 0)
            self.assertEqual(manifest.audit["background_noise_file_count"], 1)
            selected = {
                sample.path.relative_to(root).as_posix()
                for split in manifest.samples.values()
                for sample in split
            }
            self.assertEqual(
                selected,
                {
                    "no/no_train.wav",
                    "no/no_test.wav",
                    "yes/yes_train.wav",
                    "yes/yes_validation.wav",
                },
            )

    def test_feature_loading_chunks_full_manifest_without_changing_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label, names in {
                "no": ("no_train.wav", "no_test.wav"),
                "yes": ("yes_train.wav", "yes_validation.wav"),
            }.items():
                directory = root / label
                directory.mkdir()
                for name in names:
                    self._write_silence(directory / name)
            (root / "validation_list.txt").write_text(
                "yes/yes_validation.wav\n", encoding="utf-8"
            )
            (root / "testing_list.txt").write_text(
                "no/no_test.wav\n", encoding="utf-8"
            )
            config = ExperimentConfig(
                dataset_root=root,
                labels=("no", "yes"),
                train_samples_per_label=None,
                validation_samples_per_label=None,
                test_samples_per_label=None,
                dataset_profile="full",
                feature_chunk_size=1,
            )

            feature_sets, manifest, digests = load_feature_sets(config)

            self.assertEqual(manifest.audit["selected_word_sample_count"], 4)
            self.assertEqual(
                {split: values[0].shape for split, values in feature_sets.items()},
                {
                    "training": torch.Size((2, 49, 20)),
                    "validation": torch.Size((1, 49, 20)),
                    "testing": torch.Size((1, 49, 20)),
                },
            )
            self.assertTrue(
                all(torch.isfinite(features).all() for features, _ in feature_sets.values())
            )
            self.assertEqual(set(digests), {"training", "validation", "testing"})

    def test_classification_metrics_report_every_label_and_macro_scores(self) -> None:
        metrics = classification_metrics(
            torch.tensor([[2, 1], [1, 4]], dtype=torch.int64), ("no", "yes")
        )

        self.assertEqual(metrics["sample_count"], 8)
        self.assertEqual(metrics["correct_count"], 6)
        self.assertEqual(metrics["confusion_matrix"], [[2, 1], [1, 4]])
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["macro_recall"], (2 / 3 + 4 / 5) / 2)
        self.assertAlmostEqual(metrics["per_class"]["no"]["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["per_class"]["yes"]["f1"], 0.8)


if __name__ == "__main__":
    unittest.main()
