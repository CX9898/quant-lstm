"""Full 35-class Speech Commands v0.02 training and evaluation gate."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

import torch

from pretrained_quality_assertions import QUALITY_POLICY, assert_pretrained_quality

from speech_commands_lstm_training import (
    FULL_SPEECH_COMMAND_LABELS,
    ExperimentConfig,
    run_training_comparison,
)


ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "tests/results/speech_commands_lstm_full_training.json"


class SpeechCommandsFullTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if os.environ.get("QUANT_LSTM_RUN_FULL_SPEECH_COMMANDS") != "1":
            raise unittest.SkipTest(
                "set QUANT_LSTM_RUN_FULL_SPEECH_COMMANDS=1 to run full training"
            )
        dataset_root = os.environ.get("QUANT_LSTM_SPEECH_COMMANDS_ROOT")
        if not dataset_root:
            raise RuntimeError("QUANT_LSTM_SPEECH_COMMANDS_ROOT is required")
        torch.set_num_threads(8)
        cls.dataset_root = Path(dataset_root)
        if not torch.cuda.is_available():
            raise unittest.SkipTest("full QuantLSTM training requires CUDA")

    def test_all_official_samples_train_and_evaluate_all_lstm_modes(self) -> None:
        report = run_training_comparison(
            ExperimentConfig(
                dataset_root=self.dataset_root,
                labels=FULL_SPEECH_COMMAND_LABELS,
                train_samples_per_label=None,
                validation_samples_per_label=None,
                test_samples_per_label=None,
                dataset_profile="full",
                feature_chunk_size=512,
                feature_cache=None,
                hidden_size=64,
                batch_size=256,
                pretrain_epochs=50,
                minimum_pretrain_validation_accuracy=0.90,
                epochs=20,
                learning_rate=3.0e-4,
                calibration_batches=4,
                quant_bitwidths=(8, 16),
                seed=20260921,
                quality_gate_seeds=(20260921, 20260922, 20260923, 20261008),
                extended_diagnostics=False,
            )
        )
        report["quality_policy"] = QUALITY_POLICY["full"]
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        dataset = report["dataset"]
        self.assertEqual(dataset["profile"], "full")
        self.assertEqual(dataset["labels"], list(FULL_SPEECH_COMMAND_LABELS))
        self.assertEqual(dataset["training_samples"], 84_843)
        self.assertEqual(dataset["validation_samples"], 9_981)
        self.assertEqual(dataset["test_samples"], 11_005)
        self.assertEqual(dataset["feature_shape"], [49, 20])
        audit = dataset["audit"]
        self.assertEqual(audit["selected_word_sample_count"], 105_829)
        self.assertEqual(audit["omitted_word_sample_count"], 0)
        self.assertEqual(audit["split_overlap_count"], 0)

        assert_pretrained_quality(self, report)
        for calibration in report["quantization"]["calibration"].values():
            self.assertEqual(calibration["sample_count"], 1024)
            self.assertEqual(sum(calibration["label_counts"]), 1024)


if __name__ == "__main__":
    unittest.main()
