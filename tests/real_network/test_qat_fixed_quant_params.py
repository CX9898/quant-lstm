"""QAT updates master weights while preserving the initial PTQ quantizer."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
import warnings
from pathlib import Path

import torch

from speech_commands_lstm_training import (
    ExperimentConfig,
    SpeechCommandsLstmClassifier,
    _calibrate_quant_lstm,
    _train,
)


@unittest.skipUnless(torch.cuda.is_available(), "requires the native CUDA QAT path")
class FixedQuantParamsTest(unittest.TestCase):
    def test_training_and_reload_preserve_initial_ptq_parameters(self) -> None:
        torch.set_num_threads(2)
        device = torch.device("cuda")
        config = ExperimentConfig(
            dataset_root=Path("unused"), labels=("no", "yes"),
            hidden_size=8, batch_size=4, epochs=2, calibration_batches=2,
        )
        for bits, cell16 in ((8, False), (16, False), (8, True)):
            with self.subTest(bits=bits, cell16=cell16), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                torch.manual_seed(314)
                data = (torch.randn(8, 5, 20), torch.arange(8) % 2)
                features = {split: data for split in ("training", "validation", "testing")}
                model = SpeechCommandsLstmClassifier(
                    8, 2, use_quant_lstm=True, device=device
                )
                _calibrate_quant_lstm(
                    model, data, config, device, bits,
                    operator_bitwidths={"cell_state": 16} if cell16 else None,
                )
                ptq_params = model.lstm.export_quant_params()
                original_weight = model.lstm.weight_ih_l0.detach().clone()
                observed_training = []
                observed_weight_updates = []

                def check_quantizer(module, inputs):
                    self.assertFalse(module.calibrating, "QAT recalibrated after initial PTQ")
                    self.assertEqual(module.export_quant_params(), ptq_params)
                    observed_training.append(module.training)
                    observed_weight_updates.append(not torch.equal(original_weight, module.weight_ih_l0))

                handle = model.lstm.register_forward_pre_hook(check_quantizer)
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = _train(
                            "fixed_qat", model, features, config, device,
                            qat_bitwidth=bits, extended_diagnostics=False,
                        )
                finally:
                    handle.remove()
                self.assertIn(True, observed_training)
                self.assertIn(False, observed_training)
                self.assertTrue(result["native_qat_checkpoint_observed"])
                self.assertTrue(any(observed_weight_updates))
                self.assertGreater(result["parameter_update_norm"], 0)
                self.assertEqual(model.lstm.export_quant_params(), ptq_params)
                model.eval()
                with torch.no_grad():
                    expected = model(data[0].to(device))
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "trained.pt"
                    torch.save({"weights": model.state_dict(), "quant_params": ptq_params}, path)
                    saved = torch.load(path, weights_only=True)
                restored = SpeechCommandsLstmClassifier(
                    8, 2, use_quant_lstm=True, device=device
                )
                restored.load_state_dict(saved["weights"])
                restored.lstm.load_quant_params(saved["quant_params"])
                restored.lstm.use_quantization = True
                restored.eval()
                with torch.no_grad():
                    actual = restored(data[0].to(device))
                self.assertEqual(restored.lstm.export_quant_params(), ptq_params)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
