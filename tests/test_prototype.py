from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from flip_observer.core import (
    calibrate_identity_vs_reversal,
    fixed_weights,
    hadamard_calibration,
    readout,
    reverse_channels,
    undo_reversal,
)
from flip_observer.experiment import _one_trial, render_report

ROOT = Path(__file__).resolve().parents[1]


class FlipObserverTests(unittest.TestCase):
    def test_reversal_twice_restores_identity(self) -> None:
        rng = np.random.default_rng(29)
        values = rng.normal(size=(31, 512))
        np.testing.assert_array_equal(reverse_channels(reverse_channels(values)), values)

    def test_known_inverse_recovers_noiseless_channels_and_readout_exactly(self) -> None:
        rng = np.random.default_rng(44)
        values = rng.normal(size=(67, 512))
        weights = fixed_weights(512)
        target = readout(values, weights)  # target fixed before hidden reversal
        observed = reverse_channels(values)
        corrected = undo_reversal(observed)
        np.testing.assert_array_equal(corrected, values)
        np.testing.assert_array_equal(readout(corrected, weights), target)
        self.assertEqual(float(np.max(np.abs(readout(corrected, weights) - target))), 0.0)

    def test_full_vector_calibration_detects_identity_and_reversal(self) -> None:
        patterns = hadamard_calibration(512)
        rng = np.random.default_rng(203)
        dropout = 0.20
        noise = 0.20
        for hidden_reversal in (False, True):
            mask = rng.random(patterns.shape) >= dropout
            measured = patterns + rng.normal(0.0, noise, patterns.shape)
            if hidden_reversal:
                measured, mask = reverse_channels(measured), reverse_channels(mask)
            result = calibrate_identity_vs_reversal(patterns, measured, mask)
            expected = "reversed" if hidden_reversal else "identity"
            self.assertEqual(result.mapping, expected)

    def test_scalar_output_is_not_used_for_mapping_detection(self) -> None:
        # The calibration API requires sample-by-channel vectors, not scalar outputs.
        patterns = hadamard_calibration(512)
        with self.assertRaises(ValueError):
            calibrate_identity_vs_reversal(patterns[:, 0], patterns[:, 0])

    def test_exact_duplicate_bank_adds_no_distinct_input_channels_or_readout(self) -> None:
        rng = np.random.default_rng(71)
        bank = rng.normal(size=(80, 512))
        duplicated = np.concatenate((bank, bank), axis=1)
        self.assertEqual(duplicated.shape[1], 1024)
        self.assertEqual(np.unique(duplicated, axis=1).shape[1], 512)
        single_estimate = (bank + bank) / 2
        duplicate_fused_estimate = np.concatenate((bank, bank), axis=1).reshape(80, 2, 512).mean(axis=1)
        np.testing.assert_array_equal(duplicate_fused_estimate, single_estimate)

    def test_trial_metrics_reproduce_exactly_for_same_seed(self) -> None:
        first = _one_trial(seed=5, noise_std=0.05, dropout=0.05)
        second = _one_trial(seed=5, noise_std=0.05, dropout=0.05)
        self.assertEqual(first, second)

    def test_generated_sweep_meets_predeclared_pass_criteria(self) -> None:
        path = ROOT / "results" / "metrics.json"
        self.assertTrue(path.exists(), "run `python3 run.py` before acceptance tests")
        results = json.loads(path.read_text(encoding="utf-8"))
        for condition in results["conditions"]:
            label = f"noise={condition['noise_std']}, dropout={condition['dropout']}"
            self.assertGreaterEqual(condition["identity_calibration_accuracy"], 0.95, label)
            self.assertGreaterEqual(condition["reversal_calibration_accuracy"], 0.95, label)
            misses = condition["calibrated_not_better_than_uncorrected_count"]
            self.assertGreaterEqual(1.0 - misses / condition["trial_count"], 0.90, label)
            for trial in condition["trials"]:
                self.assertEqual(trial["duplicate_unique_channels"], 512, label)
                self.assertEqual(trial["duplicate_mask_shape_channels"], 1024, label)
                self.assertEqual(trial["duplicate_vs_single_max_abs"], 0.0, label)

        clean = next(c for c in results["conditions"] if c["noise_std"] == 0.0 and c["dropout"] == 0.0)
        self.assertEqual(clean["metrics"]["oracle_nrmse"]["max"], 0.0)

    def test_bundled_report_is_exactly_what_the_generator_writes(self) -> None:
        # Guards against hand-edited/stale report text: every section, including the
        # acceptance outcome and payload arithmetic, must come from render_report().
        results = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
        bundled = (ROOT / "results" / "sample_report.md").read_text(encoding="utf-8")
        self.assertEqual(render_report(results), bundled)
        self.assertIn("## Acceptance outcome and payload arithmetic", bundled)


if __name__ == "__main__":
    unittest.main()
