"""Focused tests for Stage 3.5 experiment design and paired evaluation."""

import unittest

import numpy as np

from ibfd_simulation import apply_multipath_channel
from qpsk_communication import generate_random_bits, qpsk_modulate
from robustness_experiments import (
    run_one_configuration,
    scale_si_channel_to_ratio,
    summarize_trials,
)


class RobustnessExperimentTests(unittest.TestCase):
    def test_si_scaling_hits_requested_power_ratio_and_preserves_shape(self):
        tx = qpsk_modulate(generate_random_bits(2000, 101))
        desired = qpsk_modulate(generate_random_bits(2000, 100))
        scaled, measured = scale_si_channel_to_ratio(tx, desired, "multipath_4", 20.0)
        self.assertAlmostEqual(measured, 20.0, places=10)
        base = .65 ** np.arange(4) * np.exp(1j * .43 * np.arange(4))
        ratios = scaled / base
        np.testing.assert_allclose(ratios, np.repeat(ratios[0], 4))
        self.assertAlmostEqual(float(np.imag(ratios[0])), 0.0, places=12)

    def test_flat_and_multipath_ratio_definitions(self):
        tx = qpsk_modulate(generate_random_bits(1000, 1))
        desired = qpsk_modulate(generate_random_bits(1000, 2))
        for channel in ("flat", "multipath_2", "multipath_8"):
            _, actual = scale_si_channel_to_ratio(tx, desired, channel, -20.0)
            self.assertAlmostEqual(actual, -20.0, places=10)

    def test_identical_paired_input_and_separate_train_val_test_ranges(self):
        row, lms_curve, nlms_curve = run_one_configuration(
            num_bits=2000, seed=8, channel="multipath_4",
            si_to_desired_db=0, noise_snr_db=20, filter_length=4,
        )
        self.assertEqual(row["lms_ber_before"], row["nlms_ber_before"])
        self.assertEqual(row["lms_ber_before"], row["no_cancellation_ber_before"])
        self.assertEqual(row["training_samples"], row["validation_start"])
        self.assertEqual(row["validation_stop"], row["test_start"])
        self.assertEqual(row["test_stop"], row["num_symbols"])
        self.assertLess(row["training_samples"], row["validation_stop"])
        self.assertLess(row["validation_stop"], row["test_stop"])
        self.assertEqual(lms_curve.size, row["training_samples"])
        self.assertEqual(nlms_curve.size, row["training_samples"])
        self.assertAlmostEqual(row["si_to_desired_db_measured"], 0.0, places=10)

    def test_deterministic_trial(self):
        kwargs = dict(num_bits=1200, seed=4, channel="flat",
                      si_to_desired_db=10, noise_snr_db=15, filter_length=2)
        first = run_one_configuration(**kwargs)
        second = run_one_configuration(**kwargs)
        self.assertEqual(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        np.testing.assert_array_equal(first[2], second[2])

    def test_summary_uses_mean_and_sample_standard_deviation(self):
        rows = [
            {"sweep": "sir", "factor": "0", "algorithm": "LMS",
             "cancellation_db": value, "ber_before": .2, "ber_after": .1 * value,
             "residual_si_power": value}
            for value in (1., 2., 3.)
        ]
        result = summarize_trials(rows, ("sweep", "factor"))[0]
        self.assertEqual(result["seed_count"], 3)
        self.assertAlmostEqual(result["mean_cancellation_db"], 2.0)
        self.assertAlmostEqual(result["std_cancellation_db"], 1.0)
        self.assertAlmostEqual(result["mean_residual_si_power"], 2.0)
        self.assertAlmostEqual(result["mean_ber_after"], .2)

    def test_channel_characteristic_taps_are_a_fixed_shape(self):
        tx = qpsk_modulate(generate_random_bits(400, 7))
        desired = np.ones_like(tx)
        taps, _ = scale_si_channel_to_ratio(tx, desired, "multipath_8", 0)
        output = apply_multipath_channel(tx, taps)
        self.assertEqual(output.size, tx.size)
        self.assertTrue(np.all(np.isfinite(output)))


if __name__ == "__main__":
    unittest.main()
