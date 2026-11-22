"""Deterministic unit checks for complex LMS/NLMS SI cancellation."""

import inspect
import unittest

import numpy as np

from adaptive_si_cancellation import (
    _ratio_db,
    adaptive_cancel,
    run_cancellation_experiment,
)
from ibfd_simulation import apply_multipath_channel
from qpsk_communication import calculate_ber, qpsk_demodulate, qpsk_modulate


class AdaptiveSITests(unittest.TestCase):
    def test_correct_complex_lms_update_and_conjugation(self):
        x = np.array([1 + 2j])
        y = np.array([.5 - .25j])
        result = adaptive_cancel(x, y, algorithm="lms", filter_length=1,
                                 step_size=.2, training_samples=1,
                                 freeze_after_training=False)
        expected_error = y[0]
        expected_weight = .2 * x[0] * np.conjugate(expected_error)
        self.assertAlmostEqual(result.estimate[0], 0j)
        np.testing.assert_allclose(result.final_weights, [expected_weight])

    def test_correct_complex_nlms_update(self):
        x = np.array([1 + 1j])
        y = np.array([.25 + .5j])
        mu, eps = .4, .1
        result = adaptive_cancel(x, y, algorithm="nlms", filter_length=1,
                                 step_size=mu, training_samples=1, epsilon=eps,
                                 freeze_after_training=False)
        expected = (mu / (eps + abs(x[0]) ** 2)) * x[0] * np.conjugate(y[0])
        np.testing.assert_allclose(result.final_weights, [expected])

    def test_reference_vector_order_is_current_then_past(self):
        x = np.array([1 + 0j, 2 + 1j, -1 + 2j, .5j])
        y = np.array([1, 0, 0, 0], dtype=complex)
        result = adaptive_cancel(x, y, algorithm="lms", filter_length=3,
                                 step_size=.1, training_samples=1,
                                 freeze_after_training=True)
        # First update uses [x[0], 0, 0], and is frozen for later samples.
        np.testing.assert_allclose(result.final_weights, [.1 + 0j, 0j, 0j])
        self.assertAlmostEqual(result.estimate[1], np.vdot(result.final_weights, [x[1], x[0], 0]))

    def test_flat_channel_cancellation_on_held_out_samples(self):
        x = np.tile(np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2), 80)
        true = (.4 - .2j) * x
        result = adaptive_cancel(x, true, algorithm="nlms", filter_length=1,
                                 step_size=.8, training_samples=200, epsilon=1e-10)
        self.assertLess(np.mean(np.abs(true[200:] - result.estimate[200:]) ** 2), 1e-5)

    def test_multipath_cancellation(self):
        x = np.tile(np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2), 300)
        taps = np.array([.5 + .1j, .2 - .15j, -.08 + .03j])
        true = apply_multipath_channel(x, taps)
        result = adaptive_cancel(x, true, algorithm="nlms", filter_length=3,
                                 step_size=.8, training_samples=800, epsilon=1e-10)
        self.assertLess(np.mean(np.abs(true[800:] - result.estimate[800:]) ** 2), 1e-4)

    def test_zero_reference_is_finite_and_does_not_adapt(self):
        result = adaptive_cancel(np.zeros(6), np.ones(6), algorithm="nlms",
                                 filter_length=3, step_size=.5,
                                 training_samples=3)
        np.testing.assert_array_equal(result.estimate, np.zeros(6))
        np.testing.assert_array_equal(result.final_weights, np.zeros(3))
        self.assertTrue(np.all(np.isfinite(result.smoothed_error_power)))

    def test_reproducible_experiment_and_lengths(self):
        kwargs = dict(num_bits=1000, seed=5, training_fraction=.6,
                      filter_length=3, si_mode="multipath",
                      si_taps=np.array([.6 + .1j, .12 - .05j]), noise_snr_db=22)
        first = run_cancellation_experiment(**kwargs)
        second = run_cancellation_experiment(**kwargs)
        np.testing.assert_array_equal(first.lms.estimate, second.lms.estimate)
        np.testing.assert_array_equal(first.nlms.residual, second.nlms.residual)
        self.assertEqual(first.ibfd.received.size, 500)
        self.assertEqual(first.evaluation_slice.start, 300)

    def test_training_evaluation_freeze_boundary(self):
        x = np.ones(10, dtype=complex)
        y = np.arange(1, 11, dtype=complex)
        training = 4
        result = adaptive_cancel(x, y, algorithm="lms", filter_length=1,
                                 step_size=.1, training_samples=training)
        np.testing.assert_array_equal(result.weights_history[training + 1:],
                                      np.repeat(result.final_weights[None, :],
                                                10 - training, axis=0))

    def test_si_metric_and_zero_power_cases(self):
        self.assertAlmostEqual(_ratio_db(1.0, .01), 20.0)
        self.assertEqual(_ratio_db(1.0, 0.0), float("inf"))
        self.assertTrue(np.isnan(_ratio_db(0.0, 0.0)))

    def test_ber_alignment_and_demodulation(self):
        bits = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.uint8)
        symbols = qpsk_modulate(bits)
        recovered = qpsk_demodulate(symbols)
        self.assertEqual(calculate_ber(bits, recovered), (8, 0, 0.0))
        experiment = run_cancellation_experiment(
            num_bits=400, seed=2, training_fraction=.5,
            si_mode="flat", si_gain=.2 + .1j, noise_snr_db=30,
            filter_length=2)
        self.assertEqual(experiment.desired_bits[2 * experiment.training_samples:].size,
                         experiment.ibfd.received[experiment.evaluation_slice].size * 2)

    def test_adaptation_api_has_no_true_si_input(self):
        self.assertNotIn("true_si", inspect.signature(adaptive_cancel).parameters)
        # The filter is called with observation only and remains well-defined.
        result = adaptive_cancel(np.ones(5), np.ones(5), algorithm="lms",
                                 filter_length=1, step_size=.1, training_samples=3)
        self.assertEqual(result.estimate.size, 5)

    def test_existing_qpsk_and_ibfd_interfaces_remain_usable(self):
        bits = np.array([0, 0, 1, 1], dtype=np.uint8)
        np.testing.assert_array_equal(qpsk_demodulate(qpsk_modulate(bits)), bits)
        experiment = run_cancellation_experiment(
            num_bits=200, seed=1, training_fraction=.5, si_mode="flat",
            si_gain=.1, noise_snr_db=20)
        self.assertEqual(experiment.ibfd.tx_reference.size, 100)


if __name__ == "__main__":
    unittest.main()
