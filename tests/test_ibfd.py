"""Checks for the Stage 2 IBFD signal path."""

import unittest

import numpy as np

from ibfd_simulation import (
    apply_flat_channel,
    apply_multipath_channel,
    apply_nonlinear_si,
    simulate_ibfd,
)
from qpsk_communication import qpsk_demodulate, qpsk_modulate


class IBFDTests(unittest.TestCase):
    def test_disabled_si_and_noise_leaves_desired(self):
        result = simulate_ibfd(num_bits=20, desired_gain=0.7 + 0.2j,
                               si_enabled=False, noise_enabled=False)
        expected = apply_flat_channel(qpsk_modulate(
            np.random.default_rng(11).integers(0, 2, 20, dtype=np.uint8)), 0.7 + 0.2j)
        np.testing.assert_allclose(result.received, expected)

    def test_disabled_desired_and_noise_leaves_si(self):
        result = simulate_ibfd(num_bits=20, desired_enabled=False, si_enabled=True,
                               si_mode="flat", si_gain=0.3 - 0.1j, noise_enabled=False)
        np.testing.assert_allclose(result.received, result.self_interference)

    def test_flat_channel_known_values(self):
        x = np.array([1 + 2j, -1j])
        np.testing.assert_allclose(apply_flat_channel(x, 2 - 1j),
                                   np.array([4 + 3j, -1 - 2j]))

    def test_multipath_matches_full_convolution_prefix(self):
        x = np.array([1 + 1j, 2 - 1j, -1j])
        h = np.array([0.5j, 1 - 0.25j])
        expected = np.convolve(x, h, mode="full")[:len(x)]
        actual = apply_multipath_channel(x, h)
        np.testing.assert_allclose(actual, expected)
        self.assertEqual(actual.size, x.size)

    def test_nonlinear_formula(self):
        x = np.array([1 + 1j, 0.5 - 2j])
        expected = (0.8 + .2j) * x + (-.1 + .05j) * x * np.abs(x) ** 2
        np.testing.assert_allclose(apply_nonlinear_si(x, .8 + .2j, -.1 + .05j), expected)

    def test_total_is_sum_and_complex_aligned_components(self):
        result = simulate_ibfd(num_bits=200, si_mode="multipath",
                               si_taps=np.array([.1 + .2j, -.03j]), noise_snr_db=15)
        np.testing.assert_allclose(result.received,
                                   result.desired + result.self_interference + result.noise)
        for component in (result.desired, result.tx_reference, result.self_interference,
                          result.noise, result.received):
            self.assertEqual(component.dtype, np.complex128)
            self.assertEqual(component.size, 100)

    def test_existing_qpsk_round_trip_is_unchanged(self):
        bits = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.uint8)
        np.testing.assert_array_equal(qpsk_demodulate(qpsk_modulate(bits)), bits)

    def test_empty_multipath_and_invalid_inputs(self):
        self.assertEqual(apply_multipath_channel(np.array([], dtype=complex), [1]).size, 0)
        with self.assertRaises(ValueError):
            apply_multipath_channel([1 + 0j], [])
        with self.assertRaises(ValueError):
            simulate_ibfd(num_bits=9)
        with self.assertRaises(ValueError):
            simulate_ibfd(si_mode="multipath")


if __name__ == "__main__":
    unittest.main()
