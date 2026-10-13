"""Validation tests for the QPSK baseband model."""

import unittest

import numpy as np

from qpsk_communication import (
    CONSTELLATION,
    awgn_channel,
    calculate_ber,
    generate_random_bits,
    qpsk_demodulate,
    qpsk_modulate,
    run_ebn0_sweep,
    theoretical_qpsk_ber,
)


class QPSKTests(unittest.TestCase):
    def test_gray_mapping_and_noiseless_round_trip(self):
        bits = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.uint8)
        symbols = qpsk_modulate(bits)
        np.testing.assert_allclose(symbols, np.array(list(CONSTELLATION.values())))
        np.testing.assert_array_equal(qpsk_demodulate(symbols), bits)

    def test_unit_average_symbol_energy(self):
        symbols = qpsk_modulate(generate_random_bits(200_000, seed=1))
        self.assertAlmostEqual(float(np.mean(np.abs(symbols) ** 2)), 1.0, places=12)

    def test_awgn_complex_variance(self):
        count, ebn0_db = 500_000, 4.0
        zeros = np.zeros(count, dtype=np.complex128)
        noise = awgn_channel(zeros, ebn0_db, np.random.default_rng(22))
        n0 = 0.5 / (10 ** (ebn0_db / 10))
        self.assertAlmostEqual(float(np.var(noise.real)), n0 / 2, delta=n0 * .01)
        self.assertAlmostEqual(float(np.var(noise.imag)), n0 / 2, delta=n0 * .01)
        self.assertAlmostEqual(float(np.mean(np.abs(noise) ** 2)), n0, delta=n0 * .01)

    def test_ber_accounting_and_length_validation(self):
        total, errors, ber = calculate_ber(np.array([0, 1, 1]), np.array([0, 0, 1]))
        self.assertEqual((total, errors), (3, 1))
        self.assertAlmostEqual(ber, 1 / 3)
        with self.assertRaises(ValueError):
            calculate_ber(np.array([0]), np.array([0, 1]))

    def test_sweep_follows_theory_and_decreases_overall(self):
        points, measured = run_ebn0_sweep([0, 2, 4, 6, 8], num_bits=400_000, seed=3)
        expected = theoretical_qpsk_ber(points)
        # Allow Monte Carlo variation while checking useful agreement and trend.
        self.assertTrue(np.all(np.diff(measured) < 0))
        np.testing.assert_allclose(measured, expected, rtol=.18, atol=2e-4)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            generate_random_bits(3)
        with self.assertRaises(ValueError):
            qpsk_modulate(np.array([0, 2]))


if __name__ == "__main__":
    unittest.main()
