"""Reproducible Gray-coded QPSK baseband simulation over complex AWGN."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BITS_PER_SYMBOL = 2
SYMBOL_ENERGY = 1.0
_SCALE = 1.0 / math.sqrt(2.0)
CONSTELLATION = {
    (0, 0): _SCALE * (1.0 + 1.0j),
    (0, 1): _SCALE * (-1.0 + 1.0j),
    (1, 1): _SCALE * (-1.0 - 1.0j),
    (1, 0): _SCALE * (1.0 - 1.0j),
}


def generate_random_bits(num_bits: int, seed: int | None = 7) -> np.ndarray:
    """Generate an even-length reproducible array of binary values."""
    if not isinstance(num_bits, (int, np.integer)) or num_bits <= 0 or num_bits % 2:
        raise ValueError("num_bits must be a positive even integer")
    return np.random.default_rng(seed).integers(0, 2, size=num_bits, dtype=np.uint8)


def qpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """Map successive bit pairs to unit-energy Gray-coded QPSK symbols."""
    bits = np.asarray(bits)
    if bits.ndim != 1 or bits.size == 0 or bits.size % 2:
        raise ValueError("bits must be a non-empty, even-length one-dimensional array")
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must contain only 0 and 1")
    pairs = bits.reshape(-1, 2)
    symbols = np.fromiter((CONSTELLATION[(int(a), int(b))] for a, b in pairs),
                          dtype=np.complex128, count=len(pairs))
    return symbols


def awgn_channel(symbols: np.ndarray, ebn0_db: float,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Add independent complex Gaussian noise at the requested Eb/N0.

    For unit-energy QPSK, Es=1 and Es=2Eb, so Eb=1/2. N0=Eb/(Eb/N0).
    Each real component has variance N0/2; hence E[|w|^2]=N0.
    """
    symbols = np.asarray(symbols, dtype=np.complex128)
    if symbols.ndim != 1:
        raise ValueError("symbols must be a one-dimensional array")
    if not np.isfinite(ebn0_db):
        raise ValueError("ebn0_db must be finite")
    ebn0_linear = 10.0 ** (float(ebn0_db) / 10.0)
    eb = SYMBOL_ENERGY / BITS_PER_SYMBOL
    n0 = eb / ebn0_linear
    component_std = math.sqrt(n0 / 2.0)
    if rng is None:
        rng = np.random.default_rng()
    noise = component_std * (rng.standard_normal(symbols.size) +
                              1j * rng.standard_normal(symbols.size))
    return symbols + noise


def qpsk_demodulate(received: np.ndarray) -> np.ndarray:
    """Hard-decision demodulation using nearest constellation point.

    Exact ties are resolved by the first point in CONSTELLATION insertion order.
    """
    received = np.asarray(received, dtype=np.complex128)
    if received.ndim != 1 or not np.all(np.isfinite(received)):
        raise ValueError("received must be a finite one-dimensional complex array")
    points = np.array(list(CONSTELLATION.values()), dtype=np.complex128)
    labels = np.array(list(CONSTELLATION.keys()), dtype=np.uint8)
    distances = np.abs(received[:, None] - points[None, :]) ** 2
    decisions = labels[np.argmin(distances, axis=1)]
    return decisions.reshape(-1)


def calculate_ber(transmitted_bits: np.ndarray, recovered_bits: np.ndarray) -> tuple[int, int, float]:
    """Return (number of bits, errors, BER), validating equal array lengths."""
    tx = np.asarray(transmitted_bits)
    rx = np.asarray(recovered_bits)
    if tx.ndim != 1 or rx.ndim != 1 or tx.size != rx.size:
        raise ValueError("transmitted and recovered bits must be 1-D arrays of equal length")
    if tx.size == 0:
        raise ValueError("BER is undefined for empty bit arrays")
    errors = int(np.count_nonzero(tx != rx))
    return int(tx.size), errors, errors / int(tx.size)


def theoretical_qpsk_ber(ebn0_db: float | np.ndarray) -> float | np.ndarray:
    """Gray-coded coherent QPSK BER, equal to coherent BPSK BER."""
    linear = np.power(10.0, np.asarray(ebn0_db, dtype=float) / 10.0)
    values = np.vectorize(math.erfc)(np.sqrt(linear)) * 0.5
    return float(values) if values.ndim == 0 else values


def run_single_simulation(num_bits: int, ebn0_db: float, seed: int = 7
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float]:
    """Run bits through modulation, AWGN, and demodulation once."""
    bits = generate_random_bits(num_bits, seed)
    tx_symbols = qpsk_modulate(bits)
    rx_symbols = awgn_channel(tx_symbols, ebn0_db, np.random.default_rng(seed + 1))
    recovered = qpsk_demodulate(rx_symbols)
    total, errors, ber = calculate_ber(bits, recovered)
    return bits, tx_symbols, rx_symbols, total, errors, ber


def run_ebn0_sweep(ebn0_values_db: np.ndarray | list[float], num_bits: int = 200_000,
                   seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Measure BER at each Eb/N0 with reusable bits and independent noise."""
    points = np.asarray(ebn0_values_db, dtype=float)
    if points.ndim != 1 or points.size == 0 or not np.all(np.isfinite(points)):
        raise ValueError("ebn0_values_db must be a non-empty finite 1-D sequence")
    bits = generate_random_bits(num_bits, seed)
    symbols = qpsk_modulate(bits)
    bers = []
    for index, value in enumerate(points):
        # SeedSequence gives each SNR point an independent, reproducible noise stream.
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        recovered = qpsk_demodulate(awgn_channel(symbols, float(value), rng))
        bers.append(calculate_ber(bits, recovered)[2])
    return points, np.asarray(bers)


def _save_figures(tx_symbols: np.ndarray, rx_symbols: np.ndarray, ebn0_db: float,
                  sweep_db: np.ndarray, simulated_ber: np.ndarray, output_dir: Path,
                  sample_count: int = 40) -> None:
    """Create the requested constellation, IQ, and BER figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ideal = np.array(list(CONSTELLATION.values()))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(tx_symbols.real, tx_symbols.imag, s=12, alpha=.55, label="Transmitted symbols")
    ax.scatter(ideal.real, ideal.imag, marker="x", s=100, linewidths=2, label="Ideal points")
    ax.set(title="Transmitted QPSK constellation", xlabel="In-phase (I)", ylabel="Quadrature (Q)")
    ax.axis("equal"); ax.grid(True); ax.legend(); fig.tight_layout()
    fig.savefig(output_dir / "transmitted_constellation.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(rx_symbols.real, rx_symbols.imag, s=12, alpha=.45, label=f"Received at {ebn0_db:g} dB")
    ax.scatter(ideal.real, ideal.imag, marker="x", s=100, linewidths=2, label="Ideal points")
    ax.set(title="Received QPSK constellation (AWGN spreads symbols)", xlabel="In-phase (I)", ylabel="Quadrature (Q)")
    ax.axis("equal"); ax.grid(True); ax.legend(); fig.tight_layout()
    fig.savefig(output_dir / "received_constellation.png", dpi=150); plt.close(fig)

    count = min(sample_count, tx_symbols.size)
    index = np.arange(count)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(index, tx_symbols[:count].real, "o-", label="Transmitted I", markersize=3)
    axes[0].plot(index, rx_symbols[:count].real, ".-", label="Received I", alpha=.8)
    axes[0].set_ylabel("Amplitude"); axes[0].grid(True); axes[0].legend()
    axes[1].plot(index, tx_symbols[:count].imag, "o-", label="Transmitted Q", markersize=3)
    axes[1].plot(index, rx_symbols[:count].imag, ".-", label="Received Q", alpha=.8)
    axes[1].set(xlabel="Symbol index", ylabel="Amplitude"); axes[1].grid(True); axes[1].legend()
    fig.suptitle("Transmitted and received complex baseband samples")
    fig.tight_layout(); fig.savefig(output_dir / "iq_samples.png", dpi=150); plt.close(fig)

    theory = theoretical_qpsk_ber(sweep_db)
    # A zero observed BER cannot be shown on a logarithmic axis; show a marker at
    # half a bit-error resolution (0.5 / N) and retain the exact value in output.
    displayed = np.maximum(simulated_ber, 0.5 / (2 * tx_symbols.size))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(sweep_db, displayed, "o-", label="Simulated BER")
    ax.semilogy(sweep_db, theory, "s--", label="Theoretical BER")
    ax.set(xlabel="Eb/N0 (dB)", ylabel="Bit Error Rate (BER)", title="Gray-coded QPSK BER over AWGN")
    ax.grid(True, which="both"); ax.legend(); fig.tight_layout()
    fig.savefig(output_dir / "ber_vs_ebn0.png", dpi=150); plt.close(fig)


def main() -> None:
    """Run the reproducible example and save figures beside this script."""
    ebn0_values = np.array([0, 2, 4, 6, 8, 10], dtype=float)
    num_bits = 200_000
    bits = generate_random_bits(num_bits, seed=7)
    symbols = qpsk_modulate(bits)
    display_ebn0_db = 6.0
    received = awgn_channel(symbols, display_ebn0_db, np.random.default_rng(8))
    recovered = qpsk_demodulate(received)
    total, errors, ber = calculate_ber(bits, recovered)
    sweep_db, simulated_ber = run_ebn0_sweep(ebn0_values, num_bits=num_bits, seed=7)
    output_dir = Path(__file__).resolve().parent / "results"
    _save_figures(symbols, received, display_ebn0_db, sweep_db, simulated_ber, output_dir)
    print(f"Single run at {display_ebn0_db:g} dB: {errors}/{total} errors, BER={ber:.6g}")
    print("Eb/N0 (dB) | Simulated BER | Theoretical BER")
    for snr, measured in zip(sweep_db, simulated_ber):
        print(f"{snr:9.1f} | {measured:13.6g} | {theoretical_qpsk_ber(snr):14.6g}")
    print(f"Figures saved to {output_dir}")
    print("Finite runs may observe zero errors at high Eb/N0 although theoretical BER is nonzero.")


if __name__ == "__main__":
    main()
