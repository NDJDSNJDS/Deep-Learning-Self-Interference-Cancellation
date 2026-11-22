"""Stage 2: simple in-band full-duplex (IBFD) complex-baseband model.

The model is symbol-rate and causal. Multipath convolution assumes a zero
initial state and returns the first N samples (same duration as the input).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qpsk_communication import generate_random_bits, qpsk_modulate


def _complex_vector(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.complex128)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return result


def apply_flat_channel(signal: np.ndarray, gain: complex) -> np.ndarray:
    """Apply a flat complex channel gain to a complex-baseband vector."""
    signal = _complex_vector(signal, "signal")
    if not np.isfinite(gain):
        raise ValueError("gain must be finite")
    return np.asarray(gain, dtype=np.complex128) * signal


def apply_multipath_channel(signal: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Apply causal FIR taps with zero state; return N samples for N inputs.

    This is the beginning of the full convolution, so the tail beyond the
    input duration is omitted to keep all IBFD paths sample-aligned.
    """
    signal = _complex_vector(signal, "signal")
    taps = _complex_vector(taps, "taps")
    if taps.size == 0:
        raise ValueError("taps must not be empty")
    if signal.size == 0:
        return np.empty(0, dtype=np.complex128)
    return np.convolve(signal, taps, mode="full")[:signal.size].astype(np.complex128)


def apply_nonlinear_si(signal: np.ndarray, h1: complex, h3: complex) -> np.ndarray:
    """Memoryless cubic SI model: h1*x + h3*x*|x|^2."""
    signal = _complex_vector(signal, "signal")
    if not np.isfinite(h1) or not np.isfinite(h3):
        raise ValueError("h1 and h3 must be finite")
    return h1 * signal + h3 * signal * np.abs(signal) ** 2


def generate_awgn_noise(num_samples: int, reference_power: float, snr_db: float,
                        rng: np.random.Generator) -> np.ndarray:
    """Generate complex AWGN for desired-signal-referenced SNR.

    SNR = mean(|d|^2) / E[|w|^2]. Each real component has half the
    resulting complex noise power. This parameter is not Eb/N0.
    """
    if not isinstance(num_samples, (int, np.integer)) or num_samples < 0:
        raise ValueError("num_samples must be a non-negative integer")
    if not np.isfinite(reference_power) or reference_power < 0:
        raise ValueError("reference_power must be finite and non-negative")
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    if reference_power == 0 and num_samples:
        raise ValueError("reference_power must be positive when generating noise")
    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    complex_noise_power = reference_power / snr_linear
    sigma = np.sqrt(complex_noise_power / 2.0)
    return (sigma * (rng.standard_normal(num_samples) +
                     1j * rng.standard_normal(num_samples))).astype(np.complex128)


@dataclass(frozen=True)
class IBFDResult:
    """Aligned complex arrays for one IBFD simulation run."""

    desired: np.ndarray
    tx_reference: np.ndarray
    self_interference: np.ndarray
    noise: np.ndarray
    received: np.ndarray


def simulate_ibfd(
    num_bits: int = 20_000,
    seed: int = 11,
    desired_gain: complex = 1.0 + 0.0j,
    si_mode: str = "flat",
    si_gain: complex = 0.1 + 0.0j,
    si_taps: np.ndarray | None = None,
    h1: complex = 0.1 + 0.0j,
    h3: complex = 0.0 + 0.0j,
    desired_enabled: bool = True,
    si_enabled: bool = True,
    noise_enabled: bool = True,
    noise_snr_db: float = 20.0,
) -> IBFDResult:
    """Construct y=d+s_SI+w using independent remote and local QPSK streams.

    ``noise_snr_db`` defines SNR relative to average power of the desired
    component after ``desired_gain``. When desired is disabled, noise is zero
    (or an error is raised if noise was explicitly requested).
    """
    if num_bits <= 0 or num_bits % 2:
        raise ValueError("num_bits must be a positive even integer")
    if si_mode not in {"flat", "multipath", "nonlinear"}:
        raise ValueError("si_mode must be 'flat', 'multipath', or 'nonlinear'")

    desired_bits = generate_random_bits(num_bits, seed)
    local_bits = generate_random_bits(num_bits, seed + 1)
    remote_symbols = qpsk_modulate(desired_bits)
    tx_reference = qpsk_modulate(local_bits).astype(np.complex128)

    desired = apply_flat_channel(remote_symbols, desired_gain) if desired_enabled else np.zeros_like(remote_symbols)
    if not si_enabled:
        self_interference = np.zeros_like(tx_reference)
    elif si_mode == "flat":
        self_interference = apply_flat_channel(tx_reference, si_gain)
    elif si_mode == "multipath":
        if si_taps is None:
            raise ValueError("si_taps are required for multipath mode")
        self_interference = apply_multipath_channel(tx_reference, si_taps)
    else:
        self_interference = apply_nonlinear_si(tx_reference, h1, h3)

    if noise_enabled:
        if not desired_enabled:
            raise ValueError("noise SNR is defined relative to desired signal power; enable desired path")
        rng = np.random.default_rng(seed + 2)
        noise = generate_awgn_noise(tx_reference.size, float(np.mean(np.abs(desired) ** 2)),
                                    noise_snr_db, rng)
    else:
        noise = np.zeros_like(tx_reference)
    received = (desired + self_interference + noise).astype(np.complex128)
    if not (desired.size == tx_reference.size == self_interference.size == noise.size == received.size):
        raise RuntimeError("IBFD component lengths are inconsistent")
    return IBFDResult(desired, tx_reference, self_interference, noise, received)


def save_ibfd_outputs(result: IBFDResult, output_dir: str | Path,
                      sample_count: int = 80) -> None:
    """Save requested plots and all complex components for later inspection."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "ibfd_components.npz", desired=result.desired,
             tx_reference=result.tx_reference, self_interference=result.self_interference,
             noise=result.noise, received=result.received)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, values, title in zip(
        axes,
        (result.desired, result.self_interference, result.received),
        ("Desired signal d[n]", "Self-interference s_SI[n]", "Total received y[n]"),
    ):
        shown = values[: min(2000, values.size)]
        ax.scatter(shown.real, shown.imag, s=9, alpha=.45)
        ax.set(title=title, xlabel="In-phase (I)", ylabel="Quadrature (Q)")
        ax.axis("equal"); ax.grid(True)
    fig.suptitle("IBFD signal constellations")
    fig.tight_layout(); fig.savefig(output_dir / "ibfd_constellations.png", dpi=150); plt.close(fig)

    count = min(sample_count, result.received.size)
    indices = np.arange(count)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for component, label in ((result.desired, "Desired"),
                             (result.self_interference, "Self-interference"),
                             (result.received, "Total received")):
        axes[0].plot(indices, component[:count].real, ".-", label=label)
        axes[1].plot(indices, component[:count].imag, ".-", label=label)
    axes[0].set_ylabel("Real amplitude"); axes[1].set_ylabel("Imaginary amplitude")
    axes[1].set_xlabel("Symbol index")
    for ax in axes:
        ax.grid(True); ax.legend()
    fig.suptitle("IBFD desired, self-interference, and received samples")
    fig.tight_layout(); fig.savefig(output_dir / "ibfd_iq_samples.png", dpi=150); plt.close(fig)

    powers = [float(np.mean(np.abs(x) ** 2)) for x in
              (result.desired, result.self_interference, result.noise)]
    labels = ["Desired", "Self-interference", "Noise"]
    reference = powers[0]
    if reference <= 0:
        raise ValueError("desired reference power must be positive for relative-power plot")
    relative_db = [10.0 * np.log10(max(power, np.finfo(float).tiny) / reference)
                   for power in powers]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, relative_db)
    ax.bar_label(bars, fmt="%.1f dB")
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(ylabel="Average component power relative to desired (dB)",
           title="Relative component powers (desired power = 0 dB)")
    ax.grid(True, axis="y"); fig.tight_layout()
    fig.savefig(output_dir / "ibfd_relative_powers.png", dpi=150); plt.close(fig)


def main() -> None:
    """Run default illustrative configuration and save outputs in results/."""
    result = simulate_ibfd(
        num_bits=20_000,
        seed=11,
        desired_gain=1.0 + 0.0j,
        si_mode="multipath",
        si_taps=np.array([0.12 + 0.04j, 0.04 - 0.02j, -0.015 + 0.01j]),
        noise_snr_db=20.0,
    )
    output_dir = Path(__file__).resolve().parent / "results" / "stage2"
    save_ibfd_outputs(result, output_dir)
    print(f"Saved IBFD plots and component arrays to {output_dir}")
    print(f"Average powers: desired={np.mean(np.abs(result.desired)**2):.6g}, "
          f"SI={np.mean(np.abs(result.self_interference)**2):.6g}, "
          f"noise={np.mean(np.abs(result.noise)**2):.6g}")


if __name__ == "__main__":
    main()
