"""Complex LMS/NLMS self-interference cancellation baseline (Stage 3).

The adaptive filters receive only the local transmit reference and the total
received observation. Ground-truth SI is used only after adaptation for
offline held-out evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from ibfd_simulation import IBFDResult, simulate_ibfd
from qpsk_communication import calculate_ber, generate_random_bits, qpsk_demodulate


Algorithm = Literal["lms", "nlms"]


@dataclass(frozen=True)
class AdaptiveFilterResult:
    """Adaptive filter outputs; weights_history[k] is the weight vector
    after processing k samples (history row zero is the zero initialization).
    """

    estimate: np.ndarray
    residual: np.ndarray
    final_weights: np.ndarray
    weights_history: np.ndarray
    instantaneous_error_power: np.ndarray
    smoothed_error_power: np.ndarray


def adaptive_cancel(
    reference: np.ndarray,
    observation: np.ndarray,
    *,
    algorithm: Algorithm,
    filter_length: int,
    step_size: float,
    training_samples: int,
    epsilon: float = 1e-8,
    freeze_after_training: bool = True,
    smoothing_alpha: float = 0.01,
) -> AdaptiveFilterResult:
    """Estimate and subtract SI using complex FIR LMS or NLMS.

    The reference vector is ``[x[n], x[n-1], ..., x[n-L+1]]`` with zero
    history before n=0. Estimate convention is ``w.conj().T @ x_n`` and the
    complex LMS update is ``w += mu * x_n * conj(e)``. The update occurs after
    producing the estimate for the current sample. With frozen evaluation,
    updates stop at ``training_samples``; evaluation samples do not update w.

    No ground-truth SI argument is accepted: adaptation can depend only on
    reference and observation. The observation error includes desired signal
    and noise disturbances and must not be called pure residual SI power.
    """
    x = _as_complex_vector(reference, "reference")
    y = _as_complex_vector(observation, "observation")
    if x.size != y.size:
        raise ValueError("reference and observation lengths must match")
    if algorithm not in ("lms", "nlms"):
        raise ValueError("algorithm must be 'lms' or 'nlms'")
    if not isinstance(filter_length, (int, np.integer)) or filter_length <= 0:
        raise ValueError("filter_length must be a positive integer")
    if not 0 <= training_samples <= x.size:
        raise ValueError("training_samples must be between zero and input length")
    if not np.isfinite(step_size) or step_size <= 0:
        raise ValueError("step_size must be finite and positive")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not np.isfinite(smoothing_alpha) or not 0 < smoothing_alpha <= 1:
        raise ValueError("smoothing_alpha must be in (0, 1]")

    length = int(filter_length)
    weights = np.zeros(length, dtype=np.complex128)
    history = np.zeros((x.size + 1, length), dtype=np.complex128)
    estimate = np.zeros(x.size, dtype=np.complex128)
    error_power = np.zeros(x.size, dtype=float)
    smoothed = np.zeros(x.size, dtype=float)

    for n in range(x.size):
        # Right-aligned buffer gives the current sample followed by its past.
        reference_vector = np.zeros(length, dtype=np.complex128)
        available = min(length, n + 1)
        reference_vector[:available] = x[n - available + 1:n + 1][::-1]
        y_hat = np.vdot(weights, reference_vector)  # w^H x_n
        error = y[n] - y_hat
        estimate[n] = y_hat
        error_power[n] = float(np.abs(error) ** 2)
        smoothed[n] = (error_power[n] if n == 0 else
                       smoothing_alpha * error_power[n] +
                       (1.0 - smoothing_alpha) * smoothed[n - 1])

        may_update = not freeze_after_training or n < training_samples
        if may_update:
            gain = step_size
            if algorithm == "nlms":
                gain /= epsilon + float(np.vdot(reference_vector, reference_vector).real)
            weights = weights + gain * reference_vector * np.conjugate(error)
        history[n + 1] = weights

    return AdaptiveFilterResult(
        estimate=estimate,
        residual=y - estimate,
        final_weights=weights.copy(),
        weights_history=history,
        instantaneous_error_power=error_power,
        smoothed_error_power=smoothed,
    )


def _as_complex_vector(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.complex128)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite one-dimensional complex array")
    return result


def _power(values: np.ndarray) -> float:
    values = np.asarray(values)
    return float(np.mean(np.abs(values) ** 2)) if values.size else 0.0


def _ratio_db(numerator: float, denominator: float) -> float:
    """Safe 10log10 ratio: NaN for 0/0 and +/-inf for nonzero/zero cases."""
    if numerator < 0 or denominator < 0:
        raise ValueError("powers must be non-negative")
    if numerator == 0 and denominator == 0:
        return float("nan")
    if denominator == 0:
        return float("inf")
    if numerator == 0:
        return float("-inf")
    return float(10.0 * np.log10(numerator / denominator))


@dataclass(frozen=True)
class MethodMetrics:
    name: str
    cancellation_db: float
    residual_si_power: float
    ber_before: float
    ber_after: float
    adaptive: AdaptiveFilterResult | None


@dataclass(frozen=True)
class CancellationExperiment:
    """Simulation and held-out results for no cancellation, LMS, and NLMS."""

    ibfd: IBFDResult
    desired_bits: np.ndarray
    training_samples: int
    evaluation_slice: slice
    methods: dict[str, MethodMetrics]
    lms: AdaptiveFilterResult
    nlms: AdaptiveFilterResult
    lms_si_error_power: np.ndarray
    nlms_si_error_power: np.ndarray


def run_cancellation_experiment(
    *,
    num_bits: int = 40_000,
    seed: int = 21,
    training_fraction: float = 0.7,
    filter_length: int = 8,
    lms_step_size: float = 0.06,
    nlms_step_size: float = 0.6,
    nlms_epsilon: float = 1e-8,
    smoothing_alpha: float = 0.01,
    freeze_after_training: bool = True,
    si_mode: str = "multipath",
    si_gain: complex = 0.1 + 0j,
    si_taps: np.ndarray | None = None,
    h1: complex = 0.1 + 0j,
    h3: complex = 0.0 + 0j,
    noise_snr_db: float = 25.0,
) -> CancellationExperiment:
    """Run Stage 2, adapt on a prefix, then score only a held-out suffix.

    For the standard evaluation use ``freeze_after_training=True``. The
    optional false mode is provided to inspect continuous adaptation, but its
    suffix is not a frozen-weight generalization evaluation.
    """
    if not np.isfinite(training_fraction) or not 0 < training_fraction < 1:
        raise ValueError("training_fraction must be strictly between 0 and 1")
    simulation = simulate_ibfd(
        num_bits=num_bits,
        seed=seed,
        si_mode=si_mode,
        si_gain=si_gain,
        si_taps=si_taps,
        h1=h1,
        h3=h3,
        noise_snr_db=noise_snr_db,
    )
    num_samples = simulation.received.size
    training_samples = int(num_samples * training_fraction)
    if training_samples <= 0 or training_samples >= num_samples:
        raise ValueError("configuration must leave at least one train and one evaluation sample")
    evaluation = slice(training_samples, num_samples)

    # Reconstruct the deterministic desired source bits used by simulate_ibfd.
    desired_bits = generate_random_bits(num_bits, seed)
    lms = adaptive_cancel(
        simulation.tx_reference, simulation.received, algorithm="lms",
        filter_length=filter_length, step_size=lms_step_size,
        training_samples=training_samples, freeze_after_training=freeze_after_training,
        smoothing_alpha=smoothing_alpha,
    )
    nlms = adaptive_cancel(
        simulation.tx_reference, simulation.received, algorithm="nlms",
        filter_length=filter_length, step_size=nlms_step_size,
        training_samples=training_samples, epsilon=nlms_epsilon,
        freeze_after_training=freeze_after_training,
        smoothing_alpha=smoothing_alpha,
    )

    lms_si_error = simulation.self_interference[evaluation] - lms.estimate[evaluation]
    nlms_si_error = simulation.self_interference[evaluation] - nlms.estimate[evaluation]
    lms_si_trace = np.full(num_samples, np.nan)
    nlms_si_trace = np.full(num_samples, np.nan)
    lms_si_trace[evaluation] = np.abs(lms_si_error) ** 2
    nlms_si_trace[evaluation] = np.abs(nlms_si_error) ** 2

    before_bits = qpsk_demodulate(simulation.received[evaluation])
    lms_bits = qpsk_demodulate(lms.residual[evaluation])
    nlms_bits = qpsk_demodulate(nlms.residual[evaluation])
    bit_start = 2 * training_samples
    expected_bits = desired_bits[bit_start:]
    if expected_bits.size != before_bits.size:
        raise RuntimeError("evaluation bit and symbol alignment mismatch")
    ber_before = calculate_ber(expected_bits, before_bits)[2]
    ber_lms = calculate_ber(expected_bits, lms_bits)[2]
    ber_nlms = calculate_ber(expected_bits, nlms_bits)[2]

    true_eval = simulation.self_interference[evaluation]
    true_power = _power(true_eval)
    methods = {
        "No cancellation": MethodMetrics(
            "No cancellation", 0.0, true_power, ber_before, ber_before, None),
        "LMS": MethodMetrics(
            "LMS", _ratio_db(true_power, _power(lms_si_error)),
            _power(lms_si_error), ber_before, ber_lms, lms),
        "NLMS": MethodMetrics(
            "NLMS", _ratio_db(true_power, _power(nlms_si_error)),
            _power(nlms_si_error), ber_before, ber_nlms, nlms),
    }
    return CancellationExperiment(
        simulation, desired_bits, training_samples, evaluation, methods,
        lms, nlms, lms_si_trace, nlms_si_trace,
    )


def print_metrics(experiment: CancellationExperiment) -> None:
    """Print held-out metrics in a compact table."""
    print(f"Training samples: {experiment.training_samples}; "
          f"held-out samples: {experiment.ibfd.received.size - experiment.training_samples}")
    print(f"{'Method':<18} {'SI cancel (dB)':>16} {'Residual SI power':>19} "
          f"{'BER before':>13} {'BER after':>13}")
    for metrics in experiment.methods.values():
        print(f"{metrics.name:<18} {metrics.cancellation_db:16.4f} "
              f"{metrics.residual_si_power:19.6g} {metrics.ber_before:13.6g} "
              f"{metrics.ber_after:13.6g}")


def save_cancellation_plots(experiment: CancellationExperiment,
                            output_dir: str | Path) -> None:
    """Write the five Stage 3 academic-style figures."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stop = experiment.training_samples
    n = np.arange(stop)

    # This is observation prediction-error power |y - s_hat|^2; d and noise
    # remain in the error and therefore it is not a pure SI-power curve.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(n, np.maximum(experiment.lms.smoothed_error_power[:stop], 1e-15),
                label="LMS smoothed |y - estimate|²")
    ax.semilogy(n, np.maximum(experiment.nlms.smoothed_error_power[:stop], 1e-15),
                label="NLMS smoothed |y - estimate|²")
    ax.axvline(stop - 1, color="black", linestyle=":", label="End of adaptation")
    ax.set(xlabel="Training sample", ylabel="Smoothed observation error power",
           title="Adaptive filter convergence (includes desired signal and noise)")
    ax.grid(True, which="both"); ax.legend(); fig.tight_layout()
    fig.savefig(out / "lms_nlms_convergence.png", dpi=160); plt.close(fig)

    labels = list(experiment.methods)
    cancellation = [experiment.methods[key].cancellation_db for key in labels]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    bars = ax.bar(labels, cancellation, color=["#888888", "#3977a8", "#e28e2c"])
    ax.bar_label(bars, fmt="%.2f dB")
    ax.set(ylabel="Held-out SI cancellation (dB)", title="SI cancellation comparison")
    ax.grid(True, axis="y"); fig.tight_layout()
    fig.savefig(out / "si_cancellation_comparison.png", dpi=160); plt.close(fig)

    before = [experiment.methods[key].ber_before for key in labels]
    after = [experiment.methods[key].ber_after for key in labels]
    xloc = np.arange(len(labels)); width = .36
    floor = 0.5 / (experiment.desired_bits.size - 2 * experiment.training_samples)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(xloc - width / 2, np.maximum(before, floor), width, label="Before")
    ax.bar(xloc + width / 2, np.maximum(after, floor), width, label="After")
    ax.set_yscale("log"); ax.set_xticks(xloc, labels)
    ax.set(ylabel="BER (log scale; zero shown at half-bit resolution)",
           title="Held-out BER before and after cancellation")
    ax.grid(True, axis="y", which="both"); ax.legend(); fig.tight_layout()
    fig.savefig(out / "ber_before_after_cancellation.png", dpi=160); plt.close(fig)

    residual_power = [experiment.methods[key].residual_si_power for key in labels]
    positive = [p for p in residual_power if p > 0]
    power_floor = min(positive) * .5 if positive else 1e-15
    fig, ax = plt.subplots(figsize=(7, 4.8))
    bars = ax.bar(labels, np.maximum(residual_power, power_floor),
                  color=["#888888", "#3977a8", "#e28e2c"])
    ax.set_yscale("log")
    ax.set(ylabel="Held-out residual SI power (linear units)",
           title="Residual SI power comparison")
    ax.grid(True, axis="y", which="both"); fig.tight_layout()
    fig.savefig(out / "residual_si_power.png", dpi=160); plt.close(fig)

    first = experiment.evaluation_slice.start
    count = min(1500, experiment.ibfd.received.size - first)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    series = (
        (experiment.ibfd.received[first:first + count], "Before cancellation"),
        (experiment.lms.residual[first:first + count], "After LMS"),
        (experiment.nlms.residual[first:first + count], "After NLMS"),
    )
    for ax, (values, title) in zip(axes, series):
        ax.scatter(values.real, values.imag, s=9, alpha=.5)
        ax.set(title=title, xlabel="In-phase (I)", ylabel="Quadrature (Q)")
        ax.grid(True); ax.axis("equal")
    fig.suptitle("Held-out received constellation before and after SI cancellation")
    fig.tight_layout(); fig.savefig(out / "ibfd_constellation_after_cancellation.png", dpi=160); plt.close(fig)

    np.savez(out / "stage3_cancellation_components.npz",
             desired=experiment.ibfd.desired,
             tx_reference=experiment.ibfd.tx_reference,
             self_interference=experiment.ibfd.self_interference,
             noise=experiment.ibfd.noise,
             received=experiment.ibfd.received,
             lms_estimate=experiment.lms.estimate,
             nlms_estimate=experiment.nlms.estimate,
             lms_residual=experiment.lms.residual,
             nlms_residual=experiment.nlms.residual,
             desired_bits=experiment.desired_bits,
             training_samples=experiment.training_samples)


def main() -> None:
    """Run a fixed-seed multipath baseline, report metrics, and save plots."""
    experiment = run_cancellation_experiment(
        num_bits=40_000,
        seed=21,
        training_fraction=0.7,
        filter_length=8,
        lms_step_size=0.06,
        nlms_step_size=0.6,
        nlms_epsilon=1e-8,
        si_mode="multipath",
        si_taps=np.array([1.0 + .2j, .22 - .12j, .08 + .04j, -.03 + .02j]),
        noise_snr_db=25.0,
        freeze_after_training=True,
    )
    output = Path(__file__).resolve().parent / "results" / "stage3"
    print_metrics(experiment)
    save_cancellation_plots(experiment, output)
    print(f"Plots and arrays saved to {output}")


if __name__ == "__main__":
    main()
