"""Stage 3.5 paired robustness sweeps for the LMS/NLMS SI baseline.

This module orchestrates existing Stage 2 and Stage 3 APIs; it does not
replace either simulation or adaptive-filter implementation.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from adaptive_si_cancellation import _ratio_db, adaptive_cancel
from ibfd_simulation import (
    apply_flat_channel,
    apply_multipath_channel,
    simulate_ibfd,
)
from qpsk_communication import calculate_ber, generate_random_bits, qpsk_demodulate, qpsk_modulate


SI_RATIOS_DB = (-20.0, 0.0, 20.0, 40.0)
NOISE_SNRS_DB = (0.0, 10.0, 20.0, 30.0)
CHANNELS = ("flat", "multipath_2", "multipath_4", "multipath_8", "multipath_16")
FILTER_LENGTHS = (2, 4, 8, 16)
DEFAULT_SEEDS = (11, 22, 33)


def _base_channel(channel: str) -> tuple[str, complex | np.ndarray]:
    if channel == "flat":
        return "flat", 1.0 + 0.2j
    if not channel.startswith("multipath_"):
        raise ValueError(f"unknown channel configuration: {channel}")
    try:
        length = int(channel.split("_", 1)[1])
    except ValueError as exc:
        raise ValueError(f"invalid multipath channel name: {channel}") from exc
    if length <= 0:
        raise ValueError("multipath channel length must be positive")
    tap_indices = np.arange(length)
    # Fixed decaying complex tap shape; each SI-ratio condition scales all taps
    # by one real factor and therefore preserves these relative tap values.
    taps = 0.65 ** tap_indices * np.exp(1j * 0.43 * tap_indices)
    return "multipath", taps.astype(np.complex128)


def scale_si_channel_to_ratio(
    tx_reference: np.ndarray,
    desired: np.ndarray,
    channel: str,
    si_to_desired_db: float,
) -> tuple[complex | np.ndarray, float]:
    """Scale a flat/FIR channel to target whole-record SI/desired power ratio.

    Ratio definition: ``10 log10(mean(|s_SI|^2)/mean(|d|^2))`` over all
    generated samples before noise. For multipath, every base FIR tap is
    multiplied by the same positive real scale, preserving the channel shape.
    Returns the channel parameter passed to Stage 2 and the measured ratio.
    """
    x = np.asarray(tx_reference, dtype=np.complex128)
    d = np.asarray(desired, dtype=np.complex128)
    if x.ndim != 1 or d.ndim != 1 or x.size != d.size or not x.size:
        raise ValueError("tx_reference and desired must be non-empty equal-length vectors")
    if not np.isfinite(si_to_desired_db):
        raise ValueError("si_to_desired_db must be finite")
    mode, base = _base_channel(channel)
    desired_power = float(np.mean(np.abs(d) ** 2))
    if desired_power <= 0:
        raise ValueError("desired signal must have positive average power")
    base_si = apply_flat_channel(x, base) if mode == "flat" else apply_multipath_channel(x, base)
    base_power = float(np.mean(np.abs(base_si) ** 2))
    if base_power <= 0:
        raise ValueError("base SI channel must have positive output power")
    target_power = desired_power * 10.0 ** (si_to_desired_db / 10.0)
    scale = np.sqrt(target_power / base_power)
    channel_parameter = base * scale
    achieved_si = (apply_flat_channel(x, channel_parameter) if mode == "flat" else
                   apply_multipath_channel(x, channel_parameter))
    achieved_db = _ratio_db(float(np.mean(np.abs(achieved_si) ** 2)), desired_power)
    return channel_parameter, achieved_db


def summarize_trials(trials: list[dict[str, Any]], group_keys: tuple[str, ...]
                     ) -> list[dict[str, Any]]:
    """Aggregate trial metrics with sample standard deviation across seeds."""
    metrics = ("cancellation_db", "ber_before", "ber_after", "residual_si_power")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in trials:
        key = tuple(row[name] for name in (*group_keys, "algorithm"))
        groups.setdefault(key, []).append(row)
    summary: list[dict[str, Any]] = []
    for key, rows in groups.items():
        item = dict(zip((*group_keys, "algorithm"), key))
        item["seed_count"] = len(rows)
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in rows], dtype=float)
            item[f"mean_{metric}"] = float(np.mean(values))
            item[f"std_{metric}"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        summary.append(item)
    return summary


def run_one_configuration(
    *,
    num_bits: int,
    seed: int,
    channel: str,
    si_to_desired_db: float,
    noise_snr_db: float,
    filter_length: int,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    lms_step_size: float = 0.06,
    nlms_step_size: float = 0.6,
    nlms_epsilon: float = 1e-8,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Run one paired LMS/NLMS condition and score validation/test separately.

    Split boundaries are fixed before simulation. Both methods receive exactly
    the same reference and observation. The adaptive calls receive no true SI.
    """
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("training and validation must leave a non-empty test split")

    bits = generate_random_bits(num_bits, seed)
    tx_ref = qpsk_modulate(generate_random_bits(num_bits, seed + 1))
    desired = qpsk_modulate(bits).astype(np.complex128)
    mode, _ = _base_channel(channel)
    channel_parameter, achieved_ratio_db = scale_si_channel_to_ratio(
        tx_ref, desired, channel, si_to_desired_db)
    _, base_parameter = _base_channel(channel)
    si_channel_scale = (float(abs(channel_parameter) / abs(base_parameter))
                        if mode == "flat" else
                        float(abs(channel_parameter[0] / base_parameter[0])))
    common = dict(num_bits=num_bits, seed=seed, si_mode=mode, noise_snr_db=noise_snr_db)
    if mode == "flat":
        simulation = simulate_ibfd(**common, si_gain=channel_parameter)
    else:
        simulation = simulate_ibfd(**common, si_taps=channel_parameter)

    samples = simulation.received.size
    train_stop = int(samples * train_fraction)
    validation_stop = int(samples * (train_fraction + validation_fraction))
    if train_stop <= 0 or validation_stop <= train_stop or validation_stop >= samples:
        raise ValueError("sample count is too small for non-empty train/validation/test splits")
    validation = slice(train_stop, validation_stop)
    test = slice(validation_stop, samples)

    lms = adaptive_cancel(
        simulation.tx_reference, simulation.received, algorithm="lms",
        filter_length=filter_length, step_size=lms_step_size,
        training_samples=train_stop, freeze_after_training=True,
    )
    nlms = adaptive_cancel(
        simulation.tx_reference, simulation.received, algorithm="nlms",
        filter_length=filter_length, step_size=nlms_step_size,
        training_samples=train_stop, epsilon=nlms_epsilon,
        freeze_after_training=True,
    )

    def evaluate(part: slice, filt: Any | None) -> dict[str, float]:
        truth = simulation.self_interference[part]
        estimate = np.zeros_like(truth) if filt is None else filt.estimate[part]
        residual_si = truth - estimate
        si_power = float(np.mean(np.abs(truth) ** 2))
        residual_power = float(np.mean(np.abs(residual_si) ** 2))
        before = qpsk_demodulate(simulation.received[part])
        after = before if filt is None else qpsk_demodulate(filt.residual[part])
        expected = bits[2 * part.start:2 * part.stop]
        if expected.size != before.size:
            raise RuntimeError("bit/symbol split alignment is invalid")
        return {
            "cancellation_db": 0.0 if filt is None else _ratio_db(si_power, residual_power),
            "residual_si_power": si_power if filt is None else residual_power,
            "ber_before": calculate_ber(expected, before)[2],
            "ber_after": calculate_ber(expected, after)[2],
        }

    validation_metrics = {name: evaluate(validation, method)
                          for name, method in (("none", None), ("lms", lms), ("nlms", nlms))}
    test_metrics = {name: evaluate(test, method)
                    for name, method in (("none", None), ("lms", lms), ("nlms", nlms))}

    row: dict[str, Any] = {
        "seed": seed,
        "channel": channel,
        "channel_length": 1 if channel == "flat" else int(channel.rsplit("_", 1)[1]),
        "si_to_desired_db_target": float(si_to_desired_db),
        "si_to_desired_db_measured": achieved_ratio_db,
        "noise_snr_db": float(noise_snr_db),
        "filter_length": int(filter_length),
        "lms_step_size": float(lms_step_size),
        "nlms_step_size": float(nlms_step_size),
        "nlms_epsilon": float(nlms_epsilon),
        "train_fraction": float(train_fraction),
        "validation_fraction": float(validation_fraction),
        "si_channel_scale": si_channel_scale,
        "num_bits": int(num_bits),
        "num_symbols": samples,
        "training_samples": train_stop,
        "validation_start": train_stop,
        "validation_stop": validation_stop,
        "test_start": validation_stop,
        "test_stop": samples,
        "validation_si_cancel_lms_db": validation_metrics["lms"]["cancellation_db"],
        "validation_si_cancel_nlms_db": validation_metrics["nlms"]["cancellation_db"],
        "validation_ber_before": validation_metrics["none"]["ber_before"],
        "validation_ber_lms": validation_metrics["lms"]["ber_after"],
        "validation_ber_nlms": validation_metrics["nlms"]["ber_after"],
    }
    for name, metric_key, algorithm in (("no_cancellation", "none", "No cancellation"),
                                        ("lms", "lms", "LMS"),
                                        ("nlms", "nlms", "NLMS")):
        metrics = test_metrics[metric_key]
        row.update({
            f"{name}_algorithm": algorithm,
            f"{name}_cancellation_db": metrics["cancellation_db"],
            f"{name}_residual_si_power": metrics["residual_si_power"],
            f"{name}_ber_before": metrics["ber_before"],
            f"{name}_ber_after": metrics["ber_after"],
        })
    # Return training curves only: each has the same train length for this study.
    return row, lms.smoothed_error_power[:train_stop], nlms.smoothed_error_power[:train_stop]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _algorithm_rows(row: dict[str, Any], sweep: str, factor: str) -> list[dict[str, Any]]:
    result = []
    for prefix, algorithm in (("no_cancellation", "No cancellation"),
                              ("lms", "LMS"), ("nlms", "NLMS")):
        result.append({
            "sweep": sweep,
            "factor": factor,
            "seed": row["seed"],
            "algorithm": algorithm,
            "cancellation_db": row[f"{prefix}_cancellation_db"],
            "residual_si_power": row[f"{prefix}_residual_si_power"],
            "ber_before": row[f"{prefix}_ber_before"],
            "ber_after": row[f"{prefix}_ber_after"],
            "channel": row["channel"],
            "channel_length": row["channel_length"],
            "filter_length": row["filter_length"],
            "si_to_desired_db_target": row["si_to_desired_db_target"],
            "noise_snr_db": row["noise_snr_db"],
        })
    return result


def _write_plots(summary: list[dict[str, Any]], convergence: list[dict[str, Any]],
                 filter_convergence: list[dict[str, Any]], out: Path,
                 ber_floor: float) -> None:
    def factor_rows(sweep: str) -> list[dict[str, Any]]:
        return [item for item in summary if item["sweep"] == sweep]

    def lineplot(sweep: str, title: str, xlabel: str, filename: str,
                 metric: str, algorithms: tuple[str, ...] = ("LMS", "NLMS"),
                 log_y: bool = False) -> None:
        rows = factor_rows(sweep)
        factors = list(dict.fromkeys(item["factor"] for item in rows))
        fig, ax = plt.subplots(figsize=(8, 5))
        for alg in algorithms:
            vals = [next(item for item in rows if item["factor"] == factor and
                         item["algorithm"] == alg) for factor in factors]
            means = [item[f"mean_{metric}"] for item in vals]
            stds = [item[f"std_{metric}"] for item in vals]
            x = np.arange(len(factors))
            ax.errorbar(x, means, yerr=stds, marker="o", capsize=3, label=alg)
        ax.set_xticks(np.arange(len(factors)), factors)
        ylabel = ("Mean held-out SI cancellation (dB)" if metric == "cancellation_db"
                  else "Mean held-out BER")
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, which="both"); ax.legend(); fig.tight_layout()
        fig.savefig(out / filename, dpi=160); plt.close(fig)

    lineplot("si_ratio", "Cancellation versus SI-to-desired power ratio",
             "SI / desired power ratio (dB)", "si_cancellation_vs_si_ratio.png",
             "cancellation_db")
    # BER plot includes uncancelled baseline as well as both adaptive methods.
    rows = factor_rows("si_ratio")
    factors = list(dict.fromkeys(item["factor"] for item in rows))
    fig, ax = plt.subplots(figsize=(8, 5))
    for alg, style in (("No cancellation", "o-"), ("LMS", "s-"), ("NLMS", "^-")):
        vals = [next(item for item in rows if item["factor"] == f and item["algorithm"] == alg)
                for f in factors]
        raw_means = np.asarray([item["mean_ber_after"] for item in vals])
        means = np.maximum(raw_means, ber_floor)
        stds = np.asarray([item["std_ber_after"] for item in vals])
        lower = np.minimum(stds, np.maximum(raw_means - ber_floor, 0.0))
        ax.errorbar(np.arange(len(factors)), means, yerr=np.vstack((lower, stds)),
                    fmt=style, capsize=3, label=alg)
    ax.set_xticks(np.arange(len(factors)), factors); ax.set_yscale("log")
    ax.set(xlabel="SI / desired power ratio (dB)", ylabel="Mean held-out BER",
           title="BER versus SI-to-desired power ratio")
    ax.grid(True, which="both"); ax.legend(); fig.tight_layout()
    fig.savefig(out / "ber_vs_si_ratio.png", dpi=160); plt.close(fig)

    lineplot("channel", "LMS and NLMS across SI channel configurations",
             "SI channel", "lms_nlms_by_channel.png", "cancellation_db")
    lineplot("filter_length", "Cancellation versus adaptive FIR length",
             "Adaptive filter length (taps)", "cancellation_vs_filter_length.png",
             "cancellation_db")
    lineplot("snr", "Cancellation versus desired-signal-referenced SNR",
             "Desired signal / noise SNR (dB)", "cancellation_vs_snr.png",
             "cancellation_db")

    fig, ax = plt.subplots(figsize=(9, 5))
    samples = [row["training_sample"] for row in convergence]
    ax.semilogy(samples, np.maximum([row["lms_mean_error_power"] for row in convergence], 1e-15),
                label="LMS mean smoothed |y - estimate|²")
    ax.semilogy(samples, np.maximum([row["nlms_mean_error_power"] for row in convergence], 1e-15),
                label="NLMS mean smoothed |y - estimate|²")
    ax.set(xlabel="Training sample", ylabel="Mean smoothed observation error power",
           title="Mean convergence over deterministic seeds (includes desired signal and noise)")
    ax.grid(True, which="both"); ax.legend(); fig.tight_layout()
    fig.savefig(out / "mean_convergence.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for length in FILTER_LENGTHS:
        curves = [row for row in filter_convergence if row["filter_length"] == length]
        samples = [row["training_sample"] for row in curves]
        axes[0].semilogy(samples,
                         np.maximum([row["lms_mean_error_power"] for row in curves], 1e-15),
                         label=f"{length} taps")
        axes[1].semilogy(samples,
                         np.maximum([row["nlms_mean_error_power"] for row in curves], 1e-15),
                         label=f"{length} taps")
    axes[0].set_ylabel("Mean smoothed |y - estimate|²")
    axes[0].set_title("LMS training convergence by adaptive filter length")
    axes[1].set_ylabel("Mean smoothed |y - estimate|²")
    axes[1].set_xlabel("Training sample")
    axes[1].set_title("NLMS training convergence by adaptive filter length")
    for ax in axes:
        ax.grid(True, which="both")
        ax.legend(title="Filter length")
    fig.suptitle("Curves average the same three deterministic seeds")
    fig.tight_layout(); fig.savefig(out / "convergence_by_filter_length.png", dpi=160); plt.close(fig)


def run_robustness_study(
    output_dir: str | Path,
    *,
    num_bits: int = 16_000,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one-factor-at-a-time sweeps and save raw/summary CSV, NPZ, and plots.

    All conditions use the same symbol count, 60/20/20 data split, desired
    SNR definition, step sizes, and deterministic seed schedule. The tuning
    parameters are fixed a priori; validation is recorded for future tuning,
    while all headline aggregation and plots use the untouched test suffix.
    """
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty sequence of distinct integers")
    if num_bits <= 0 or num_bits % 2:
        raise ValueError("num_bits must be a positive even integer")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    conditions: list[tuple[str, str, str, float, float, int]] = []
    conditions.extend(("si_ratio", f"{ratio:g}", "multipath_4", ratio, 20.0, 8)
                      for ratio in SI_RATIOS_DB)
    conditions.extend(("snr", f"{snr:g}", "multipath_4", 0.0, snr, 8)
                      for snr in NOISE_SNRS_DB)
    conditions.extend(("channel", name, name, 0.0, 20.0, 8) for name in CHANNELS)
    conditions.extend(("filter_length", str(length), "multipath_8", 0.0, 20.0, length)
                      for length in FILTER_LENGTHS)

    cache: dict[tuple[Any, ...], tuple[dict[str, Any], np.ndarray, np.ndarray]] = {}
    raw_trials: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    convergence_by_seed: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    convergence_by_filter: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for sweep, factor, channel, sir_db, snr_db, filter_length in conditions:
        for seed in seeds:
            key = (num_bits, seed, channel, float(sir_db), float(snr_db), int(filter_length))
            if key not in cache:
                cache[key] = run_one_configuration(
                    num_bits=num_bits, seed=seed, channel=channel,
                    si_to_desired_db=sir_db, noise_snr_db=snr_db,
                    filter_length=filter_length,
                )
            row, lms_curve, nlms_curve = cache[key]
            raw_trials.append({"sweep": sweep, "factor": factor, **row})
            paired_rows.extend(_algorithm_rows(row, sweep, factor))
            if (channel == "multipath_4" and sir_db == 0.0 and snr_db == 20.0
                    and filter_length == 8):
                convergence_by_seed[seed] = (lms_curve, nlms_curve)
            if sweep == "filter_length":
                convergence_by_filter.setdefault(filter_length, {})[seed] = (lms_curve, nlms_curve)

    summary = summarize_trials(paired_rows, ("sweep", "factor"))
    _write_csv(output / "raw_trials.csv", raw_trials)
    _write_csv(output / "summary.csv", summary)

    if not convergence_by_seed:
        raise RuntimeError("baseline convergence condition was not included in the study")
    ordered = [convergence_by_seed[seed] for seed in seeds]
    lms_mean = np.mean(np.stack([item[0] for item in ordered]), axis=0)
    nlms_mean = np.mean(np.stack([item[1] for item in ordered]), axis=0)
    convergence = [{"training_sample": index + 1,
                    "lms_mean_error_power": float(lms_mean[index]),
                    "nlms_mean_error_power": float(nlms_mean[index])}
                   for index in range(lms_mean.size)]
    _write_csv(output / "mean_convergence.csv", convergence)
    filter_convergence: list[dict[str, Any]] = []
    for length in FILTER_LENGTHS:
        curves = [convergence_by_filter[length][seed] for seed in seeds]
        mean_lms = np.mean(np.stack([item[0] for item in curves]), axis=0)
        mean_nlms = np.mean(np.stack([item[1] for item in curves]), axis=0)
        filter_convergence.extend({
            "filter_length": length,
            "training_sample": index + 1,
            "lms_mean_error_power": float(mean_lms[index]),
            "nlms_mean_error_power": float(mean_nlms[index]),
        } for index in range(mean_lms.size))
    _write_csv(output / "filter_length_convergence.csv", filter_convergence)
    np.savez_compressed(
        output / "raw_results.npz",
        seeds=np.asarray(seeds, dtype=int),
        num_bits=np.asarray(num_bits, dtype=int),
        convergence_sample=np.asarray([row["training_sample"] for row in convergence]),
        lms_mean_convergence=np.asarray([row["lms_mean_error_power"] for row in convergence]),
        nlms_mean_convergence=np.asarray([row["nlms_mean_error_power"] for row in convergence]),
        filter_convergence_length=np.asarray([row["filter_length"] for row in filter_convergence], dtype=int),
        filter_convergence_sample=np.asarray([row["training_sample"] for row in filter_convergence], dtype=int),
        filter_lms_mean_convergence=np.asarray([row["lms_mean_error_power"] for row in filter_convergence]),
        filter_nlms_mean_convergence=np.asarray([row["nlms_mean_error_power"] for row in filter_convergence]),
        raw_seed=np.asarray([row["seed"] for row in raw_trials], dtype=int),
        raw_sweep=np.asarray([row["sweep"] for row in raw_trials]),
        raw_factor=np.asarray([row["factor"] for row in raw_trials]),
        raw_channel=np.asarray([row["channel"] for row in raw_trials]),
        raw_si_ratio_db=np.asarray([row["si_to_desired_db_measured"] for row in raw_trials]),
        raw_snr_db=np.asarray([row["noise_snr_db"] for row in raw_trials]),
        raw_filter_length=np.asarray([row["filter_length"] for row in raw_trials], dtype=int),
        raw_lms_cancellation_db=np.asarray([row["lms_cancellation_db"] for row in raw_trials]),
        raw_nlms_cancellation_db=np.asarray([row["nlms_cancellation_db"] for row in raw_trials]),
        raw_lms_residual_si_power=np.asarray([row["lms_residual_si_power"] for row in raw_trials]),
        raw_nlms_residual_si_power=np.asarray([row["nlms_residual_si_power"] for row in raw_trials]),
        raw_ber_before=np.asarray([row["lms_ber_before"] for row in raw_trials]),
        raw_lms_ber_after=np.asarray([row["lms_ber_after"] for row in raw_trials]),
        raw_nlms_ber_after=np.asarray([row["nlms_ber_after"] for row in raw_trials]),
    )
    ber_floor = 0.5 / (num_bits * 0.2)
    _write_plots(summary, convergence, filter_convergence, output, ber_floor)
    return raw_trials, summary


def print_summary(summary: list[dict[str, Any]]) -> None:
    """Print test-set mean and sample standard deviation for key sweeps."""
    print("Held-out test metrics (mean +/- sample SD across seeds)")
    print(f"{'Sweep':<14} {'Factor':<14} {'Algorithm':<15} {'Cancel dB':>18} "
          f"{'Residual SI P':>15} {'BER before':>20} {'BER after':>20}")
    for row in summary:
        print(f"{row['sweep']:<14} {str(row['factor']):<14} {row['algorithm']:<15} "
              f"{row['mean_cancellation_db']:7.3f} +/- {row['std_cancellation_db']:<6.3f} "
              f"{row['mean_residual_si_power']:15.5g} "
              f"{row['mean_ber_before']:7.5f} +/- {row['std_ber_before']:<7.5f} "
              f"{row['mean_ber_after']:7.5f} +/- {row['std_ber_after']:.5f}")


def main() -> None:
    out = Path(__file__).resolve().parent / "results" / "stage3_5"
    raw, summary = run_robustness_study(out)
    print(f"Completed {len(raw)} scenario-seed rows across {len(DEFAULT_SEEDS)} seeds.")
    print_summary(summary)
    print(f"CSV, NPZ, and figures saved under {out}")


if __name__ == "__main__":
    main()
