# Project Overview

## Research problem

In-band full-duplex radios transmit and receive simultaneously in the same band. Local transmitter leakage can overwhelm a remote desired signal at the receiver. This project studies a known local baseband reference and estimates the associated self-interference (SI) so it can be subtracted from the received observation.

## Signal model

The receiver model is

\[
y[n] = d[n] + s_{SI}[n] + w[n],
\]

where `d[n]` is the desired signal, `s_SI[n]` is SI, and `w[n]` is complex AWGN. The local transmit signal `x[n]` is available to the adaptive canceller as a reference. For a linear FIR SI channel, `s_SI[n] = sum_k h[k]x[n-k]`.

## Simulation architecture

`qpsk_communication.py` generates seeded bits, maps them to unit-energy Gray-coded QPSK symbols, adds AWGN, demodulates, measures BER, and compares simulated/theoretical curves. `ibfd_simulation.py` combines separate local and remote streams through flat desired and configurable SI channels. `adaptive_si_cancellation.py` estimates SI with complex LMS or NLMS FIR filters. `robustness_experiments.py` orchestrates paired multi-seed sweeps using those existing modules.

The adaptive output uses `s_hat[n] = wᴴx_n`, with reference vector ordered from the current sample backwards. LMS updates weights by `mu*x_n*conj(e[n])`; NLMS normalizes this by `epsilon + ||x_n||²`. Algorithms receive only the reference and total received signal. The known simulated SI is used only after adaptation for offline evaluation.

## Implemented algorithms and stages

- **Stage 1:** QPSK over ideal complex AWGN with BER and theoretical BER comparison.
- **Stage 2:** IBFD signal composition, flat/multipath SI channels, optional simplified cubic SI, and desired-referenced noise SNR.
- **Stage 3:** Complex LMS/NLMS, adaptive FIR, frozen held-out evaluation, cancellation and BER metrics.
- **Stage 3.5:** SI/desired ratio, SNR, channel length/type, and filter length sweeps over three deterministic seeds, with separate train/validation/test partitions and sample mean/standard deviation summaries.

## Experimental methodology and key findings

Stage 3.5 uses 16,000 bits (8,000 symbols) per condition with seeds 11, 22, and 33. Channel, SI level, filter length, and noise are varied one at a time. LMS and NLMS share the exact reference and observation in each trial and weights are frozen after the training subset. SI/desired power is measured from average noiseless component powers over the finite generated record. Noise SNR is desired component power divided by complex noise power, not Eb/N0.

At 0 dB SI/desired, 20 dB desired-referenced SNR, a four-tap SI channel, and eight adaptive taps, mean test cancellation was 5.455 ± 1.347 dB for LMS and 4.521 ± 0.978 dB for NLMS. Test BER changed from 0.1900 ± 0.0101 before subtraction to 0.03167 ± 0.01566 with LMS and 0.04781 ± 0.01660 with NLMS. These are results for this simulated setup, not physical measurements or a general algorithm ranking. The full per-condition dataset and summary are stored in `results/stage3_5/`.

## Current limitations

The implementation is discrete-time symbol-rate baseband, assumes ideal timing and carrier synchronization, and uses static illustrative channel coefficients. It has no pulse shaping, RF impairments, measured IQ, SDR integration, or hardware validation. LMS/NLMS are linear filters and do not provide general nonlinear cancellation. Three random seeds are useful for reproducibility but insufficient to establish broad statistical confidence.

## Future milestones

1. Expand channel/noise characterization and determine conventional filter settings using training and validation data only.
2. Develop a deep-learning SI estimator and compare it to LMS/NLMS on identical held-out signals and metrics.
3. Validate with measured SDR/USRP complex IQ and integrate through a GNU Radio or equivalent real-time flowgraph.

These milestones are planned, not implemented in this repository.
