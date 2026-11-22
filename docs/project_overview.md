# Project Overview

## Research problem

In-band full-duplex (IBFD) radios transmit and receive simultaneously within the same frequency band. Because the local transmitter and receiver are co-located, local transmitter leakage—referred to as self-interference (SI)—can overwhelm a weak desired signal arriving from a remote node. This project uses a known local baseband transmit reference to estimate the self-interference channel and subtract the reconstructed interference from the received signal.

## Signal model

At discrete symbol index $n$, the complex baseband received signal $y[n] \in \mathbb{C}$ is modeled as:

$$y[n] = d[n] + s_{\text{SI}}[n] + w[n]$$

where:
* $d[n] \in \mathbb{C}$ is the desired remote signal passing through a flat-fading channel,
* $s_{\text{SI}}[n] \in \mathbb{C}$ is the local self-interference component,
* $w[n] \sim \mathcal{CN}(0, \sigma_w^2)$ is circularly symmetric complex additive white Gaussian noise (AWGN).

The local baseband transmit sequence $x[n] \in \mathbb{C}$ is known at the receiver and serves as the reference input to the adaptive canceller. For a causal linear finite impulse response (FIR) SI channel of length $L$ with impulse response $h[k]$, the linear self-interference is:

$$s_{\text{SI}}[n] = \sum_{k=0}^{L-1} h[k] x[n - k]$$

## Simulation architecture

* `qpsk_communication.py`: Generates seeded bit streams, maps them to unit-energy Gray-coded QPSK symbols ($\mathbb{E}[|x[n]|^2] = 1$), adds complex AWGN, performs minimum-distance demodulation, measures bit error rate (BER), and compares simulated performance against the theoretical QPSK curve $\text{BER} = Q\left(\sqrt{2 \cdot \text{SNR}}\right)$.
* `ibfd_simulation.py`: Combines independent local and remote QPSK streams through a flat desired channel and a configurable SI channel (flat, multipath FIR, or optional simplified cubic nonlinearity).
* `adaptive_si_cancellation.py`: Estimates and cancels self-interference using complex Least Mean Squares (LMS) or Normalized LMS (NLMS) adaptive FIR filters.
* `robustness_experiments.py`: Orchestrates paired multi-seed parameter sweeps across the simulation modules.

### Adaptive cancellation formulation

For an adaptive FIR filter of length $M$, define the tap-input reference vector $\mathbf{x}[n] \in \mathbb{C}^M$ ordered from the current sample backwards and the adaptive weight vector $\mathbf{w}[n] \in \mathbb{C}^M$:

$$\mathbf{x}[n] = \begin{bmatrix} x[n] & x[n-1] & \cdots & x[n-M+1] \end{bmatrix}^T, \qquad \mathbf{w}[n] = \begin{bmatrix} w_0[n] & w_1[n] & \cdots & w_{M-1}[n] \end{bmatrix}^T$$

At each step $n$, the estimated self-interference $\hat{s}_{\text{SI}}[n]$ and the residual error signal $e[n]$ are computed as:

$$\hat{s}_{\text{SI}}[n] = \mathbf{w}^H[n] \mathbf{x}[n]$$

$$e[n] = y[n] - \hat{s}_{\text{SI}}[n] = d[n] + \left(s_{\text{SI}}[n] - \hat{s}_{\text{SI}}[n]\right) + w[n]$$

where $(\cdot)^H$ denotes the Hermitian (conjugate) transpose. Because $d[n]$ and $w[n]$ are uncorrelated with the local reference $\mathbf{x}[n]$, minimizing $\mathbb{E}[|e[n]|^2]$ drives $\hat{s}_{\text{SI}}[n] \to s_{\text{SI}}[n]$, leaving $e[n] \approx d[n] + w[n]$ as the cleaned signal passed to the QPSK demodulator.

The filter weights are updated iteratively using either **Complex LMS** or **Complex NLMS**:

* **Complex LMS:**

$$\mathbf{w}[n+1] = \mathbf{w}[n] + \mu \, \mathbf{x}[n] \, e^*[n]$$

* **Complex NLMS:**

$$\mathbf{w}[n+1] = \mathbf{w}[n] + \frac{\mu}{\epsilon + \lVert\mathbf{x}[n]\rVert_2^2} \, \mathbf{x}[n] \, e^*[n]$$

where $\mu > 0$ is the step size, $e^*[n]$ is the complex conjugate of the error, $\lVert\mathbf{x}[n]\rVert_2^2 = \mathbf{x}^H[n]\mathbf{x}[n]$ is the instantaneous input vector energy, and $\epsilon > 0$ is a small regularization constant preventing division by zero. During adaptation, the algorithms observe only $\mathbf{x}[n]$ and $y[n]$; the true noiseless $s_{\text{SI}}[n]$ is reserved strictly for offline evaluation.

## Implemented algorithms and stages

* **Stage 1:** QPSK transmission over ideal complex AWGN with empirical vs. theoretical BER verification.
* **Stage 2:** IBFD composite signal generation, flat and multipath FIR SI channels, optional simplified cubic SI distortion, and desired-referenced noise scaling.
* **Stage 3:** Complex LMS and NLMS adaptive FIR filtering, frozen-weight evaluation on held-out test partitions, SI cancellation gain, and post-cancellation BER measurement.
* **Stage 3.5:** One-at-a-time sweeps over SI-to-desired power ratio, SNR, channel length/type, and adaptive filter length $M$ across three deterministic seeds, reporting sample mean and standard deviation ($\text{mean} \pm \text{SD}$) across separate train, validation, and test partitions.

## Experimental methodology and key findings

Stage 3.5 evaluates $16{,}000$ bits ($8{,}000$ QPSK symbols) per condition using deterministic seeds `11`, `22`, and `33`. Parameters (SI channel, SI power, adaptive filter length $M$, and SNR) are varied one at a time. In every trial, LMS and NLMS operate on identical realizations of $x[n]$ and $y[n]$, and the learned weight vector $\mathbf{w}$ is frozen after the training partition before evaluating the held-out test partition.

Power ratios and cancellation performance over a finite evaluation window $\mathcal{N}_{\text{test}}$ are defined using empirical average noiseless powers:

$$\text{SIR}_{\text{in}} \text{ (dB)} = 10 \log_{10} \left( \frac{\sum_{n} |d[n]|^2}{\sum_{n} |s_{\text{SI}}[n]|^2} \right), \qquad \text{SNR} \text{ (dB)} = 10 \log_{10} \left( \frac{\frac{1}{N}\sum_{n} |d[n]|^2}{\sigma_w^2} \right)$$

$$\text{SI Cancellation (dB)} = 10 \log_{10} \left( \frac{\sum_{n \in \mathcal{N}_{\text{test}}} |s_{\text{SI}}[n]|^2}{\sum_{n \in \mathcal{N}_{\text{test}}} \left|s_{\text{SI}}[n] - \hat{s}_{\text{SI}}[n]\right|^2} \right)$$

> **Note:** $\text{SNR}$ is defined with respect to the desired signal power and complex noise variance $\sigma_w^2$, rather than $E_b/N_0$.

For the baseline condition ($\text{SI/Desired} = 0\text{ dB}$, $\text{SNR} = 20\text{ dB}$, channel length $L = 4$ taps, filter length $M = 8$ taps):
* **Mean Test SI Cancellation:** $5.455 \pm 1.347\text{ dB}$ (LMS) and $4.521 \pm 0.978\text{ dB}$ (NLMS).
* **Mean Test BER:** Reduced from $0.1900 \pm 0.0101$ (before cancellation) to $0.03167 \pm 0.01566$ (LMS) and $0.04781 \pm 0.01660$ (NLMS).

These figures reflect this specific simulated baseband configuration rather than physical hardware measurements or a general ranking between LMS and NLMS. Complete per-condition datasets and summary tables are stored in `results/stage3_5/`.

## Current limitations

The simulation operates at discrete-time symbol-rate baseband, assuming ideal symbol timing and carrier synchronization alongside static illustrative channel taps. It does not model pulse-shaping filters, analog/RF impairments (such as I/Q imbalance, phase noise, or power amplifier saturation), measured IQ captures, or SDR hardware integration. Furthermore, LMS and NLMS are strictly linear FIR filters and cannot cancel general nonlinear self-interference components, and three random seeds provide reproducibility rather than broad statistical confidence bounds.

## Future milestones

1. Expand channel and noise characterization and tune conventional filter hyperparameters ($\mu, M$) strictly on training and validation partitions.
2. Develop a neural/deep-learning SI estimator to capture nonlinear distortion and benchmark it against LMS/NLMS on identical held-out signals and metrics.
3. Validate cancellation performance using measured SDR/USRP complex IQ data and integrate the estimator into a real-time GNU Radio flowgraph.

> These milestones represent planned future work and are not yet implemented in this repository.
