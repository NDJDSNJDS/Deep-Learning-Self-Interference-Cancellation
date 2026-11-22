# Deep Learning-Based Self-Interference Cancellation for In-Band Full-Duplex Communication Using SDR

**B.Tech Electronics and Communication Engineering (ECE) final-year project — pre-project implementation**

This repository contains a reproducible, symbol-rate complex-baseband study of QPSK transmission, in-band full-duplex (IBFD) self-interference (SI) modelling, and conventional complex LMS/NLMS cancellation. It establishes the measured simulation baseline for planned deep-learning and software-defined-radio work. Current outputs are software simulations; no SDR/USRP hardware measurements are included.

## Research motivation and problem statement

An IBFD radio transmits and receives at the same time in the same frequency band. Its local transmit signal can couple into its receiver as self-interference, potentially masking a weaker remote desired signal. The research problem is to model that coupling and estimate its contribution from a known local transmit reference so that it can be subtracted before detection.

This pre-project builds the baseband link and a conventional adaptive-filter reference point. It provides reproducible signals, channel conditions, held-out metrics, and plots that later learned estimators can be compared against. Deep learning, GNU Radio, and SDR/USRP processing are planned work and are not implemented here.

## Project objectives

- Build and validate a Gray-coded QPSK communication chain over AWGN.
- Model an IBFD receiver with separate desired and local transmit paths.
- Implement complex linear LMS and NLMS SI estimation and subtraction.
- Measure held-out SI cancellation, residual SI power, BER, and convergence.
- Assess sensitivity to SI level, desired-referenced SNR, channel, filter length, and deterministic random seed.
- Preserve a modular baseline suitable for future comparative work.

## IBFD concept and signal model

The local transmitter and remote transmitter send independently generated QPSK symbol streams. The receiver observes:

$$y[n] = d[n] + s_{\mathrm{SI}}[n] + w[n]$$

where $d[n] \in \mathbb{C}$ is the desired remote signal after its configured flat complex gain, $s_{\mathrm{SI}}[n] \in \mathbb{C}$ is local signal leakage after the SI channel, and $w[n] \sim \mathcal{CN}(0, \sigma_w^2)$ is complex AWGN. Stage 2 offers a flat channel, a causal FIR multipath channel ($s_{\mathrm{SI}}[n] = \sum_{k=0}^{L-1} h[k]x[n-k]$), and a simplified memoryless cubic option. The Stage 3.5 sweeps use the flat and linear FIR cases.

```mermaid
flowchart LR
    A[Remote bits] --> B[QPSK modulator]
    B --> C[Desired flat channel]
    D[Local bits] --> E[QPSK modulator]
    E --> F[Known local reference x n]
    F --> G[SI channel]
    C --> H[Sum with SI and AWGN]
    G --> H
    I[Complex AWGN] --> H
    H --> J[Observation y n]
    F --> K[Complex LMS or NLMS]
    J --> K
    K --> L[SI estimate]
    J --> M[Subtract estimate]
    L --> M
    M --> N[Residual and BER evaluation]
