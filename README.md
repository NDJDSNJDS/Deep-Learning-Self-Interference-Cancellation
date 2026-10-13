# QPSK Baseband Communication Simulation

A reproducible discrete-time complex baseband model of Gray-coded QPSK over an additive white Gaussian noise (AWGN) channel. It covers random bits, modulation, complex noise, hard-decision demodulation, BER measurement, an Eb/N0 sweep, and theoretical BER comparison. The code is intentionally limited to this first project stage; it contains no self-interference canceller, neural network, GNU Radio flowgraph, or USRP interface.

## What QPSK does

Quadrature Phase Shift Keying carries two bits in each complex symbol: its in-phase (I) and quadrature (Q) coordinates jointly select one of four phases. The mapping here is Gray coded so neighboring points differ by one bit:

| Bits | Symbol |
|---|---|
| 00 | (1 + j)/sqrt(2) |
| 01 | (-1 + j)/sqrt(2) |
| 11 | (-1 - j)/sqrt(2) |
| 10 | (1 - j)/sqrt(2) |

Each symbol has energy one, so average symbol energy is Es=1. Two uncoded bits per symbol gives Es=2Eb and Eb=1/2.

## Channel and BER

For an Eb/N0 setting in dB, the model converts to linear ratio `10**(EbN0_dB/10)`, calculates `N0 = Eb/(Eb/N0)`, and independently adds real and imaginary zero-mean Gaussian samples, each with variance `N0/2`. Thus the total complex noise power is `E[|w|^2]=N0`. The receiver chooses the nearest of the four ideal constellation points and recovers the associated bit pair. BER is the count of differing transmitted/recovered bit positions divided by total transmitted bits.

For coherent Gray-coded QPSK in AWGN, theoretical BER equals BPSK BER: `0.5 * erfc(sqrt(Eb/N0))`, with Eb/N0 in linear units. The sweep uses a common reproducible bit sequence and independent reproducible noise at each point. Finite runs fluctuate; BER need not decrease at every adjacent SNR point, and a high-SNR run can observe zero errors while theoretical BER remains nonzero. Zero simulated points are drawn at half a bit-error resolution on the logarithmic chart; the printed BER remains exact.

## Installation and run

From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python qpsk_communication.py
```

The default experiment uses 200,000 bits and Eb/N0 values 0, 2, 4, 6, 8, and 10 dB. Adjust `num_bits` and `ebn0_values` in `main()` for a different run. Figures are saved under `results/`.

## Generated plots

- `results/transmitted_constellation.png`: transmitted symbols at the four ideal points.
- `results/received_constellation.png`: received symbols spread around ideal points by AWGN (default display point 6 dB).
- `results/iq_samples.png`: a short symbol-index view of transmitted and received I and Q components.
- `results/ber_vs_ebn0.png`: simulated BER with the theoretical curve on a logarithmic BER axis.

## Code structure

- `generate_random_bits`: creates even-length seeded bit arrays.
- `qpsk_modulate`: groups bits in order and applies the documented mapping.
- `awgn_channel`: converts Eb/N0 into complex noise variance and adds noise.
- `qpsk_demodulate`: makes nearest-point hard decisions; exact ties follow constellation insertion order.
- `calculate_ber`: validates lengths and returns bit count, errors, and BER.
- `theoretical_qpsk_ber`: evaluates the closed-form Gray-coded QPSK AWGN BER.
- `run_single_simulation`: runs one seeded end-to-end case.
- `run_ebn0_sweep`: measures reproducible BER over multiple SNR points.
- `_save_figures`: writes the four plots; `main` configures and runs the example.

## Validation

Run the test suite with:

```powershell
python -m unittest discover -s tests -v
```

It checks mapping and noiseless round-trip, unit symbol energy, measured real/imaginary AWGN variance, BER accounting and size validation, input validation, and agreement/trend of a seeded sweep against theory. Monte Carlo values can fluctuate; the trend assertion applies to this fixed test seed and sample count, not as a general mathematical guarantee.

## Scope and next stage

This is a discrete-time symbol-rate complex baseband simulation, not a continuous-time RF waveform or complete model of a radio link. It assumes perfect timing and carrier synchronization and includes no pulse shaping, fading, multipath, phase/frequency offset, hardware impairments, coding, or measured USRP IQ data. The complex-array signal path and separately defined channel function provide a clean point to add a channel model for `y[n] = d[n] + s_SI[n] + w[n]`, then compare conventional or learned cancellation methods while keeping modulation and demodulation reusable.
