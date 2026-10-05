# Assumptions and limitations

## Model

- **Signals:** 512 real-valued, dimensionless synthetic channel traces per held-out trial. Eight smooth sinusoidal latent traces are mixed into channels with seeded random coefficients; a small seeded channel-local term prevents identical traces.
- **Observer:** one deterministic unit-L2 vector `w`, held fixed across trials and conditions. The clean target `z(t) = wᵀx(t)` is computed before mapping faults are applied. This ordering makes the reversal observable rather than co-permuting the observer.
- **Flip:** the second bank's channel axis is reversed exactly. The reversal matrix `P` is an involution (`P⁻¹ = P`).
- **Calibration:** known 512-by-512 orthogonal ±1 patterns, measured with separate seeded noise/dropout. The detector compares only identity and reversal residuals using the full channel vector. Calibration samples do not overlap held-out signal samples.
- **Two banks:** each bank is a separate noisy/dropout-affected observation of the same underlying 512 signals, not a new independent latent signal basis. Available per-channel observations are averaged; if both are missing, the prototype inserts zero.
- **Fault range:** additive independent zero-mean Gaussian noise with standard deviation 0, 0.05, or 0.20 in arbitrary units; independent per-entry dropout probability 0%, 5%, or 20%.
- **Reproducibility:** all generated traces, fault draws, calibration patterns, and fixed weights are seed-controlled. Results use 12 held-out seeds per regime.

## Metrics and assumptions

Readout RMSE is normalized by the RMS of the clean scalar target for the same trial. Means and sample standard deviations are reported across seeds; worst/best ranges and per-seed values are available in JSON/CSV. Mapping accuracy measures selection of the correct identity-versus-reversal hypothesis; it is not evidence of general arbitrary-permutation decoding.

Throughput arithmetic assumes 16-bit values sampled at 1,000 samples/second/channel, giving 8.192 Mbit/s for 512 channels and 16.384 Mbit/s for two 512-channel banks. It excludes headers, timestamps, retransmission, compression, and other overhead.

## What the results do not establish

No result is biological, medical, physical, or quantum evidence. The prototype does not demonstrate an actual observer, a membrane, superposition, entanglement, or 1,024 independent measurements. It tests the software consequences of one fixed classical weighted sum under a narrowly defined channel-order fault. The detector does not recover arbitrary permutations; a scalar output cannot reveal a full hidden 512-channel mapping. Permanent sensor failures, crosstalk, gain mismatch, clock drift, adversarial faults, hardware cost, power, and real-world throughput are out of scope. Synthetic performance should not be treated as empirical system performance.
