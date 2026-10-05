# Sample benchmark results

All entries are deterministic synthetic results; values are dimensionless.
Each regime summarizes 12 independent seeded held-out trials. Calibration patterns are separate from held-out mixtures.
Errors are normalized RMSE (readout RMSE divided by the clean target RMS); `mean ± SD` is across seeds.

| Noise SD | Element dropout | Identity calibration accuracy | Reversal calibration accuracy | Single bank | Uncorrected flip | Calibrated flip | Oracle map |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0% | 100% | 100% | 0.0000 ± 0.0000 | 0.6955 ± 0.1275 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| 0.00 | 5% | 100% | 100% | 0.2294 ± 0.0390 | 0.7286 ± 0.1260 | 0.0510 ± 0.0085 | 0.0510 ± 0.0085 |
| 0.00 | 20% | 100% | 100% | 0.4616 ± 0.0591 | 0.8351 ± 0.1348 | 0.2053 ± 0.0337 | 0.2053 ± 0.0337 |
| 0.05 | 0% | 100% | 100% | 0.0728 ± 0.0118 | 0.6966 ± 0.1267 | 0.0518 ± 0.0091 | 0.0518 ± 0.0091 |
| 0.05 | 5% | 100% | 100% | 0.2436 ± 0.0435 | 0.7371 ± 0.1299 | 0.0761 ± 0.0165 | 0.0761 ± 0.0165 |
| 0.05 | 20% | 100% | 100% | 0.4679 ± 0.0675 | 0.8346 ± 0.1313 | 0.2193 ± 0.0381 | 0.2193 ± 0.0381 |
| 0.20 | 0% | 100% | 100% | 0.2914 ± 0.0496 | 0.7226 ± 0.1269 | 0.2056 ± 0.0370 | 0.2056 ± 0.0370 |
| 0.20 | 5% | 100% | 100% | 0.3760 ± 0.0679 | 0.7730 ± 0.1292 | 0.2220 ± 0.0379 | 0.2220 ± 0.0379 |
| 0.20 | 20% | 100% | 100% | 0.5349 ± 0.0800 | 0.8703 ± 0.1469 | 0.3089 ± 0.0546 | 0.3089 ± 0.0546 |

## Failure counts

Counts are out of 12 seeds per regime. Empty calibration-failure lists mean all seeds selected the right identity/reversal hypothesis.

| Noise SD | Dropout | Reversal calibration misses | Identity-control misses | Calibrated trials not better than uncorrected |
|---:|---:|---:|---:|---:|
| 0.00 | 0% | 0 | 0 | 0 |
| 0.00 | 5% | 0 | 0 | 0 |
| 0.00 | 20% | 0 | 0 | 0 |
| 0.05 | 0% | 0 | 0 | 0 |
| 0.05 | 5% | 0 | 0 | 0 |
| 0.05 | 20% | 0 | 0 | 0 |
| 0.20 | 0% | 0 | 0 | 0 |
| 0.20 | 5% | 0 | 0 | 0 |
| 0.20 | 20% | 0 | 0 | 0 |

## Reading the result

The oracle column uses the known inverse reversal and is an upper bound on the calibration decision, not a deployable detector. The calibrated column uses only a separate known-pattern calibration phase, then evaluates on held-out signal mixtures. The single scalar observer output is never used to infer the channel map.

The 1,024-row option consists of two noisy/dropout-affected banks observing the same 512 underlying channel identities. It is repeated measurement capacity, not 1,024 independent latent channels. Exact copied-bank duplication leaves the single-bank estimate unchanged.

See `visualization.html` for an offline interactive comparison and `metrics.json` / `trials.csv` for the full trial-level values.


## Acceptance outcome and payload arithmetic

The default sweep passed all predeclared criteria: 108 trials across 9 regimes; identity and reversal calibration each achieved 100% accuracy in every regime; calibrated recovery beat the uncorrected flip in 108/108 trials; and exact-copy duplication represented 512 unique source channels with zero change to the fused estimate. In the noiseless/no-dropout regime, the oracle corrected readout had exactly zero normalized error. This report is generated from `metrics.json`; it does not record unit-test results (run `PYTHONPATH=src python -m unittest discover -s tests`).

For an illustrative raw throughput only, assume 16-bit samples at 1,000 samples/second/channel, excluding all framing and protocol overhead: 512 channels = 8.192 Mbit/s; two 512-channel banks = 16.384 Mbit/s. The second figure is twice the traffic, not twice the number of independent latent signals.
