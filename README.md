# Flip Observer 1024 — synthetic 512-input calibration prototype

[![CI](https://github.com/sparkainlp-x/flip-observer-1024/actions/workflows/ci.yml/badge.svg)](https://github.com/sparkainlp-x/flip-observer-1024/actions/workflows/ci.yml)
[![License: AGPL-3.0-only](https://img.shields.io/badge/License-AGPL--3.0--only-blue.svg)](LICENSE)
[![Evidence: SYNTHETIC](https://img.shields.io/badge/evidence-SYNTHETIC-blue.svg)](#scope-and-limitations)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23173449.svg)](https://doi.org/10.5281/zenodo.23173449)
[![Demo](https://img.shields.io/badge/demo-GitHub%20Pages-245a9b.svg)](https://sparkainlp-x.github.io/flip-observer-1024/)

Offline Python prototype for an **ordinary classical weighted readout** under a hidden **channel-order reversal**. All inputs are **SYNTHETIC**. It is a software experiment only—not a biological membrane model, not a quantum model, and not a claim of entanglement or superposition.

**Live demo (static visualization):** <https://sparkainlp-x.github.io/flip-observer-1024/>

> **Framing.** `flip` = exact reversal of channel order. `1024` = two noisy/dropout-affected measurement banks observing the **same** 512 underlying synthetic inputs (repeated-measurement capacity), **not** 1,024 independent latent channels. Evidence class: **SYNTHETIC**.

## What it implements

The first layer has 512 dimensionless real-valued synthetic signals, `x(t) ∈ R^512`. A fixed weight vector is generated once, and the target is defined **before any hidden reversal**:

```text
z(t) = wᵀ x(t)
y(t) = P x(t)
corrected z(t) = wᵀ P⁻¹ y(t)
```

Here `P` reverses the channel order exactly. Because reversal is its own inverse, `P⁻¹ = P`. The scalar `z` is an ordinary weighted sum—not a quantum superposition. The detector never uses that scalar to infer a mapping.

A separate calibration phase sends 512 known orthogonal full-channel patterns and observes the full channel vector. It compares the **identity** hypothesis with **exact reversal**. Calibration data and synthetic held-out signal mixtures are generated separately. The practical detector is intentionally limited to those two hypotheses; it does **not** claim to recover an arbitrary unknown 512-channel permutation.

> **What the perfect result means.** The 100% calibration accuracy and zero noiseless error test **only identity versus exact reversal after known full-vector calibration**. They do **not** show recovery of an arbitrary permutation from a scalar observer: the scalar `z = wᵀx` is never used for mapping detection, and in general a single scalar output cannot identify an unknown 512-channel permutation.

## The 1,024-slot extension

The extension has two transparently represented banks of 512 observations each. Both observe the same 512 underlying synthetic signal channels, but each bank receives independent synthetic measurement noise and independent per-entry dropout. One bank can be hidden-reversed, calibrated, corrected, and fused with the other by averaging available measurements.

This is repeated-measurement capacity, **not 1,024 independent latent channels**. The included duplication control copies the same bank exactly and confirms that copying does not add signal information or improve the estimate.

## Install and run

Python 3.10+ and NumPy are required. From this folder:

```bash
git clone https://github.com/sparkainlp-x/flip-observer-1024.git
cd flip-observer-1024
python3 -m pip install "numpy>=1.24"
python3 run.py
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Or install the package editable from `pyproject.toml`:

```bash
python3 -m pip install -e .
python3 run.py
python3 -m unittest discover -s tests -v
```

`run.py` writes `results/metrics.json`, `results/trials.csv`, `results/sample_report.md`, and `results/visualization.html`. Open `results/visualization.html` (or the Pages copy at `docs/index.html`) directly in a browser; it is self-contained and works offline. No account, server, or external service is used at runtime.

## Reproducible evaluation

The sweep uses seeds `0–11` at every combination of dimensionless additive-noise standard deviation `0.00, 0.05, 0.20` and independent element-dropout probability `0%, 5%, 20%`—108 held-out trials overall. Each held-out trial contains 512 time samples across 512 channels. The synthetic signals are mixtures of eight smooth latent sinusoids plus a small channel-local component. All traces, noise, and dropout are deterministic given their seeds; nothing is measured from people, biology, hardware, or an external dataset.

Reported mapping accuracy is the fraction of trials where the calibration phase chooses the correct identity/reversal hypothesis. Readout error is RMSE divided by the clean target RMS; each regime reports the mean and sample standard deviation across its 12 seeds, plus failure counts and trial-level values. The principal comparisons are:

- **Single bank:** bank A alone.
- **Identity control:** both banks aligned, without a hidden reversal.
- **Uncorrected hidden flip:** average bank A with bank B while B is still reversed.
- **Calibrated reversal:** infer identity/reversal from independent full-vector calibration, then align and fuse held-out data.
- **Oracle upper bound:** correct with the true known inverse; not a deployable detector.

### Pass criteria (declared before running tests)

1. Applying reversal twice restores every channel exactly; noiseless inverse correction has maximum absolute error `0`.
2. Identity and reversal calibration accuracy are each at least 95% **in every noise/dropout regime**.
3. Calibrated recovery has lower normalized readout error than the uncorrected hidden flip in at least 90% of trials in each regime.
4. The exact duplicate bank counts as 512 distinct underlying input channels; its fused estimate equals the single-bank estimate exactly.
5. Re-running a trial with the same seeds produces identical metric values.

The test suite checks these criteria against the generated `results/metrics.json`; run `python3 run.py` before the test command if you regenerate artifacts. Checked-in sample outputs already satisfy the suite for CI.

`results/sample_report.md` is produced entirely by `render_report()` in `src/flip_observer/experiment.py`, including the “Acceptance outcome and payload arithmetic” section (derived from `metrics.json`, not hand-maintained); a unit test checks that the bundled report is byte-for-byte what the generator writes. The checked-in data files regenerate byte-identically with NumPy 2.4/2.5; other NumPy/BLAS builds may differ in the last floating-point digit of `metrics.json`/`trials.csv` without changing the rounded report.

## Illustrative raw payload arithmetic

For a throughput comparison only, assume 16-bit samples and 1,000 samples/second/channel, with no framing or protocol overhead. A 512-channel bank carries `512 × 16 × 1,000 = 8,192,000 bit/s` (8.192 Mbit/s). Two banks (1,024 measurement slots) carry 16.384 Mbit/s. Copying a bank doubles transmitted bits but not independent signal content. These are arithmetic assumptions, not measured performance or an implementation requirement.

## Files

| Path | Contents |
|---|---|
| `src/flip_observer/core.py` | Reversal, fixed readout, orthogonal calibration patterns, calibration detector |
| `src/flip_observer/experiment.py` | Seeded synthetic traces, two-bank fault injection, benchmark, report/visualization |
| `tests/test_prototype.py` | Correctness, recovery, duplication, reproducibility, sweep acceptance, and report-sync tests |
| `run.py` | Local benchmark entry point |
| `docs/index.html` | Static visualization served by GitHub Pages |
| `results/` | Checked-in sample outputs from the default run |
| `ASSUMPTIONS.md` | Model choices and limitations |
| `CITATION.cff`, `.zenodo.json` | Citation and archive metadata |
| `LICENSE`, `COMMERCIAL-LICENSE.md` | AGPL-3.0-only and commercial licensing |
| `SECURITY.md` | Security policy |

## Scope and limitations

This is a compact falsifiable software toy. It does **not** model a physical observer, a membrane, sensing hardware, biological processes, quantum superposition/entanglement, or a communications protocol. The reversal is a known structured fault; calibration chooses only between identity and reversal. It cannot use one scalar readout to infer an arbitrary permutation. Dropout is independent per measurement entry, not a permanently dead hardware channel. Gains, crosstalk, calibration drift, latency, implementation costs, and real payload overhead are deliberately omitted. The oracle is an upper bound, not the practical method.

See [ASSUMPTIONS.md](ASSUMPTIONS.md) for the full model and “what the results do not establish” list.

## Related

- [**spark-oes512-demo**](https://github.com/sparkainlp-x/spark-oes512-demo) ([demo](https://sparkainlp-x.github.io/spark-oes512-demo/)): synthetic 512-channel browser explorer.
- [**evidence-passport**](https://github.com/sparkainlp-x/evidence-passport): offline evidence-passport MVP for one experiment-run manifest.
- [**quantum-claims-passport**](https://github.com/sparkainlp-x/quantum-claims-passport) ([report](https://sparkainlp-x.github.io/quantum-claims-passport/report.html)): claims audit that keeps unlike evidence types separate.

## Cite

See [`CITATION.cff`](CITATION.cff) (GitHub’s “Cite this repository” button). Concept DOI (all versions): [10.5281/zenodo.23173449](https://doi.org/10.5281/zenodo.23173449). Version DOI for v0.1.0: [10.5281/zenodo.23173450](https://doi.org/10.5281/zenodo.23173450).

> Brisson, Jean-François. *flip-observer-1024: synthetic classical channel-reversal observer calibration prototype (512 inputs, two noisy banks)* (v0.1.0). Spark AI NLP. https://doi.org/10.5281/zenodo.23173449

Author: Jean-François Brisson, Spark AI NLP · ORCID [0009-0000-9778-5374](https://orcid.org/0009-0000-9778-5374)

## License

Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. Released under the GNU Affero General Public License v3.0 only (AGPL-3.0-only); see [LICENSE](LICENSE). For proprietary use without AGPL obligations, see [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md).

## Security

See [SECURITY.md](SECURITY.md).
