# Changelog

All notable changes to this project. Evidence tag: SYNTHETIC (all traces, noise, and dropout are generated and seed-controlled).

Cite all versions via the concept DOI [10.5281/zenodo.23173449](https://doi.org/10.5281/zenodo.23173449).

## 0.1.1 — 2026-10-05

Independent-review fixes. No change to the model, the default sweep, or any reported number.

### Fixed
- `results/sample_report.md` is now produced entirely by `render_report()`, including the “Acceptance outcome and payload arithmetic” section (derived from `metrics.json` instead of hand-maintained text). The generated report no longer asserts unit-test outcomes; it points to the test command instead.
- README scope note: the 100% calibration accuracy and zero noiseless error test **only identity versus exact reversal after known full-vector calibration**. They do not show recovery of an arbitrary permutation from a scalar observer.

### Added
- Report-sync test: the bundled `results/sample_report.md` must equal the generator output byte-for-byte.

### Unchanged
- Regenerated `metrics.json`, `trials.csv`, and `visualization.html` are byte-identical to 0.1.0 (NumPy 2.4/2.5).
- License remains AGPL-3.0-only (commercial licensing: see `COMMERCIAL-LICENSE.md`).

## 0.1.0 — 2026-10-05

- Initial public release: deterministic offline Python prototype for a classical weighted readout under a hidden channel-order reversal (512 synthetic inputs; optional two noisy banks on the same 512 signals), identity-vs-reversal calibration, seeded benchmark sweep (108 held-out trials across 9 noise/dropout regimes), sample results, static visualization, CI, `CITATION.cff`, and `.zenodo.json`. Archived on Zenodo: version DOI [10.5281/zenodo.23173450](https://doi.org/10.5281/zenodo.23173450).
