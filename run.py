#!/usr/bin/env python3
"""Run the deterministic synthetic benchmark and write local results."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from flip_observer.experiment import run_benchmark, write_artifacts  # noqa: E402

RESULTS = ROOT / "results"

if __name__ == "__main__":
    results = run_benchmark()
    write_artifacts(RESULTS, results)
    print(f"Wrote {RESULTS / 'metrics.json'}")
    print(f"Wrote {RESULTS / 'sample_report.md'}")
    print(f"Wrote {RESULTS / 'visualization.html'}")
    print(f"Regimes: {len(results['conditions'])}; seeded trials: {sum(c['trial_count'] for c in results['conditions'])}")
