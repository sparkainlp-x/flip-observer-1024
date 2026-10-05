"""Reproducible synthetic benchmark and report-data generation."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .core import (
    calibrate_identity_vs_reversal,
    fixed_weights,
    hadamard_calibration,
    readout,
    reverse_channels,
    undo_reversal,
)

CHANNELS = 512
DEFAULT_SEEDS = tuple(range(12))
NOISE_LEVELS = (0.0, 0.05, 0.20)
DROPOUT_LEVELS = (0.0, 0.05, 0.20)
SAMPLES = 512

# Predeclared before test execution; see README for the plain-language criteria.
PASS_CRITERIA = {
    "exact_noiseless_correction_max_abs_error": 0.0,
    "calibration_accuracy_min_per_regime": 0.95,
    "calibrated_vs_uncorrected_win_fraction_min": 0.90,
    "duplicate_unique_channel_count": CHANNELS,
    "reproducibility": "same inputs/seeds produce byte-equivalent JSON values",
}


def synthetic_mixture(seed: int, samples: int = SAMPLES, channels: int = CHANNELS) -> np.ndarray:
    """Make a deterministic, dimensionless mixture of eight smooth latent traces."""
    rng = np.random.default_rng(seed)
    time = np.arange(samples, dtype=float) / samples
    frequencies = np.array([3, 5, 9, 13, 17, 23, 29, 37], dtype=float)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=frequencies.size)
    latent = np.stack(
        [np.sin(2.0 * np.pi * f * time + phase) for f, phase in zip(frequencies, phases)],
        axis=1,
    )
    mixing = rng.normal(size=(frequencies.size, channels)) / np.sqrt(frequencies.size)
    # Small channel-local deterministic variation keeps the traces non-identical.
    local = 0.05 * rng.normal(size=(samples, channels))
    return latent @ mixing + local


def _measurement(
    signal: np.ndarray, noise_std: float, dropout: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    mask = rng.random(signal.shape) >= dropout
    measured = signal + rng.normal(0.0, noise_std, size=signal.shape)
    return np.where(mask, measured, 0.0), mask


def _fuse(
    first: np.ndarray,
    first_mask: np.ndarray,
    second: np.ndarray,
    second_mask: np.ndarray,
) -> np.ndarray:
    """Average available same-channel bank observations; missing both -> zero."""
    count = first_mask.astype(np.int8) + second_mask.astype(np.int8)
    total = np.where(first_mask, first, 0.0) + np.where(second_mask, second, 0.0)
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0)


def _rmse(actual: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(actual - target))))


def _nrmse(actual: np.ndarray, target: np.ndarray) -> float:
    denominator = float(np.sqrt(np.mean(np.square(target))))
    return _rmse(actual, target) / max(denominator, np.finfo(float).eps)


def _one_trial(seed: int, noise_std: float, dropout: float) -> dict[str, Any]:
    # The observer weights and target are fixed before either bank's hidden map.
    weights = fixed_weights(CHANNELS, seed=1024)
    x = synthetic_mixture(seed)
    target = readout(x, weights)

    rng = np.random.default_rng(seed * 1009 + int(noise_std * 10000) * 31 + int(dropout * 10000))
    bank_a, mask_a = _measurement(x, noise_std, dropout, rng)
    bank_b_aligned, mask_b_aligned = _measurement(x, noise_std, dropout, rng)
    bank_b_hidden = reverse_channels(bank_b_aligned)
    mask_b_hidden = reverse_channels(mask_b_aligned)

    # Calibration uses known full vectors and independent calibration noise/dropout.
    patterns = hadamard_calibration(CHANNELS)
    calibration_rng = np.random.default_rng(seed * 9176 + int(noise_std * 10000) * 19 + int(dropout * 10000) + 17)
    cal_mask_identity = calibration_rng.random(patterns.shape) >= dropout
    cal_noise_identity = calibration_rng.normal(0.0, noise_std, size=patterns.shape)
    observed_identity = np.where(cal_mask_identity, patterns + cal_noise_identity, 0.0)
    identity_result = calibrate_identity_vs_reversal(patterns, observed_identity, cal_mask_identity)

    cal_mask_aligned = calibration_rng.random(patterns.shape) >= dropout
    cal_noise_reverse = calibration_rng.normal(0.0, noise_std, size=patterns.shape)
    observed_reverse = reverse_channels(np.where(cal_mask_aligned, patterns + cal_noise_reverse, 0.0))
    observed_mask_reverse = reverse_channels(cal_mask_aligned)
    reverse_result = calibrate_identity_vs_reversal(patterns, observed_reverse, observed_mask_reverse)

    inferred_b = undo_reversal(bank_b_hidden) if reverse_result.mapping == "reversed" else bank_b_hidden
    inferred_mask = undo_reversal(mask_b_hidden) if reverse_result.mapping == "reversed" else mask_b_hidden
    oracle_b = undo_reversal(bank_b_hidden)
    oracle_mask = undo_reversal(mask_b_hidden)

    estimate_a = bank_a
    estimate_uncorrected = _fuse(bank_a, mask_a, bank_b_hidden, mask_b_hidden)
    estimate_calibrated = _fuse(bank_a, mask_a, inferred_b, inferred_mask)
    estimate_oracle = _fuse(bank_a, mask_a, oracle_b, oracle_mask)
    estimate_identity_control = _fuse(bank_a, mask_a, bank_b_aligned, mask_b_aligned)

    # Exact same-observation duplication is included as a control, not a new source.
    duplicated_values = np.concatenate([bank_a, bank_a], axis=1)
    duplicated_mask = np.concatenate([mask_a, mask_a], axis=1)
    duplicate_mean = _fuse(bank_a, mask_a, bank_a, mask_a)

    return {
        "seed": int(seed),
        "noise_std": float(noise_std),
        "dropout": float(dropout),
        "identity_control_correct": identity_result.mapping == "identity",
        "reverse_calibration_correct": reverse_result.mapping == "reversed",
        "reverse_calibration_mapping": reverse_result.mapping,
        "identity_calibration_mse": identity_result.identity_mse,
        "identity_calibration_flip_mse": identity_result.reversed_mse,
        "reverse_calibration_identity_mse": reverse_result.identity_mse,
        "reverse_calibration_flip_mse": reverse_result.reversed_mse,
        "reverse_calibration_confidence_gap": reverse_result.confidence_gap,
        "identity_map_accuracy": float(identity_result.mapping == "identity"),
        "reverse_map_accuracy": float(reverse_result.mapping == "reversed"),
        "single_bank_rmse": _rmse(readout(estimate_a, weights), target),
        "single_bank_nrmse": _nrmse(readout(estimate_a, weights), target),
        "uncorrected_flip_rmse": _rmse(readout(estimate_uncorrected, weights), target),
        "uncorrected_flip_nrmse": _nrmse(readout(estimate_uncorrected, weights), target),
        "calibrated_rmse": _rmse(readout(estimate_calibrated, weights), target),
        "calibrated_nrmse": _nrmse(readout(estimate_calibrated, weights), target),
        "oracle_rmse": _rmse(readout(estimate_oracle, weights), target),
        "oracle_nrmse": _nrmse(readout(estimate_oracle, weights), target),
        "identity_control_nrmse": _nrmse(readout(estimate_identity_control, weights), target),
        "duplicate_unique_channels": int(np.unique(duplicated_values, axis=1).shape[1]),
        "duplicate_mask_shape_channels": int(duplicated_mask.shape[1]),
        "duplicate_vs_single_max_abs": float(np.max(np.abs(duplicate_mean - bank_a))),
        "duplicate_nrmse": _nrmse(readout(duplicate_mean, weights), target),
    }


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def run_benchmark(seeds: tuple[int, ...] = DEFAULT_SEEDS) -> dict[str, Any]:
    """Run the full seeded noise/dropout grid, returning JSON-serializable values."""
    conditions: list[dict[str, Any]] = []
    for noise_std in NOISE_LEVELS:
        for dropout in DROPOUT_LEVELS:
            trials = [_one_trial(seed, noise_std, dropout) for seed in seeds]
            metrics = (
                "single_bank_nrmse",
                "uncorrected_flip_nrmse",
                "calibrated_nrmse",
                "oracle_nrmse",
                "identity_control_nrmse",
                "duplicate_nrmse",
            )
            conditions.append({
                "noise_std": noise_std,
                "dropout": dropout,
                "trial_count": len(trials),
                "identity_calibration_accuracy": float(np.mean([t["identity_map_accuracy"] for t in trials])),
                "reversal_calibration_accuracy": float(np.mean([t["reverse_map_accuracy"] for t in trials])),
                "metrics": {name: _stats([float(t[name]) for t in trials]) for name in metrics},
                "calibrated_not_better_than_uncorrected_count": int(sum(
                    t["calibrated_nrmse"] >= t["uncorrected_flip_nrmse"] for t in trials
                )),
                "calibration_failure_seeds": [
                    t["seed"] for t in trials if not t["reverse_calibration_correct"]
                ],
                "identity_control_failure_seeds": [
                    t["seed"] for t in trials if not t["identity_control_correct"]
                ],
                "trials": trials,
            })
    return {
        "title": "Flip Observer 512-to-1,024 synthetic benchmark",
        "version": "1.0.0",
        "model": {
            "channels_per_bank": CHANNELS,
            "samples_per_trial": SAMPLES,
            "weights_seed": 1024,
            "observer": "z(t) = w^T x(t), fixed before hidden reversal",
            "flip": "exact channel-order reversal; P^-1 = P",
            "calibration": "known orthogonal full-vector patterns; identity-vs-reversal only",
            "banks": "two independently noisy/dropout-affected observations of the same 512 latent channel signals",
        },
        "sweep": {
            "seeds": list(seeds),
            "noise_std_dimensionless": list(NOISE_LEVELS),
            "element_dropout_probability": list(DROPOUT_LEVELS),
            "seed_count_per_regime": len(seeds),
        },
        "pass_criteria": PASS_CRITERIA,
        "conditions": conditions,
    }


def write_artifacts(output_dir: str | Path, results: dict[str, Any]) -> None:
    """Write deterministic JSON/CSV, a concise Markdown report and offline HTML."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "metrics.json"
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    flat_rows: list[dict[str, Any]] = []
    for condition in results["conditions"]:
        for trial in condition["trials"]:
            flat_rows.append(trial)
    csv_path = output / "trials.csv"
    if flat_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flat_rows)

    lines = [
        "# Sample benchmark results",
        "",
        "All entries are deterministic synthetic results; values are dimensionless.",
        "Each regime summarizes 12 independent seeded held-out trials. Calibration patterns are separate from held-out mixtures.",
        "Errors are normalized RMSE (readout RMSE divided by the clean target RMS); `mean ± SD` is across seeds.",
        "",
        "| Noise SD | Element dropout | Identity calibration accuracy | Reversal calibration accuracy | Single bank | Uncorrected flip | Calibrated flip | Oracle map |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in results["conditions"]:
        m = c["metrics"]
        def fmt(name: str) -> str:
            return f"{m[name]['mean']:.4f} ± {m[name]['std']:.4f}"
        lines.append(
            f"| {c['noise_std']:.2f} | {c['dropout']:.0%} | "
            f"{c['identity_calibration_accuracy']:.0%} | {c['reversal_calibration_accuracy']:.0%} | "
            f"{fmt('single_bank_nrmse')} | {fmt('uncorrected_flip_nrmse')} | "
            f"{fmt('calibrated_nrmse')} | {fmt('oracle_nrmse')} |"
        )
    lines.extend([
        "",
        "## Failure counts",
        "",
        "Counts are out of 12 seeds per regime. Empty calibration-failure lists mean all seeds selected the right identity/reversal hypothesis.",
        "",
        "| Noise SD | Dropout | Reversal calibration misses | Identity-control misses | Calibrated trials not better than uncorrected |",
        "|---:|---:|---:|---:|---:|",
    ])
    for c in results["conditions"]:
        lines.append(
            f"| {c['noise_std']:.2f} | {c['dropout']:.0%} | {len(c['calibration_failure_seeds'])} | "
            f"{len(c['identity_control_failure_seeds'])} | {c['calibrated_not_better_than_uncorrected_count']} |"
        )
    lines.extend([
        "",
        "## Reading the result",
        "",
        "The oracle column uses the known inverse reversal and is an upper bound on the calibration decision, not a deployable detector. The calibrated column uses only a separate known-pattern calibration phase, then evaluates on held-out signal mixtures. The single scalar observer output is never used to infer the channel map.",
        "",
        "The 1,024-row option consists of two noisy/dropout-affected banks observing the same 512 underlying channel identities. It is repeated measurement capacity, not 1,024 independent latent channels. Exact copied-bank duplication leaves the single-bank estimate unchanged.",
        "",
        "See `visualization.html` for an offline interactive comparison and `metrics.json` / `trials.csv` for the full trial-level values.",
    ])
    (output / "sample_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    data = json.dumps(results, separators=(",", ":"), sort_keys=True).replace("</", "<\\/")
    html = _offline_html(data)
    (output / "visualization.html").write_text(html, encoding="utf-8")


def _offline_html(data: str) -> str:
    return '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Flip Observer — synthetic benchmark</title>
<style>
:root{color-scheme:light;--ink:#172b4d;--muted:#536579;--line:#d9e2ec;--blue:#2763a5;--teal:#16877b;--orange:#c46628;--bg:#f6f8fb}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1040px;margin:0 auto;padding:28px 20px 48px}h1{font-size:1.8rem;line-height:1.2;margin:0 0 8px}.sub{color:var(--muted);margin:0 0 22px}.notice{background:#fff7e8;border:1px solid #f0d5a8;border-radius:10px;padding:12px 15px;margin:18px 0}
.controls{display:flex;gap:14px;flex-wrap:wrap;align-items:end;background:white;padding:16px;border:1px solid var(--line);border-radius:12px}.controls label{display:grid;gap:4px;font-weight:650}.controls select{font:inherit;padding:7px 10px;border:1px solid #aab8c7;border-radius:7px;background:white}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}.card,section{background:white;border:1px solid var(--line);border-radius:12px;padding:16px}.card .label{color:var(--muted);font-size:.9rem}.card strong{display:block;font-size:1.35rem;margin-top:3px}section{margin:14px 0}canvas{width:100%;height:320px;display:block}p{color:var(--muted)}.small{font-size:.9rem}table{border-collapse:collapse;width:100%;font-size:.91rem}th,td{text-align:left;border-bottom:1px solid var(--line);padding:8px}th{color:var(--muted)}code{background:#edf2f7;padding:2px 4px;border-radius:4px}
</style></head><body><main>
<h1>Flip Observer: 512 → 1,024</h1><p class="sub">Offline synthetic benchmark · fixed classical weighted readout · exact channel-order reversal</p>
<div class="notice"><strong>Interpretation guardrail:</strong> the second bank repeats the same 512 underlying signal channels with independent synthetic measurement noise/dropout. This tests repeated-readout reliability, not 1,024 independent inputs or any biological/quantum mechanism.</div>
<div class="controls"><label>Noise SD<select id="noise"></select></label><label>Element dropout<select id="drop"></select></label><span id="trial-count" class="small"></span></div>
<div class="cards" id="cards"></div>
<section><h2>Held-out normalized readout error</h2><p>Bars show the mean across seeded trials; error bars show ±1 sample SD. Lower is better.</p><canvas id="chart" width="960" height="340" aria-label="Readout error comparison chart"></canvas><p class="small">Single bank: bank A only. Uncorrected: average A with still-reversed B. Calibrated: choose identity/reversal from separate full-vector calibration then fuse. Oracle: use the known inverse as an upper bound.</p></section>
<section><h2>Calibration and failure checks</h2><div id="details"></div></section>
<section><h2>What the detector can claim</h2><p>It sees known calibration vectors during a separate phase and distinguishes only identity from exact reversal. A scalar output <code>wᵀx</code> is not used to infer the mapping; generally it cannot identify an unknown 512-channel permutation.</p><p class="small">Calibration is not retrained on the held-out mixtures. All signals and values here are dimensionless and synthetic.</p></section>
</main><script>
const DATA = __DATA__;
const conditions=DATA.conditions;
const noiseSel=document.getElementById('noise'),dropSel=document.getElementById('drop');
[...new Set(conditions.map(c=>c.noise_std))].forEach(v=>{let o=document.createElement('option');o.value=v;o.textContent=v.toFixed(2);noiseSel.append(o)});
[...new Set(conditions.map(c=>c.dropout))].forEach(v=>{let o=document.createElement('option');o.value=v;o.textContent=(100*v).toFixed(0)+'%';dropSel.append(o)});
function current(){return conditions.find(c=>c.noise_std===Number(noiseSel.value)&&c.dropout===Number(dropSel.value))||conditions[0]}
function render(){const c=current(),m=c.metrics;document.getElementById('trial-count').textContent=c.trial_count+' seeded trials';
const names=[['Reversal map accuracy',c.reversal_calibration_accuracy],['Identity control',c.identity_calibration_accuracy],['Calibration misses',c.calibration_failure_seeds.length],['Duplicated unique channels',c.trials[0].duplicate_unique_channels]];
document.getElementById('cards').innerHTML=names.map(([a,b])=>`<div class="card"><span class="label">${a}</span><strong>${typeof b==='number'&&b<=1?(100*b).toFixed(0)+'%':b}</strong></div>`).join('');
const labels=[['Single bank','single_bank_nrmse','#718096'],['Uncorrected flip','uncorrected_flip_nrmse','#c46628'],['Calibrated flip','calibrated_nrmse','#2763a5'],['Oracle map','oracle_nrmse','#16877b']];
const canvas=document.getElementById('chart'),ctx=canvas.getContext('2d'),W=canvas.width,H=canvas.height;ctx.clearRect(0,0,W,H);const max=Math.max(...labels.map(x=>m[x[1]].mean+m[x[1]].std),.01)*1.15;const left=60,bottom=H-48,top=24,plotH=bottom-top;ctx.strokeStyle='#d9e2ec';ctx.fillStyle='#536579';ctx.font='13px system-ui';for(let i=0;i<=4;i++){let y=bottom-plotH*i/4;ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(W-15,y);ctx.stroke();ctx.fillText((max*i/4).toFixed(3),8,y+4)}const bw=104,gap=40,start=left+32;labels.forEach(([label,key,color],i)=>{const x=start+i*(bw+gap),v=m[key].mean,e=m[key].std,h=v/max*plotH,eh=e/max*plotH;ctx.fillStyle=color;ctx.fillRect(x,bottom-h,bw,h);ctx.strokeStyle='#172b4d';ctx.beginPath();ctx.moveTo(x+bw/2,bottom-h-eh);ctx.lineTo(x+bw/2,bottom-h+eh);ctx.moveTo(x+bw/2-10,bottom-h-eh);ctx.lineTo(x+bw/2+10,bottom-h-eh);ctx.moveTo(x+bw/2-10,bottom-h+eh);ctx.lineTo(x+bw/2+10,bottom-h+eh);ctx.stroke();ctx.fillStyle='#172b4d';ctx.textAlign='center';ctx.fillText(label,x+bw/2,bottom+20);ctx.fillText(v.toFixed(4)+' ± '+e.toFixed(4),x+bw/2,Math.max(top,bottom-h-eh-7))});ctx.textAlign='start';
document.getElementById('details').innerHTML=`<table><tbody><tr><th>Regime</th><td>noise SD ${c.noise_std.toFixed(2)}, per-entry dropout ${(100*c.dropout).toFixed(0)}%</td></tr><tr><th>Calibration wrong-map seeds</th><td>${c.calibration_failure_seeds.length?c.calibration_failure_seeds.join(', '):'None'}</td></tr><tr><th>Identity-control misses</th><td>${c.identity_control_failure_seeds.length?c.identity_control_failure_seeds.join(', '):'None'}</td></tr><tr><th>Calibrated not better than uncorrected</th><td>${c.calibrated_not_better_than_uncorrected_count} / ${c.trial_count}</td></tr><tr><th>RMSE variability</th><td>Bars report mean ± sample standard deviation across seeds.</td></tr></tbody></table>`}
noiseSel.value=conditions[0].noise_std;dropSel.value=conditions[0].dropout;noiseSel.addEventListener('change',render);dropSel.addEventListener('change',render);render();
</script></body></html>'''.replace("__DATA__", data)
