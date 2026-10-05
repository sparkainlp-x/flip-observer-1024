"""Classical synthetic channel-routing primitives for Flip Observer."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CalibrationResult:
    """Result of distinguishing the documented identity/reversal hypotheses."""

    mapping: str
    identity_mse: float
    reversed_mse: float
    confidence_gap: float
    observed_fraction: float


def reverse_channels(values: np.ndarray) -> np.ndarray:
    """Apply the exact channel-order reversal P along the final axis."""
    array = np.asarray(values)
    if array.ndim == 0:
        raise ValueError("values must have a channel axis")
    return np.flip(array, axis=-1).copy()


def readout(channels: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Ordinary classical weighted sum z(t) = w^T x(t)."""
    values = np.asarray(channels, dtype=float)
    vector = np.asarray(weights, dtype=float)
    if values.shape[-1] != vector.size:
        raise ValueError("channel count and weight count differ")
    return values @ vector


def fixed_weights(channel_count: int = 512, seed: int = 1024) -> np.ndarray:
    """Create one deterministic, unit-L2 observer vector, fixed before flips."""
    rng = np.random.default_rng(seed)
    weights = rng.normal(size=channel_count)
    return weights / np.linalg.norm(weights)


def hadamard_calibration(channel_count: int = 512) -> np.ndarray:
    """Return orthogonal, known +/-1 calibration patterns (rows are trials)."""
    if channel_count < 1 or channel_count & (channel_count - 1):
        raise ValueError("channel_count must be a positive power of two")
    matrix = np.ones((1, 1), dtype=float)
    while matrix.shape[0] < channel_count:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix


def calibrate_identity_vs_reversal(
    expected: np.ndarray,
    observed: np.ndarray,
    observed_mask: np.ndarray | None = None,
) -> CalibrationResult:
    """Use full channel vectors from known calibration patterns to choose I or P.

    This detector deliberately solves only the two stated hypotheses: identity
    and exact channel-order reversal. It does not claim to identify an arbitrary
    unknown permutation. A missing calibration element is excluded by the mask.
    """
    expected = np.asarray(expected, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if expected.shape != observed.shape or expected.ndim != 2:
        raise ValueError("expected and observed must be equal 2-D sample-by-channel arrays")
    if observed_mask is None:
        mask = np.ones(expected.shape, dtype=bool)
    else:
        mask = np.asarray(observed_mask, dtype=bool)
        if mask.shape != expected.shape:
            raise ValueError("observed_mask must have the same shape as calibration arrays")
    count = int(mask.sum())
    if count == 0:
        raise ValueError("calibration has no observed entries")

    identity_residual = np.where(mask, observed - expected, 0.0)
    reverse_residual = np.where(mask, observed - reverse_channels(expected), 0.0)
    identity_mse = float(np.square(identity_residual).sum() / count)
    reversed_mse = float(np.square(reverse_residual).sum() / count)
    mapping = "reversed" if reversed_mse < identity_mse else "identity"
    best = min(identity_mse, reversed_mse)
    worst = max(identity_mse, reversed_mse)
    gap = (worst - best) / max(worst, np.finfo(float).eps)
    return CalibrationResult(
        mapping=mapping,
        identity_mse=identity_mse,
        reversed_mse=reversed_mse,
        confidence_gap=float(gap),
        observed_fraction=float(count / mask.size),
    )


def undo_reversal(observed: np.ndarray) -> np.ndarray:
    """Apply the exact inverse P^-1; for reversal, P^-1 = P."""
    return reverse_channels(observed)
