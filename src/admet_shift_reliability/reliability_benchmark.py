from __future__ import annotations

import math

import numpy as np


def resolve_rf_n_jobs(requested: int | None = None, cpu_count: int | None = None) -> int:
    available = max(1, int(cpu_count or 1))
    if requested is None:
        return max(1, min(192, available - 8))
    return max(1, min(int(requested), available))


def binary_entropy_confidence(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 1e-8, 1.0 - 1e-8)
    entropy = -(probs * np.log(probs) + (1.0 - probs) * np.log(1.0 - probs)) / math.log(2.0)
    return (1.0 - entropy).astype(np.float32)


def selective_error_at_coverage(
    y_true: np.ndarray,
    probs: np.ndarray,
    confidence: np.ndarray,
    coverage: float,
) -> tuple[float, int]:
    y_true = np.asarray(y_true).astype(np.int64)
    probs = np.asarray(probs).astype(np.float64)
    confidence = np.asarray(confidence).astype(np.float64)
    if not (len(y_true) == len(probs) == len(confidence)):
        raise ValueError("y_true, probs, and confidence must have the same length")
    if not (0.0 < coverage <= 1.0):
        raise ValueError("coverage must be in (0, 1]")

    keep = max(1, int(round(len(y_true) * coverage)))
    order = np.argsort(-confidence, kind="mergesort")[:keep]
    pred = (probs[order] >= 0.5).astype(np.int64)
    error = float(np.mean(pred != y_true[order]))
    return error, keep
