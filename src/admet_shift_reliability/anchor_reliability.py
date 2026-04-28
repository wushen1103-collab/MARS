from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


def summarize_anchor_neighbors(
    neighbor_labels: np.ndarray,
    neighbor_distances: np.ndarray,
) -> dict[str, float]:
    if neighbor_labels.size == 0:
        raise ValueError("neighbor_labels must be non-empty")
    if neighbor_labels.shape != neighbor_distances.shape:
        raise ValueError("neighbor_labels and neighbor_distances must have the same shape")

    similarities = 1.0 - neighbor_distances.astype(np.float64)
    weights = np.clip(similarities, a_min=1e-6, a_max=None)
    weight_sum = float(np.sum(weights))
    anchor_prob = float(np.sum(weights * neighbor_labels) / weight_sum)
    anchor_disagreement = float(np.sum(weights * np.square(neighbor_labels - anchor_prob)) / weight_sum)
    return {
        "anchor_prob": anchor_prob,
        "anchor_disagreement": anchor_disagreement,
        "anchor_distance_mean": float(np.mean(neighbor_distances)),
        "anchor_distance_min": float(np.min(neighbor_distances)),
        "anchor_neighbor_label_mean": float(np.mean(neighbor_labels)),
    }


def compute_anchor_features(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    n_neighbors: int = 15,
) -> dict[str, np.ndarray]:
    if train_x.shape[0] != train_y.shape[0]:
        raise ValueError("train_x and train_y must align on the first axis")
    if train_x.shape[0] == 0:
        raise ValueError("train_x must be non-empty")

    k = max(1, min(int(n_neighbors), int(train_x.shape[0])))
    nbrs = NearestNeighbors(metric="jaccard", algorithm="brute", n_neighbors=k)
    nbrs.fit(train_x.astype(bool))
    distances, indices = nbrs.kneighbors(query_x.astype(bool), return_distance=True)

    keys = [
        "anchor_prob",
        "anchor_disagreement",
        "anchor_distance_mean",
        "anchor_distance_min",
        "anchor_neighbor_label_mean",
    ]
    outputs = {key: np.zeros(query_x.shape[0], dtype=np.float32) for key in keys}

    for row_idx in range(query_x.shape[0]):
        summary = summarize_anchor_neighbors(
            neighbor_labels=train_y[indices[row_idx]].astype(np.float64),
            neighbor_distances=distances[row_idx].astype(np.float64),
        )
        for key in keys:
            outputs[key][row_idx] = summary[key]

    return outputs


def risk_coverage_curve(
    y_true: np.ndarray,
    probs: np.ndarray,
    confidence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(np.int64)
    probs = np.asarray(probs).astype(np.float64)
    confidence = np.asarray(confidence).astype(np.float64)

    if not (len(y_true) == len(probs) == len(confidence)):
        raise ValueError("y_true, probs, and confidence must have the same length")

    pred = (probs >= 0.5).astype(np.int64)
    correct = (pred == y_true).astype(np.float64)
    order = np.argsort(-confidence, kind="mergesort")
    ordered_correct = correct[order]
    coverage = np.arange(1, len(y_true) + 1, dtype=np.float32) / float(len(y_true))
    risk = 1.0 - (np.cumsum(ordered_correct) / np.arange(1, len(y_true) + 1))
    return coverage, risk.astype(np.float32)


def risk_coverage_auc(
    y_true: np.ndarray,
    probs: np.ndarray,
    confidence: np.ndarray,
) -> float:
    coverage, risk = risk_coverage_curve(y_true=y_true, probs=probs, confidence=confidence)
    trapz = getattr(np, "trapezoid", np.trapz)
    return float(trapz(risk, coverage))


def error_detection_auroc(
    y_true: np.ndarray,
    probs: np.ndarray,
    confidence: np.ndarray,
) -> float:
    y_true = np.asarray(y_true).astype(np.int64)
    probs = np.asarray(probs).astype(np.float64)
    confidence = np.asarray(confidence).astype(np.float64)
    errors = ((probs >= 0.5).astype(np.int64) != y_true).astype(np.int64)
    if np.unique(errors).size < 2:
        return float("nan")
    return float(roc_auc_score(errors, 1.0 - confidence))
