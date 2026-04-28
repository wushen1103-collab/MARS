import numpy as np

from admet_shift_reliability.anchor_reliability import (
    compute_anchor_features,
    risk_coverage_auc,
    risk_coverage_curve,
    summarize_anchor_neighbors,
)


def test_summarize_anchor_neighbors_reports_low_disagreement_for_consistent_neighbors():
    summary = summarize_anchor_neighbors(
        neighbor_labels=np.array([1, 1, 1], dtype=np.int64),
        neighbor_distances=np.array([0.0, 0.1, 0.2], dtype=np.float32),
    )

    assert summary["anchor_prob"] > 0.99
    assert summary["anchor_disagreement"] < 1e-6
    assert summary["anchor_distance_min"] == 0.0


def test_compute_anchor_features_shapes_and_mixed_neighbors_raise_disagreement():
    train_x = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 0],
        ],
        dtype=bool,
    )
    train_y = np.array([1, 1, 0, 0], dtype=np.int64)
    query_x = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
        ],
        dtype=bool,
    )

    features = compute_anchor_features(train_x, train_y, query_x, n_neighbors=2)

    assert set(features) == {
        "anchor_prob",
        "anchor_disagreement",
        "anchor_distance_mean",
        "anchor_distance_min",
        "anchor_neighbor_label_mean",
    }
    assert features["anchor_prob"].shape == (2,)
    assert features["anchor_disagreement"][0] < 1e-6
    assert features["anchor_disagreement"][1] > 0.0


def test_risk_coverage_curve_tracks_cumulative_error_rate():
    y_true = np.array([1, 0, 1, 0], dtype=np.int64)
    probs = np.array([0.9, 0.8, 0.6, 0.1], dtype=np.float32)
    confidence = np.array([0.95, 0.2, 0.9, 0.8], dtype=np.float32)

    coverage, risk = risk_coverage_curve(y_true, probs, confidence)

    assert np.allclose(coverage, np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32))
    assert risk[0] == 0.0
    assert risk[-1] == 0.25
    assert 0.0 <= risk_coverage_auc(y_true, probs, confidence) <= 1.0
