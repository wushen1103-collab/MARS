import numpy as np

from admet_shift_reliability.reliability_benchmark import (
    binary_entropy_confidence,
    resolve_rf_n_jobs,
    selective_error_at_coverage,
)


def test_binary_entropy_confidence_prefers_extreme_probabilities():
    probs = np.array([0.5, 0.1, 0.9], dtype=np.float32)

    conf = binary_entropy_confidence(probs)

    assert conf.shape == (3,)
    assert conf[0] < conf[1]
    assert conf[0] < conf[2]
    assert np.all((conf >= 0.0) & (conf <= 1.0))


def test_selective_error_at_coverage_keeps_most_confident_examples():
    y_true = np.array([1, 0, 1, 0], dtype=np.int64)
    probs = np.array([0.9, 0.8, 0.6, 0.2], dtype=np.float32)
    confidence = np.array([0.95, 0.3, 0.9, 0.8], dtype=np.float32)

    error, kept = selective_error_at_coverage(y_true, probs, confidence, coverage=0.5)

    assert kept == 2
    assert error == 0.0


def test_resolve_rf_n_jobs_respects_requested_value_and_cpu_cap():
    assert resolve_rf_n_jobs(requested=32, cpu_count=128) == 32
    assert resolve_rf_n_jobs(requested=512, cpu_count=64) == 64
