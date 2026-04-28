from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_neural_calibration_analysis_script_exports_paper_ready_artifacts():
    source = (ROOT / "scripts" / "analyze_neural_calibration_outputs.py").read_text()

    for marker in [
        "single_run_mean_std.csv",
        "single_run_delta_vs_uncalibrated.csv",
        "deep_ensemble_delta_vs_single.csv",
        "best_reliability_by_task.csv",
        "method_level_summary.csv",
        "headline_summary.json",
        "neural_calibration_report.md",
        "calibration_delta_summary",
        "deep_ensemble_tradeoff",
    ]:
        assert marker in source


def test_neural_calibration_analysis_script_tracks_metric_directions():
    source = (ROOT / "scripts" / "analyze_neural_calibration_outputs.py").read_text()

    assert "LOWER_IS_BETTER" in source
    assert "HIGHER_IS_BETTER" in source
    assert "risk_coverage_auc" in source
    assert "error_detection_auroc" in source
    assert "ece" in source
    assert "auprc" in source
