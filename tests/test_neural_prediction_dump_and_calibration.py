from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_training_scripts_expose_per_sample_prediction_dump_interfaces():
    gin_source = (ROOT / "scripts" / "train_gin_baseline.py").read_text()
    schnet_source = (ROOT / "scripts" / "train_schnet_baseline.py").read_text()

    assert "--prediction-dir" in gin_source
    assert "--mc-dropout-passes" in gin_source
    assert "mc_prob_std" in gin_source
    assert "row_index" in gin_source
    assert "smiles" in gin_source

    assert "--prediction-dir" in schnet_source
    assert "schnet" in schnet_source
    assert "row_index" in schnet_source
    assert "smiles" in schnet_source


def test_true_neural_calibration_aggregator_has_required_calibrators_and_outputs():
    source = (ROOT / "scripts" / "aggregate_neural_calibration.py").read_text()

    for marker in [
        "temperature",
        "platt",
        "isotonic",
        "mc_dropout",
        "calibration_results.csv",
        "deep_ensemble_results.csv",
        "risk_coverage_auc",
        "error_detection_auroc",
    ]:
        assert marker in source


def test_neural_prediction_dump_launcher_targets_true_neural_outputs():
    source = (ROOT / "scripts" / "launch_neural_prediction_dump.py").read_text()

    assert "neural_prediction_dump" in source
    assert "--prediction-dir" in source
    assert "--mc-dropout-passes" in source
    assert "gin" in source
    assert "gat" in source
    assert "mpnn" in source
    assert "schnet" in source
