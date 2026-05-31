from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_followup_experiment_scripts_cover_all_seven_recommended_blocks():
    expected = {
        "run_activity_cliff_novelty.py": ["novelty_bucket_metrics.csv", "activity_cliff_metrics.csv"],
        "run_calibration_benchmark.py": ["calibration_results.csv", "temperature", "isotonic"],
        "run_realistic_ood_splits.py": ["pca_cluster", "molecular_weight_reverse", "fingerprint_density"],
        "run_selective_screening_utility.py": ["selective_coverage_metrics.csv", "enrichment_metrics.csv"],
        "run_strong_descriptor_baselines.py": ["chemprop_rdkit_proxy", "xgb_rdkit"],
        "run_conformer_sensitivity.py": ["num_conformers", "conformer_sensitivity_results.csv"],
        "build_ours_variant_tables.py": ["Ours-Base", "Ours+Anchor", "Ours+Reliability", "Ours-Full"],
    }

    for script_name, markers in expected.items():
        source = (ROOT / "scripts" / script_name).read_text()
        for marker in markers:
            assert marker in source, f"{script_name} missing marker {marker}"


def test_followup_scripts_guard_against_observed_runtime_regressions():
    activity_source = (ROOT / "scripts" / "run_activity_cliff_novelty.py").read_text()
    conformer_source = (ROOT / "scripts" / "run_conformer_sensitivity.py").read_text()

    assert "np.asarray(pd.cut" in activity_source
    assert "np.zeros(12" in conformer_source
    assert "ProcessPoolExecutor" in conformer_source


def test_topjournal_followup_scripts_cover_new_reviewer_gaps():
    expected = {
        "run_strict_ood_model_matrix.py": ["pca_cluster", "fingerprint_density", "molecular_weight_reverse", "rf_ensemble", "learned_shift_error_model"],
        "run_conformal_risk_control.py": ["conformal_set_metrics.csv", "risk_control_metrics.csv", "classwise_calibration_metrics.csv"],
        "run_cross_dataset_transfer.py": ["TRANSFER_PAIRS", "herg_endpoint", "bbb_endpoint"],
        "run_pretrained_smiles_baselines.py": ["DeepChem/ChemBERTa-77M-MTR", "pretrained_smiles_baseline_metrics.csv"],
        "run_chemprop_strict_ood_baseline.py": ["chemprop_strict_ood", "strict-split"],
        "run_anchor_case_studies.py": ["anchor_rescue", "top_anchors"],
        "launch_topjournal_followup_batch.py": ["HF_ENDPOINT", "chemprop_strict_ood_metrics_20260422"],
    }

    for script_name, markers in expected.items():
        source = (ROOT / "scripts" / script_name).read_text()
        for marker in markers:
            assert marker in source, f"{script_name} missing marker {marker}"


def test_strict_ood_legacy_aliases_remain_compatible_with_fixed_artifacts():
    source = (ROOT / "scripts" / "run_realistic_ood_splits.py").read_text()
    assert "make_umap_split" in source
    assert "make_lohi_split" in source
