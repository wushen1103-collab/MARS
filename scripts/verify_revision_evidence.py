from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "revision"


def csv_rows(path: Path) -> int:
    return int(len(pd.read_csv(path)))


def main() -> None:
    checks = {
        "strict_ood_rows": (csv_rows(OUT / "aggregate" / "strict_ood_all_seeds.csv"), 420),
        "strict_ood_confidence_rows": (csv_rows(OUT / "aggregate" / "strict_ood_confidence_all_seeds.csv"), 525),
        "transfer_rows": (csv_rows(OUT / "aggregate" / "transfer_all_seeds.csv"), 135),
        "external_admet_rows": (csv_rows(OUT / "aggregate" / "external_admet_all_seeds.csv"), 90),
        "reliability_rows": (csv_rows(OUT / "aggregate" / "reliability_mean_std.csv"), 5),
        "paired_statistical_comparisons": (csv_rows(OUT / "aggregate" / "paired_cluster_bootstrap_stats.csv"), 30),
        "scalability_bank_sizes": (csv_rows(OUT / "retrieval_scalability" / "retrieval_scalability_summary.csv"), 6),
        "anchor_stratified_metric_rows": (csv_rows(OUT / "anchor_stratified" / "anchor_stratified_metrics.csv"), 630),
        "anchor_stratified_benefit_rows": (csv_rows(OUT / "anchor_stratified" / "anchor_stratified_benefits.csv"), 270),
        "neural_scaffold_rows": (csv_rows(OUT / "neural_multiseed_seed1_5_scaffold_aggregate" / "all_results.csv"), 245),
        "neural_scaffold_missing_files": (
            json.loads((OUT / "neural_multiseed_seed1_5_scaffold_aggregate" / "summary.json").read_text())["num_missing_files"],
            0,
        ),
        "ours_variant_rows": (csv_rows(OUT / "ours_variant_tables_seed1_5" / "ours_variant_main_table.csv"), 28),
        "component_paired_comparisons": (csv_rows(OUT / "component_evidence" / "component_paired_stats.csv"), 28),
        "conformal_base_rows": (csv_rows(OUT / "conformal_risk_control_aggregate" / "base_model_metrics_all_seeds.csv"), 105),
        "conformal_classwise_rows": (csv_rows(OUT / "conformal_risk_control_aggregate" / "classwise_calibration_all_seeds.csv"), 105),
        "conformal_set_rows": (csv_rows(OUT / "conformal_risk_control_aggregate" / "conformal_set_metrics_all_seeds.csv"), 315),
        "risk_control_rows": (csv_rows(OUT / "conformal_risk_control_aggregate" / "risk_control_metrics_all_seeds.csv"), 315),
        "conformal_paired_comparisons": (csv_rows(OUT / "conformal_risk_control_aggregate" / "conformal_risk_control_paired_stats.csv"), 6),
    }
    payload = {
        "checks": {
            name: {"actual": int(actual), "expected": int(expected), "ok": bool(actual == expected)}
            for name, (actual, expected) in checks.items()
        },
        "environment_snapshot_exists": (OUT / "environment_snapshot.txt").exists(),
    }
    payload["all_checks_pass"] = bool(
        all(check["ok"] for check in payload["checks"].values()) and payload["environment_snapshot_exists"]
    )
    (OUT / "revision_evidence_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["all_checks_pass"]:
        raise SystemExit("Revision evidence verification failed")


if __name__ == "__main__":
    main()
