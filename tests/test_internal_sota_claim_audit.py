from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_internal_sota_claim_audit_script_exports_guardrail_tables():
    source = (ROOT / "scripts" / "build_internal_sota_claim_audit.py").read_text()

    for marker in [
        "internal_sota_audit_by_task.csv",
        "baseline_pool.csv",
        "candidate_pool.csv",
        "claim_audit_summary.json",
        "claim_recommendations.md",
        "performance_win_rate",
        "reliability_win_rate",
        "Do not claim universal SOTA",
    ]:
        assert marker in source
