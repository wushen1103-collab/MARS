from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_anchor_k_sensitivity_script_has_paper_required_k_values_and_isolated_output():
    script = ROOT / "scripts" / "run_anchor_k_sensitivity.py"
    source = script.read_text()

    assert "1,3,5,10" in source
    assert "anchor_k_sensitivity" in source
    assert "results.csv" in source
