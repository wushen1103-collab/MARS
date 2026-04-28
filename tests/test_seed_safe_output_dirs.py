from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_long_running_neural_scripts_accept_output_dir_to_avoid_seed_overwrites():
    scripts = [
        ROOT / "scripts" / "train_gin_baseline.py",
        ROOT / "scripts" / "train_schnet_baseline.py",
        ROOT / "scripts" / "run_gin_embedding_anchor_probe.py",
    ]

    for script in scripts:
        source = script.read_text()
        assert "--output-dir" in source, f"{script.name} is missing --output-dir"
        assert "args.output_dir" in source, f"{script.name} does not write through args.output_dir"
