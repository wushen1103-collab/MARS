import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "launch_neural_multiseed.py"


def load_launcher_module():
    spec = importlib.util.spec_from_file_location("launch_neural_multiseed", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_multiseed_plan_covers_core_and_anchor_without_overwriting_main_outputs():
    launcher = load_launcher_module()

    jobs = launcher.build_jobs(seeds=[1, 2, 3], include_random=True, include_anchor=True)

    assert len(jobs) == 189
    assert sum(1 for job in jobs if job.stage == "core_scaffold") == 84
    assert sum(1 for job in jobs if job.stage == "anchor_scaffold") == 21
    assert sum(1 for job in jobs if job.stage == "core_random") == 84
    assert all("outputs/neural_multiseed" in job.output_dir for job in jobs)
    assert all("outputs/gin_baseline" not in job.command for job in jobs)
    assert all("outputs/schnet_baseline" not in job.command for job in jobs)
    assert all(f"seed{job.seed}" in job.output_dir for job in jobs)
