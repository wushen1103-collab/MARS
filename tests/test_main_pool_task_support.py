import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
MAIN_POOL = {"bbbp", "clintox", "tox21", "ames", "herg", "dili"}


def _load_module_ast(script_name: str) -> ast.Module:
    return ast.parse((SCRIPTS / script_name).read_text(encoding="utf-8"))


def _extract_dataset_config_keys(script_name: str, assign_name: str) -> set[str]:
    tree = _load_module_ast(script_name)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == assign_name and isinstance(node.value, ast.Dict):
                    keys = set()
                    for key in node.value.keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            keys.add(key.value)
                    return keys
    raise AssertionError(f"Could not find dict assignment {assign_name!r} in {script_name}")


def _extract_task_datasets(script_name: str) -> set[str]:
    tree = _load_module_ast(script_name)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TASKS" and isinstance(node.value, ast.List):
                    datasets = set()
                    for task_node in node.value.elts:
                        if not isinstance(task_node, ast.Dict):
                            continue
                        for key_node, value_node in zip(task_node.keys, task_node.values):
                            if (
                                isinstance(key_node, ast.Constant)
                                and key_node.value == "dataset"
                                and isinstance(value_node, ast.Constant)
                                and isinstance(value_node.value, str)
                            ):
                                datasets.add(value_node.value)
                    return datasets
    raise AssertionError(f"Could not find TASKS assignment in {script_name}")


def test_gnn_and_3d_training_scripts_cover_main_pool():
    for script_name in ("train_gin_baseline.py", "train_schnet_baseline.py", "run_gin_embedding_anchor_probe.py"):
        supported = _extract_dataset_config_keys(script_name, "DATASET_CONFIGS")
        assert MAIN_POOL.issubset(supported), f"{script_name} missing {sorted(MAIN_POOL - supported)}"


def test_anchor_probe_scripts_cover_main_pool():
    for script_name in ("run_reliability_benchmark.py", "run_anchor_reliability_probe.py", "run_anchor_hybrid_probe.py"):
        supported = _extract_task_datasets(script_name)
        assert MAIN_POOL.issubset(supported), f"{script_name} missing {sorted(MAIN_POOL - supported)}"

