# MARS

This repository contains the experiment code for **MARS: Multi-anchor
Reasoning for Reliable Toxicity Prediction Under Distribution Shift**.
The code supports random/scaffold evaluation, strict out-of-distribution
splits, cross-dataset transfer, calibration, conformal risk-control,
anchor sensitivity, and neural baseline experiments.

## Repository layout

- `src/admet_shift_reliability/`: reusable dataset, split, feature,
  anchor-reliability, graph, and model utilities.
- `scripts/`: runnable experiment and aggregation entry points.
- `tests/`: regression tests for split construction, featurization,
  reliability utilities, launch plans, and result aggregation.

Generated artifacts are intentionally excluded from Git. This includes
`data/`, `outputs/`, `logs/`, local environments, caches, checkpoints,
and model weights.

## Environment

Use Python 3.10 or newer. A minimal CPU environment for fingerprint and
reliability experiments can be installed with:

```bash
python -m pip install -e ".[dev,tdc]"
```

Neural and optional baselines require additional extras:

```bash
python -m pip install -e ".[dev,tdc,neural,chemprop,smiles,plot]"
```

For RDKit and PyTorch Geometric, a conda-based installation may be more
stable on some systems.

## Data

The repository does not vendor public benchmark datasets. Place raw
tables under `data/raw/` or let scripts that use Therapeutics Data
Commons fetch and cache datasets locally. Generated split files, cached
3D conformers, logs, and results should remain untracked.

Typical local layout:

```text
data/
  raw/
  tdc_external/
outputs/
logs/
```

## Examples

Run quick fingerprint baselines:

```bash
python scripts/quick_fingerprint_baselines.py --output-dir outputs/fingerprint_baselines
```

Run anchor number sensitivity:

```bash
python scripts/run_anchor_k_sensitivity.py --output-dir outputs/anchor_k_sensitivity
```

Plan neural multiseed jobs without launching workers:

```bash
python scripts/launch_neural_multiseed.py
```

Use `--start` only when the required data, dependencies, and GPU/CPU
resources are available.

## Tests

```bash
python -m pytest
```

