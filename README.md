# MARS

This repository contains the experiment code for **MARS: Multi-anchor
Reasoning for Reliable Toxicity Prediction Under Distribution Shift**.
The code supports random/scaffold evaluation, strict out-of-distribution
splits, cross-dataset transfer, calibration, conformal risk-control,
anchor sensitivity, and neural baseline experiments.

The strict OOD protocol uses three explicit split families:

- `fingerprint_density`: hold out the lowest 20% of molecules ranked by
  the number of active Morgan-fingerprint bits.
- `molecular_weight_reverse`: hold out the heaviest 20% of molecules.
- `pca_cluster`: project fingerprints to ten principal components,
  cluster the projection with five-means clustering, and hold out the
  smallest cluster.

The historical aliases `lohi` and `umap` remain accepted only so that
fixed pre-revision artifacts can still be regenerated. They map to
`fingerprint_density` and `pca_cluster`, respectively.

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

The seven-task main benchmark expects the public MoleculeNet files
`BBBP.csv`, `clintox.csv.gz`, and `tox21.csv.gz` under `data/raw/`.
The AMES, hERG, and DILI tasks are fetched through Therapeutics Data
Commons and cached under `data/raw/`. External ADMET probes and
cross-dataset transfer sources are fetched through Therapeutics Data
Commons and cached under `data/tdc_external/`.

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

Run the strict OOD model matrix with the manuscript split names and
five fixed seeds:

```bash
python scripts/run_strict_ood_model_matrix.py \
  --splits fingerprint_density,molecular_weight_reverse,pca_cluster \
  --seeds 42,43,44,45,46 \
  --output-dir outputs/strict_ood_model_matrix
```

The strict OOD and transfer `Ours-Anchor` rows use radius-2, 2048-bit
Morgan fingerprints, a class-balanced 500-tree random forest, exact
top-15 anchor retrieval, and class-balanced logistic reasoning and
reliability models fitted from validation predictions.

Run the additional revision analyses:

```bash
python scripts/run_anchor_stratified_analysis.py \
  --seeds 42,43,44,45,46 \
  --output-dir outputs/anchor_stratified_analysis
python scripts/summarize_anchor_stratified_analysis.py \
  --input-dir outputs/anchor_stratified_analysis
python scripts/run_retrieval_scalability.py \
  --output-dir outputs/retrieval_scalability
```

When the fixed rerun shards are placed under
`outputs/revision_20260531/`, generate the statistical summaries and
verify their row-level coverage with:

```bash
python scripts/aggregate_revision_evidence.py
python scripts/aggregate_component_evidence.py
python scripts/aggregate_conformal_risk_control_multiseed.py
python scripts/verify_revision_evidence.py
```

The manuscript statistical audit uses paired endpoint-cluster bootstrap
intervals with 10,000 resamples, two-sided Wilcoxon signed-rank tests on
endpoint-cluster means, and Benjamini-Hochberg correction.

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
