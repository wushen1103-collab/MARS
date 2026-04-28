from __future__ import annotations

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

rdBase.DisableLog("rdApp.warning")
rdBase.DisableLog("rdApp.error")


def _optimize_conformer(mol: Chem.Mol, conf_id: int) -> None:
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
        return

    AllChem.UFFOptimizeMolecule(mol, confId=conf_id)


def generate_conformer(smiles: str, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    conf_id = AllChem.EmbedMolecule(mol, params)
    if conf_id < 0:
        params.useRandomCoords = True
        conf_id = AllChem.EmbedMolecule(mol, params)
    if conf_id < 0:
        raise ValueError(f"Unable to generate conformer for SMILES: {smiles}")

    _optimize_conformer(mol, conf_id)
    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()
    atomic_numbers = np.asarray([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int64)
    positions = np.asarray(
        [
            [conf.GetAtomPosition(idx).x, conf.GetAtomPosition(idx).y, conf.GetAtomPosition(idx).z]
            for idx in range(mol.GetNumAtoms())
        ],
        dtype=np.float32,
    )
    return atomic_numbers, positions
