from __future__ import annotations

import torch
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.utils.smiles import from_smiles


def smiles_to_pyg_graph(smiles: str, y: int | float | None = None) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    data = from_smiles(smiles)
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float32)
    return data
