from __future__ import annotations

import torch
from torch_geometric.data import Data

from admet_shift_reliability.conformers import generate_conformer


def smiles_to_3d_pyg_graph(smiles: str, y: float | int | None = None, seed: int = 42) -> Data:
    atomic_numbers, positions = generate_conformer(smiles=smiles, seed=seed)
    data = Data(
        z=torch.tensor(atomic_numbers, dtype=torch.long),
        pos=torch.tensor(positions, dtype=torch.float32),
    )
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float32)
    return data
