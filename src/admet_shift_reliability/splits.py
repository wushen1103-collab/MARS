from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


@dataclass(slots=True)
class BemisMurckoScaffoldSplitter:
    valid_frac: float = 0.1
    test_frac: float = 0.1

    def __post_init__(self) -> None:
        if not 0 <= self.valid_frac < 1:
            raise ValueError("valid_frac must be in [0, 1).")
        if not 0 <= self.test_frac < 1:
            raise ValueError("test_frac must be in [0, 1).")
        if self.valid_frac + self.test_frac >= 1:
            raise ValueError("valid_frac + test_frac must be < 1.")

    def split(self, smiles: list[str]) -> dict[str, list[int]]:
        scaffold_to_indices: dict[str, list[int]] = {}
        for idx, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"Invalid SMILES at index {idx}: {smi}")
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or "__NO_SCAFFOLD__"
            scaffold_to_indices.setdefault(scaffold, []).append(idx)

        groups = sorted(
            scaffold_to_indices.values(),
            key=lambda values: (-len(values), values[0]),
        )

        total = len(smiles)
        valid_target = int(round(total * self.valid_frac))
        test_target = int(round(total * self.test_frac))

        split = {"train": [], "valid": [], "test": []}

        valid_count = 0
        test_count = 0
        for group in groups:
            if test_count < test_target:
                split["test"].extend(group)
                test_count += len(group)
            elif valid_count < valid_target:
                split["valid"].extend(group)
                valid_count += len(group)
            else:
                split["train"].extend(group)

        for name in split:
            split[name].sort()

        return split
