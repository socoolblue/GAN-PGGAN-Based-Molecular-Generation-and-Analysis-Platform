"""생성 분자 평가 지표: validity / uniqueness / novelty + 기능기 검출."""
from __future__ import annotations

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

# 수계 전해질 / 리튬염 관점에서 의미 있는 기능기 (SMARTS)
FUNCTIONAL_GROUPS = {
    "sulfonyl": "S(=O)(=O)",
    "sulfonimide_N-": "[N-](S(=O)(=O))S(=O)(=O)",
    "sulfonate": "S(=O)(=O)[O-,OH]",
    "CF3": "C(F)(F)F",
    "fluoro": "[F]",
    "carbonate": "O=C(O)O",
    "ether": "[CX4][OX2][CX4]",
    "hydroxyl": "[OX2H]",
    "carboxyl": "C(=O)[O-,OH]",
    "nitrile": "C#N",
    "amide": "C(=O)N",
    "borate": "[B-]",
    "phosphate": "P(=O)(O)(O)O",
    "anion": "[-]",
    "cation": "[+]",
    "quaternary_N": "[N+;X4]",
    "imidazolium": "c1c[n+]cn1",
    "ring": "[R]",
}
_FG_PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in FUNCTIONAL_GROUPS.items()}


def canonical(smiles: str) -> str | None:
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m) if m is not None else None


def evaluate_samples(samples: list[str], train_set: set[str]) -> dict:
    valid = [c for c in (canonical(s) for s in samples) if c]
    uniq = set(valid)
    novel = uniq - train_set
    n = max(len(samples), 1)
    return {
        "validity": len(valid) / n,
        "uniqueness": len(uniq) / max(len(valid), 1),
        "novelty": len(novel) / max(len(uniq), 1),
        "n_valid": len(valid), "n_unique": len(uniq), "n_novel": len(novel),
        "valid_smiles": sorted(uniq),
    }


def functional_groups(smiles: str) -> dict[str, int]:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return {k: 0 for k in FUNCTIONAL_GROUPS}
    return {k: len(m.GetSubstructMatches(p)) for k, p in _FG_PATTERNS.items()}


def descriptors(smiles: str) -> dict[str, float]:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return {}
    return {
        "MolWt": Descriptors.MolWt(m),
        "LogP": Descriptors.MolLogP(m),
        "TPSA": rdMolDescriptors.CalcTPSA(m),
        "HBD": rdMolDescriptors.CalcNumHBD(m),
        "HBA": rdMolDescriptors.CalcNumHBA(m),
        "RotB": rdMolDescriptors.CalcNumRotatableBonds(m),
        "HeavyAtoms": m.GetNumHeavyAtoms(),
        "FormalCharge": Chem.GetFormalCharge(m),
        "nF": sum(a.GetSymbol() == "F" for a in m.GetAtoms()),
        "nO": sum(a.GetSymbol() == "O" for a in m.GetAtoms()),
        "nN": sum(a.GetSymbol() == "N" for a in m.GetAtoms()),
        "nS": sum(a.GetSymbol() == "S" for a in m.GetAtoms()),
    }
