"""후보 필터 (설정 가능) + Protic 구조 규칙 표시.

* `FilterConfig` : 유효성 / 중복 / 최소 중원자 수 / 전하 / 라디칼 / 조각 수 / 헤테로원자 비율 조건
* `PRESETS`      : 생성 대상에 따라 다른 조건
    - "anion"   : 리튬염 음이온 후보 (단일 조각, 총 전하 -1)
    - "salt"    : 완성된 염 (양이온.음이온 2조각, 총 전하 0)
    - "neutral" : 중성 분자 (용매·첨가제, 단일 조각, 전하 0)
* `apply_filters` : 각 분자의 통과 여부와 **제외 이유** 를 DataFrame 으로 반환
* `protic_flags`  : Protic 분자 판별에 쓰는 구조 규칙(SMARTS) 을 별도로 표시.
    -> 이는 "O-H / N-H 등 산성 양성자를 가진 구조" 라는 **구조적 표시**일 뿐,
       수계 전해질로 실제 사용 가능한지에 대한 검증이 아니다. (anion 과 같은 기준으로 다루지 않는다)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")


@dataclass
class FilterConfig:
    name: str = "anion"
    min_heavy_atoms: int = 3            # [N-], [P-] 같은 단일/극소 원자 구조 제외
    max_heavy_atoms: int = 60
    allowed_total_charge: tuple = (-1,)  # 분자 전체 전하 허용값
    allow_radicals: bool = False
    n_fragments: tuple = (1,)           # '.' 로 분리된 조각 수 허용값
    min_hetero_frac: float = 0.25       # 중원자 중 헤테로원자 비율
    require_fragment_charges: tuple | None = None  # salt: 조각별 전하 집합 (예: (+1, -1))
    exclude_train_set: bool = True      # 학습 데이터에 이미 있는 구조 제외
    note: str = ""

    def to_dict(self):
        return asdict(self)


PRESETS: dict[str, FilterConfig] = {
    "anion": FilterConfig(name="anion", allowed_total_charge=(-1,), n_fragments=(1,),
                          note="리튬염 음이온 후보: 단일 조각, 총 전하 -1"),
    "salt": FilterConfig(name="salt", allowed_total_charge=(0,), n_fragments=(2,),
                         require_fragment_charges=(1, -1), min_heavy_atoms=4,
                         note="완성된 염: 양이온+음이온 2조각, 총 전하 0"),
    "neutral": FilterConfig(name="neutral", allowed_total_charge=(0,), n_fragments=(1,), min_hetero_frac=0.2,
                            note="중성 분자(용매·첨가제): 단일 조각, 전하 0"),
}

# Protic 판별용 구조 규칙 — 구조 표시 목적 (실제 사용 가능성 검증 아님)
PROTIC_RULES = {
    "O-H (hydroxyl/acid)": "[OX2H]",
    "N-H": "[NX3;H1,H2,H3]",
    "N+-H": "[N+;!H0]",
    "S-H": "[SX2H]",
    "P-OH": "[PX4][OX2H]",
    "B-OH": "[B][OX2H]",
}
_PROTIC = {k: Chem.MolFromSmarts(v) for k, v in PROTIC_RULES.items()}


def protic_flags(smiles: str) -> dict:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return {"is_protic_rule": False, "protic_rules_matched": ""}
    hit = [k for k, p in _PROTIC.items() if m.HasSubstructMatch(p)]
    return {"is_protic_rule": bool(hit), "protic_rules_matched": "; ".join(hit)}


def _check_one(smiles: str, cfg: FilterConfig, seen: set, train_set: set) -> dict:
    reasons = []
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return {"smiles": smiles, "canonical": None, "passed": False, "reasons": "invalid_smiles"}
    can = Chem.MolToSmiles(m)
    if can in seen:
        reasons.append("duplicate")
    seen.add(can)
    if cfg.exclude_train_set and can in train_set:
        reasons.append("in_train_set")
    n_heavy = m.GetNumHeavyAtoms()
    if n_heavy < cfg.min_heavy_atoms:
        reasons.append(f"too_small(heavy<{cfg.min_heavy_atoms})")
    if n_heavy > cfg.max_heavy_atoms:
        reasons.append(f"too_large(heavy>{cfg.max_heavy_atoms})")
    charge = Chem.GetFormalCharge(m)
    if charge not in cfg.allowed_total_charge:
        reasons.append(f"charge={charge}∉{cfg.allowed_total_charge}")
    n_rad = Descriptors.NumRadicalElectrons(m)
    if n_rad > 0 and not cfg.allow_radicals:
        reasons.append("radical_electrons>0")
    frags = Chem.GetMolFrags(m, asMols=True)
    if len(frags) not in cfg.n_fragments:
        reasons.append(f"fragments={len(frags)}∉{cfg.n_fragments}")
    elif cfg.require_fragment_charges is not None:
        fc = sorted(Chem.GetFormalCharge(f) for f in frags)
        if fc != sorted(cfg.require_fragment_charges):
            reasons.append(f"fragment_charges={fc}≠{sorted(cfg.require_fragment_charges)}")
    hetero = sum(a.GetSymbol() not in ("C", "H") for a in m.GetAtoms()) / max(n_heavy, 1)
    if hetero < cfg.min_hetero_frac:
        reasons.append(f"hetero_frac<{cfg.min_hetero_frac}")
    return {"smiles": smiles, "canonical": can, "passed": not reasons, "reasons": "; ".join(reasons),
            "n_heavy": n_heavy, "total_charge": charge, "n_fragments": len(frags), "hetero_frac": round(hetero, 3)}


FILTER_COLUMNS = ["smiles", "canonical", "passed", "reasons", "n_heavy", "total_charge", "n_fragments",
                  "hetero_frac", "is_protic_rule", "protic_rules_matched"]


def apply_filters(smiles_list: list[str], cfg: FilterConfig, train_set: set | None = None) -> pd.DataFrame:
    """모든 분자에 대해 통과 여부(passed)와 제외 이유(reasons)를 기록해 반환.
    입력이 비어 있어도 FILTER_COLUMNS 스키마(열 순서·dtype)를 유지한다."""
    seen, train_set = set(), (train_set or set())
    rows = [_check_one(s, cfg, seen, train_set) for s in smiles_list]
    df = pd.DataFrame(rows, columns=FILTER_COLUMNS[:8])
    pf = pd.DataFrame([protic_flags(c) if isinstance(c, str) else protic_flags("") for c in df["canonical"]],
                      columns=["is_protic_rule", "protic_rules_matched"])
    df = pd.concat([df, pf], axis=1)
    df["passed"] = df["passed"].astype(bool)
    df["is_protic_rule"] = df["is_protic_rule"].astype(bool)
    df["reasons"] = df["reasons"].fillna("").astype(str)
    df["protic_rules_matched"] = df["protic_rules_matched"].fillna("").astype(str)
    return df[FILTER_COLUMNS]


def filter_summary(df: pd.DataFrame) -> pd.DataFrame:
    """제외 이유별 건수 요약."""
    if len(df) == 0:
        return pd.DataFrame([{"reason": "PASSED", "count": 0}])
    reasons = df.loc[~df["passed"], "reasons"].str.split("; ").explode()
    reasons = reasons[reasons != ""]
    out = reasons.value_counts().rename_axis("reason").reset_index(name="count")
    return pd.concat([pd.DataFrame([{"reason": "PASSED", "count": int(df["passed"].sum())}]), out],
                     ignore_index=True)
