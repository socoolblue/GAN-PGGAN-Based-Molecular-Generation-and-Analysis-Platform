"""생성된 리튬염/전해질 후보의 특성 평가용 머신러닝 모델.

* 입력 : RDKit 디스크립터 + Morgan fingerprint + 기능기 카운트
* 모델 : RandomForestRegressor
* 타깃 : 데이터 CSV 의 `target` 열 (예: 측정 이온전도도).
         target 이 비어 있으면 `proxy_score` (탐색용 휴리스틱 점수, 실제 물성 예측값 아님) 로 대체.
         실험 결과가 쌓이면 feedback.py 로 재학습한다.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from .metrics import descriptors, functional_groups

FP_BITS = 512


def featurize(smiles: str) -> np.ndarray | None:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    fp = np.zeros(FP_BITS, dtype=np.float32)
    bv = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=FP_BITS)
    from rdkit import DataStructs
    DataStructs.ConvertToNumpyArray(bv, fp)
    d = descriptors(smiles); g = functional_groups(smiles)
    return np.concatenate([fp, np.array(list(d.values()), np.float32), np.array(list(g.values()), np.float32)])


def proxy_score(smiles: str) -> float:
    """탐색용 휴리스틱 점수: 수계 리튬염 음이온 관련 구조 특징의 가중합 (0~1).
    실제 측정값(target)이 없을 때만 사용. 약배위(weakly-coordinating) 음이온 특징·
    물 친화성·분자 크기를 단순 가중합한 값이며, 실험 데이터로 반드시 대체해야 한다."""
    d, g = descriptors(smiles), functional_groups(smiles)
    if not d:
        return 0.0
    s = 0.0
    s += 0.30 * (d["FormalCharge"] == -1)                     # 단일 음이온
    s += 0.20 * min(g["sulfonyl"], 2) / 2                       # 전하 비편재화 (설포닐)
    s += 0.15 * min(g["CF3"] + 0.5 * g["fluoro"], 6) / 6        # 불소화 -> 약배위, 전기화학 안정성
    s += 0.15 * (1 - min(abs(d["TPSA"] - 60) / 60, 1))          # 적정 극성 (수용성)
    s += 0.10 * (1 - min(abs(d["MolWt"] - 250) / 250, 1))       # 적정 분자량
    s += 0.10 * (g["cation"] == 0)                              # 양이온 아님
    return round(float(s), 4)


class PropertyModel:
    def __init__(self, seed: int = 0):
        self.rf = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
        self.target_name = "proxy_score"
        self.score_type = "heuristic_proxy"   # "heuristic_proxy" | "ml_from_measured"
        self.n_train = 0
        self.cv_r2 = None

    @property
    def score_label(self) -> str:
        """결과표에 쓰는 점수 이름 — 실측 학습 여부를 이름에 명시."""
        return ("heuristic_exploration_score" if self.score_type == "heuristic_proxy"
                else f"ml_pred_{self.target_name}")

    def describe(self) -> str:
        if self.score_type == "heuristic_proxy":
            return ("탐색용 휴리스틱 점수 (proxy_score): 실측 데이터를 학습하지 않은 규칙 기반 점수. "
                    "실제 물성 예측값이 아님.")
        return f"실측 '{self.target_name}' {self.n_train}건으로 학습한 RandomForest 예측값 (CV R2={self.cv_r2})"

    def fit(self, df: pd.DataFrame) -> "PropertyModel":
        has_target = "target" in df and df["target"].notna().any()
        if has_target:
            d = df.dropna(subset=["target"])
            y = d["target"].astype(float).values; self.target_name = "target"; self.score_type = "ml_from_measured"
        else:
            d = df; y = np.array([proxy_score(s) for s in d["smiles"]]); self.score_type = "heuristic_proxy"
        self.n_train = len(d)
        X = np.stack([featurize(s) for s in d["smiles"]])
        if len(d) >= 10:
            self.cv_r2 = round(float(cross_val_score(self.rf, X, y, cv=min(5, len(d)), scoring="r2").mean()), 3)
        self.rf.fit(X, y)
        return self

    def predict(self, smiles: list[str]) -> np.ndarray:
        X = np.stack([featurize(s) for s in smiles])
        return self.rf.predict(X)

    def save(self, path: Path):
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "PropertyModel":
        return joblib.load(path)
