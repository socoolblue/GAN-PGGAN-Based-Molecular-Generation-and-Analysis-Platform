"""SMILES 토크나이저 / 데이터 증강 / PyTorch Dataset."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from torch.utils.data import Dataset

RDLogger.DisableLog("rdApp.*")

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
# 다문자 토큰(Cl, Br, [Li+], [N-] ...) 을 하나의 토큰으로 묶는 정규식
_TOKEN_RE = re.compile(
    r"(\[[^\]]+\]|Br|Cl|Si|@@|%\d{2}|[A-Za-z]|\d|\(|\)|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%|\.)"
)


def tokenize(smiles: str) -> list[str]:
    toks = _TOKEN_RE.findall(smiles)
    assert "".join(toks) == smiles, f"tokenization mismatch: {smiles}"
    return toks


@dataclass
class SmilesVocab:
    itos: list[str] = field(default_factory=list)
    stoi: dict[str, int] = field(default_factory=dict)

    @classmethod
    def build(cls, smiles_list: list[str]) -> "SmilesVocab":
        tokens = set()
        for s in smiles_list:
            tokens.update(tokenize(s))
        itos = [PAD, BOS, EOS] + sorted(tokens)
        return cls(itos=itos, stoi={t: i for i, t in enumerate(itos)})

    def __len__(self):
        return len(self.itos)

    @property
    def pad_idx(self):
        return self.stoi[PAD]

    def encodable(self, smiles: str, max_len: int | None = None) -> bool:
        toks = tokenize(smiles)
        return all(t in self.stoi for t in toks) and (max_len is None or len(toks) + 2 <= max_len)

    def encode(self, smiles: str, max_len: int) -> np.ndarray:
        ids = [self.stoi[BOS]] + [self.stoi[t] for t in tokenize(smiles)] + [self.stoi[EOS]]
        ids = ids[:max_len]
        ids += [self.pad_idx] * (max_len - len(ids))
        return np.asarray(ids, dtype=np.int64)

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            t = self.itos[int(i)]
            if t == EOS:
                break
            if t in (PAD, BOS):
                continue
            out.append(t)
        return "".join(out)

    def save(self, path: Path):
        Path(path).write_text("\n".join(self.itos), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SmilesVocab":
        itos = Path(path).read_text(encoding="utf-8").split("\n")
        return cls(itos=itos, stoi={t: i for i, t in enumerate(itos)})


# ------------------------------------------------------------------ 증강
def randomize_smiles(smiles: str, n: int, seed: int = 0) -> list[str]:
    """SMILES enumeration (원자 순서 셔플) 으로 같은 분자의 다른 표기를 생성."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    rng = np.random.default_rng(seed)
    out = set()
    for _ in range(n * 4):
        if len(out) >= n:
            break
        idx = list(range(mol.GetNumAtoms()))
        rng.shuffle(idx)
        m2 = Chem.RenumberAtoms(mol, [int(i) for i in idx])
        out.add(Chem.MolToSmiles(m2, canonical=False))
    return sorted(out)


class DataError(ValueError):
    """입력 데이터 문제 (파일 누락 / 필수 열 누락 / 유효 분자 없음)."""


def load_smiles_csv(path: Path, smiles_col: str = "smiles", required_cols=("smiles",)) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DataError(f"데이터 파일이 없습니다: {path.resolve()}")
    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise DataError(f"필수 열 누락: {missing}  (현재 열: {list(df.columns)})  파일: {path}")
    if len(df) == 0:
        raise DataError(f"데이터 파일이 비어 있습니다: {path}")
    valid, n_invalid = [], 0
    for s in df[smiles_col].astype(str):
        m = Chem.MolFromSmiles(s)
        valid.append(Chem.MolToSmiles(m) if m is not None else None)
        n_invalid += m is None
    df = df.assign(smiles=valid).dropna(subset=["smiles"]).reset_index(drop=True)
    if len(df) == 0:
        raise DataError(f"유효한 SMILES 가 하나도 없습니다 ({n_invalid}건 모두 파싱 실패): {path}")
    if n_invalid:
        print(f"[data] {n_invalid}개 SMILES 파싱 실패로 제외, {len(df)}개 사용: {path.name}")
    for c in ("name", "category"):
        if c not in df.columns:
            df[c] = "unknown"
    if "target" not in df.columns:
        df["target"] = np.nan
    return df


class SmilesDataset(Dataset):
    def __init__(self, smiles: list[str], vocab: SmilesVocab, max_len: int,
                 augment: int = 0, seed: int = 0):
        base = list(smiles)
        if augment > 0:
            extra = []
            for i, s in enumerate(base):
                extra += randomize_smiles(s, augment, seed + i)
            base = sorted(set(base + extra))
        # 증강으로 새로 나온 토큰은 vocab 에 없을 수 있으므로 필터링
        self.smiles = [s for s in base
                       if all(t in vocab.stoi for t in tokenize(s))
                       and len(tokenize(s)) + 2 <= max_len]
        self.vocab, self.max_len = vocab, max_len
        if not self.smiles:
            raise DataError("토큰화/길이 필터 후 남은 학습 분자가 없습니다 "
                            f"(max_len={max_len}, vocab={len(vocab)}). max_len 을 늘리거나 데이터를 확인하세요.")
        self.data = torch.from_numpy(np.stack([vocab.encode(s, max_len) for s in self.smiles]))
        # 토큰 수 + BOS/EOS : 단계별(길이별) 완결 분자 선택에 사용
        self.lengths = torch.tensor([len(tokenize(s)) + 2 for s in self.smiles])

    def subset_for_length(self, L: int) -> torch.Tensor:
        """길이 L 이하로 완결되는 분자만 (BOS..EOS 포함) 앞 L 토큰으로 반환 — 절단 없음."""
        return self.data[self.lengths <= L][:, :L]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def to_onehot(ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """(B, L) -> (B, V, L) float one-hot  (Conv1d 입력 형태)."""
    return torch.nn.functional.one_hot(ids, vocab_size).permute(0, 2, 1).float()
