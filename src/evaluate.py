"""GAN / PGGAN 공통 평가 (동일 샘플 수 · 동일 온도 · 동일 잠재벡터).

EvalConfig 로 고정한 조건에서 두 모델을 같은 z 로 평가해 공정하게 비교한다.
기록 항목:
  validity            유효 SMILES 비율
  n_valid_unique      유효 고유 구조 수
  n_not_in_train      학습 데이터에 없는 고유 구조 수  (※ '신규 물질 검증' 이 아니라 단순 비포함 여부)
  n_filter_pass       후보 필터(FilterConfig) 통과 수
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .filters import FilterConfig, PRESETS, apply_filters
from .metrics import canonical


@dataclass
class EvalConfig:
    n_samples: int = 512
    tau: float = 0.0          # 0 = argmax(결정적), >0 = 온도 샘플링
    seed: int = 1234          # 평가용 잠재벡터 seed (학습 seed 와 별도)
    sampler: str = "normal"   # "normal" | "sobol"
    filter_preset: str = "anion"

    def to_dict(self):
        return asdict(self)


def eval_latents(cfg: EvalConfig, latent_dim: int, device="cpu") -> torch.Tensor:
    """cfg.seed 로 결정되는 고정 잠재벡터 — 모델에 관계없이 동일."""
    if cfg.sampler == "sobol":
        eng = torch.quasirandom.SobolEngine(latent_dim, scramble=True, seed=cfg.seed)
        u = eng.draw(cfg.n_samples).clamp(1e-6, 1 - 1e-6)
        return torch.distributions.Normal(0, 1).icdf(u).to(device)
    g = torch.Generator().manual_seed(cfg.seed)
    return torch.randn(cfg.n_samples, latent_dim, generator=g).to(device)


@torch.no_grad()
def evaluate_model(G, vocab, cfg: EvalConfig, train_set: set, z: torch.Tensor | None = None,
                   stage=None, alpha: float = 1.0, filter_cfg: FilterConfig | None = None) -> dict:
    was_training = G.training; G.eval()
    dev = next(G.parameters()).device
    z = eval_latents(cfg, G.latent_dim, dev) if z is None else z
    if cfg.tau > 0:
        torch.manual_seed(cfg.seed)  # 온도 샘플링의 난수도 고정
    ids = G.sample_ids(z, stage, alpha, tau=cfg.tau).cpu().numpy()
    if was_training:
        G.train()
    raw = [vocab.decode(r) for r in ids]
    cans = [canonical(s) for s in raw]
    valid = [c for c in cans if c]
    uniq = sorted(set(valid))
    not_in_train = [c for c in uniq if c not in train_set]
    fcfg = filter_cfg or PRESETS[cfg.filter_preset]
    fdf = apply_filters(uniq, fcfg, train_set)
    return {
        "n_samples": len(raw), "tau": cfg.tau, "eval_seed": cfg.seed,
        "validity": round(len(valid) / max(len(raw), 1), 4),
        "n_valid": len(valid), "n_valid_unique": len(uniq),
        "uniqueness": round(len(uniq) / max(len(valid), 1), 4),
        "n_not_in_train": len(not_in_train),
        "n_filter_pass": int(fdf["passed"].sum()) if len(fdf) else 0,
        "filter_preset": fcfg.name,
        "valid_unique_smiles": uniq,
        "filter_table": fdf,
    }


def save_eval(result: dict, path: Path):
    r = {k: v for k, v in result.items() if k != "filter_table"}
    Path(path).write_text(json.dumps(r, ensure_ascii=False, indent=2))
