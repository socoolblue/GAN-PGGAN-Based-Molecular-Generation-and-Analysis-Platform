"""후보 생성 → 필터(사유 기록) → 벡터공간 매핑 → 점수 → 소규모 후보군 선정.

  python -m src.candidates --ckpt outputs/pggan/checkpoint.pt --n 4096 --top 20 --filter_preset anion

출력 (out/):
  generation_config.json   후보 생성 조건 (seed, n, tau, sampler, filter preset) — 학습 평가 조건과 별개로 명시
  filter_results.csv       생성된 모든 고유 구조의 필터 통과 여부 + 제외 이유 + Protic 규칙 표시 (후보 없어도 헤더 유지)
  filter_summary.csv       제외 이유별 건수
  generated_passed.csv     필터 통과·임베딩 가능 구조 전체 (좌표·클러스터·기준군 영역 여부·점수·디스크립터·기능기)
  candidates_top.csv       선정 후보 (구조, 선정 점수, 필터 결과, 선정 이유) — 후보 없어도 헤더 유지
  STATUS.json / STATUS.txt 상태 코드와 메시지
  latent_map.png / zspace_map.png   D feature space 지도 / G z-space 지도

STATUS.code:
  NO_VALID_STRUCTURES  생성 결과에 유효한 SMILES 가 하나도 없음
  NO_FILTER_PASS       유효 구조는 있으나 필터를 통과한 구조가 없음 (filter_summary.csv 참고)
  NO_EMBEDDABLE        필터는 통과했으나 임베딩 조건(vocab 외 토큰 / 길이 초과)으로 모두 제외됨 — 필터 탈락과 구분
  OK_PARTIAL           후보는 있으나 요청 수(top)보다 적음
  OK                   요청 수만큼 선정
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import load_smiles_csv
from .filters import PRESETS, FilterConfig, apply_filters, filter_summary
from .latent_analysis import LatentSpaceAnalyzer, generate, plot_latent_map, plot_zspace_map, sobol_latents
from .metrics import canonical, descriptors, functional_groups
from .property_model import PropertyModel
from .train import load_checkpoint

CANDIDATE_COLUMNS = ["smiles", "selection_score", "score_type", "filter_passed", "filter_reasons",
                     "selection_reason", "cluster", "in_reference_region", "is_protic_rule", "protic_rules_matched"]


def validate_top(top) -> int:
    """top 은 0 이상의 정수여야 한다 (bool, float, 음수는 입력 오류)."""
    if isinstance(top, bool) or not isinstance(top, (int, np.integer)):
        raise TypeError(f"top 은 정수여야 합니다 (받은 값: {top!r}, type={type(top).__name__})")
    if top < 0:
        raise ValueError(f"top 은 0 이상이어야 합니다 (받은 값: {top})")
    return int(top)


def select_top(ranked: pd.DataFrame, top: int, score_col: str = "selection_score",
               cluster_col: str = "cluster", id_col: str = "smiles") -> pd.DataFrame:
    """클러스터 다양성을 고려한 상위 top 개 선정. 최종 수 = min(top, 고유 후보 수).

    정렬 기준(재현성): score 내림차순 → cluster 오름차순 → smiles 오름차순 (동점이면 사전순).
    * top < 클러스터 수 : 각 클러스터의 최고 점수 후보끼리 비교해 상위 top 개
    * top ≥ 클러스터 수 : 클러스터별 상위 top//n_cluster 개를 먼저 뽑고, 남은 자리는 미선정 후보 점수순
    """
    top = validate_top(top)
    cols = list(ranked.columns) + ["selection_reason"]
    if top == 0 or len(ranked) == 0:
        return pd.DataFrame(columns=cols)
    r = ranked.drop_duplicates(id_col).sort_values(
        [score_col, cluster_col, id_col], ascending=[False, True, True], kind="mergesort").reset_index(drop=True)
    n_cl = r[cluster_col].nunique()
    cluster_best = r.groupby(cluster_col, sort=False).head(1)   # 정렬 유지 → 각 클러스터 최고점
    if top < n_cl:
        picked = cluster_best.head(top).copy()
        picked["selection_reason"] = f"top {top} among per-cluster best candidates ({n_cl} clusters)"
    else:
        per_cluster = top // n_cl
        picked = r.groupby(cluster_col, sort=False).head(per_cluster).copy()
        picked["selection_reason"] = f"per-cluster top {per_cluster} by {score_col} ({n_cl} clusters)"
        remaining = top - len(picked)
        if remaining > 0:
            extra = r.drop(picked.index).head(remaining).copy()
            extra["selection_reason"] = f"fill by global {score_col} rank"
            picked = pd.concat([picked, extra])
    picked = picked.sort_values([score_col, cluster_col, id_col], ascending=[False, True, True],
                                kind="mergesort").reset_index(drop=True)
    assert len(picked) == min(top, len(r)) and picked[id_col].is_unique
    return picked


def _write_status(out: Path, code: str, msg: str, **extra):
    status = {"code": code, "message": msg, **extra}
    (out / "STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2))
    (out / "STATUS.txt").write_text(f"{code}: {msg}")
    print(f"[candidates] {code}: {msg}")
    return status


def _empty_candidates(out: Path) -> pd.DataFrame:
    df = pd.DataFrame(columns=CANDIDATE_COLUMNS)
    df.to_csv(out / "candidates_top.csv", index=False)
    pd.DataFrame(columns=["smiles", "cluster", "in_reference_region", "PC1", "PC2"]).to_csv(
        out / "generated_passed.csv", index=False)
    return df


def run(ckpt: str, data: str, out: str, n: int = 4096, top: int = 20, tau: float = 0.6,
        region_categories=("anion",), n_clusters: int = 5, seed: int = 0,
        property_model: str | None = None, filter_preset: str = "anion",
        filter_cfg: FilterConfig | None = None, region_bonus: float = 0.1) -> pd.DataFrame:
    top = validate_top(top)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    G, D, vocab, args = load_checkpoint(ckpt)
    df = load_smiles_csv(Path(data))
    train_set = set(df["smiles"])
    fcfg = filter_cfg or PRESETS[filter_preset]

    gen_cfg = {"checkpoint": str(ckpt), "n_samples": n, "top": top, "tau": tau, "sampler": "sobol", "seed": seed,
               "filter": fcfg.to_dict(), "region_categories": list(region_categories), "n_clusters": n_clusters,
               "note": "후보 생성 조건. 학습 중 평가 조건(checkpoint['eval_config'])과는 별개.",
               "train_eval_config": getattr(G, "eval_config", None)}
    (out / "generation_config.json").write_text(json.dumps(gen_cfg, ensure_ascii=False, indent=2))

    # 1) Generator z-space Sobol 탐색으로 생성
    z = sobol_latents(n, args.latent_dim, seed)
    if tau > 0:
        torch.manual_seed(seed)
    raw = generate(G, vocab, z, tau)
    cans = [canonical(s) for s in raw]
    n_valid = sum(c is not None for c in cans)
    uniq = sorted({c for c in cans if c})
    counts = {"n_generated": n, "n_valid": n_valid, "n_valid_unique": len(uniq),
              "n_not_in_train": sum(c not in train_set for c in uniq)}
    print(f"generated={n}  valid={n_valid} ({n_valid/max(n,1):.3f})  valid_unique={len(uniq)}  "
          f"not_in_train={counts['n_not_in_train']}")

    # 2) 필터 — 유효 구조가 0개여도 스키마가 유지된 빈 표를 저장
    fres = apply_filters(uniq, fcfg, train_set)
    fres.to_csv(out / "filter_results.csv", index=False)
    filter_summary(fres).to_csv(out / "filter_summary.csv", index=False)
    passed = fres.loc[fres["passed"], "canonical"].tolist()
    counts["n_filter_pass"] = len(passed)
    print(f"filter '{fcfg.name}' passed: {len(passed)} / {len(uniq)}")
    pass_set = set(passed)
    plot_zspace_map(z, raw, out / "zspace_map.png", [c in pass_set for c in cans])

    # 3) Discriminator feature space 매핑 / 구획화 (실제 데이터 기준) — 생성 결과와 무관하게 항상 수행
    an = LatentSpaceAnalyzer(G, D, vocab, args.max_len, n_clusters=min(n_clusters, len(df)), seed=seed)
    an.fit(df, region_categories=region_categories)
    an.save(out / "latent")
    gen_map = an.map_generated(passed)
    plot_latent_map(an.real_df, gen_map, an.region, out / "latent_map.png")
    counts["n_embeddable"] = len(gen_map)
    not_embeddable = sorted(pass_set - set(gen_map["smiles"])) if len(passed) else []
    counts["n_not_embeddable"] = len(not_embeddable)
    (out / "excluded_by_embedding.txt").write_text("\n".join(not_embeddable))

    if n_valid == 0:
        _write_status(out, "NO_VALID_STRUCTURES", f"생성 {n}개 중 유효한 SMILES 가 없습니다. "
                      "학습 epoch 를 늘리거나 --tau 를 낮추세요.", **counts)
        return _empty_candidates(out)
    if len(passed) == 0:
        _write_status(out, "NO_FILTER_PASS", f"유효 고유 구조 {len(uniq)}개 중 필터 '{fcfg.name}' 통과 구조가 없습니다. "
                      "filter_summary.csv 의 제외 이유를 확인하고 필터 프리셋/조건 또는 --n, --tau 를 조정하세요.", **counts)
        return _empty_candidates(out)
    if len(gen_map) == 0:
        _write_status(out, "NO_EMBEDDABLE", f"필터 통과 {len(passed)}개가 모두 임베딩 조건(모델 vocab 외 토큰 또는 "
                      f"max_len={args.max_len} 초과)으로 제외되었습니다 — 필터 탈락이 아님. excluded_by_embedding.txt 참고.", **counts)
        return _empty_candidates(out)
    if not_embeddable:
        print(f"[note] 필터 통과 {len(passed)}개 중 {len(not_embeddable)}개는 임베딩 조건으로 제외 (필터 탈락 아님): "
              f"excluded_by_embedding.txt")

    # 4) 점수 (실측 학습 여부를 열 이름과 score_type 에 명시)
    pm = PropertyModel.load(property_model) if property_model else PropertyModel(seed).fit(df)
    pm.save(out / "property_model.joblib")
    print("score:", pm.describe())
    gen_map[pm.score_label] = pm.predict(gen_map["smiles"].tolist())
    gen_map["score_type"] = pm.score_type
    gen_map["selection_score"] = gen_map[pm.score_label] + region_bonus * gen_map["in_reference_region"]
    gen_map = gen_map.merge(fres[["canonical", "passed", "reasons", "is_protic_rule", "protic_rules_matched",
                                  "n_heavy", "total_charge", "n_fragments", "hetero_frac"]]
                            .rename(columns={"canonical": "smiles", "passed": "filter_passed", "reasons": "filter_reasons"}),
                            on="smiles", how="left")
    gen_map = pd.concat([gen_map,
                         pd.DataFrame([descriptors(s) for s in gen_map["smiles"]]),
                         pd.DataFrame([functional_groups(s) for s in gen_map["smiles"]])], axis=1)
    gen_map = gen_map.loc[:, ~gen_map.columns.duplicated()]
    gen_map["filter_reasons"] = gen_map["filter_reasons"].fillna("")
    gen_map["protic_rules_matched"] = gen_map["protic_rules_matched"].fillna("")
    gen_map.to_csv(out / "generated_passed.csv", index=False)

    # 5) 선정: min(top, 고유 후보 수), 클러스터 다양성, 재현 가능한 정렬
    picked = select_top(gen_map, top, "selection_score", "cluster", "smiles")
    if len(picked):
        picked["selection_reason"] = picked["selection_reason"] + np.where(
            picked["in_reference_region"], f"; +{region_bonus} in reference region", "")
    picked = picked[CANDIDATE_COLUMNS + [c for c in picked.columns if c not in CANDIDATE_COLUMNS]]
    picked.to_csv(out / "candidates_top.csv", index=False)
    counts["n_selected"] = len(picked)
    if top == 0:
        _write_status(out, "OK", "top=0: 후보 선정 없이 생성·필터·매핑 결과만 저장했습니다.", **counts)
    elif len(picked) < top:
        _write_status(out, "OK_PARTIAL", f"요청 {top}개 중 {len(picked)}개만 선정됨 (선정 가능한 고유 후보 {len(gen_map)}개).",
                      score=pm.score_label, score_type=pm.score_type, **counts)
    else:
        _write_status(out, "OK", f"{len(picked)}개 선정 (요청 {top}).", score=pm.score_label, score_type=pm.score_type, **counts)

    if len(picked):
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
            img = Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in picked["smiles"]], molsPerRow=5,
                                       subImgSize=(220, 180),
                                       legends=[f"{r.selection_score:.3f} ({pm.score_type})" for r in picked.itertuples()])
            if hasattr(img, "save"):
                img.save(out / "candidates_top.png")
            else:
                (out / "candidates_top.png").write_bytes(img.data)
        except Exception as e:
            print("grid image skipped:", e)
    return picked


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default="data/sample_electrolytes.csv")
    p.add_argument("--out", default="outputs/candidates")
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--tau", type=float, default=0.6, help="0=argmax(결정적), >0 온도 샘플링")
    p.add_argument("--n_clusters", type=int, default=5)
    p.add_argument("--region_categories", nargs="+", default=["anion"])
    p.add_argument("--property_model", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--filter_preset", default="anion", choices=list(PRESETS))
    a = p.parse_args()
    run(a.ckpt, a.data, a.out, a.n, a.top, a.tau, tuple(a.region_categories), a.n_clusters, a.seed,
        a.property_model, a.filter_preset)
