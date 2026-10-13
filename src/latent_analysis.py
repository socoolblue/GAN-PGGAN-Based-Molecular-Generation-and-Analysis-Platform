"""GAN 벡터 공간 분석 프로그램.

두 공간을 구분한다:
  * **Generator z-space** : 생성 입력 노이즈 공간. Sobol 준난수로 탐색하고 slerp 로 보간한다.
    (실제 분자를 이 공간에 직접 투영할 수는 없다 — 생성 방향으로만 쓰는 공간)
  * **Discriminator feature space** : D 의 중간 레이어(`features()`) 임베딩. 실제·생성 분자를 모두
    투영할 수 있어 PCA / KMeans 구획화 · 축-기능기 상관 · 기준군 분포 영역 분석에 쓴다.

"기준군 분포 영역(reference region)" 은 기준 카테고리(예: anion) 임베딩의 PCA 평균 ± kσ 상자이다.
이는 **실제 사용 가능한 영역이 아니라** 기준군이 분포하는 범위를 나타내는 참고 지표다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .data import to_onehot
from .metrics import FUNCTIONAL_GROUPS, canonical, descriptors, functional_groups


@torch.no_grad()
def embed_smiles(D, vocab, smiles: list[str], max_len: int) -> np.ndarray:
    ids = torch.from_numpy(np.stack([vocab.encode(s, max_len) for s in smiles]))
    return D.features(to_onehot(ids, len(vocab))).cpu().numpy()


@torch.no_grad()
def sobol_latents(n: int, dim: int, seed: int = 0) -> torch.Tensor:
    """Sobol 저불일치 수열 -> 정규분포 잠재벡터 (고차원 공간 균일 탐색)."""
    eng = torch.quasirandom.SobolEngine(dim, scramble=True, seed=seed)
    u = eng.draw(n).clamp(1e-6, 1 - 1e-6)
    return torch.distributions.Normal(0, 1).icdf(u)


@torch.no_grad()
def generate(G, vocab, z: torch.Tensor, tau: float = 0.0) -> list[str]:
    return [vocab.decode(r) for r in G.sample_ids(z, tau=tau).cpu().numpy()]


def slerp(z0, z1, t):
    """구면 선형보간 (가우시안 잠재공간에서 선형보간보다 분포 밀도를 유지)."""
    z0n, z1n = z0 / z0.norm(), z1 / z1.norm()
    omega = torch.acos((z0n * z1n).sum().clamp(-1, 1))
    so = torch.sin(omega)
    if so.abs() < 1e-6:
        return (1 - t) * z0 + t * z1
    return torch.sin((1 - t) * omega) / so * z0 + torch.sin(t * omega) / so * z1


@torch.no_grad()
def interpolate_path(G, vocab, z0, z1, n_steps=10, tau=0.0) -> list[str]:
    zs = torch.stack([slerp(z0, z1, t) for t in np.linspace(0, 1, n_steps)])
    return generate(G, vocab, zs, tau)


def interpolation_table(G, vocab, z0, z1, n_steps=10, tau=0.0,
                        track_groups=("fluoro", "sulfonyl", "ether", "hydroxyl", "anion", "ring")) -> pd.DataFrame:
    """z-space 보간 경로의 각 단계에 대해 생성 SMILES · 유효성 · 기능기/구조 특성 변화를 표로 반환.
    유효 구조가 하나도 없거나 같은 구조만 반복돼도 오류 없이 결과를 돌려준다."""
    raws = interpolate_path(G, vocab, z0, z1, n_steps, tau)
    rows, prev = [], None
    for i, (t, raw) in enumerate(zip(np.linspace(0, 1, n_steps), raws)):
        can = canonical(raw)
        row = {"step": i, "t": round(float(t), 3), "generated": raw, "canonical": can, "valid": can is not None,
               "changed_vs_prev": (can != prev) if i > 0 else None}
        if can:
            fg, d = functional_groups(can), descriptors(can)
            row.update({g: fg.get(g, 0) for g in track_groups})
            row.update({"HeavyAtoms": d["HeavyAtoms"], "MolWt": round(d["MolWt"], 1), "FormalCharge": d["FormalCharge"]})
        rows.append(row); prev = can
    df = pd.DataFrame(rows)
    n_valid, n_unique = int(df["valid"].sum()), df["canonical"].dropna().nunique()
    df.attrs["summary"] = (f"valid {n_valid}/{n_steps} steps, unique valid structures {n_unique}"
                           + ("  — 유효 구조 없음" if n_valid == 0 else "")
                           + ("  — 모든 유효 단계가 같은 구조" if n_valid > 1 and n_unique == 1 else ""))
    return df


class LatentSpaceAnalyzer:
    def __init__(self, G, D, vocab, max_len: int, n_components: int = 2, n_clusters: int = 5, seed: int = 0):
        self.G, self.D, self.vocab, self.max_len = G, D, vocab, max_len
        self.pca = PCA(n_components=n_components, random_state=seed)
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
        self.region = None

    # ---------- 1. 매핑 ----------
    def fit(self, df: pd.DataFrame, region_categories=("anion",), k_sigma: float = 1.0) -> pd.DataFrame:
        """실제 데이터로 PCA/KMeans 를 적합하고, 기준 카테고리 분포로 '가용 범위' 설정."""
        emb = embed_smiles(self.D, self.vocab, df["smiles"].tolist(), self.max_len)
        self.pca.fit(emb)
        pcs = self.pca.transform(emb)
        self.kmeans.fit(emb)
        out = df.copy()
        out["cluster"] = self.kmeans.labels_
        for i in range(pcs.shape[1]):
            out[f"PC{i+1}"] = pcs[:, i]
        self.real_emb, self.real_df = emb, out
        # 기준군 분포 영역: 기준 카테고리(예: 리튬염 음이온) 임베딩의 PCA 평균 ± k·표준편차 (각 PC 축)
        mask = out["category"].isin(region_categories) if "category" in out else np.ones(len(out), bool)
        if "category" in out and not mask.any():
            print(f"[warn] 기준 카테고리 {region_categories} 가 데이터에 없어 전체 분포로 영역을 잡습니다.")
        ref = pcs[mask.values] if mask.any() else pcs
        self.region = {"space": "discriminator_feature_space_PCA", "meaning": "reference-group distribution region (NOT a validated usable region)",
                       "center": ref.mean(0).tolist(), "std": ref.std(0).tolist(), "k": k_sigma,
                       "categories": list(region_categories), "n_reference": int(len(ref))}
        return out

    def in_reference_region(self, pcs: np.ndarray) -> np.ndarray:
        c, s = np.array(self.region["center"]), np.array(self.region["std"]) + 1e-8
        return (np.abs(pcs - c) / s <= self.region["k"]).all(axis=1)

    def map_generated(self, smiles: list[str]) -> pd.DataFrame:
        """생성 분자를 같은 공간에 투영하고 클러스터·가용범위 여부를 부여."""
        # canonical 표기가 vocab 밖 토큰을 포함하면 임베딩 불가 -> 제외
        can = [c for c in (canonical(s) for s in smiles) if c and self.vocab.encodable(c, self.max_len)]
        if not can:
            return pd.DataFrame(columns=["smiles", "cluster", "in_reference_region", "PC1", "PC2"])
        emb = embed_smiles(self.D, self.vocab, can, self.max_len)
        pcs = self.pca.transform(emb)
        df = pd.DataFrame({"smiles": can, "cluster": self.kmeans.predict(emb),
                           "in_reference_region": self.in_reference_region(pcs)})
        for i in range(pcs.shape[1]):
            df[f"PC{i+1}"] = pcs[:, i]
        return df

    # ---------- 2. 구획화 / 클러스터 해석 ----------
    def cluster_summary(self) -> pd.DataFrame:
        fg = pd.DataFrame([functional_groups(s) for s in self.real_df["smiles"]])
        fg["cluster"] = self.real_df["cluster"].values
        summ = fg.groupby("cluster").mean().round(2)
        summ["n"] = fg.groupby("cluster").size()
        if "category" in self.real_df:
            summ["categories"] = self.real_df.groupby("cluster")["category"] \
                .agg(lambda s: ", ".join(f"{k}:{v}" for k, v in s.value_counts().items()))
        summ["examples"] = self.real_df.groupby("cluster")["smiles"].agg(lambda s: " | ".join(s.head(3)))
        return summ

    # ---------- 3. 방향성 ↔ 기능기 상관 ----------
    def direction_correlations(self, top_k: int = 6) -> pd.DataFrame:
        """각 주성분 축 좌표와 기능기 개수·물성 디스크립터 간 Spearman 상관."""
        feats = pd.concat([
            pd.DataFrame([functional_groups(s) for s in self.real_df["smiles"]]),
            pd.DataFrame([descriptors(s) for s in self.real_df["smiles"]]),
        ], axis=1)
        rows = []
        for i in range(self.pca.n_components_):
            axis = self.real_df[f"PC{i+1}"].values
            for col in feats.columns:
                v = feats[col].values
                if np.std(v) == 0:
                    continue
                rho, p = spearmanr(axis, v)
                rows.append({"axis": f"PC{i+1}", "feature": col, "spearman_rho": round(float(rho), 3),
                             "p_value": float(p)})
        corr = pd.DataFrame(rows)
        corr["abs"] = corr["spearman_rho"].abs()
        return corr.sort_values(["axis", "abs"], ascending=[True, False]).groupby("axis").head(top_k) \
                   .drop(columns="abs").reset_index(drop=True)

    def interpret_axes(self) -> dict[str, str]:
        """축 방향의 의미를 '양(+)방향: ..., 음(-)방향: ...' 형태로 요약."""
        corr = self.direction_correlations(top_k=20)
        out = {}
        for axis, g in corr.groupby("axis"):
            pos = g[g.spearman_rho > 0.3].feature.tolist()[:4]
            neg = g[g.spearman_rho < -0.3].feature.tolist()[:4]
            out[axis] = f"(+) {', '.join(pos) or '-'}   |   (-) {', '.join(neg) or '-'}"
        return out

    def save(self, out_dir: Path):
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        self.real_df.to_csv(out_dir / "latent_map_real.csv", index=False)
        self.cluster_summary().to_csv(out_dir / "cluster_summary.csv")
        self.direction_correlations().to_csv(out_dir / "direction_correlations.csv", index=False)
        (out_dir / "axis_interpretation.json").write_text(
            json.dumps(self.interpret_axes(), ensure_ascii=False, indent=2))
        (out_dir / "reference_region.json").write_text(json.dumps(self.region, ensure_ascii=False, indent=2))
        (out_dir / "pca_explained_variance.json").write_text(
            json.dumps(self.pca.explained_variance_ratio_.tolist()))


def plot_latent_map(real_df: pd.DataFrame, gen_df: pd.DataFrame | None, region: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(7, 6))
    cats = real_df["category"].unique() if "category" in real_df else ["real"]
    for c in cats:
        m = real_df["category"] == c if "category" in real_df else np.ones(len(real_df), bool)
        ax.scatter(real_df.loc[m, "PC1"], real_df.loc[m, "PC2"], s=28, alpha=0.8, label=f"real:{c}")
    if gen_df is not None and len(gen_df):
        ax.scatter(gen_df["PC1"], gen_df["PC2"], marker="x", s=22, c="k", alpha=0.6, label="generated")
    c, s, k = np.array(region["center"]), np.array(region["std"]), region["k"]
    ax.add_patch(Rectangle((c[0] - k * s[0], c[1] - k * s[1]), 2 * k * s[0], 2 * k * s[1],
                           fill=False, ls="--", lw=1.2,
                           label=f"reference region ({'/'.join(region['categories'])} mean±{k}σ)"))
    ax.set_xlabel("PC1 (D feature space)"); ax.set_ylabel("PC2 (D feature space)"); ax.legend(fontsize=8)
    ax.set_title("Discriminator feature space (PCA) — not Generator z-space")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_zspace_map(z: torch.Tensor, smiles: list[str], path: Path, filter_pass: list[bool] | None = None):
    """Generator z-space 지도: 샘플링한 z 를 PCA 2D 로 투영하고 생성 결과(유효/무효/필터 통과)를 색으로 표시.
    (D feature space 지도와 구분되는 별도 공간)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zz = z.cpu().numpy()
    pcs = PCA(n_components=2).fit_transform(zz)
    valid = np.array([canonical(s) is not None for s in smiles])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(pcs[~valid, 0], pcs[~valid, 1], s=8, c="lightgray", label=f"invalid ({(~valid).sum()})")
    ax.scatter(pcs[valid, 0], pcs[valid, 1], s=12, c="tab:blue", alpha=0.7, label=f"valid ({valid.sum()})")
    if filter_pass is not None:
        fp = np.array(filter_pass) & valid
        ax.scatter(pcs[fp, 0], pcs[fp, 1], s=30, c="tab:red", marker="*", label=f"filter pass ({fp.sum()})")
    ax.set_xlabel("PC1 (Generator z-space)"); ax.set_ylabel("PC2 (Generator z-space)")
    ax.set_title("Generator z-space (sampled latents) — not D feature space"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
