"""실험 결과 피드백 → 데이터 병합 → 특성모델 재학습 → GAN 이어서 재학습.

experiments.csv 형식 : smiles,target[,name,category]
  (실제 합성·측정한 후보의 결과. target = 측정값, 예: 이온전도도 mS/cm)

  python -m src.feedback --experiments data/experiments.csv --ckpt outputs/pggan/checkpoint.pt \
                         --out outputs/round2 --epochs 100
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import load_smiles_csv
from .property_model import PropertyModel
from .train import get_args, train


def merge(base_csv: Path, exp_csv: Path, out_csv: Path, keep_quantile: float = 0.5) -> pd.DataFrame:
    base = load_smiles_csv(base_csv)
    exp = load_smiles_csv(exp_csv, required_cols=("smiles", "target"))
    exp = exp.dropna(subset=["target"])
    if len(exp) == 0:
        raise ValueError(f"실험 파일에 target 값이 있는 행이 없습니다: {exp_csv}")
    exp["category"] = exp.get("category", "experiment")
    exp["name"] = exp.get("name", "exp")
    merged = pd.concat([base, exp], ignore_index=True)
    merged = merged.drop_duplicates("smiles", keep="last")
    # GAN 학습용 데이터: 측정값 상위 (keep_quantile 이상) 후보만 포함해 생성 분포를 우수 영역으로 이동
    thr = exp["target"].astype(float).quantile(keep_quantile)
    gan_df = pd.concat([base, exp[exp["target"].astype(float) >= thr]], ignore_index=True) \
        .drop_duplicates("smiles", keep="last")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    gan_df.to_csv(out_csv.with_name(out_csv.stem + "_gan.csv"), index=False)
    print(f"merged {len(base)} + {len(exp)} -> {len(merged)} (gan subset {len(gan_df)}, target>={thr:.3g})")
    return merged


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="data/sample_electrolytes.csv")
    p.add_argument("--experiments", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="outputs/feedback")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--keep_quantile", type=float, default=0.5)
    a, rest = p.parse_known_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    is_example = "example" in Path(a.experiments).name.lower()
    src_label = ("EXAMPLE DATA — 실제 실험 결과가 아닌 예시 파일로 실행된 결과" if is_example
                 else f"experimental data: {a.experiments}")
    (out / "DATA_SOURCE.txt").write_text(src_label, encoding="utf-8")
    print("[feedback] data source:", src_label)
    merged_csv = out / "data_merged.csv"
    merged = merge(Path(a.base), Path(a.experiments), merged_csv, a.keep_quantile)

    pm = PropertyModel().fit(merged)
    out.mkdir(parents=True, exist_ok=True)
    pm.save(out / "property_model.joblib")
    print(f"property model retrained on '{pm.target_name}'  CV R2={pm.cv_r2}")

    targs = get_args(["--data", str(merged_csv.with_name("data_merged_gan.csv")), "--out", str(out / "gan"),
                      "--resume", a.ckpt, "--epochs", str(a.epochs)] + rest)
    train(targs)


if __name__ == "__main__":
    main()
