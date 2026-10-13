"""학습 안정성을 위한 하이퍼파라미터 랜덤 서치.

  python -m src.tune --mode pggan --trials 8 --epochs 120 --out outputs/tune
결과: outputs/tune/results.csv (validity*uniqueness*novelty 기준 정렬)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .train import get_args, train

SPACE = {
    "lr": [1e-4, 2e-4, 5e-4],
    "n_critic": [3, 5],
    "latent_dim": [32, 64, 128],
    "channels": [64, 128],
    "tau_min": [0.3, 0.5, 0.8],
    "gp_lambda": [5.0, 10.0],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="pggan")
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--out", default="outputs/tune")
    p.add_argument("--data", default="data/sample_electrolytes.csv")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    rows = []
    for t in range(a.trials):
        hp = {k: rng.choice(v).item() for k, v in SPACE.items()}
        argv = ["--mode", a.mode, "--epochs", str(a.epochs), "--data", a.data,
                "--out", f"{a.out}/trial{t}", "--eval_every", str(a.epochs)]
        for k, v in hp.items():
            argv += [f"--{k}", str(v)]
        hist = train(get_args(argv))
        last = hist[-1]
        # 점수: 동일 평가 조건(n_eval, eval_tau, eval_seed)에서의 필터 통과 수 우선, 동률이면 유효 고유 구조 수
        score = last["n_filter_pass"] + 0.001 * last["n_valid_unique"]
        rows.append({"trial": t, **hp, **{k: last[k] for k in ("validity", "n_valid_unique", "n_not_in_train",
                                                              "n_filter_pass", "lossD")},
                     "score": round(score, 4)})
        print("TRIAL", t, rows[-1])
    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    df.to_csv(Path(a.out) / "results.csv", index=False)
    (Path(a.out) / "best.json").write_text(json.dumps(df.iloc[0].to_dict(), indent=2))
    print(df.to_string())


if __name__ == "__main__":
    main()
