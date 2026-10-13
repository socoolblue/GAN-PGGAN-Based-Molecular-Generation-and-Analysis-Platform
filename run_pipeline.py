"""엔드투엔드: GAN vs PGGAN 학습·비교 -> 벡터공간 분석 -> 후보 선정.

  python run_pipeline.py --epochs 300
"""
import argparse
import json
from pathlib import Path

from src.candidates import run as select_candidates
from src.train import get_args, train

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--data", default="data/sample_electrolytes.csv")
p.add_argument("--out", default="outputs")
p.add_argument("--n_gen", type=int, default=8192)
p.add_argument("--top", type=int, default=20)
p.add_argument("--filter_preset", default="anion")
a, rest = p.parse_known_args()
out = Path(a.out)

summary = {}
for mode in ("gan", "pggan"):
    hist = train(get_args(["--mode", mode, "--epochs", str(a.epochs), "--data", a.data,
                           "--out", str(out / mode)] + rest))
    summary[mode] = hist[-1]
print("\n=== GAN vs PGGAN (final epoch) ===")
for k, v in summary.items():
    print(k, v)
(out / "comparison.json").write_text(json.dumps(summary, indent=2))

# 동일 평가 조건(n_eval / eval_tau / eval_seed 공통)에서 필터 통과 수 → 유효 고유 구조 수 순으로 비교
best = max(summary, key=lambda k: (summary[k]["n_filter_pass"], summary[k]["n_valid_unique"]))
print(f"\n>>> best model: {best}  -> candidate selection")
select_candidates(str(out / best / "checkpoint.pt"), a.data, str(out / "candidates"), n=a.n_gen, top=a.top,
                  filter_preset=a.filter_preset)
