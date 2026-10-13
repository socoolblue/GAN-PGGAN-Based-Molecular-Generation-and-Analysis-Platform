"""electrolyte_gan_research.ipynb 생성 스크립트 (nbformat)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# 수계 전해질용 신규 리튬염 후보 탐색 — GAN / PGGAN 분자 생성과 벡터 공간 분석

연구 흐름

1. **데이터**: 수계 전해질 관련 분자(리튬염 음이온, 이온성 액체 양이온, 용매·첨가제) SMILES
2. **GAN / PGGAN 학습**: SMILES 시퀀스용 WGAN-GP. PGGAN 은 시퀀스 길이를 16 → 32 → 64 로 단계 성장
3. **생성 결과 평가**: 두 모델을 동일한 샘플 수·샘플링 온도·고정 잠재벡터로 비교
4. **분자 생성과 후보 필터링**: 구조 조건(전하·조각·라디칼·크기·헤테로원자 비율)별 프리셋 필터
5. **벡터 공간 분석**: Generator z-space 와 Discriminator feature space 구분, 클러스터·축 해석·보간
6. **후보 선정**: 클러스터 다양성을 고려한 상위 후보 추출

해석상의 전제

* 점수 `heuristic_exploration_score` 는 구조 규칙에 기반한 **탐색용 휴리스틱 점수**이며 실제 물성 예측값이 아닙니다.
  실측 `target` 을 학습한 경우에만 `ml_pred_target` 으로 표기됩니다.
* 구조 필터 통과는 SMILES 수준의 구조 조건을 만족한다는 뜻일 뿐, **합성 가능성이나 수계 전해질 적용 가능성을 검증한 것이 아닙니다.**
* 학습 데이터에 없는 구조라는 것은 미포함 여부일 뿐 신규 물질로 검증된 것이 아닙니다.
* `data/experiments_example.csv` 는 예시 데이터이며 실제 측정 데이터와 구분해 표시됩니다.
* `QUICK = True` 는 짧은 학습 설정입니다. 이 조건의 유효율·후보 수는 낮으며 결과 해석에 사용할 수 없습니다.""")

code("""# !pip install -r requirements.txt   # 최초 1회
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from IPython.display import display, Image

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)

QUICK = True
EPOCHS = 30 if QUICK else 300
SEED = 0
OUT = Path("outputs_run"); OUT.mkdir(exist_ok=True)
print("project root:", ROOT, "| torch", torch.__version__, "| device:", "cuda" if torch.cuda.is_available() else "cpu")""")

md("""## 0. 데이터 로딩 (입력 검증 포함)

`load_smiles_csv` 는 파일 누락 · 필수 열 누락 · 유효 분자 0개에 대해 `DataError` 로 명확히 알립니다.""")

code("""from src.data import load_smiles_csv, DataError, SmilesVocab, SmilesDataset

DATA = Path("data/sample_electrolytes.csv")
df = load_smiles_csv(DATA)
display(df.groupby("category").size().rename("n"))
print("target(실측) 값 보유 행:", int(df["target"].notna().sum()), "/", len(df))

# 오류 메시지 확인
for bad in ["data/없는파일.csv"]:
    try:
        load_smiles_csv(bad)
    except DataError as e:
        print("DataError:", e)""")

md("""## 1. GAN / PGGAN 학습 — 단계별 완결 분자 (절단 없음)

PGGAN stage k 의 길이 `L_k` 에 대해 `(BOS + 토큰 + EOS) ≤ L_k` 인 **완결된 분자만** 학습합니다 (`SmilesDataset.subset_for_length`).
학습 전 stage 별 분자 수를 출력하고, 0개면 `DataError`, `--min_stage_data` 미만이면 오류, batch 보다 적으면 경고를 냅니다.

**한계(데이터 분포 변화)**: stage 0 은 짧은 분자(예: 소형 용매·단순 음이온)만, 마지막 stage 는 전체 분자를 학습하므로
단계마다 학습 분포가 다릅니다. 짧은 분자가 적은 데이터셋에서는 초기 stage 가 소수 분자에 과적합할 수 있고,
긴 분자 위주 데이터셋이면 초기 stage 데이터가 부족해 오류가 날 수 있습니다 (`--max_len`, `--n_stages` 로 조정).

**평가 조건 통일**: 두 모델 모두 `--n_eval`, `--eval_tau`, `--eval_seed` 로 정해진 **같은 잠재벡터** 로 평가합니다.""")

code("""from src.train import get_args, train

EVAL = dict(n_eval=512, eval_tau=0.0, eval_seed=1234, filter_preset="anion")   # 학습 중 평가 조건 (두 모델 공통)
common = ["--data", str(DATA), "--epochs", str(EPOCHS), "--eval_every", str(max(EPOCHS // 5, 1)),
          "--augment", "10", "--seed", str(SEED)] + sum([[f"--{k}", str(v)] for k, v in EVAL.items()], [])
hist_gan   = train(get_args(["--mode", "gan",   "--out", str(OUT / "gan")]   + common))
hist_pggan = train(get_args(["--mode", "pggan", "--out", str(OUT / "pggan")] + common))""")

code("""# stage 별 학습 분자 수 / 저장된 설정 확인
cfg_pg = json.load(open(OUT / "pggan/config.json"))
print("PGGAN stage 별 완결 학습 분자 수:", cfg_pg["stage_train_counts"])
print("평가 조건:", cfg_pg["eval_config"], "| 학습 seed:", cfg_pg["train_args"]["seed"])""")

md("""### 1-1. 동일 조건 비교 (같은 z · 같은 온도 · 같은 샘플 수)

`n_not_in_train` 은 "학습 데이터에 없는 구조의 수"일 뿐, **신규 물질로 검증됐다는 뜻이 아닙니다.**""")

code("""from src.evaluate import EvalConfig, eval_latents, evaluate_model, save_eval
from src.train import load_checkpoint

ecfg = EvalConfig(n_samples=EVAL["n_eval"], tau=EVAL["eval_tau"], seed=EVAL["eval_seed"], filter_preset="anion")
train_set = set(df["smiles"])
rows = {}
for name in ("gan", "pggan"):
    G, D, vocab, args = load_checkpoint(OUT / name / "checkpoint.pt")
    z_eval = eval_latents(ecfg, args.latent_dim)          # 두 모델에 동일한 z
    r = evaluate_model(G, vocab, ecfg, train_set, z=z_eval)
    save_eval(r, OUT / name / "final_eval.json")
    rows[name] = {k: r[k] for k in ("n_samples", "tau", "eval_seed", "validity", "n_valid_unique", "n_not_in_train", "n_filter_pass")}
cmp = pd.DataFrame(rows).T
display(cmp)
BEST = cmp.sort_values(["n_filter_pass", "n_valid_unique"], ascending=False).index[0]
print("선택된 모델 (필터 통과 수 → 유효 고유 구조 수 순):", BEST)

fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
for name, h in [("GAN", hist_gan), ("PGGAN", hist_pggan)]:
    hd = pd.DataFrame(h)
    for ax, k in zip(axes, ["lossD", "validity", "n_valid_unique"]):
        ax.plot(hd["epoch"], hd[k], marker="o", label=name); ax.set_title(k); ax.set_xlabel("epoch")
axes[0].legend(); plt.tight_layout(); plt.show()""")

md("""## 2. 후보 필터 (설정 가능, 사유 기록)

* `PRESETS["anion"]` : 단일 조각, 총 전하 −1 (리튬염 음이온 후보)
* `PRESETS["salt"]`  : 2조각(양이온+음이온), 총 전하 0, 조각 전하 (+1, −1)
* `PRESETS["neutral"]` : 단일 조각, 전하 0 (용매·첨가제)

음이온 생성과 완성된 염 생성에 같은 조건을 적용하지 않도록 프리셋을 분리했으며, `FilterConfig` 로 직접 수정할 수 있습니다.
**Protic 규칙**(`PROTIC_RULES`)은 O-H / N-H / S-H 등 구조 표시용이며, anion 기준과 별개이고 실제 사용 가능성 검증이 아닙니다.""")

code("""from src.filters import FilterConfig, PRESETS, PROTIC_RULES, apply_filters, filter_summary

print("Protic 구조 규칙 (표시용):", PROTIC_RULES)
demo = ["[N-]", "[P-]", "FC(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F", "CCCCCCCC", "[Li+].FS(=O)(=O)[N-]S(=O)(=O)F",
        "C[CH2]", "OCCO", "not_a_smiles", "OCCO", "OC(=O)C(F)(F)F"]
for preset in ("anion", "salt", "neutral"):
    print(f"\\n=== preset: {preset} — {PRESETS[preset].note}")
    display(apply_filters(demo, PRESETS[preset], train_set)[["smiles", "passed", "reasons", "is_protic_rule", "protic_rules_matched"]])

# 사용자 정의 예: 라디칼 허용 + 최소 중원자 5
custom = FilterConfig(name="anion_custom", allowed_total_charge=(-1,), n_fragments=(1,), min_heavy_atoms=5, allow_radicals=True)
print(custom.to_dict())""")

md("""## 3. 후보 생성 → 필터 → 벡터 공간 매핑 → 점수 → 선정

점수 열 이름이 실측 학습 여부를 나타냅니다: `heuristic_exploration_score` (score_type = `heuristic_proxy`, 탐색용 휴리스틱) 또는 `ml_pred_target` (실측 학습).
후보 생성 조건(seed, n, tau, sampler, 필터)은 `generation_config.json` 에 저장되며 학습 중 평가 조건과 별개입니다.

후보가 0개이거나 요청 수보다 적어도 처리는 중단되지 않고 `STATUS.json` 에 상태 코드를 남깁니다.

| 코드 | 의미 |
|---|---|
| `NO_VALID_STRUCTURES` | 생성 결과에 유효한 SMILES 가 없음 |
| `NO_FILTER_PASS` | 유효 구조는 있으나 필터 통과 구조가 없음 |
| `NO_EMBEDDABLE` | 필터는 통과했으나 임베딩 조건(모델 vocab 외 토큰 / max_len 초과)으로 제외됨 — **필터 탈락과 구분** |
| `OK_PARTIAL` | 후보가 요청 수보다 적음 |
| `OK` | 요청 수만큼 선정 |""")

code("""from src.candidates import run as select_candidates

GEN = dict(n=4096, top=20, tau=0.6, seed=SEED)      # 최종 후보 생성 조건 (학습 평가 조건과 별개로 명시)
results = {}
for preset in ("anion", "neutral"):
    print(f"\\n##### filter preset = {preset}")
    results[preset] = select_candidates(str(OUT / BEST / "checkpoint.pt"), str(DATA), str(OUT / f"candidates_{preset}"),
                                        filter_preset=preset, region_categories=("anion",), n_clusters=5, **GEN)
    st = json.load(open(OUT / f"candidates_{preset}/STATUS.json"))
    print("STATUS:", st["code"], "-", st["message"])
    print({k: st[k] for k in ("n_valid", "n_valid_unique", "n_filter_pass", "n_embeddable", "n_not_embeddable") if k in st})
    display(pd.read_csv(OUT / f"candidates_{preset}/filter_summary.csv"))""")

code("""# 선정 후보표: 구조 · 선정 점수 · 점수 종류 · 필터 결과 · 선정 이유 · Protic 규칙 표시
# 구조 필터 통과는 구조 조건 충족을 뜻할 뿐 합성 가능성·수계 적용 가능성 검증이 아님.
SHOW = ["smiles", "selection_score", "score_type", "filter_passed", "filter_reasons", "selection_reason",
        "cluster", "in_reference_region", "is_protic_rule", "protic_rules_matched"]
for preset, picked in results.items():
    print(f"=== {preset}: {len(picked)} candidates")
    if len(picked) == 0:
        st = json.load(open(OUT / f"candidates_{preset}/STATUS.json"))
        print("  선정된 후보 없음 —", st["code"], ":", st["message"])
        continue
    display(picked[SHOW].head(10))
    png = OUT / f"candidates_{preset}/candidates_top.png"
    if png.exists():
        display(Image(str(png)))
    else:
        print("  (구조 그림 없음)")""")

md("""### 3-1. 후보 선정 수 동작 확인

`select_top` 은 항상 `min(top, 선정 가능한 고유 후보 수)` 개를 선정합니다.
`top` 이 클러스터 수보다 작으면 각 클러스터 최고 점수 후보끼리 비교하고, 그 이상이면 클러스터별로 나눈 뒤 남은 자리를 점수순으로 채웁니다.
정렬 기준은 점수 내림차순 → 클러스터 오름차순 → SMILES 오름차순이므로 동점에서도 결과가 재현됩니다.""")

code("""from src.candidates import select_top

pools = {k: pd.read_csv(OUT / f"candidates_{k}/generated_passed.csv") for k in results}
pools = {k: v for k, v in pools.items() if len(v) and "cluster" in v}
if not pools:
    print("선정 가능한 후보가 없어 선정 수 동작 확인을 건너뜁니다.")
else:
    key = max(pools, key=lambda k: pools[k]["cluster"].nunique())   # 클러스터가 가장 많은 풀로 확인
    src_df = pools[key]
    print(f"후보 풀('{key}'):", len(src_df), "개 / 클러스터", src_df["cluster"].nunique(), "개")
    for t in [0, 2, 5, 10 ** 6]:
        sel = select_top(src_df, t)
        print(f"top={t:>7} -> 선정 {len(sel):>3}개"
              + (f"  클러스터 {sorted(int(c) for c in sel['cluster'].unique())}"
                 f" | {sel['selection_reason'].iloc[0][:52]}" if len(sel) else ""))
    shuffled = select_top(src_df.sample(frac=1, random_state=42), 5)
    print("입력 순서를 섞어도 동일 선정:", shuffled["smiles"].tolist() == select_top(src_df, 5)["smiles"].tolist())
    for bad in (-1, 2.5):
        try:
            select_top(src_df, bad)
        except (ValueError, TypeError) as e:
            print(f"top={bad!r} ->", type(e).__name__, str(e)[:70])

# 생성 후보가 한 클러스터에 몰리는 경우 top < 클러스터 수 분기를 실제 후보로 확인할 수 없으므로,
# 아래는 해당 분기 동작만 보이기 위한 합성 입력 (실제 생성 결과 아님).
synthetic = pd.DataFrame([{"smiles": f"C{c}_{i}", "cluster": c, "selection_score": 10 - c - 0.1 * i}
                          for c in range(5) for i in range(3)])
for t in (2, 7):
    sel = select_top(synthetic, t)
    print(f"[합성 입력] 클러스터 5개, top={t} -> {len(sel)}개, 클러스터 {sorted(int(c) for c in sel['cluster'].unique())}"
          f" | {sel['selection_reason'].iloc[0][:52]}")""")

md("""## 4. 벡터 공간 분석 — 두 공간을 구분

| 공간 | 무엇인가 | 용도 |
|---|---|---|
| **Generator z-space** | 생성 입력 노이즈 (Sobol 샘플) | 탐색·보간. 실제 분자를 직접 투영할 수 없음 |
| **Discriminator feature space** | D 중간 레이어 임베딩 | 실제·생성 분자 공동 투영 → PCA/KMeans, 축-기능기 상관, 기준군 분포 영역 |

점선 상자는 기준군(anion) 임베딩의 PCA 평균 ± kσ 인 **기준군 분포 영역**입니다. 실제 사용 가능 영역을 뜻하지 않습니다.""")

code("""from src.latent_analysis import LatentSpaceAnalyzer

pick_dir = OUT / f"candidates_{list(results)[0]}"
for title, fname in [("Generator z-space 지도 (샘플링한 z 의 PCA, 유효/필터통과 표시)", "zspace_map.png"),
                     ("Discriminator feature space 지도 (실제 분자 + 필터 통과 생성 분자 + 기준군 분포 영역)", "latent_map.png")]:
    print(title)
    display(Image(str(pick_dir / fname))) if (pick_dir / fname).exists() else print("  (그림 없음 — 생성 결과 부족)")
print(json.load(open(pick_dir / "latent/reference_region.json"))["meaning"])
display(pd.read_csv(pick_dir / "latent/cluster_summary.csv").head())
print(json.dumps(json.load(open(pick_dir / "latent/axis_interpretation.json")), ensure_ascii=False, indent=2))""")

md("""### 4-1. z-space 보간 — 생성 구조 · 유효성 · 기능기 변화

유효 구조가 없거나 같은 구조만 반복돼도 표와 요약이 그대로 출력됩니다.""")

code("""from src.latent_analysis import sobol_latents, generate, interpolation_table
from src.metrics import canonical

G, D, vocab, args = load_checkpoint(OUT / BEST / "checkpoint.pt")
z = sobol_latents(512, args.latent_dim, seed=SEED)
det = generate(G, vocab, z, tau=0.0)
ok = [i for i, s in enumerate(det) if canonical(s)]
pairs = [(z[ok[0]], z[ok[-1]]) if len(ok) >= 2 else (z[0], z[1]),   # 유효 구조 사이
         (z[2], z[3])]                                             # 임의 z 사이 (유효 구조가 없을 수 있음)
for z0, z1 in pairs:
    tbl = interpolation_table(G, vocab, z0, z1, n_steps=8, tau=0.0)
    print("요약:", tbl.attrs["summary"])
    display(tbl)""")

md("""## 5. 측정 결과 피드백 재학습 (아래 실행은 예시 데이터 — 실제 측정 결과 아님)

`data/experiments_example.csv` 는 **예시 파일**입니다. 결과 폴더의 `DATA_SOURCE.txt` 에 예시 데이터임이 기록되며,
실제 측정 데이터 파일(`smiles,target`)로 바꾸면 특성 모델이 해당 측정값을 학습해 `ml_pred_target` 으로 표기됩니다.""")

code("""from src.feedback import main as feedback_main
feedback_main(["--experiments", "data/experiments_example.csv", "--ckpt", str(OUT / BEST / "checkpoint.pt"),
               "--out", str(OUT / "feedback_example"), "--epochs", str(max(EPOCHS // 3, 2)),
               "--eval_every", str(max(EPOCHS // 3, 2))] + sum([[f"--{k}", str(v)] for k, v in EVAL.items()], []))
print((OUT / "feedback_example/DATA_SOURCE.txt").read_text())""")

md("""## 6. 체크포인트 구조/vocab 일치 검사""")

code("""from src.train import load_checkpoint
import torch
ck = torch.load(OUT / BEST / "checkpoint.pt", map_location="cpu")
bad = dict(ck); bad["vocab"] = ck["vocab"][:-1]           # vocab 크기를 일부러 어긋나게
torch.save(bad, OUT / "bad_ckpt.pt")
for pth in [OUT / "bad_ckpt.pt", OUT / "missing.pt"]:
    try:
        load_checkpoint(pth)
    except (ValueError, FileNotFoundError) as e:
        print(type(e).__name__, "->", str(e)[:140])
print("정상 checkpoint 로드:", load_checkpoint(OUT / BEST / "checkpoint.pt")[3].mode)""")

md("""## 결과물 요약

* `outputs_run/{gan,pggan}/` : checkpoint, config.json(학습 설정·평가 조건·stage별 데이터 수), history.json, final_eval.json
* `outputs_run/candidates_{anion,neutral}/` : generation_config.json, filter_results.csv, filter_summary.csv, generated_passed.csv, candidates_top.csv(+png), zspace_map.png, latent_map.png, latent/
* `outputs_run/feedback_example/` : DATA_SOURCE.txt (예시 데이터 표시), 재학습 결과

필터·선정 로직의 단위 테스트는 `python -m pytest tests -q` 로 실행합니다 (빈 입력, 전량 무효, 필터 통과 0, 임베딩 제외,
top < 클러스터 수, 후보 부족, top=0, 잘못된 top, 동점 재현성).""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
nbf.write(nb, "electrolyte_gan_research.ipynb")
print("written electrolyte_gan_research.ipynb")
