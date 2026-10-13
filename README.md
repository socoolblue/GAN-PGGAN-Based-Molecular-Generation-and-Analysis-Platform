# 수계 고체전해질 · 리튬염 분자 생성 GAN / PGGAN

"딥러닝 기반 알고리즘 타당성 조사" 문서의 3개 과업을 그대로 코드로 옮긴 PyTorch 프로젝트입니다.

| 문서 항목 | 구현 |
|---|---|
| GAN 모델 구축 (Generator / Discriminator) | `src/models.py` – SMILES 시퀀스용 WGAN-GP (Gumbel-Softmax) |
| PGGAN (Progressive Growing) | 같은 모델을 `--mode pggan` 으로 실행: 시퀀스 길이 16 → 32 → 64 로 단계 성장 + fade-in |
| 학습 안정성 하이퍼파라미터 튜닝 | `src/tune.py` – lr / n_critic / latent_dim / channels / tau / GP 랜덤 서치 |
| Tensorboard 실시간 모니터링 | `src/train.py` – loss, stage/alpha, validity·uniqueness·novelty, 생성 SMILES 텍스트 |
| 중간 레이어를 Latent space 로 활용한 벡터 공간 분석 | `src/latent_analysis.py` – `Discriminator.features()` 임베딩 → PCA / KMeans |
| 랜덤 입력 벡터 공간 Mapping, 고차원 잠재공간 효율 탐색 | `sobol_latents()` (Sobol 준난수) + `slerp` 보간 |
| 가용 벡터 공간 범위 설정 | `LatentSpaceAnalyzer.fit(region_categories=...)` – 기준군(음이온) 평균 ± kσ |
| 구획화·클러스터링, 방향 ↔ 기능기 상관 | `cluster_summary()`, `direction_correlations()`, `interpret_axes()` |
| 생성 소재 특성 평가 ML 모델 | `src/property_model.py` – RandomForest (디스크립터 + Morgan FP + 기능기) |
| 소규모 후보군 도출 | `src/filters.py` (설정 가능한 필터·제외 사유·Protic 규칙 표시) + `src/candidates.py` – 탐색용 휴리스틱 점수/실측 ML 예측 구분, 클러스터 다양성 기준 선정 |
| GAN·PGGAN 동일 조건 평가 | `src/evaluate.py` – 같은 샘플 수·온도·고정 잠재벡터로 유효율 / 유효 고유 수 / 학습데이터 미포함 수 / 필터 통과 수 기록 |
| 실험 결과 피드백 · 재학습 | `src/feedback.py` – experiments.csv 병합 → 특성모델 재학습 → GAN `--resume` 재학습 |
| 데이터 증강 | `src/data.py` – SMILES randomization (`--augment N`) |

## 설치

```bash
pip install -r requirements.txt
```

## 빠른 실행 (전체 파이프라인)

```bash
python data/make_sample_data.py          # 샘플 CSV 생성 (이미 포함되어 있음)
python run_pipeline.py --epochs 300      # GAN vs PGGAN 학습·비교 → 벡터공간 분석 → 후보 20개
tensorboard --logdir outputs             # 학습 모니터링
```

산출물 (`outputs/`)

```
gan/, pggan/            checkpoint.pt, history.json, vocab.txt, tb/ (TensorBoard)
comparison.json         두 모델 최종 지표 비교
candidates/
  generated_all.csv     유효·신규 생성 분자 전체 (좌표, 클러스터, 가용범위 여부, 예측 특성, 디스크립터, 기능기)
  candidates_top.csv    선정된 소규모 후보군  (+ candidates_top.png 구조 그림)
  latent_map.png        Discriminator feature space 지도 (실제 분자 / 필터 통과 생성 분자 / 기준군 분포 영역)
  zspace_map.png        Generator z-space 지도
  filter_results.csv    모든 생성 구조의 필터 통과 여부·제외 이유·Protic 규칙 표시
  generation_config.json 후보 생성 조건(seed, n, tau, 필터)
  latent/               cluster_summary.csv, direction_correlations.csv, axis_interpretation.json, region.json
  property_model.joblib 특성 예측 모델
```

## 개별 단계

```bash
# 1) 학습
python -m src.train --mode gan   --epochs 300 --out outputs/gan
python -m src.train --mode pggan --epochs 300 --out outputs/pggan --n_stages 3 --max_len 64
# 2) 하이퍼파라미터 튜닝
python -m src.tune --mode pggan --trials 8 --epochs 120
# 3) 후보 생성·분석·선정
python -m src.candidates --ckpt outputs/pggan/checkpoint.pt --n 4096 --top 20
# 4) 실험 결과 피드백 후 재학습  (data/experiments.csv: smiles,target)
python -m src.feedback --experiments data/experiments.csv --ckpt outputs/pggan/checkpoint.pt --out outputs/round2
```

## 실제 데이터로 교체하기

`data/sample_electrolytes.csv` 와 같은 열(`smiles, name, category, target`)의 CSV 를 만들고 `--data` 로 지정하면 됩니다.

* `category` : 가용 벡터 공간 범위를 정할 기준군 이름 (`--region_categories anion` 등).
* `target` : 측정값(이온전도도 등). 값이 있으면 특성 예측 모델이 이 값을 학습하고,
  없으면 `property_model.proxy_score()` (**탐색용 휴리스틱 점수**, 열 이름 `heuristic_exploration_score`) 를 대신 사용하며 실제 물성 예측값이 아닙니다.
  실제 연구에서는 반드시 측정 데이터로 대체하세요.

## 설계 메모

* **WGAN-GP + Gumbel-Softmax(straight-through)**: 이산 토큰 생성에서도 Generator 로 기울기가 흐르고,
  Discriminator 가 "soft vs one-hot" 차이로 쉽게 구분하는 문제를 막습니다.
* **PGGAN 의 시퀀스 치환**: 이미지의 해상도 대신 SMILES 길이를 단계적으로 늘립니다.
  각 stage 에서는 그 길이 안에 완결되는 분자만 학습하고(절단 없음), 새 블록은 `alpha` 로 fade-in 됩니다. 단계별 데이터 분포가 달라지는 한계가 있습니다.
  일반 GAN 모드는 마지막 stage 만 사용하므로 두 모델이 **동일 코드·동일 파라미터 수**로 공정 비교됩니다.
* **벡터 공간**: Generator 입력 z 공간(탐색용)과 Discriminator 중간층 feature 공간(분석용) 두 가지를 씁니다.
  실제·생성 분자를 같은 feature 공간에 투영해 클러스터·축 의미·가용 범위를 해석합니다.
* 포함된 107종 예시 데이터는 학습·생성·분석 절차의 실행을 확인하기 위한 데이터입니다.
  생성 구조의 유효율은 데이터 구성, 모델, 학습 횟수 및 샘플링 조건에 따라 달라지므로, 예시 실행 결과를 모델의 일반적인 성능으로 해석하지 않아야 합니다.
  연구 데이터에 적용할 때는 `src/tune.py` 등을 활용하여 학습 조건을 탐색하고, 동일한 평가 조건에서 성능을 확인해야 합니다.

## 노트북

`electrolyte_gan_research.ipynb` — 데이터 로딩부터 학습·평가·후보 필터링·벡터 공간 분석·후보 선정까지 실행하는 연구 노트북 (실행 결과 포함).
프로젝트 루트에서 `jupyter lab` 실행 후 열면 됩니다. 첫 셀의 `QUICK = True` 는 짧은 학습 설정이며, 이 조건의 유효율·후보 수는 결과 해석에 사용할 수 없습니다. 충분한 학습은 `QUICK = False` 로 실행합니다.

## 테스트

```bash
python -m pytest tests -q
```
필터 스키마(빈 입력 포함)와 후보 선정 수·재현성·입력 검증을 확인합니다.

## 결과 해석 시 유의

- `heuristic_exploration_score` 는 구조 규칙 기반의 탐색용 점수이며 실제 물성 예측값이 아닙니다. 실측 `target` 학습 시에만 `ml_pred_target` 으로 표기됩니다.
- 구조 필터 통과는 SMILES 수준의 구조 조건 충족을 의미하며, 합성 가능성이나 수계 전해질 적용 가능성을 검증한 것이 아닙니다.
- `n_not_in_train` 은 학습 데이터 미포함 여부일 뿐 신규 물질 검증이 아닙니다.
- `data/experiments_example.csv` 는 예시 데이터이며, 이를 사용한 결과 폴더에는 `DATA_SOURCE.txt` 로 표시됩니다.
