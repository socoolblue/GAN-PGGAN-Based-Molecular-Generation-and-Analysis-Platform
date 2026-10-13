"""후보 필터·선정 로직 테스트.   실행: python -m pytest tests -q   또는  python tests/test_candidates.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import candidates as C                     # noqa: E402
from src.filters import PRESETS, FILTER_COLUMNS, apply_filters, filter_summary  # noqa: E402


# ---------------------------------------------------------------- filters
def test_apply_filters_empty_input_keeps_schema():
    df = apply_filters([], PRESETS["anion"])
    assert list(df.columns) == FILTER_COLUMNS and len(df) == 0
    assert df["passed"].dtype == bool
    assert df[df["passed"]]["canonical"].tolist() == []        # KeyError 없이 동작
    s = filter_summary(df)
    assert list(s.columns) == ["reason", "count"] and s.iloc[0]["count"] == 0


def test_apply_filters_passed_is_bool():
    df = apply_filters(["[N-]", "CCO", "bad"], PRESETS["neutral"])
    assert df["passed"].dtype == bool and df["passed"].tolist() == [False, True, False]


# ---------------------------------------------------------------- select_top
def _ranked(n_per_cluster=3, n_cl=5, tie=False):
    rows = []
    for c in range(n_cl):
        for i in range(n_per_cluster):
            score = 1.0 if tie else 10 - c - 0.1 * i
            rows.append({"smiles": f"C{c}_{i}", "cluster": c, "selection_score": score})
    return pd.DataFrame(rows)


def test_top_smaller_than_clusters():
    picked = C.select_top(_ranked(), top=2)
    assert len(picked) == 2 and picked["smiles"].is_unique
    assert picked["smiles"].tolist() == ["C0_0", "C1_0"]           # 클러스터 최고점끼리 비교
    assert picked["selection_reason"].str.contains("among per-cluster best").all()


def test_fewer_candidates_than_top():
    r = _ranked(n_per_cluster=1, n_cl=3)
    picked = C.select_top(r, top=20)
    assert len(picked) == 3


def test_more_candidates_than_top_fills_by_rank():
    picked = C.select_top(_ranked(n_per_cluster=4, n_cl=3), top=8)
    assert len(picked) == 8 and picked["smiles"].is_unique
    assert (picked["selection_reason"].str.startswith("per-cluster top 2")).sum() == 6
    assert (picked["selection_reason"] == "fill by global selection_score rank").sum() == 2


def test_top_zero_and_invalid_top():
    assert len(C.select_top(_ranked(), 0)) == 0
    with pytest.raises(ValueError):
        C.select_top(_ranked(), -1)
    for bad in (2.5, "3", True, None):
        with pytest.raises(TypeError):
            C.select_top(_ranked(), bad)


def test_duplicates_not_selected_twice():
    r = pd.concat([_ranked(2, 2), _ranked(2, 2)])                 # 같은 후보 중복
    picked = C.select_top(r, 10)
    assert picked["smiles"].is_unique and len(picked) == 4


def test_tie_reproducible():
    r = _ranked(3, 5, tie=True)
    a = C.select_top(r.sample(frac=1, random_state=1), 4)
    b = C.select_top(r.sample(frac=1, random_state=7), 4)
    assert a["smiles"].tolist() == b["smiles"].tolist() == ["C0_0", "C1_0", "C2_0", "C3_0"]


# ---------------------------------------------------------------- run() 빈 결과 경로 (소형 체크포인트 + generate 패치)
@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    from src.train import get_args, train
    out = tmp_path_factory.mktemp("ckpt")
    train(get_args(["--mode", "gan", "--epochs", "2", "--eval_every", "2", "--n_eval", "16",
                    "--augment", "0", "--data", str(ROOT / "data/sample_electrolytes.csv"), "--out", str(out)]))
    return out / "checkpoint.pt"


def _run_with(monkeypatch, tmp_path, ckpt, fake_generate, top=5, preset="anion"):
    monkeypatch.setattr(C, "generate", fake_generate)
    out = tmp_path / "cand"
    res = C.run(str(ckpt), str(ROOT / "data/sample_electrolytes.csv"), str(out), n=8, top=top,
                tau=0.0, filter_preset=preset, n_clusters=2)
    return res, out


def _check_empty_outputs(out):
    import json
    for f in ("filter_results.csv", "filter_summary.csv", "candidates_top.csv", "generated_passed.csv"):
        df = pd.read_csv(out / f)                                  # 헤더 유지 → 읽기 가능
        assert len(df.columns) > 0
    assert (out / "latent_map.png").exists() and (out / "zspace_map.png").exists()
    return json.loads((out / "STATUS.json").read_text())


def test_run_all_invalid(monkeypatch, tmp_path, tiny_ckpt):
    res, out = _run_with(monkeypatch, tmp_path, tiny_ckpt, lambda G, v, z, tau=0.0: ["((("] * len(z))
    st = _check_empty_outputs(out)
    assert len(res) == 0 and st["code"] == "NO_VALID_STRUCTURES" and st["n_valid"] == 0


def test_run_valid_but_no_filter_pass(monkeypatch, tmp_path, tiny_ckpt):
    res, out = _run_with(monkeypatch, tmp_path, tiny_ckpt, lambda G, v, z, tau=0.0: ["CCCCCC"] * len(z), preset="anion")
    st = _check_empty_outputs(out)
    assert len(res) == 0 and st["code"] == "NO_FILTER_PASS" and st["n_valid_unique"] == 1 and st["n_filter_pass"] == 0


def test_run_not_embeddable_distinguished(monkeypatch, tmp_path, tiny_ckpt):
    # 필터는 통과하지만 모델 vocab 에 없는 원소(I) 포함 → 임베딩 제외 (필터 탈락과 구분)
    res, out = _run_with(monkeypatch, tmp_path, tiny_ckpt, lambda G, v, z, tau=0.0: ["ICCO"] * len(z), preset="neutral")
    st = _check_empty_outputs(out)
    assert st["code"] == "NO_EMBEDDABLE" and st["n_filter_pass"] == 1 and st["n_not_embeddable"] == 1
    assert (out / "excluded_by_embedding.txt").read_text().strip() == "OCCI"   # canonical 표기


def test_run_partial_and_top_zero(monkeypatch, tmp_path, tiny_ckpt):
    mols = ["OCCO", "OCCOCCO", "COCCOC", "OCCN", "CC(O)CO", "OCCCO", "COCCO", "OCC(O)CO"]
    gen = lambda G, v, z, tau=0.0: (mols * (len(z) // len(mols) + 1))[:len(z)]
    res, out = _run_with(monkeypatch, tmp_path, tiny_ckpt, gen, top=20, preset="neutral")
    import json
    st = json.loads((out / "STATUS.json").read_text())
    assert st["code"] == "OK_PARTIAL" and 0 < len(res) < 20 and res["smiles"].is_unique
    assert len(res) == len(pd.read_csv(out / "candidates_top.csv"))
    res0, out0 = _run_with(monkeypatch, tmp_path / "z", tiny_ckpt, gen, top=0, preset="neutral")
    assert len(res0) == 0 and len(pd.read_csv(out0 / "candidates_top.csv").columns) > 0
    with pytest.raises(ValueError):
        _run_with(monkeypatch, tmp_path / "neg", tiny_ckpt, gen, top=-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
