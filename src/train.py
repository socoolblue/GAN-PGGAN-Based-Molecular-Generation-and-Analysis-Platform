"""GAN / PGGAN 학습 스크립트 (WGAN-GP, TensorBoard 모니터링).

예)
  python -m src.train --mode gan   --epochs 300 --out outputs/gan
  python -m src.train --mode pggan --epochs 300 --out outputs/pggan
  tensorboard --logdir outputs

PGGAN 단계별 데이터:
  stage k 의 시퀀스 길이 L_k 에 대해, (BOS + 토큰 + EOS) 길이가 L_k 이하인 **완결된 분자만** 학습에 쓴다
  (prefix 절단 없음). 단계별 사용 가능 분자 수를 기록하며,
  0 개이면 오류, batch 보다 적으면 경고를 낸다.
  한계: 단계마다 데이터 분포(짧은 분자 → 전체)가 달라진다. 노트북 설명 참고.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from .data import DataError, SmilesDataset, SmilesVocab, load_smiles_csv, to_onehot
from .evaluate import EvalConfig, eval_latents, evaluate_model
from .models import Discriminator, Generator, gradient_penalty


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/sample_electrolytes.csv")
    p.add_argument("--mode", choices=["gan", "pggan"], default="gan")
    p.add_argument("--out", default="outputs/run")
    p.add_argument("--epochs", type=int, default=300, help="총 epoch (pggan 은 stage 별로 균등 분배)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--n_stages", type=int, default=3, help="pggan stage 수 (base_len = max_len / 2^(n-1))")
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--feat_dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--betas", type=float, nargs=2, default=(0.5, 0.9))
    p.add_argument("--n_critic", type=int, default=5)
    p.add_argument("--gp_lambda", type=float, default=10.0)
    p.add_argument("--tau", type=float, default=1.0, help="Gumbel-softmax 온도 (학습)")
    p.add_argument("--tau_min", type=float, default=0.5, help="온도 어닐링 하한")
    p.add_argument("--augment", type=int, default=10, help="분자당 SMILES 랜덤화 증강 수")
    p.add_argument("--eval_every", type=int, default=25)
    # ---- 평가 조건 (GAN / PGGAN 공통, 최종 후보 생성과 별개로 명시) ----
    p.add_argument("--n_eval", type=int, default=512, help="평가 샘플 수")
    p.add_argument("--eval_tau", type=float, default=0.0, help="평가 샘플링 온도 (0=argmax)")
    p.add_argument("--eval_seed", type=int, default=1234, help="평가용 잠재벡터 seed")
    p.add_argument("--filter_preset", default="anion", help="평가 시 후보 필터 프리셋 (anion|salt|neutral)")
    p.add_argument("--min_stage_data", type=int, default=8, help="stage 별 최소 학습 분자 수 (미만이면 오류)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", default=None, help="재학습(피드백) 시 이어받을 checkpoint")
    return p.parse_args(argv)


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_models(vocab_size, args):
    base_len = args.max_len // (2 ** (args.n_stages - 1))
    assert base_len * 2 ** (args.n_stages - 1) == args.max_len, "max_len 은 base_len*2^(n_stages-1) 이어야 함"
    G = Generator(vocab_size, args.latent_dim, base_len, args.n_stages, args.channels)
    D = Discriminator(vocab_size, base_len, args.n_stages, args.channels, args.feat_dim)
    return G, D


def stage_schedule(args):
    """epoch -> (stage, alpha). gan 모드는 항상 마지막 stage."""
    if args.mode == "gan":
        return lambda ep: (args.n_stages - 1, 1.0)
    per = max(args.epochs // args.n_stages, 1)

    def f(ep):
        stage = min(ep // per, args.n_stages - 1)
        if stage == 0:
            return 0, 1.0
        pos = (ep - stage * per) / max(per // 2, 1)  # 전반부 fade-in
        return stage, float(min(1.0, pos))
    return f


def stage_data_report(ds: SmilesDataset, G: Generator, args) -> dict:
    """stage 별 완결 분자 수 기록 + 부족 시 오류/경고."""
    report = {}
    stages = range(args.n_stages) if args.mode == "pggan" else [args.n_stages - 1]
    for st in stages:
        L = G.seq_len(st)
        n = int((ds.lengths <= L).sum())
        report[f"stage{st}_L{L}"] = n
        if n == 0:
            raise DataError(f"stage {st} (길이 {L}) 에 맞는 완결 분자가 0개입니다. "
                            f"--max_len/--n_stages 를 조정하거나 짧은 분자를 추가하세요. (전체 {len(ds)})")
        if n < args.min_stage_data:
            raise DataError(f"stage {st} (길이 {L}) 학습 분자가 {n}개로 --min_stage_data={args.min_stage_data} 미만입니다.")
        if n < args.batch_size:
            print(f"[warn] stage {st} (L={L}) 분자 {n}개 < batch_size {args.batch_size}: 작은 배치로 학습합니다.")
    print("stage 별 완결 학습 분자 수:", report)
    return report


def train(args):
    set_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)

    df = load_smiles_csv(Path(args.data))
    smiles = df["smiles"].tolist()
    if args.resume:  # 재학습 시 모델 구조 유지를 위해 기존 vocab 사용 (미등록 토큰 분자는 제외됨)
        ck_v = torch.load(args.resume, map_location="cpu")["vocab"]
        vocab = SmilesVocab(itos=ck_v, stoi={t: i for i, t in enumerate(ck_v)})
    else:
        vocab = SmilesVocab.build(smiles)
    ds = SmilesDataset(smiles, vocab, args.max_len, augment=args.augment, seed=args.seed)
    train_set = set(smiles)
    V = len(vocab)
    print(f"molecules={len(smiles)}  augmented={len(ds)}  vocab={V}  device={dev}  mode={args.mode}")

    G, D = build_models(V, args)
    G.to(dev); D.to(dev)
    stage_report = stage_data_report(ds, G, args)
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=tuple(args.betas))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=tuple(args.betas))
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        _check_ckpt_compat(ck, args, V)
        G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"])
        print("resumed from", args.resume)

    writer = SummaryWriter(log_dir=str(out / "tb"))
    writer.add_text("config", json.dumps(vars(args), ensure_ascii=False, indent=2))
    sched = stage_schedule(args)
    eval_cfg = EvalConfig(n_samples=args.n_eval, tau=args.eval_tau, seed=args.eval_seed,
                          filter_preset=args.filter_preset)
    fixed_z = eval_latents(eval_cfg, args.latent_dim, dev)
    history, step, t0 = [], 0, time.time()
    cur_stage, dl = None, None

    for ep in range(args.epochs):
        stage, alpha = sched(ep)
        L = G.seq_len(stage)
        if stage != cur_stage:  # stage 가 바뀌면 그 길이에 완결되는 분자만으로 DataLoader 재구성
            sub = ds.subset_for_length(L)
            dl = DataLoader(TensorDataset(sub), batch_size=args.batch_size, shuffle=True,
                            drop_last=len(sub) > args.batch_size)
            cur_stage = stage
        tau = max(args.tau_min, args.tau * (1 - ep / args.epochs))  # 온도 어닐링
        lossD_acc, lossG_acc, nD, nG = 0.0, 0.0, 0, 0
        for (real_ids,) in dl:
            real = to_onehot(real_ids.to(dev), V)   # (B, V, L) — 완결 분자, 절단 없음
            B = real.size(0)
            z = torch.randn(B, args.latent_dim, device=dev)
            fake = G(z, stage, alpha, tau).detach()
            lossD = D(fake, stage, alpha).mean() - D(real, stage, alpha).mean() \
                + args.gp_lambda * gradient_penalty(D, real, fake, stage, alpha)
            optD.zero_grad(); lossD.backward(); optD.step()
            lossD_acc += lossD.item(); nD += 1; step += 1
            if step % args.n_critic == 0:
                z = torch.randn(B, args.latent_dim, device=dev)
                lossG = -D(G(z, stage, alpha, tau), stage, alpha).mean()
                optG.zero_grad(); lossG.backward(); optG.step()
                lossG_acc += lossG.item(); nG += 1

        writer.add_scalar("loss/D", lossD_acc / max(nD, 1), ep)
        writer.add_scalar("loss/G", lossG_acc / max(nG, 1), ep)
        writer.add_scalar("prog/stage", stage, ep); writer.add_scalar("prog/alpha", alpha, ep)
        writer.add_scalar("prog/tau", tau, ep)

        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            m = evaluate_model(G, vocab, eval_cfg, train_set, z=fixed_z, stage=stage, alpha=alpha)
            for k in ("validity", "n_valid_unique", "n_not_in_train", "n_filter_pass"):
                writer.add_scalar(f"eval/{k}", m[k], ep)
            writer.add_text("samples", "\n\n".join(m["valid_unique_smiles"][:20]) or "(no valid)", ep)
            rec = {"epoch": ep + 1, "stage": stage, "alpha": round(alpha, 3), "L": L,
                   "n_train_stage": int((ds.lengths <= L).sum()),
                   "lossD": round(lossD_acc / max(nD, 1), 4), "lossG": round(lossG_acc / max(nG, 1), 4),
                   "validity": m["validity"], "n_valid_unique": m["n_valid_unique"],
                   "n_not_in_train": m["n_not_in_train"], "n_filter_pass": m["n_filter_pass"],
                   "sec": round(time.time() - t0)}
            history.append(rec); print(rec)
            torch.save({"G": G.state_dict(), "D": D.state_dict(), "args": vars(args),
                        "vocab": vocab.itos, "eval_config": eval_cfg.to_dict()}, out / "checkpoint.pt")

    writer.close()
    vocab.save(out / "vocab.txt")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "config.json").write_text(json.dumps({
        "train_args": vars(args), "eval_config": eval_cfg.to_dict(),
        "stage_train_counts": stage_report, "n_molecules": len(smiles), "n_augmented": len(ds),
        "vocab_size": V, "torch": torch.__version__,
    }, ensure_ascii=False, indent=2))
    return history


def _check_ckpt_compat(ck: dict, args, vocab_size: int):
    a = ck["args"]
    for k in ("latent_dim", "channels", "feat_dim", "max_len", "n_stages"):
        if a.get(k) != getattr(args, k):
            raise ValueError(f"checkpoint 와 설정이 다릅니다: {k} ckpt={a.get(k)} vs 요청={getattr(args, k)}")
    if len(ck["vocab"]) != vocab_size:
        raise ValueError(f"checkpoint vocab 크기({len(ck['vocab'])}) 와 현재 vocab({vocab_size}) 불일치")


def load_checkpoint(path: str | Path, device="cpu"):
    """checkpoint 에 저장된 args/vocab 으로 모델을 재구성 — 구조·vocab 불일치 시 명확한 오류."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint 가 없습니다: {path.resolve()}")
    ck = torch.load(path, map_location=device)
    for k in ("G", "D", "args", "vocab"):
        if k not in ck:
            raise ValueError(f"checkpoint 형식 오류: '{k}' 항목 없음 ({path})")
    args = argparse.Namespace(**ck["args"])
    vocab = SmilesVocab(itos=ck["vocab"], stoi={t: i for i, t in enumerate(ck["vocab"])})
    G, D = build_models(len(vocab), args)
    try:
        G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"])
    except RuntimeError as e:
        raise ValueError(f"checkpoint 가중치와 모델 구조가 맞지 않습니다 ({path}): {e}") from e
    G.eval(); D.eval()
    G.eval_config = ck.get("eval_config")
    return G, D, vocab, args


if __name__ == "__main__":
    train(get_args())
