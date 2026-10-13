"""SMILES 시퀀스용 GAN / PGGAN (Progressive Growing) 모델.

* Generator : 노이즈 z -> (V, L) 토큰 확률 분포 (Gumbel-Softmax)
* Discriminator : (V, L) one-hot(soft) 시퀀스 -> Wasserstein score
* Progressive growing : 이미지 PGGAN 의 "해상도" 를 "시퀀스 길이" 로 치환.
    stage 0 -> L0,  stage 1 -> 2*L0, ... stage S-1 -> max_len
  stage 전환 시 새 블록을 alpha(0->1) 로 fade-in 한다.
  일반 GAN 모드에서는 항상 마지막 stage, alpha=1 로 동작한다 (= 동일 코드 재사용).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelNorm(nn.Module):
    def forward(self, x, eps=1e-8):
        return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + eps)


def _gblock(c_in, c_out):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv1d(c_in, c_out, 3, padding=1), nn.LeakyReLU(0.2), PixelNorm(),
        nn.Conv1d(c_out, c_out, 3, padding=1), nn.LeakyReLU(0.2), PixelNorm(),
    )


def _dblock(c_in, c_out):
    return nn.Sequential(
        nn.Conv1d(c_in, c_in, 3, padding=1), nn.LeakyReLU(0.2),
        nn.Conv1d(c_in, c_out, 3, padding=1), nn.LeakyReLU(0.2),
        nn.AvgPool1d(2),
    )


class Generator(nn.Module):
    def __init__(self, vocab_size: int, latent_dim: int = 64, base_len: int = 16,
                 n_stages: int = 3, channels: int = 128):
        super().__init__()
        self.vocab_size, self.latent_dim = vocab_size, latent_dim
        self.base_len, self.n_stages, self.channels = base_len, n_stages, channels
        self.fc = nn.Sequential(nn.Linear(latent_dim, channels * base_len), nn.LeakyReLU(0.2))
        self.stem = nn.Sequential(nn.Conv1d(channels, channels, 3, padding=1),
                                  nn.LeakyReLU(0.2), PixelNorm())
        self.blocks = nn.ModuleList([_gblock(channels, channels) for _ in range(n_stages - 1)])
        self.to_logits = nn.ModuleList([nn.Conv1d(channels, vocab_size, 1) for _ in range(n_stages)])

    def seq_len(self, stage: int) -> int:
        return self.base_len * (2 ** stage)

    def logits(self, z, stage: int | None = None, alpha: float = 1.0):
        """returns (B, V, L_stage) 토큰 logits."""
        stage = self.n_stages - 1 if stage is None else stage
        h = self.fc(z).view(-1, self.channels, self.base_len)
        h = self.stem(h)
        h_prev = None
        for k in range(stage):
            h_prev = h
            h = self.blocks[k](h)
        logits = self.to_logits[stage](h)
        if stage > 0 and alpha < 1.0:
            prev = F.interpolate(self.to_logits[stage - 1](h_prev), scale_factor=2, mode="nearest")
            logits = alpha * logits + (1 - alpha) * prev
        return logits

    def forward(self, z, stage: int | None = None, alpha: float = 1.0,
                tau: float = 1.0, hard: bool = True):
        """학습용: Gumbel-Softmax (hard=True 면 straight-through one-hot, 미분 가능)."""
        return F.gumbel_softmax(self.logits(z, stage, alpha), tau=tau, hard=hard, dim=1)

    @torch.no_grad()
    def sample_ids(self, z, stage=None, alpha=1.0, tau: float = 0.0):
        """샘플링: tau=0 이면 argmax(결정적), tau>0 이면 온도 샘플링."""
        lg = self.logits(z, stage, alpha)
        if tau <= 0:
            return lg.argmax(dim=1)  # (B, L)
        probs = F.softmax(lg / tau, dim=1).permute(0, 2, 1)  # (B, L, V)
        return torch.multinomial(probs.flatten(0, 1), 1).view(lg.size(0), lg.size(2))


class Discriminator(nn.Module):
    def __init__(self, vocab_size: int, base_len: int = 16, n_stages: int = 3,
                 channels: int = 128, feat_dim: int = 64):
        super().__init__()
        self.base_len, self.n_stages, self.channels = base_len, n_stages, channels
        self.from_onehot = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(vocab_size, channels, 1), nn.LeakyReLU(0.2)) for _ in range(n_stages)])
        self.blocks = nn.ModuleList([_dblock(channels, channels) for _ in range(n_stages - 1)])
        self.final_conv = nn.Sequential(nn.Conv1d(channels, channels, 3, padding=1), nn.LeakyReLU(0.2))
        # 중간 레이어 (feature space) -> 벡터 공간 분석에 사용
        self.feat = nn.Sequential(nn.Linear(channels * base_len, feat_dim), nn.LeakyReLU(0.2))
        self.out = nn.Linear(feat_dim, 1)

    def features(self, x, stage: int | None = None, alpha: float = 1.0):
        """x: (B, V, L_stage) -> (B, feat_dim) 중간 레이어 임베딩."""
        stage = self.n_stages - 1 if stage is None else stage
        h = self.from_onehot[stage](x)
        for k in range(stage - 1, -1, -1):
            h = self.blocks[k](h)
            if k == stage - 1 and alpha < 1.0:
                h_prev = self.from_onehot[stage - 1](F.avg_pool1d(x, 2))
                h = alpha * h + (1 - alpha) * h_prev
        h = self.final_conv(h)
        return self.feat(h.flatten(1))

    def forward(self, x, stage=None, alpha=1.0):
        return self.out(self.features(x, stage, alpha)).squeeze(1)


def gradient_penalty(D, real, fake, stage, alpha):
    B = real.size(0)
    eps = torch.rand(B, 1, 1, device=real.device)
    inter = (eps * real + (1 - eps) * fake).requires_grad_(True)
    score = D(inter, stage, alpha)
    grad = torch.autograd.grad(score.sum(), inter, create_graph=True)[0]
    return ((grad.flatten(1).norm(2, dim=1) - 1) ** 2).mean()
