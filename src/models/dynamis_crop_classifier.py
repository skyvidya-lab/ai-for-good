"""
DynamisCropClassifier — MKM + ChaosAttention + dual-head classifier.

Architecture:
    (B, T, F_in) raw sequence
        -> input projection
        -> MKM rollout (state_dim = 7, A initialised with phenology prior)
                -> posterior state trajectory (B, T, 7)
                -> innovations (B, T, 7)
        -> ChaosAttention over projected + state-augmented features
                (physics_state per sample: [chaos_score, hurst])
        -> pooled representation
        -> crop head (3 classes) + phenophase head (7 classes)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..dynamis.chaos_attention import ChaosAttention
from ..dynamis.dynamis_core import Executor, MarkovKalmanModule
from ..dynamis.phenology_prior import N_PHENOPHASES, build_phenology_prior_tensor


@dataclass
class DynamisModelConfig:
    input_dim: int = 17      # 12 bands + 5 vegetation indices
    state_dim: int = N_PHENOPHASES  # 7 phenophases
    hidden_dim: int = 64
    attn_heads: int = 4
    n_crops: int = 3
    use_phenology_prior: bool = True
    lambda_prior_strength: float = 0.5  # 0=random init, 1=hard prior
    p_init_scale: float = 0.1  # initial P = p_init_scale * eye(state_dim)
    mkm_noise_spread: float = 0.5  # heterogeneity in log_Q / log_R init


class DynamisCropClassifier(nn.Module):
    """Physics-informed classifier for crop type + phenophase."""

    def __init__(self, cfg: DynamisModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or DynamisModelConfig()
        c = self.cfg

        self.input_proj = nn.Linear(c.input_dim, c.hidden_dim)
        self.executor = Executor(
            input_dim=c.hidden_dim,
            state_dim=c.state_dim,
            hidden_dim=c.hidden_dim,
            output_dim=None,
        )
        self.mkm = MarkovKalmanModule(state_dim=c.state_dim, noise_spread=c.mkm_noise_spread)

        if c.use_phenology_prior:
            with torch.no_grad():
                A_prior = build_phenology_prior_tensor()
                blended = (1 - c.lambda_prior_strength) * self.mkm.A + c.lambda_prior_strength * A_prior
                self.mkm.A.copy_(blended)

        # Attention operates over projected features concatenated with Kalman state
        self.attn_in_dim = c.hidden_dim + c.state_dim
        self.attn_proj = nn.Linear(self.attn_in_dim, c.hidden_dim)
        self.chaos_attn = ChaosAttention(
            embed_dim=c.hidden_dim,
            num_heads=c.attn_heads,
            n_physics=2,
            dropout=0.1,
        )
        self.pool_norm = nn.LayerNorm(c.hidden_dim)
        self.head_crop = nn.Linear(c.hidden_dim, c.n_crops)
        self.head_pheno = nn.Linear(c.hidden_dim, c.state_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        hurst: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T, F_in)
            mask: (B, T) bool — True where the timestep is valid.
            hurst: (B,) precomputed Hurst exponent per point (0..1). If None, defaults to 0.5.

        Returns dict:
            crop_logits: (B, n_crops)
            pheno_logits: (B, T, n_phenos) — per-timestep phenophase classification
            innovations: (B, T, state_dim)
            uncertainty: (B,) trace of final P matrix
            final_state: (B, state_dim)
        """
        B, T, _ = x.shape
        device = x.device

        h = self.input_proj(x)  # (B, T, hidden)

        # MKM rollout — start with a smaller-than-identity P so the filter is
        # not saturated at t=0 (the first run had trace(P) ≈ 1.07 for every
        # sample because P_init = I led to instant equilibrium).
        h_exec = torch.zeros(B, self.cfg.hidden_dim, device=device)
        x_mkm = torch.zeros(B, self.cfg.state_dim, device=device)
        P_mkm = (
            self.cfg.p_init_scale
            * torch.eye(self.cfg.state_dim, device=device)
            .unsqueeze(0)
            .expand(B, -1, -1)
            .clone()
        )

        state_seq = []
        innov_seq = []
        for t in range(T):
            x_pred, P_pred = self.mkm.predict(x_mkm, P_mkm)
            h_exec, measurement, _ = self.executor(h[:, t, :], x_pred, h_exec)
            x_mkm, P_mkm, innovation = self.mkm.update(x_pred, P_pred, measurement)
            state_seq.append(x_mkm)
            innov_seq.append(innovation)

        state_traj = torch.stack(state_seq, dim=1)  # (B, T, state_dim)
        innovations = torch.stack(innov_seq, dim=1)  # (B, T, state_dim)

        # Physics state per sample
        chaos_score = innovations.pow(2).mean(dim=(1, 2)).clamp(0, 10) / 10.0  # (B,)
        if hurst is None:
            hurst_vec = torch.full((B,), 0.5, device=device)
        else:
            hurst_vec = hurst.to(device=device, dtype=x.dtype)
        physics_state = torch.stack([chaos_score, hurst_vec], dim=-1)  # (B, 2)

        # Attention over [hidden | state]
        attn_in = torch.cat([h, state_traj], dim=-1)
        attn_in = self.attn_proj(attn_in)
        attn_out = self.chaos_attn(attn_in, physics_state=physics_state, attn_mask=mask)

        # Pool (masked mean)
        if mask is not None:
            m = mask.to(attn_out.dtype).unsqueeze(-1)
            pooled = (attn_out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)
        else:
            pooled = attn_out.mean(dim=1)
        pooled = self.pool_norm(pooled)

        crop_logits = self.head_crop(pooled)
        pheno_logits = self.head_pheno(attn_out)  # (B, T, 7)

        uncertainty = P_mkm.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # trace per sample

        return {
            "crop_logits": crop_logits,
            "pheno_logits": pheno_logits,
            "innovations": innovations,
            "uncertainty": uncertainty,
            "final_state": x_mkm,
            "state_trajectory": state_traj,
        }


__all__ = ["DynamisCropClassifier", "DynamisModelConfig"]
