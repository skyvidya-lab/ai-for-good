"""
DynamisCropClassifier — MKM + ChaosAttention + physics-injected dual head.

Architecture:
    (B, T, F_in) raw sequence
        -> input projection + LayerNorm
        -> MKM rollout (state_dim = 7, A initialised with phenology prior)
                -> posterior state trajectory (B, T, 7)
                -> innovations (B, T, 7)
                -> P trajectory (B, T, 7) via diagonal
        -> ChaosAttention over projected + state-augmented features
                (physics_state per sample: [chaos_score, hurst])
        -> masked-mean pool
        -> Physics-vector injection into crop head:
             [pooled | innov_stats | final_state | trace(P) stats | hurst]
        -> crop head MLP (3 classes)
        -> phenophase head (7 classes per timestep)

Why the physics injection?
    The v2 run on 184 points showed Dynamis losing to a LightGBM baseline by
    ~7pp OA. Root cause: MKM computed innovations and Kalman uncertainty
    (the signals that make Dynamis different) but only fed them into the
    regularisation loss / attention bias. The classification head was
    operating on plain pooled attention, so LightGBM with phenology-aware
    features had richer signal than Dynamis itself.

    This rewrite wires the Dynamis physics OUTPUTS directly into the head.
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
    input_dim: int = 17                     # 12 bands + 5 vegetation indices
    state_dim: int = N_PHENOPHASES          # 7 phenophases
    hidden_dim: int = 64
    attn_heads: int = 4
    n_crops: int = 3
    use_phenology_prior: bool = True
    lambda_prior_strength: float = 0.5      # 0=random init, 1=hard prior
    p_init_scale: float = 0.1               # initial P = p_init_scale * eye
    mkm_noise_spread: float = 0.5           # heterogeneity in log_Q / log_R init
    crop_head_dropout: float = 0.3          # regularises the MLP head on small data
    # Physics feature dim breakdown (used for the crop MLP):
    # innov_mean_mag(1) + innov_max_mag(1) + innov_std_mag(1) + innov_peak_idx(1)
    # + final_state(state_dim) + trace_P_mean(1) + trace_P_max(1) + trace_P_range(1)
    # + hurst(1)
    # = 8 + state_dim


def _physics_dim(state_dim: int) -> int:
    return 8 + state_dim


class DynamisCropClassifier(nn.Module):
    """Physics-informed classifier for crop type + phenophase."""

    def __init__(self, cfg: DynamisModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or DynamisModelConfig()
        c = self.cfg

        self.input_proj = nn.Linear(c.input_dim, c.hidden_dim)
        self.input_norm = nn.LayerNorm(c.hidden_dim)

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

        # Crop head: MLP over [pooled | physics_vector]
        physics_dim = _physics_dim(c.state_dim)
        self.physics_norm = nn.LayerNorm(physics_dim)
        self.head_crop = nn.Sequential(
            nn.Linear(c.hidden_dim + physics_dim, c.hidden_dim),
            nn.GELU(),
            nn.Dropout(c.crop_head_dropout),
            nn.Linear(c.hidden_dim, c.n_crops),
        )
        # Phenophase head is per-timestep, small, no physics injection
        self.head_pheno = nn.Linear(c.hidden_dim, c.state_dim)

    def _extract_physics(
        self,
        innovations: torch.Tensor,   # (B, T, state_dim)
        P_trajectory: torch.Tensor,  # (B, T, state_dim) diagonals
        final_state: torch.Tensor,   # (B, state_dim)
        mask: torch.Tensor | None,   # (B, T) bool
        hurst_vec: torch.Tensor,     # (B,)
    ) -> torch.Tensor:
        """Build the (B, _physics_dim(state_dim)) physics feature vector."""
        B, T, _ = innovations.shape

        # Per-timestep innovation magnitude
        innov_mag = innovations.pow(2).sum(dim=-1)  # (B, T)

        if mask is not None:
            mf = mask.to(innov_mag.dtype)
            mf_sum = mf.sum(dim=-1, keepdim=True).clamp(min=1.0)
            innov_mean = (innov_mag * mf).sum(dim=-1, keepdim=True) / mf_sum  # (B, 1)
            # For max: set masked positions to -inf
            masked_mag_for_max = torch.where(mf > 0, innov_mag, torch.full_like(innov_mag, float("-inf")))
            innov_max = masked_mag_for_max.max(dim=-1, keepdim=True).values
            # std over valid: compute weighted variance
            centered = (innov_mag - innov_mean) * mf
            innov_std = torch.sqrt((centered.pow(2).sum(dim=-1, keepdim=True) / mf_sum).clamp(min=1e-8))
            # argmax over valid
            peak_idx = masked_mag_for_max.argmax(dim=-1).float().unsqueeze(-1)
            peak_idx_norm = peak_idx / max(T - 1, 1)
        else:
            innov_mean = innov_mag.mean(dim=-1, keepdim=True)
            innov_max = innov_mag.max(dim=-1, keepdim=True).values
            innov_std = innov_mag.std(dim=-1, keepdim=True, unbiased=False)
            peak_idx_norm = innov_mag.argmax(dim=-1).float().unsqueeze(-1) / max(T - 1, 1)

        # Uncertainty trajectory stats (trace over diag)
        trace_P = P_trajectory.sum(dim=-1)  # (B, T)
        trace_mean = trace_P.mean(dim=-1, keepdim=True)
        trace_max = trace_P.max(dim=-1, keepdim=True).values
        trace_range = trace_max - trace_P.min(dim=-1, keepdim=True).values

        physics = torch.cat(
            [
                innov_mean, innov_max, innov_std, peak_idx_norm,   # (B, 4)
                final_state,                                        # (B, state_dim)
                trace_mean, trace_max, trace_range,                 # (B, 3)
                hurst_vec.unsqueeze(-1),                            # (B, 1)
            ],
            dim=-1,
        )
        return physics  # (B, 8 + state_dim)

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
            hurst: (B,) precomputed Hurst exponent per point (0..1).
                   If None, defaults to 0.5.

        Returns dict:
            crop_logits: (B, n_crops)
            pheno_logits: (B, T, n_phenos)  — per-timestep phenophase classification
            innovations: (B, T, state_dim)
            uncertainty: (B,) final trace(P)
            final_state: (B, state_dim)
            state_trajectory: (B, T, state_dim)
            P_trajectory: (B, T, state_dim)  — per-timestep P diagonals
            physics_vector: (B, 8 + state_dim)
        """
        B, T, _ = x.shape
        device = x.device

        h = self.input_proj(x)
        h = self.input_norm(h)

        # MKM rollout with P trajectory collection
        h_exec = torch.zeros(B, self.cfg.hidden_dim, device=device)
        x_mkm = torch.zeros(B, self.cfg.state_dim, device=device)
        P_mkm = (
            self.cfg.p_init_scale
            * torch.eye(self.cfg.state_dim, device=device)
            .unsqueeze(0)
            .expand(B, -1, -1)
            .clone()
        )

        state_seq: list[torch.Tensor] = []
        innov_seq: list[torch.Tensor] = []
        P_diag_seq: list[torch.Tensor] = []
        for t in range(T):
            x_pred, P_pred = self.mkm.predict(x_mkm, P_mkm)
            h_exec, measurement, _ = self.executor(h[:, t, :], x_pred, h_exec)
            x_mkm, P_mkm, innovation = self.mkm.update(x_pred, P_pred, measurement)
            state_seq.append(x_mkm)
            innov_seq.append(innovation)
            P_diag_seq.append(P_mkm.diagonal(dim1=-2, dim2=-1))

        state_traj = torch.stack(state_seq, dim=1)  # (B, T, state_dim)
        innovations = torch.stack(innov_seq, dim=1)  # (B, T, state_dim)
        P_trajectory = torch.stack(P_diag_seq, dim=1)  # (B, T, state_dim)

        # Attention physics state (coarse — used as per-head bias)
        chaos_score = innovations.pow(2).mean(dim=(1, 2)).clamp(0, 10) / 10.0  # (B,)
        if hurst is None:
            hurst_vec = torch.full((B,), 0.5, device=device)
        else:
            hurst_vec = hurst.to(device=device, dtype=x.dtype)
        attn_physics = torch.stack([chaos_score, hurst_vec], dim=-1)  # (B, 2)

        # Attention over [hidden | state]
        attn_in = torch.cat([h, state_traj], dim=-1)
        attn_in = self.attn_proj(attn_in)
        attn_out = self.chaos_attn(attn_in, physics_state=attn_physics, attn_mask=mask)

        # Masked-mean pool
        if mask is not None:
            m = mask.to(attn_out.dtype).unsqueeze(-1)
            pooled = (attn_out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)
        else:
            pooled = attn_out.mean(dim=1)
        pooled = self.pool_norm(pooled)

        # Build the RICH physics vector and inject into crop head
        physics_vec = self._extract_physics(
            innovations, P_trajectory, x_mkm, mask, hurst_vec
        )
        physics_vec_n = self.physics_norm(physics_vec)

        crop_logits = self.head_crop(torch.cat([pooled, physics_vec_n], dim=-1))
        pheno_logits = self.head_pheno(attn_out)

        uncertainty = P_mkm.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

        return {
            "crop_logits": crop_logits,
            "pheno_logits": pheno_logits,
            "innovations": innovations,
            "uncertainty": uncertainty,
            "final_state": x_mkm,
            "state_trajectory": state_traj,
            "P_trajectory": P_trajectory,
            "physics_vector": physics_vec,
        }


__all__ = ["DynamisCropClassifier", "DynamisModelConfig"]
