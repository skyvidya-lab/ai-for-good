"""
DynamisTerraV9 — Dual-timescale HRM-inspired crop classifier.

Key improvements over v8:
  1. H-module (slow): MarkovKalmanModule updates every T_slow=4 timesteps.
     Tracks broad phenological state — rice stages change over weeks, not days.
  2. L-module (fast): GRUCell updates every timestep, receives H-state as context.
     Captures fine-grained spectral variation at Sentinel-2's 5-day revisit.
  3. Rice-specific A-prior: self_loop=0.85, forward=0.15 (slower transitions).
     ~15% chance of advancing per 5-day window matches real rice phenology.
  4. Attention over [L-hidden | H-state] fuses both timescales for crop head.
  5. Ensemble-ready: identical interface, just instantiate N times with diff seeds.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..dynamis.chaos_attention import ChaosAttention
from ..dynamis.dynamis_core import MarkovKalmanModule
from ..dynamis.phenology_prior import N_PHENOPHASES, build_phenology_prior_tensor


@dataclass
class DynamisV9Config:
    input_dim: int = 17                  # 12 bands + 5 vegetation indices
    state_dim: int = N_PHENOPHASES       # 7 phenophases
    hidden_dim: int = 128                # wider than v8 (64) — still tiny
    attn_heads: int = 4
    n_crops: int = 3
    T_slow: int = 4                      # H-module update interval (timesteps)
    use_phenology_prior: bool = True
    lambda_prior_strength: float = 0.7   # stronger prior blend than v8 (0.5)
    prior_self_loop: float = 0.85        # rice-specific: ~85% stay per 5-day revisit
    prior_forward: float = 0.15
    prior_wrap: float = 0.05             # Senescence → Dormancy wrap
    p_init_scale: float = 0.1
    mkm_noise_spread: float = 0.3
    crop_head_dropout: float = 0.3


def _physics_dim(state_dim: int) -> int:
    """8 scalar summaries + state_dim final-state dims."""
    return 8 + state_dim


class DynamisTerraV9(nn.Module):
    """
    Dual-timescale phenology-aware crop classifier.

    Forward contract (same as DynamisCropClassifier for drop-in compatibility):
        x:     (B, T, F_in)
        mask:  (B, T) bool — True where timestep is valid (non-cloud)
        hurst: (B,) precomputed Hurst exponent

    Returns dict with: crop_logits, pheno_logits, innovations, uncertainty,
                       final_state, state_trajectory, P_trajectory, physics_vector.
    """

    def __init__(self, cfg: DynamisV9Config | None = None):
        super().__init__()
        self.cfg = cfg or DynamisV9Config()
        c = self.cfg

        # ── Input projection ──────────────────────────────────────────────
        self.input_proj = nn.Linear(c.input_dim, c.hidden_dim)
        self.input_norm = nn.LayerNorm(c.hidden_dim)

        # ── L-module: fast GRU (every timestep) ───────────────────────────
        # Input = [projected_observation | current H-state]
        self.gru_L = nn.GRUCell(c.hidden_dim + c.state_dim, c.hidden_dim)
        self.l_to_meas = nn.Linear(c.hidden_dim, c.state_dim)

        # ── H-module: slow MKM (every T_slow timesteps) ──────────────────
        self.mkm = MarkovKalmanModule(state_dim=c.state_dim,
                                      noise_spread=c.mkm_noise_spread)

        # Initialise A with rice-specific prior (stronger self-loop)
        if c.use_phenology_prior:
            with torch.no_grad():
                A_prior = build_phenology_prior_tensor(
                    self_loop=c.prior_self_loop,
                    forward=c.prior_forward,
                    wrap_to_dormancy=c.prior_wrap,
                )
                blended = ((1.0 - c.lambda_prior_strength) * self.mkm.A
                           + c.lambda_prior_strength * A_prior)
                self.mkm.A.copy_(blended)

        # ── Attention over fused [L-hidden | H-state] ─────────────────────
        self.attn_in_dim = c.hidden_dim + c.state_dim
        self.attn_proj = nn.Linear(self.attn_in_dim, c.hidden_dim)
        self.chaos_attn = ChaosAttention(
            embed_dim=c.hidden_dim,
            num_heads=c.attn_heads,
            n_physics=2,
            dropout=0.1,
        )
        self.pool_norm = nn.LayerNorm(c.hidden_dim)

        # ── Heads ─────────────────────────────────────────────────────────
        physics_dim = _physics_dim(c.state_dim)
        self.physics_norm = nn.LayerNorm(physics_dim)
        self.head_crop = nn.Sequential(
            nn.Linear(c.hidden_dim + physics_dim, c.hidden_dim),
            nn.GELU(),
            nn.Dropout(c.crop_head_dropout),
            nn.Linear(c.hidden_dim, c.n_crops),
        )
        # Per-timestep phenophase — use L-output (fine-grained temporal signal)
        self.head_pheno = nn.Linear(c.hidden_dim, c.state_dim)

    # ─────────────────────────────────────────────────────────────────────────
    def _extract_physics(
        self,
        innovations: torch.Tensor,   # (B, T, state_dim)
        P_trajectory: torch.Tensor,  # (B, T, state_dim)  diagonal
        final_state: torch.Tensor,   # (B, state_dim)
        mask: torch.Tensor | None,   # (B, T) bool
        hurst_vec: torch.Tensor,     # (B,)
    ) -> torch.Tensor:
        """Build (B, _physics_dim(state_dim)) physics feature vector."""
        B, T, _ = innovations.shape
        innov_mag = innovations.pow(2).sum(dim=-1)  # (B, T)

        if mask is not None:
            mf = mask.to(innov_mag.dtype)
            mf_sum = mf.sum(dim=-1, keepdim=True).clamp(min=1.0)
            innov_mean = (innov_mag * mf).sum(dim=-1, keepdim=True) / mf_sum
            masked_for_max = torch.where(
                mf > 0, innov_mag, torch.full_like(innov_mag, float('-inf'))
            )
            innov_max = masked_for_max.max(dim=-1, keepdim=True).values
            centered = (innov_mag - innov_mean) * mf
            innov_std = torch.sqrt(
                (centered.pow(2).sum(dim=-1, keepdim=True) / mf_sum).clamp(min=1e-8)
            )
            peak_idx_norm = (masked_for_max.argmax(dim=-1).float().unsqueeze(-1)
                             / max(T - 1, 1))
        else:
            innov_mean = innov_mag.mean(dim=-1, keepdim=True)
            innov_max = innov_mag.max(dim=-1, keepdim=True).values
            innov_std = innov_mag.std(dim=-1, keepdim=True, unbiased=False)
            peak_idx_norm = (innov_mag.argmax(dim=-1).float().unsqueeze(-1)
                             / max(T - 1, 1))

        trace_P = P_trajectory.sum(dim=-1)
        trace_mean = trace_P.mean(dim=-1, keepdim=True)
        trace_max = trace_P.max(dim=-1, keepdim=True).values
        trace_range = trace_max - trace_P.min(dim=-1, keepdim=True).values

        return torch.cat([
            innov_mean, innov_max, innov_std, peak_idx_norm,  # (B, 4)
            final_state,                                        # (B, state_dim)
            trace_mean, trace_max, trace_range,                 # (B, 3)
            hurst_vec.unsqueeze(-1),                            # (B, 1)
        ], dim=-1)

    # ─────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        hurst: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, T, _ = x.shape
        device = x.device
        c = self.cfg

        # Project input
        h = self.input_proj(x)
        h = self.input_norm(h)  # (B, T, hidden_dim)

        # ── Dual-timescale loop ──────────────────────────────────────────
        h_L = torch.zeros(B, c.hidden_dim, device=device)
        x_H = torch.zeros(B, c.state_dim, device=device)
        P_H = (c.p_init_scale
               * torch.eye(c.state_dim, device=device)
               .unsqueeze(0).expand(B, -1, -1).clone())

        meas_accum = torch.zeros(B, c.state_dim, device=device)
        t_in_window = 0
        last_innov = torch.zeros(B, c.state_dim, device=device)

        state_seq: list[torch.Tensor] = []
        innov_seq: list[torch.Tensor] = []
        P_diag_seq: list[torch.Tensor] = []
        hL_seq: list[torch.Tensor] = []

        for t in range(T):
            # L-step (every timestep): GRU guided by current H-state
            l_in = torch.cat([h[:, t, :], x_H], dim=-1)  # (B, hidden+state)
            h_L = self.gru_L(l_in, h_L)                   # (B, hidden)
            meas_t = self.l_to_meas(h_L)                   # (B, state_dim)
            meas_accum = meas_accum + meas_t
            t_in_window += 1

            # H-step: MKM updates every T_slow steps or at final timestep
            if t_in_window == c.T_slow or t == T - 1:
                x_H_pred, P_H_pred = self.mkm.predict(x_H, P_H)
                avg_meas = meas_accum / t_in_window
                x_H, P_H, last_innov = self.mkm.update(x_H_pred, P_H_pred, avg_meas)
                meas_accum = torch.zeros_like(meas_accum)
                t_in_window = 0

            # Record: H-state is constant between updates (slow planner)
            state_seq.append(x_H)
            innov_seq.append(last_innov)
            P_diag_seq.append(P_H.diagonal(dim1=-2, dim2=-1))
            hL_seq.append(h_L)

        state_traj = torch.stack(state_seq, dim=1)    # (B, T, state_dim)
        innovations = torch.stack(innov_seq, dim=1)   # (B, T, state_dim)
        P_trajectory = torch.stack(P_diag_seq, dim=1) # (B, T, state_dim)
        h_L_traj = torch.stack(hL_seq, dim=1)         # (B, T, hidden_dim)

        # ── Attention physics scalars ────────────────────────────────────
        chaos_score = innovations.pow(2).mean(dim=(1, 2)).clamp(0, 10) / 10.0
        hurst_vec = (hurst.to(device=device, dtype=x.dtype)
                     if hurst is not None
                     else torch.full((B,), 0.5, device=device))
        attn_physics = torch.stack([chaos_score, hurst_vec], dim=-1)

        # ── Attention over fused [L-hidden | H-state] ───────────────────
        attn_in = torch.cat([h_L_traj, state_traj], dim=-1)  # (B, T, hidden+state)
        attn_in = self.attn_proj(attn_in)                      # (B, T, hidden)
        attn_out = self.chaos_attn(attn_in, physics_state=attn_physics, attn_mask=mask)

        # ── Masked-mean pool ─────────────────────────────────────────────
        if mask is not None:
            m = mask.to(attn_out.dtype).unsqueeze(-1)
            pooled = (attn_out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)
        else:
            pooled = attn_out.mean(dim=1)
        pooled = self.pool_norm(pooled)

        # ── Physics vector → crop head ───────────────────────────────────
        physics_vec = self._extract_physics(
            innovations, P_trajectory, x_H, mask, hurst_vec
        )
        physics_vec_n = self.physics_norm(physics_vec)
        crop_logits = self.head_crop(torch.cat([pooled, physics_vec_n], dim=-1))

        # ── Per-timestep phenophase head (uses L's fine-grained output) ──
        pheno_logits = self.head_pheno(h_L_traj)

        uncertainty = P_H.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

        return {
            'crop_logits': crop_logits,
            'pheno_logits': pheno_logits,
            'innovations': innovations,
            'uncertainty': uncertainty,
            'final_state': x_H,
            'state_trajectory': state_traj,
            'P_trajectory': P_trajectory,
            'physics_vector': physics_vec,
        }


__all__ = ['DynamisTerraV9', 'DynamisV9Config']
