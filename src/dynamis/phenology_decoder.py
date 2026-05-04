"""
Phenology Viterbi Decoder — crop-conditioned post-hoc smoothing of L-TAE logits.

Replaces argmax with a structured decoder that combines three log-likelihood terms:
    score(t, k) = log_emission(t, k)
                + λ_trans * log_transition(prev_k -> k)
                + λ_timing * log_timing_prior(doy_t | k, crop, anchor)

Zero-leakage by design: priors are population-level (per-crop), the Greenup anchor
is estimated from the encoder logits themselves, no ground-truth labels are used.

References:
    - Per-crop interval priors: phenology_prior.get_crop_interval_prior
    - Canonical sequence: Greenup → MidGreenup → Maturity → Peak → Senescence
                          → MidSenescence → Dormancy
"""
from __future__ import annotations

import numpy as np

from .phenology_prior import (
    N_PHENOPHASES,
    build_phenology_transition_matrix,
    get_crop_interval_prior,
)

_NEG_INF = -1e9


def get_crop_cumulative_days(crop_type: str | None) -> np.ndarray:
    """Cumulative days from Greenup for each phase, given a crop."""
    means, _ = get_crop_interval_prior(crop_type)
    return np.cumsum(means).astype(np.float32)


def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(x, eps, None))


def estimate_greenup_anchor(
    logits: np.ndarray,
    doy: np.ndarray,
    valid_mask: np.ndarray,
    greenup_idx: int = 0,
) -> float | None:
    """
    Estimate Greenup DOY for the point as the DOY of the timestep with the highest
    softmax probability for Greenup among valid timesteps.

    Returns None when there are no valid timesteps.
    """
    if not valid_mask.any():
        return None
    probs = _softmax(logits)
    masked = np.where(valid_mask, probs[:, greenup_idx], -1.0)
    t_star = int(np.argmax(masked))
    return float(doy[t_star])


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def viterbi_pheno_decode(
    logits: np.ndarray,
    doy: np.ndarray,
    valid_mask: np.ndarray,
    crop_type: str | None,
    sigma_timing: float = 12.0,
    log_trans_weight: float = 0.5,
    log_timing_weight: float = 0.3,
    self_loop: float = 0.70,
    forward: float = 0.30,
    wrap_to_dormancy: float = 0.10,
) -> np.ndarray:
    """
    Viterbi decoder for a single point's phenophase sequence.

    Args:
        logits: (T, 7) raw encoder logits.
        doy: (T,) day-of-year for each timestep.
        valid_mask: (T,) bool — invalid (padded) timesteps get -100.
        crop_type: predicted crop name; None falls back to global prior.
        sigma_timing: std (days) of the Gaussian timing likelihood.
        log_trans_weight: λ on transition log-prior.
        log_timing_weight: λ on timing log-prior.
        self_loop, forward, wrap_to_dormancy: transition matrix shape parameters.

    Returns:
        (T,) int64 array — decoded phase indices, or -100 on invalid steps.
    """
    T = logits.shape[0]
    K = N_PHENOPHASES
    out = np.full(T, -100, dtype=np.int64)

    valid_idx = np.where(valid_mask)[0]
    if valid_idx.size == 0:
        return out

    log_emit = _safe_log(_softmax(logits))               # (T, K)
    log_trans = _safe_log(
        build_phenology_transition_matrix(self_loop, forward, wrap_to_dormancy)
    ).astype(np.float32)                                 # (K, K)

    cumulative = get_crop_cumulative_days(crop_type)     # (K,)
    anchor = estimate_greenup_anchor(logits, doy, valid_mask)

    if anchor is None:
        log_timing = np.zeros((T, K), dtype=np.float32)
    else:
        expected = anchor + cumulative                   # (K,)
        diffs = doy[:, None].astype(np.float32) - expected[None, :]   # (T, K)
        # Wrap-around for Dormancy (next season): pick the smaller distance.
        diffs_wrap = np.abs(diffs)
        diffs_wrap = np.minimum(diffs_wrap, 365.0 - diffs_wrap)
        log_timing = -0.5 * (diffs_wrap / sigma_timing) ** 2

    # Run Viterbi on valid timesteps only.
    n = valid_idx.size
    delta = np.full((n, K), _NEG_INF, dtype=np.float32)
    psi = np.zeros((n, K), dtype=np.int64)

    t0 = valid_idx[0]
    delta[0] = (
        log_emit[t0]
        + log_timing_weight * log_timing[t0]
    )

    for step in range(1, n):
        t = valid_idx[step]
        # scores[i, j] = delta[step-1, i] + λ * log_trans[i, j]
        scores = delta[step - 1, :, None] + log_trans_weight * log_trans  # (K, K)
        psi[step] = scores.argmax(axis=0)
        delta[step] = (
            scores.max(axis=0)
            + log_emit[t]
            + log_timing_weight * log_timing[t]
        )

    # Backtrack.
    path = np.zeros(n, dtype=np.int64)
    path[-1] = int(delta[-1].argmax())
    for step in range(n - 2, -1, -1):
        path[step] = psi[step + 1, path[step + 1]]

    out[valid_idx] = path
    return out


def viterbi_pheno_decode_batch(
    logits_b: np.ndarray,
    doy_b: np.ndarray,
    valid_mask_b: np.ndarray,
    crop_types: list[str | None],
    **kwargs,
) -> np.ndarray:
    """Batch wrapper: applies viterbi_pheno_decode to (B, T, K) inputs."""
    B, T, _ = logits_b.shape
    out = np.full((B, T), -100, dtype=np.int64)
    for i in range(B):
        out[i] = viterbi_pheno_decode(
            logits_b[i], doy_b[i], valid_mask_b[i], crop_types[i], **kwargs
        )
    return out
