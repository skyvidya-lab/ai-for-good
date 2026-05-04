"""
Phenology Transition Prior for the Markov-Kalman Module (MKM).

The 7 canonical phenophases follow this chronological sequence in the dataset:
    Greenup → MidGreenup → Maturity → Peak → Senescence → MidSenescence → Dormancy

IMPORTANT NOTE on naming convention: In this dataset, "Maturity" refers to an early
grain-fill stage (onset of physiological maturity) that precedes the NDVI maximum
("Peak"). This is consistent across 100% of all 778 training points and all 3 crop
types (rice, corn, soybean). Do NOT reorder or rename — the evaluation matches
these labels exactly.

Per-crop empirical intervals (days) — 778 training points, verified 2026-05-03:

    Transition               Rice(n=367)   Corn(n=229)   Soybean(n=182)
    Greenup → MidGreenup    20.1 ± 1.5    22.3 ± 1.5    22.5 ± 1.8
    MidGreenup → Maturity   23.8 ± 1.3    24.1 ± 1.1    23.0 ± 1.5
    Maturity → Peak         16.7 ± 0.8    16.7 ± 1.1    14.6 ± 0.8  ← soybean 2d shorter
    Peak → Senescence       19.4 ± 1.0    19.1 ± 1.5    15.4 ± 0.8  ← soybean 4d shorter
    Senescence → MidSen     29.9 ± 1.6    27.7 ± 2.7    21.4 ± 0.9  ← soybean 8.5d shorter!
    MidSenescence → Dorm    27.7 ± 2.0    25.7 ± 2.4    20.2 ± 0.9  ← soybean 7.5d shorter!

Regional variation (intra-crop, by longitude band): small (< 2d per band),
captured at inference by using the actual observed intervals from each labelled point.

This module builds a 7x7 transition prior matrix used to initialise the learnable
A matrix of the MKM. The prior injects agronomic knowledge (valid transitions only)
while letting backprop learn exact weights.
"""
from __future__ import annotations

import numpy as np
import torch

# Canonical chronological sequence as it appears in the dataset labels
PHENOPHASES: tuple[str, ...] = (
    "Greenup",
    "MidGreenup",
    "Maturity",
    "Peak",
    "Senescence",
    "MidSenescence",
    "Dormancy",
)

# ─── Global (pooled across all crops) fallback prior ───────────────────────
# Empirical mean interval (days) between consecutive phenophases.
# Derived from all 778 training points.
PHENOPHASE_INTERVALS_MEAN: tuple[float, ...] = (
    0.0,    # Greenup (anchor — first event)
    21.3,   # MidGreenup
    23.6,   # Maturity
    16.2,   # Peak
    18.4,   # Senescence
    27.2,   # MidSenescence
    25.3,   # Dormancy
)

# Cumulative days from Greenup for each phase (useful for absolute positioning)
PHENOPHASE_CUMULATIVE_DAYS: tuple[float, ...] = (
    0.0,    # Greenup
    21.3,   # MidGreenup
    44.9,   # Maturity
    61.1,   # Peak
    79.5,   # Senescence
   106.7,   # MidSenescence
   132.0,   # Dormancy
)

# Global std deviation of intervals
PHENOPHASE_INTERVALS_STD: tuple[float, ...] = (
    0.0,  1.97, 1.35, 1.26, 2.00, 3.84, 3.54,
)

# ─── Per-crop empirical priors ───────────────────────────────────────────────
# Format: (interval_mean_tuple, interval_std_tuple)
# Index: [0]=Greenup anchor, [1..6]=intervals to next phase in canonical order
_CROP_INTERVALS: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "rice": (
        (0.0, 20.1, 23.8, 16.7, 19.4, 29.9, 27.7),   # means
        (0.0,  1.5,  1.3,  0.8,  1.0,  1.6,  2.0),   # stds
    ),
    "corn": (
        (0.0, 22.3, 24.1, 16.7, 19.1, 27.7, 25.7),
        (0.0,  1.5,  1.1,  1.1,  1.5,  2.7,  2.4),
    ),
    "soybean": (
        (0.0, 22.5, 23.0, 14.6, 15.4, 21.4, 20.2),
        (0.0,  1.8,  1.5,  0.8,  0.8,  0.9,  0.9),
    ),
    "background": (   # unknown crop — use global prior
        (0.0, 21.3, 23.6, 16.2, 18.4, 27.2, 25.3),
        (0.0,  1.97, 1.35, 1.26, 2.00, 3.84, 3.54),
    ),
}


def get_crop_interval_prior(crop_type: str | None) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (mean, std) interval arrays (length 7) for a given crop type.

    Falls back to the global pooled prior for unknown crop types.

    Args:
        crop_type: one of 'rice' | 'corn' | 'soybean' | 'background' | None.

    Returns:
        means (7,) float32, stds (7,) float32 — intervals in days.
    """
    key = (crop_type or "background").lower()
    means, stds = _CROP_INTERVALS.get(key, _CROP_INTERVALS["background"])
    return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)


PHENO_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(PHENOPHASES)}
N_PHENOPHASES = len(PHENOPHASES)


def build_phenology_transition_matrix(
    self_loop: float = 0.70,
    forward: float = 0.30,
    wrap_to_dormancy: float = 0.10,
) -> np.ndarray:
    """
    Build a 7x7 transition matrix encoding the canonical phenology sequence.

    Canonical sequence (as found in the dataset, all 778 points):
        Greenup → MidGreenup → Maturity → Peak → Senescence → MidSenescence → Dormancy

    Rows sum to 1. Transitions allowed:
        - Self-loop (weight `self_loop`): stage persists across observations.
        - Forward step (weight `forward`): advance to next stage.
        - Wrap (Dormancy → Greenup, weight `wrap_to_dormancy`): next growing season.

    Args:
        self_loop: probability of staying in current stage.
        forward: probability of advancing one stage.
        wrap_to_dormancy: probability of cycling Dormancy back to Greenup.

    Returns:
        (7, 7) float64 row-stochastic matrix.
    """
    n = N_PHENOPHASES
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        if i < n - 1:
            A[i, i] = self_loop
            A[i, i + 1] = forward
        else:
            # Dormancy (last): persist + wrap back to Greenup for next season
            A[i, i] = 1.0 - wrap_to_dormancy
            A[i, 0] = wrap_to_dormancy

    # Renormalise rows for numerical safety
    A = A / A.sum(axis=1, keepdims=True)
    return A


def build_phenology_prior_tensor(
    self_loop: float = 0.70,
    forward: float = 0.30,
    wrap_to_dormancy: float = 0.10,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Torch version of `build_phenology_transition_matrix`."""
    A = build_phenology_transition_matrix(self_loop, forward, wrap_to_dormancy)
    return torch.as_tensor(A, dtype=dtype)


def build_phenology_interval_embedding(
    observed_intervals: list[float] | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Build a (7, 2) embedding of [cumulative_days, interval_days] for the Transformer.

    When `observed_intervals` is provided (e.g. from a labelled training point),
    it overrides the empirical mean values; otherwise the empirical priors are used.
    This embedding can be concatenated to the Transformer token at each phenophase
    timestep as a temporal anchor.

    Args:
        observed_intervals: list of 7 per-phase intervals in days (0.0 for first phase).
                            If None, uses PHENOPHASE_INTERVALS_MEAN.
        normalize: if True, divides cumulative by 132.0 (total season ≈ 132 days)
                   and intervals by 30.0 (max expected interval) to [0,1] range.

    Returns:
        (7, 2) float32 array of [cum_days_norm, interval_norm] per phase.
    """
    intervals = list(observed_intervals) if observed_intervals is not None else list(PHENOPHASE_INTERVALS_MEAN)
    cumulative = np.cumsum([0.0] + intervals[1:])  # cumulative from Greenup
    arr = np.stack([cumulative, np.array(intervals, dtype=np.float32)], axis=1).astype(np.float32)
    if normalize:
        arr[:, 0] /= 132.0  # season length normalization
        arr[:, 1] /= 30.0   # interval normalization
    return arr


def phenophase_name_to_index(name: str) -> int:
    """Lookup with case-insensitive fallback."""
    if name in PHENO_TO_IDX:
        return PHENO_TO_IDX[name]
    normalised = name.strip().lower()
    for key, idx in PHENO_TO_IDX.items():
        if key.lower() == normalised:
            return idx
    raise KeyError(f"Unknown phenophase: {name!r}. Known: {PHENOPHASES}")


def phenophase_index_to_name(idx: int) -> str:
    return PHENOPHASES[int(idx)]


if __name__ == "__main__":
    A = build_phenology_transition_matrix()
    print("Phenology transition prior (rows sum to 1):")
    print(A.round(3))
    print("Row sums:", A.sum(axis=1))
