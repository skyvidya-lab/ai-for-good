"""
Phenology Transition Prior for the Markov-Kalman Module (MKM).

The 7 canonical phenophases form a near-cyclical state machine:
    Dormancy -> Greenup -> MidGreenup -> Peak -> Maturity -> MidSenescence -> Senescence -> (Dormancy)

This module builds a 7x7 transition prior matrix used to initialise the learnable
A matrix of the MKM. The prior injects agronomic knowledge (valid transitions only)
while letting backprop learn exact weights.
"""
from __future__ import annotations

import numpy as np
import torch

PHENOPHASES: tuple[str, ...] = (
    "Dormancy",
    "Greenup",
    "MidGreenup",
    "Peak",
    "Maturity",
    "MidSenescence",
    "Senescence",
)

PHENO_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(PHENOPHASES)}
N_PHENOPHASES = len(PHENOPHASES)


def build_phenology_transition_matrix(
    self_loop: float = 0.70,
    forward: float = 0.30,
    wrap_to_dormancy: float = 0.10,
) -> np.ndarray:
    """
    Build a 7x7 transition matrix encoding the canonical phenology sequence.

    Rows sum to 1. Transitions allowed:
        - Self-loop (weight `self_loop`): stage persists across observations.
        - Forward step (weight `forward`): advance to next stage.
        - Wrap (Senescence -> Dormancy, weight `wrap_to_dormancy`): year-to-year cycle.

    Args:
        self_loop: probability of staying in current stage.
        forward: probability of advancing one stage.
        wrap_to_dormancy: probability of cycling Senescence back to Dormancy.

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
            # Senescence (last): persist + wrap to Dormancy
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
