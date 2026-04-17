# The Transition Matrix

All the chain's dynamics live in one matrix. Understanding its structure unlocks everything else.

## Row-stochastic

`P ∈ R^{N × N}` is **row-stochastic** iff:

1. `P_ij >= 0` for all `i, j`
2. `sum_j P_ij = 1` for every `i`

**Always check this** before proceeding — a non-row-stochastic matrix doesn't define a valid Markov chain.

```python
import numpy as np

def is_row_stochastic(P, atol=1e-8):
    return (P >= 0).all() and np.allclose(P.sum(axis=1), 1.0, atol=atol)
```

Our [phenology_prior.py::build_phenology_transition_matrix](../../../../src/dynamis/phenology_prior.py) normalises rows explicitly to guarantee this.

## n-step transitions

`(P^n)_ij` = probability of going from state `i` to state `j` in exactly `n` steps.

```python
P_10 = np.linalg.matrix_power(P, 10)
```

For large `n`, `P^n` converges (if the chain is ergodic) — every row approaches the stationary distribution `π`.

**Diagnostic**: if all rows of `P^100` are essentially identical, the chain is ergodic and you've converged to `π`.

## Learnable P — the core of our MKM

Our Markov chain is **not** a fixed `P`. The Kalman transition `A` in our `MarkovKalmanModule` is an `nn.Parameter` initialised from the phenology prior:

```python
# Blend phenology prior with identity at init
A_prior = build_phenology_prior_tensor()          # (7, 7) row-stochastic
blended = (1 - lambda_prior) * identity + lambda_prior * A_prior
self.mkm.A.copy_(blended)
```

During training, `A` drifts away from strict row-stochasticity — it becomes a real-valued matrix that *started* structured but is now free. The Kalman update still works (`A` just needs to be a linear map), but `A` is no longer a literal transition matrix.

**Consequence**: if we want to interpret the learned `A` as a Markov chain, we need to re-normalise:

```python
A_normalized = A_learned / A_learned.sum(dim=1, keepdim=True).clamp(min=1e-6)
# Project to probability simplex per row
A_normalized = torch.clamp(A_normalized, min=0)
A_normalized = A_normalized / A_normalized.sum(dim=1, keepdim=True)
```

Useful for:
- Visualising the learned transitions.
- Computing the learned stationary distribution.
- Checking whether training has drifted toward a sensible structure.

## Common structures

### Identity
`P = I`. Each state is a self-loop with probability 1. Absorbing — once in state `i`, stays there forever.

### Permutation
Each row is a one-hot at a different column. Deterministic cycle. Periodic — no unique stationary distribution.

### Doubly-stochastic
Both rows AND columns sum to 1. The uniform distribution is stationary for a doubly-stochastic chain. Useful for MCMC samplers.

### Sparse (banded)
Only local transitions allowed. Our phenology is tridiagonal (with a wrap-around) — each phenophase can only go to itself or the next one.

### Block-structured
States group into clusters with fast intra-cluster mixing, slow inter-cluster transitions. Useful for modelling weather regimes or long-term dynamics.

## Powers and limiting behaviour

For an **ergodic** chain (irreducible + aperiodic), `P^n` converges row-wise to a matrix whose every row is the stationary distribution `π`:

```python
P_inf = np.linalg.matrix_power(P, 10000)
print(P_inf[0])              # ≈ π
print(P_inf - np.outer(np.ones(N), π))   # ≈ 0 matrix
```

**Rate of convergence** is governed by the second-largest eigenvalue of `P` (in absolute value). If `|λ_2| = 0.9`, you need ~`log(0.01) / log(0.9) ≈ 44` steps to get within 1% of `π`. If `|λ_2| = 0.99`, you need ~460 steps.

For our phenology prior, `λ_2 ≈ 0.85`, so mixing time is ~30 steps. With `time_scale = 5` in our HRM_MKM, that's 6 full cycles — plenty of time for the chain to equilibrate during inference.

## Eigenanalysis

```python
vals, vecs = np.linalg.eig(P.T)
```

- `vals = [1, λ_2, λ_3, ...]` — the leading eigenvalue is always 1 (row-stochastic).
- The left eigenvector for `λ = 1` is the **stationary distribution** (after normalisation).
- Complex eigenvalues → periodic components.
- Magnitude of `λ_2` → mixing time (smaller = faster mixing).

## References

- Norris ch. 1 — matrix treatment.
- Grimmett & Stirzaker ch. 6.
- Gallager, *Discrete Stochastic Processes* — pedagogical with good engineering examples.
