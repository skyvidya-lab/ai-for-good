# Pattern: Markov Chain in NumPy (what we actually use)

pgmpy is a nice validation tool, but day-to-day we roll our own in 10 lines of numpy/torch. This is what `phenology_prior.py` does.

## Build the transition matrix

```python
import numpy as np
from src.dynamis.phenology_prior import (
    PHENOPHASES,
    N_PHENOPHASES,
    build_phenology_transition_matrix,
)

P = build_phenology_transition_matrix(
    self_loop=0.70,
    forward=0.30,
    wrap_to_dormancy=0.10,
)
# P: (7, 7) row-stochastic numpy array
```

Implementation (simplified):

```python
def build_phenology_transition_matrix(self_loop=0.70, forward=0.30,
                                        wrap_to_dormancy=0.10):
    n = N_PHENOPHASES
    A = np.zeros((n, n))
    for i in range(n):
        if i < n - 1:
            A[i, i] = self_loop
            A[i, i + 1] = forward
        else:
            A[i, i] = 1.0 - wrap_to_dormancy
            A[i, 0] = wrap_to_dormancy
    # Renormalise for numerical safety
    A = A / A.sum(axis=1, keepdims=True)
    return A
```

## Simulate a single trajectory

```python
def sample_trajectory(P, length, start_state=0, rng=None):
    rng = rng or np.random.default_rng()
    traj = [start_state]
    state = start_state
    for _ in range(length):
        state = rng.choice(P.shape[0], p=P[state])
        traj.append(state)
    return np.array(traj)

traj = sample_trajectory(P, length=30)
# traj[i] is the phenophase index at step i
print([PHENOPHASES[k] for k in traj])
```

## Simulate many trajectories at once (vectorised)

```python
def sample_trajectories_batch(P, n_chains, length, start_states=None, rng=None):
    rng = rng or np.random.default_rng()
    n_states = P.shape[0]
    if start_states is None:
        start_states = np.zeros(n_chains, dtype=int)

    trajectories = np.zeros((n_chains, length + 1), dtype=int)
    trajectories[:, 0] = start_states

    for t in range(length):
        current = trajectories[:, t]                     # (n_chains,)
        # For each chain, sample next state from P[current[c]]
        probs = P[current]                                # (n_chains, n_states)
        cumulative = np.cumsum(probs, axis=1)
        u = rng.random(n_chains)[:, None]
        trajectories[:, t + 1] = (u < cumulative).argmax(axis=1)

    return trajectories

many = sample_trajectories_batch(P, n_chains=1000, length=30)
```

## Compute the stationary distribution

```python
def stationary_distribution(P, n_iter=1000):
    pi = np.ones(P.shape[0]) / P.shape[0]
    for _ in range(n_iter):
        pi = pi @ P
    return pi

pi = stationary_distribution(P)
for name, p in zip(PHENOPHASES, pi):
    print(f"{name:15s}  {p:.4f}")
```

## Empirical transition matrix from data

```python
def empirical_transition_matrix(sequences, n_states):
    """sequences: list of lists of state indices."""
    counts = np.zeros((n_states, n_states), dtype=int)
    for seq in sequences:
        for a, b in zip(seq[:-1], seq[1:]):
            counts[a, b] += 1
    row_sums = counts.sum(axis=1, keepdims=True).clip(min=1)
    return counts / row_sums, counts

# Usage with our labels data
import pandas as pd
df = pd.read_csv("data/points_train_label.csv")
seqs = []
for pid, g in df.groupby("point_id"):
    seq = (g.sort_values("phenophase_date")
             ["phenophase_name"]
             .map(phenophase_name_to_index)
             .tolist())
    seqs.append(seq)

P_empirical, counts = empirical_transition_matrix(seqs, N_PHENOPHASES)
```

## Torch version (for use inside the MKM)

Our `build_phenology_prior_tensor` (in `phenology_prior.py`) is just the torch equivalent:

```python
def build_phenology_prior_tensor(**kwargs) -> torch.Tensor:
    return torch.as_tensor(
        build_phenology_transition_matrix(**kwargs),
        dtype=torch.float32,
    )
```

Used during MKM initialisation:

```python
A_prior = build_phenology_prior_tensor()
with torch.no_grad():
    blended = (1 - lambda_prior) * mkm.A + lambda_prior * A_prior
    mkm.A.copy_(blended)
```

## When to use numpy vs pgmpy

| Task | numpy | pgmpy |
|---|---|---|
| Build small transition matrix | ✓ (10 lines) | ✓ (verbose) |
| Simulate 1 trajectory | ✓ | ✓ |
| Simulate 10k trajectories | ✓ (vectorised) | ✗ (slow) |
| Multi-variable joint transitions | painful | ✓ (intended use case) |
| Stationarity check | ✓ (eigendecomp) | ✓ (`is_stationarity`) |
| Full Bayesian network inference | ✗ | ✓ |

For our 1-variable 7-state phenology chain, **numpy is faster and simpler**. pgmpy shines when you have multi-variable discrete dynamics (e.g. [phenophase, weather, soil moisture] jointly).

## Tests

```python
import numpy as np
from src.dynamis.phenology_prior import build_phenology_transition_matrix

def test_row_stochastic():
    P = build_phenology_transition_matrix()
    assert P.shape == (7, 7)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert (P >= 0).all()

def test_irreducible():
    """Every state reachable from every other in at most 7 steps."""
    P = build_phenology_transition_matrix()
    reachability = np.linalg.matrix_power(P, 7) > 0
    assert reachability.all()
```

## References

- Our implementation: [src/dynamis/phenology_prior.py](../../../../src/dynamis/phenology_prior.py)
- Tests: [tests/test_dynamis_modules.py](../../../../tests/test_dynamis_modules.py)
