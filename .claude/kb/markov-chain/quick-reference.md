# Markov Chain Quick Reference

## Definition

A discrete-time Markov chain on state space `S = {1, ..., N}` is a sequence `X_0, X_1, X_2, ...` with:

```
P(X_{t+1} = j | X_t = i, X_{t-1}, ..., X_0) = P(X_{t+1} = j | X_t = i) = P_ij
```

The **Markov property** — the future depends only on the present, not the past.

## Transition matrix

```
P ∈ R^{N × N}    P_ij = P(X_{t+1} = j | X_t = i)
```

- Every row sums to 1: `sum_j P_ij = 1`
- All entries non-negative: `P_ij >= 0`

`P` is called **row-stochastic**. Our phenology prior in [phenology_prior.py](../../../../src/dynamis/phenology_prior.py):

```python
# Simplified
P = [
    # Dorm Grn Mid Pk  Mat MidS Sen
    [0.7, 0.3, 0,   0,   0,   0,   0],   # Dormancy
    [0,   0.7, 0.3, 0,   0,   0,   0],   # Greenup
    [0,   0,   0.7, 0.3, 0,   0,   0],   # MidGreenup
    [0,   0,   0,   0.7, 0.3, 0,   0],   # Peak
    [0,   0,   0,   0,   0.7, 0.3, 0],   # Maturity
    [0,   0,   0,   0,   0,   0.7, 0.3], # MidSenescence
    [0.1, 0,   0,   0,   0,   0,   0.9], # Senescence (wraps back)
]
```

## n-step transition

```
P^(n)_ij = P(X_{t+n} = j | X_t = i) = (P^n)_ij
```

The n-th power of `P` gives the n-step transitions. In Python:

```python
import numpy as np
P_n = np.linalg.matrix_power(P, n)
```

## Stationary distribution `π`

A distribution `π` over states is **stationary** if:

```
π P = π            (row vector × matrix)
sum_i π_i = 1
```

Interpretation: if you start the chain with probabilities `π`, it stays at `π` forever. For many "nice" chains, `π` is the long-run fraction of time spent in each state.

Compute via eigendecomposition:

```python
vals, vecs = np.linalg.eig(P.T)
# Find eigenvalue = 1
idx = np.argmin(np.abs(vals - 1))
pi = vecs[:, idx].real
pi = pi / pi.sum()
```

## Classification of states

| Concept | Meaning | Example in our A |
|---|---|---|
| **Accessible** (`i → j`) | Possible to reach j from i eventually | Dormancy → Peak is accessible |
| **Communicates** (`i ↔ j`) | Both directions accessible | All 7 phenophases communicate (Senescence wraps) |
| **Irreducible** | Single communicating class | Yes — our chain is irreducible |
| **Absorbing** | State with `P_ii = 1` | None in ours |
| **Recurrent** | Returns to state infinitely often w.p. 1 | All states |
| **Transient** | May not return | None |
| **Periodic** | Returns only at multiples of some `d > 1` | No (our chain is aperiodic due to self-loops) |
| **Ergodic** | Irreducible + aperiodic + positive recurrent | Yes — unique stationary dist exists |

## Detailed balance (optional but useful)

A distribution `π` is a **detailed balance** (reversible) distribution if:

```
π_i P_ij = π_j P_ji   for all i, j
```

Detailed balance implies stationarity. Not every stationary distribution satisfies detailed balance (our phenology prior doesn't — it's one-directional).

## With pgmpy

```python
from pgmpy.models import MarkovChain as MC
from pgmpy.factors.discrete import State

PHENOPHASES = ["Dormancy", "Greenup", "MidGreenup", "Peak",
               "Maturity", "MidSenescence", "Senescence"]

model = MC(["phenophase"], [7])

# Transition dict: {from_state: {to_state: prob}}
trans = {0: {0: 0.7, 1: 0.3},
         1: {1: 0.7, 2: 0.3},
         # ... up to:
         6: {6: 0.9, 0: 0.1}}
model.add_transition_model("phenophase", trans)

model.set_start_state([State("phenophase", 0)])
trajectories = model.sample(size=100)           # pandas DataFrame
```

## With numpy (what we actually use)

```python
from src.dynamis import build_phenology_transition_matrix
P = build_phenology_transition_matrix()         # (7, 7) numpy array

# Simulate a single trajectory of length T
state = 0                                         # start Dormancy
trajectory = [state]
for _ in range(T):
    state = np.random.choice(7, p=P[state])
    trajectory.append(state)
```

## Gotchas

- Non-row-stochastic `P` → meaningless. Always check `P.sum(axis=1) == 1`.
- **Reducible chains** have multiple stationary distributions. Phenology wrap to Dormancy is what makes ours irreducible.
- **Periodic chains** don't converge to a single `π` by Cesaro averaging. Self-loops kill periodicity.
- Computing `P^n` for large `n` is unstable; use `np.linalg.matrix_power` or eigendecomposition.

## References

- Norris, *Markov Chains* (1998) — cleanest graduate-level reference.
- University of Copenhagen lecture notes: <https://www.math.ku.dk/bibliotek/arkivet/noter/stoknoter.pdf>
- pgmpy: <https://pgmpy.org/models/markovchain.html>
