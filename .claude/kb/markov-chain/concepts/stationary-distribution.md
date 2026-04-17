# Stationary Distribution

The long-run "time-average" of a Markov chain. Captures equilibrium behaviour.

## Definition

A distribution `π` on the state space `S` is **stationary** iff:

```
π P = π                (row vector times matrix)
π_i >= 0,  sum_i π_i = 1
```

Equivalently, `π^T` is a left-eigenvector of `P` with eigenvalue 1.

**Intuition**: if the chain starts with state distribution `π`, it stays at `π` at every step — the distribution is invariant under the dynamics.

## Existence and uniqueness

Depends on the chain's structure:

| Property | Stationary distribution |
|---|---|
| Irreducible, aperiodic (ergodic) | Unique `π`, reached from any start |
| Irreducible, periodic | Unique `π`, reached in Cesaro sense only |
| Reducible | Multiple stationary distributions (one per recurrent class) |
| Transient-only | No stationary distribution (mass escapes to infinity in non-finite case) |
| Absorbing | Stationary distribution concentrated on absorbing states |

For our phenology prior: irreducible (Senescence wraps to Dormancy) + aperiodic (self-loops) + finite state space → **unique ergodic stationary distribution**.

## Computing `π`

### Power iteration
```python
import numpy as np
pi = np.ones(N) / N                  # uniform start
for _ in range(1000):
    pi = pi @ P
print(pi)                             # converged stationary
```

Simple and robust. Converges geometrically at rate `|λ_2|`.

### Eigendecomposition
```python
vals, vecs = np.linalg.eig(P.T)
idx = np.argmin(np.abs(vals - 1.0))
pi = vecs[:, idx].real
pi = pi / pi.sum()
```

Faster for small state spaces. `vals` should contain one eigenvalue exactly 1.

### Linear system
`π` satisfies `π (P - I) = 0` with `sum π = 1`. Solve the linear system:

```python
# Replace one row of (P.T - I) with all-ones to impose sum constraint
A_sys = P.T - np.eye(N)
A_sys[-1] = 1
b = np.zeros(N)
b[-1] = 1
pi = np.linalg.solve(A_sys, b)
```

Most numerically stable for near-singular cases.

## Ergodic theorem

For an ergodic chain, **time averages = state averages**:

```
lim_{n→∞} (1/n) sum_{t=1}^{n} f(X_t) = sum_i π_i f(i)          w.p. 1
```

Implications for us:
- Over a long-enough sequence, the fraction of time spent in phenophase `k` converges to `π_k`.
- If our learned `A` has stationary `π` that disagrees with the empirical phenophase distribution in training data, something is wrong with the dynamics.

## Our phenology `π`

Computing for the standard init:

```python
from src.dynamis import build_phenology_transition_matrix
P = build_phenology_transition_matrix()

pi = np.ones(7) / 7
for _ in range(1000):
    pi = pi @ P

# Result (approximate):
# Dormancy      ~0.32
# Greenup       ~0.10
# MidGreenup    ~0.10
# Peak          ~0.10
# Maturity      ~0.10
# MidSenescence ~0.10
# Senescence    ~0.18
```

Dormancy dominates because it's the "entry point" (Senescence wraps to it with probability 0.1, and we spend most of the time there). Senescence is second due to its 0.9 self-loop.

This is a **sanity check** — the stationary distribution matches the intuition that crops spend more time in dormant/senescent phases than in active-growth phases.

## Detailed balance (reversible chains)

A chain with stationary `π` is **reversible** if:

```
π_i P_ij = π_j P_ji        for all i, j
```

Detailed balance → stationarity (but not vice versa).

**Our phenology chain is NOT reversible** — it's one-directional. Dormancy → Greenup has high probability, but Greenup → Dormancy has zero. This violates detailed balance.

Reversibility matters for MCMC algorithms (Metropolis-Hastings, Gibbs). For our purposes, reversibility is not required — we just need ergodicity.

## Mixing time

How fast the chain converges to `π` from any start:

```
t_mix(ε) = min { t : max_i || P^t(i, ·) - π ||_TV <= ε }
```

Governed by the **spectral gap** `γ = 1 - |λ_2|`:

```
t_mix(ε) ≈ (1/γ) log(1/ε)
```

For our phenology prior: `|λ_2| ≈ 0.85`, `γ ≈ 0.15`, so reaching 1% accuracy takes `log(100)/0.15 ≈ 30` steps.

## References

- Norris ch. 1.4 — stationary distributions.
- Levin & Peres, *Markov Chains and Mixing Times* — the modern reference.
- KU lecture notes — classical proofs.
