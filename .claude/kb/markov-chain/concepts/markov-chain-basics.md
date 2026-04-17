# Markov Chain Basics

## The Markov Property

A stochastic process `{X_t}_{t ≥ 0}` on a countable state space `S` satisfies the **Markov property** iff:

```
P(X_{n+1} = j | X_n = i, X_{n-1} = i_{n-1}, ..., X_0 = i_0)
    = P(X_{n+1} = j | X_n = i)
```

Only the current state matters for the next — history beyond the current state is irrelevant.

**Memorylessness** is *not* the same as "independent increments". A random walk is Markov but its increments are certainly not independent of position.

## Time homogeneity

If `P(X_{n+1} = j | X_n = i)` doesn't depend on `n`, the chain is **time-homogeneous**:

```
P_ij := P(X_{n+1} = j | X_n = i)   for all n
```

The pgmpy `MarkovChain` class assumes this (as does our phenology prior). Time-inhomogeneous chains have `P_ij(n)` varying with time — useful if, say, you model early-season vs late-season dynamics differently.

## State space

For us: finite, with `|S| = 7` (phenophases). The transition matrix is `7 × 7`.

For continuous state spaces (e.g. a Kalman filter's latent state), the "transition matrix" becomes a transition kernel `K(x, dy)` — a density. Our MKM is the continuous-state analogue of what this KB describes.

## Initial distribution

The chain's law is determined by:
1. The transition matrix `P`.
2. The initial distribution `μ_0` (a probability vector over `S`).

Common choices:
- **Delta**: `μ_0 = e_k` (start in known state k).
- **Uniform**: `μ_0 = 1/|S|` (no prior knowledge).
- **Stationary**: `μ_0 = π` (stationary distribution, if you want the chain to be in equilibrium immediately).

For our crop classification, at inference we don't know the initial phenophase — we typically use uniform `μ_0 = 1/7` or a small-variance `P_0` in the Kalman-state analogue.

## Chapman-Kolmogorov equation

For any `m, n ≥ 0`:

```
P^{(m+n)}_ij = sum_k  P^{(m)}_ik * P^{(n)}_kj
```

In matrix form: `P^{m+n} = P^m P^n`. Trivially holds because matrix multiplication is associative.

**Consequence**: n-step probabilities are just `P^n`. Easy to compute for finite-state chains, expensive in general.

## Mental model: a random walker

Visualise `N` states as nodes of a graph. Edges have weights `P_ij`, the probability of moving from `i` to `j` in one step. A "walker" sits on a node and picks an outgoing edge at random (weighted by probabilities). This is exactly a Markov chain.

For our phenology prior, the graph looks like:

```
Dormancy ──0.3──▶ Greenup ──0.3──▶ MidGreenup ──0.3──▶ Peak
   ▲                                                     │
   │                                                    0.3
   │0.1                                                  ▼
   │                                              Maturity
   │                                                     │
   │                                                    0.3
   │                                                     ▼
   │                                               MidSenescence
   │                                                     │
   │                                                    0.3
   │                                                     ▼
   └────────────────────────────────────────────  Senescence (self-loop 0.9, wrap 0.1)
```

Self-loops (e.g. 0.7 probability to stay in Dormancy) are what make the chain **aperiodic** — without them, you'd return to Dormancy only every 7 steps and the stationary distribution wouldn't converge.

## Why Markov is a big deal

1. **Tractable** — closed-form computation for stationary distribution, hitting times, absorption probabilities.
2. **Composable** — product of Markov chains is Markov (on product state space).
3. **Universal approximator with hidden state** — any stochastic process can be made Markov by augmenting the state to contain enough history. This is why HMMs, state-space models, and Kalman filters all reduce to Markov chain theory.

For Dynamis Terra, the phenology is Markov on 7 states. The Kalman filter's continuous state is Markov on `R^7`. Both inherit the analytical tools from Markov chain theory.

## References

- Norris ch. 1 — rigorous treatment with examples.
- KU lecture notes: <https://www.math.ku.dk/bibliotek/arkivet/noter/stoknoter.pdf> — the reference we're adopting.
- Grimmett & Stirzaker ch. 6 — pedagogical.
