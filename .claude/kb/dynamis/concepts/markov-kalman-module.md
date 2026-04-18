# Markov-Kalman Module (MKM)

A differentiable Kalman filter where `A`, `H`, `Q`, `R` are `nn.Parameter`s learned via backprop. This is the **physics engine** of Dynamis — the "cerebellum" that tracks latent state evolution while the rest of the model handles semantics.

Our implementation: [src/dynamis/dynamis_core.py::MarkovKalmanModule](../../../../src/dynamis/dynamis_core.py).

## Intellectual ancestry

The MKM sits inside a larger `HRM_MKM` architecture — the **Hierarchical Reasoning Model** of Wang et al. 2025 ([arXiv:2506.21734](https://arxiv.org/abs/2506.21734)) extended with a Kalman state-space prior:

- **HRM original**: two recurrent modules operating on different timescales — a "slow abstract planner" (L2) and a "fast tactical computer" (L0). Achieves strong reasoning with only 27M params on 1k samples.
- **HRM + MKM** (ours): replace the slow planner's latent dynamics with an **explicit** Kalman state transition. The filter acts as a structured prior on how the latent state can evolve; the classical KF closed-form update replaces part of the learned recurrence.

Why this matters: HRM's power comes from hierarchical depth in a single forward pass. Adding a Kalman filter keeps that depth **uncertainty-aware** — every intermediate state has an explicit covariance `P` we can read out.

## What the MKM does

Given a sequence of observations (measurements derived from the Executor), the MKM produces at each timestep:

1. **Prior** `(x_pred, P_pred)` — what the dynamics predict.
2. **Posterior** `(x_post, P_post)` — what Bayes' update yields after seeing the measurement.
3. **Innovation** `y = z - H x_pred` — the "surprise" signal (see [innovation-loss.md](innovation-loss.md)).

The two equations you must know:

```
PREDICT: x_pred = A x_post
         P_pred = A P_post A^T + Q

UPDATE:  y = z - H x_pred
         S = H P_pred H^T + R
         K = P_pred H^T S^-1
         x_post = x_pred + K y
         P_post = (I - K H) P_pred
```

For the rigorous theory, see [kalman-bayesian/concepts/kalman-basics.md](../../kalman-bayesian/concepts/kalman-basics.md).

## Why it's learnable (and what's new)

Classical KF: `A`, `H`, `Q`, `R` are hand-designed from physics. Our MKM promotes them to `nn.Parameter` and trains them via:

- The **innovation loss** (minimise `||y||²` — see [innovation-loss.md](innovation-loss.md)).
- Downstream **classification loss** (cross-entropy on the crop head, back-propagated through the rollout).

Backprop threads through every KF update step. The resulting `A`, `H` are still mathematically valid linear maps — just fit to data.

## Key implementation decisions (our codebase)

### 1. Phenology transition prior on `A`

We blend the identity with an agronomic 7×7 transition matrix at init:

```python
A_prior = build_phenology_prior_tensor()        # Dormancy → Greenup → ... → Senescence
blended = (1 - λ) * eye(7) + λ * A_prior
self.mkm.A.copy_(blended)
```

See [patterns/phenology-transition-prior.md](../patterns/phenology-transition-prior.md). The prior encodes the canonical phenology state machine; backprop refines it.

### 2. Heterogeneous Q/R initialisation (v3 fix)

```python
# src/dynamis/dynamis_core.py
q_init = -2.0 + noise_spread * (2 * torch.rand(state_dim) - 1)
r_init = -1.0 + noise_spread * (2 * torch.rand(state_dim) - 1)
self.log_Q_diag = nn.Parameter(q_init)
self.log_R_diag = nn.Parameter(r_init)
```

Without the spread, v2 had `trace(P) = 1.07` for every sample — the filter saturated to a common equilibrium and couldn't express per-sample uncertainty. Spreading the init gives each state dim its own dynamic.

### 3. Small P₀ (v3 fix)

Initial covariance is `0.1 * eye(state_dim)` rather than identity. Identity saturates after 2-3 updates on short sequences; `0.1 I` lets uncertainty grow naturally from observations. See `DynamisModelConfig.p_init_scale`.

### 4. Physics vector exposed to the head (v3 fix)

The MKM internals (innovations, state trajectory, P trajectory, final state) are stacked into a (B, 8+state_dim) vector and **concatenated** with the pooled attention before the crop head. This is the v3 breakthrough — without it, Dynamis was discarding its own physics. See [patterns/physics-vector-injection.md](../patterns/physics-vector-injection.md).

## Shape reference

For state dim = 7, batch = B, sequence = T:

| Quantity | Shape | Meaning |
|---|---|---|
| `x_post` | (B, 7) | Posterior mean |
| `P_post` | (B, 7, 7) | Posterior covariance |
| `A` | (7, 7) | Learnable transition |
| `H` | (7, 7) | Learnable observation |
| `Q` | (7, 7) diag | Process noise, diagonal from `log_Q_diag` |
| `R` | (7, 7) diag | Measurement noise, diagonal from `log_R_diag` |
| `y` | (B, 7) | Innovation |
| `innovations` trajectory | (B, T, 7) | All innovations, used by `innovation_loss` |

## Diagnostics

Three checks any production MKM should pass:

1. **Innovations whiteness** — `y_t` should be zero-mean, uncorrelated across t.
2. **NIS** (normalised innovation squared) — `y^T S^-1 y` should be χ²(state_dim)-distributed.
3. **Uncertainty discriminates errors** — mean `trace(P)` on errors > mean on correct (our v3 delta: +0.08).

See [kalman-bayesian/concepts/innovation-and-gain.md](../../kalman-bayesian/concepts/innovation-and-gain.md).

## Related reading

- Wang et al. 2025 — HRM parent paper: <https://arxiv.org/abs/2506.21734>
- Labbe, *Kalman and Bayesian Filters in Python* — the canonical practical reference.
- Krishnan, Shalit & Sontag 2015, "Deep Kalman Filters" — first clean "KF-inside-a-neural-net" paper.
- Bayesian extension (v5+): [pyro/patterns/probabilistic-dynamis-model.md](../../pyro/patterns/probabilistic-dynamis-model.md).
