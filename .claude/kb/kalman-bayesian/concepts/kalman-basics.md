# Kalman Filter Basics

The Kalman filter answers one question: **what is the optimal (linear, Gaussian) estimate of a latent state given all observations so far?**

## Mental model — two beliefs combine

At each timestep you have:

1. **A prediction** of the new state, based on the old state and the physics: `x_pred = A x_old`. Comes with its own uncertainty `P_pred`.
2. **A measurement** of something related to the state: `z = H x_true + noise`. Has uncertainty `R`.

The KF combines them by weighting inversely to variance:

- If the measurement is precise (small `R`) → trust the measurement → `x_post ≈ (measurement-informed)`.
- If the prediction is precise (small `P_pred`) → trust the prediction → `x_post ≈ x_pred`.

The exact weight is the **Kalman gain** `K = P_pred H^T (H P_pred H^T + R)^-1`.

## Why it's "optimal"

Under 3 assumptions:

1. Dynamics are linear: `x_t = A x_{t-1} + w` with `w ~ N(0, Q)`.
2. Observations are linear: `z_t = H x_t + v` with `v ~ N(0, R)`.
3. `w` and `v` are Gaussian and independent.

...the KF produces the **minimum-variance unbiased** estimator. No other algorithm gives a lower-variance estimate under these assumptions.

When the assumptions fail (non-linear, non-Gaussian), the KF is still a reasonable heuristic but loses optimality.

## Predict-update cycle

```
┌─────────────────────────────────────────────────────────────────┐
│                        PREDICT (a priori)                       │
│                                                                 │
│   x_pred = A · x_post                                            │
│   P_pred = A · P_post · A^T + Q                                  │
│                                                                 │
│   interpretation: roll the state forward by the physics,        │
│   inflate uncertainty because dynamics are noisy                │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                        UPDATE (a posteriori)                    │
│                                                                 │
│   y = z - H · x_pred              ← innovation ("surprise")     │
│   S = H · P_pred · H^T + R        ← innovation covariance       │
│   K = P_pred · H^T · S^-1         ← Kalman gain                  │
│                                                                 │
│   x_post = x_pred + K · y                                        │
│   P_post = (I - K · H) · P_pred                                  │
│                                                                 │
│   interpretation: the new observation shifts x_pred toward the  │
│   measurement by K·y and reduces uncertainty by (I - K H)       │
└─────────────────────────────────────────────────────────────────┘
```

## The "everything lives on a Gaussian" simplification

At every step, the state estimate is a Gaussian with mean `x` and covariance `P`. This is the reason:

- **Predict** just propagates a Gaussian through a linear map — stays Gaussian.
- **Update** combines two Gaussians via Bayes' rule — stays Gaussian.
- We only ever track `(mean, cov)` — closed-form, no sampling.

When the noise isn't Gaussian (e.g. heavy-tailed cloud contamination in satellite imagery), the filter still runs but the uncertainty estimates lose their probabilistic interpretation. For our Sentinel-2 setup the Gaussian approximation is acceptable because (a) L2A atmospheric correction normalises most outliers, and (b) our QA filtering removes extreme cloudiness.

## Why the 7-phenophase state is a natural fit

Phenophases form a linear-ish sequence: Dormancy → Greenup → ... → Senescence. The transition matrix `A` encodes which stage follows which, and `Q` models "agronomic slack" (some points accelerate through MidGreenup faster than others). `H` maps the 7-dim phenological state to observable spectral quantities — a learnable linear combination that captures how reflectance changes across stages.

## Relation to HMMs

An HMM is a Kalman filter with **discrete** state. The predict-update cycle is the same, but:

- State is `P(state=k)` for `k = 1..K` (categorical, not Gaussian).
- `A` becomes a row-stochastic transition matrix.
- Update uses Bayes' rule on categorical probabilities.

Our [phenology_prior.py](../../../../src/dynamis/phenology_prior.py) uses exactly this idea — it's an HMM-style transition matrix blended into the KF's `A` as a prior. See `.claude/kb/dynamis/patterns/phenology-transition-prior.md`.

## Further reading

- Labbe ch. 04 (scalar KF) — build intuition without linear algebra.
- Labbe ch. 06 (multivariate KF) — our setup.
- Kalman's original 1960 paper — concise, 11 pages, free online.
