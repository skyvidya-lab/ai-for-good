# Kalman Filter Quick Reference

## The six quantities

| Symbol | Name | Shape | Role |
|---|---|---|---|
| `x` | State estimate | `(state_dim,)` | What we believe the latent state is |
| `P` | State covariance | `(state_dim, state_dim)` | Uncertainty of that belief |
| `A` | Transition matrix | `(state_dim, state_dim)` | How state evolves: `x_t = A x_{t-1}` |
| `H` | Observation matrix | `(obs_dim, state_dim)` | How state maps to measurement: `z = H x` |
| `Q` | Process noise cov | `(state_dim, state_dim)` | Uncertainty in the dynamics |
| `R` | Measurement noise cov | `(obs_dim, obs_dim)` | Uncertainty in the sensor |

Derived:

| Symbol | Name | Meaning |
|---|---|---|
| `K` | Kalman gain | Weight between prediction and measurement |
| `y` | Innovation (residual) | `z - H x_pred` — the "surprise" |
| `S` | Innovation covariance | `H P_pred H^T + R` |

## The two-step cycle

```
        ┌──────── PREDICT ─────────┐    ┌──────── UPDATE ────────┐
  x_0 → │ x_pred = A x              │ → │ y = z - H x_pred        │ → x_post
  P_0 → │ P_pred = A P A^T + Q      │ → │ S = H P_pred H^T + R    │ → P_post
        └───────────────────────────┘    │ K = P_pred H^T S^-1     │
                                         │ x_post = x_pred + K y   │
                                         │ P_post = (I - K H) P_pred
                                         └─────────────────────────┘
```

## Tuning Q and R — rules of thumb

| Symptom | Likely cause | Fix |
|---|---|---|
| Filter ignores measurements (tracks prediction only) | `R` too large relative to `Q` | Decrease `R` or increase `Q` |
| Filter overfits to noise (jumpy) | `R` too small relative to `Q` | Increase `R` or decrease `Q` |
| `trace(P)` → 0 too fast, stuck | `Q` too small → filter thinks it knows everything | Increase `Q` to inject uncertainty |
| `trace(P)` → constant for all samples | Q and R both too large, P saturates | Lower both, especially `P_0` |
| `r_hat` diverges over time | `A` has eigenvalues > 1 | Re-check `A` — is it stable? |

Our v2 bug (`trace(P) ≈ 1.07` uniform) was the **P saturation** row. Fix in [src/models/dynamis_crop_classifier.py](../../../../src/models/dynamis_crop_classifier.py): `p_init_scale=0.1` instead of identity.

## Innovation diagnostics

If the filter is correctly specified, innovations `y_t` should be:

- **Zero-mean** (otherwise there's a bias in `A` or `H`).
- **Uncorrelated across time** (otherwise the dynamics are wrong).
- Have **covariance equal to the predicted `S_t`** (Normalised Innovation Squared, NIS, should be χ²-distributed).

NIS test:
```python
nis = y @ torch.linalg.solve(S, y)
# Should be ~ chi_square(obs_dim)
# mean(nis) ≈ obs_dim, std(nis) ≈ sqrt(2 * obs_dim)
```

When NIS deviates wildly → model is mis-specified. Tune Q/R or revisit A/H.

## Extensions (beyond linear KF)

| Variant | When |
|---|---|
| **EKF** (Extended KF) | Non-linear `A(x)` or `H(x)` — linearise around current estimate |
| **UKF** (Unscented KF) | Same as EKF but via sigma-points; better when linearisation is poor |
| **Particle Filter** | Highly non-linear / non-Gaussian noise; expensive |
| **Differentiable KF** | What WE do — `A`, `H`, `Q`, `R` are learned via backprop |

For Sentinel-2 reflectance, the observation is approximately linear in the latent phenophase state, so our linear KF is appropriate.

## Book chapters most relevant to us

| Chapter | Why |
|---|---|
| 04 — One-Dimensional Kalman Filter | Pure intuition, no matrices |
| 06 — Multivariate Kalman Filter | Our setup (7×7 `A`) |
| 08 — Designing Kalman Filters | How to pick state variables, Q, R |
| 14 — Adaptive Filtering | When Q/R themselves are unknown — relevant to our learnable approach |

## Within our repo

The canonical KF equations live in `MarkovKalmanModule.predict` and `.update`:

```python
# src/dynamis/dynamis_core.py (ported verbatim from skyvidya_dynamis)
def predict(self, x_post, P_post):
    Q = torch.diag_embed(torch.exp(self.log_Q_diag))
    x_pred = F.linear(x_post, self.A)
    P_pred = A @ P_post @ A.T + Q
    return x_pred, P_pred

def update(self, x_pred, P_pred, measurement):
    R = torch.diag_embed(torch.exp(self.log_R_diag))
    y = measurement - F.linear(x_pred, self.H)       # innovation
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ S^-1
    x_post = x_pred + K @ y
    P_post = (I - K @ H) @ P_pred
    return x_post, P_post, y  # y returned for innovation_loss
```
