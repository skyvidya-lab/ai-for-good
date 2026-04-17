# Innovation and Kalman Gain

Two of the most operationally important quantities in a KF. Understanding them unlocks debugging and interpretation.

## Innovation `y` — the "surprise" signal

```
y_t = z_t - H · x_pred_t
```

The measurement minus what the filter *expected* to see. Properties:

- **Zero-mean** if the filter is correctly specified.
- **Uncorrelated across time** (white) if `A` captures the true dynamics.
- **Covariance** `S_t = H P_pred H^T + R`.

**In our MKM**: innovations are what `innovation_loss` minimises:

```python
# src/dynamis/innovation_loss.py
def innovation_loss(innovations: torch.Tensor) -> torch.Tensor:
    return torch.mean(innovations**2)
```

Minimising `‖y‖²` is equivalent to fitting `A` and `H` to explain the data — **this IS the inference signal** in a differentiable KF. A well-trained MKM has small, white-noise-like innovations.

## Why innovation ≠ error

Easy to confuse:

- **Innovation**: `z - H x_pred` — measurement minus prediction, **what the filter didn't know yet**.
- **Error**: `x_true - x_post` — true state minus filter estimate, **unknowable at runtime** (you don't have ground truth).

The Kalman filter minimises *expected* error by using innovation as its only observable feedback signal.

## Kalman gain `K`

```
K_t = P_pred · H^T · S^-1 = P_pred · H^T · (H P_pred H^T + R)^-1
```

Interpretation:

- `K ≈ 0` when `R` is huge (measurement is garbage) — ignore it, trust prediction.
- `K ≈ H^-1` when `R` is tiny (measurement is perfect) — snap to measurement.
- Scales with `P_pred` — the more uncertain the prediction, the more weight on the measurement.

**`x_post = x_pred + K · y`** — read as "move x_pred toward what the innovation suggests, weighted by how trustworthy the measurement is".

## Why `y` is the physics signal for Dynamis

Different crops produce different innovation *patterns* over the phenophase cycle:

- **Rice** — big innovation at Greenup (paddy flooding is not explained by a vanilla phenology `A`).
- **Corn** — big innovation at MidGreenup → Peak (aggressive canopy closure).
- **Soybean** — small, evenly-distributed innovations (smooth growth matches the baseline `A`).

Our `physics_vector` in [src/models/dynamis_crop_classifier.py](../../../../src/models/dynamis_crop_classifier.py) exposes `{innov_mean, innov_max, innov_std, innov_peak_idx}` precisely so the classifier head sees these patterns directly.

## NIS — the diagnostic you should always run

Normalised Innovation Squared:

```python
nis = y @ torch.linalg.solve(S, y)    # shape (B,) per timestep
# Expected: mean ≈ obs_dim, std ≈ sqrt(2 * obs_dim), chi_square(obs_dim) distribution
```

If actual `mean(NIS) >> obs_dim` → innovations are larger than the filter thinks they should be → `Q` or `R` is too small, filter is overconfident.
If `mean(NIS) << obs_dim` → `Q`/`R` too large, filter is sluggish.

Gold-standard test: `P(chi_square(obs_dim) > observed_NIS)` should be a uniform distribution over [0, 1].

## References

- Labbe ch. 06 — "Tuning the filter" section.
- Bar-Shalom, Li, Kirubarajan — *Estimation with Applications to Tracking and Navigation* — canonical reference on NIS.
