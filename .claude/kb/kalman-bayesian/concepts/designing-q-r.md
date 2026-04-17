# Designing Q and R

Labbe's ch. 08 is essentially about this. Q and R are rarely known a priori — you either measure them, tune them, or learn them.

## Three strategies

### 1. Measure from data (classical)

**`R`** — run the sensor in a known-state condition, collect the variance of repeat measurements.

**`Q`** — harder. Often set as a small diagonal "driven by physics":
- For a position-velocity KF: `Q = sigma_a² * dt² * G G^T` where `G` is the driving-noise model.
- For our phenology state: no closed-form — `Q` encodes "how much can the phenophase distribution shift between observations we didn't see".

### 2. Tune by hand (pragmatic)

Start from order-of-magnitude guesses and inspect the filter output:

- Too smooth / lagging measurements → decrease `R` or increase `Q`.
- Too jumpy / over-reactive → increase `R` or decrease `Q`.
- `trace(P)` saturates at the same value for every sample → reduce `P_0` (we hit this in our v2 run — fix in `DynamisModelConfig.p_init_scale=0.1`).

### 3. Learn them (our approach)

Parametrise `Q` and `R` as diagonal matrices with learnable log-diagonals:

```python
# src/dynamis/dynamis_core.py
self.log_Q_diag = nn.Parameter(torch.zeros(state_dim) - 2.0)
self.log_R_diag = nn.Parameter(torch.zeros(state_dim) - 1.0)
# Exponentiate at use-time to ensure positivity
Q = torch.diag_embed(torch.exp(self.log_Q_diag))
```

Trained via `innovation_loss` (see [innovation-and-gain.md](innovation-and-gain.md)) plus the downstream classification loss.

**v3 refinement**: initialise with **spread** instead of uniform values, so each state dim gets a different learned dynamic:

```python
# src/dynamis/dynamis_core.py — MarkovKalmanModule v3
q_init = -2.0 + noise_spread * (2 * torch.rand(state_dim) - 1)
self.log_Q_diag = nn.Parameter(q_init)
```

This fixes the v2 bug where every sample ended with identical `trace(P) ≈ 1.07`.

## Common pathologies

| Symptom | Probable cause | Fix |
|---|---|---|
| Filter never responds to measurements | `K ≈ 0` — means `R` dominates `P_pred` | Decrease `R` or increase process noise `Q` |
| Filter follows measurements exactly (no smoothing) | `K ≈ H^-1` — measurement dominates | Increase `R` |
| `P` collapses to zero rapidly | No process noise — filter believes dynamics are perfect | Add small Q, especially if system is known imperfect |
| `P` explodes | `A` unstable (eigenvalues > 1) or `H` rank-deficient | Check `A`, ensure observable states |
| Filter diverges silently | Mis-specified model (wrong `A` or `H`) | NIS test — if innovations don't match `S`, model is wrong |
| All samples end with same `trace(P)` | `P_0` too large, saturates after a few updates | Lower `P_0` — was our v2 bug |

## Heuristics from Labbe ch. 08

- **`Q` should scale with `dt`**. If your observations are days apart, `Q` should be roughly `q_rate × dt` where `q_rate` is "how much the state drifts per unit time".
- **`R` should be slightly larger than sensor spec**. Better to be mildly under-confident than to trust sensors absolutely.
- **Diagonal-only Q and R are almost always fine**. Full covariance is theoretically more expressive but practically unidentifiable.
- **Scale matters**. If state components have wildly different magnitudes, normalise first — otherwise `P` is ill-conditioned.

## Adaptive filtering (Labbe ch. 14)

If Q and R themselves drift over time (e.g. sensor quality degrades), use an **adaptive** approach:

- **IAE** (Innovation-based Adaptive Estimation) — update `R` from a sliding window of recent innovations.
- **MMAE** (Multiple Model Adaptive Estimation) — run several filters with different Q/R in parallel, weight by likelihood.

Our approach — backprop through the filter — is essentially a slow form of adaptive filtering across the training distribution. At inference time, Q/R are fixed.

## Specific advice for Dynamis Terra

- Start `log_Q_diag` around `-2` (Q ≈ 0.13) — phenophase can shift by ~10% per observation.
- Start `log_R_diag` around `-1` (R ≈ 0.37) — Sentinel-2 L2A reflectance has substantial noise.
- **Always** spread the init (`noise_spread=0.5`) — homogeneous init caused the v2 saturation bug.
- Clamp `log_Q_diag >= -6` and `log_R_diag >= -6` during training (prevents numerical collapse).
- Run the NIS diagnostic at validation time. If mean NIS > 2× state_dim, your learned Q/R is over-confident.

## References

- Labbe ch. 08, 14.
- Mehra 1970, "On the identification of variances and adaptive Kalman filtering."
- Zarchan & Musoff — *Fundamentals of Kalman Filtering: A Practical Approach*.
