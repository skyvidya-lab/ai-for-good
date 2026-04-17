# Physics Vector Injection (v3 fix)

## Problem

After v2 sample run (184 pts, 4 regions), Dynamis **underperformed LightGBM by 7pp OA / 5pp F1m** despite:
- balanced class weights ✓
- stratified region sampling ✓
- Hurst regional feature ✓
- phenology transition prior ✓

Root cause, isolated after reviewing `skyvidya_dynamis/dynamis-core-gsm-poc/dynamis_core.py`:

> The Dynamis **physics outputs** (innovations, Kalman P trajectory, final latent state) were computed but only fed into the *regularisation loss* and the *attention bias*. The classification head received only a pooled attention vector, discarding the signals that make Dynamis different from an LSTM.

LightGBM with the phenology-aware features (`src/data/phenology_features.py`) had explicit per-point features (NDVI peak, greenup slope, NDWI flood) that were *implicitly* present in the Kalman innovations but never surfaced to the classifier.

## Pattern

**The crop classification head must receive the full Dynamis physics vector as explicit features**, alongside the pooled attention representation.

```
crop_logits = head_crop( concat[ pooled_attn, physics_vector ] )
```

Where `physics_vector` (shape `(B, 8 + state_dim)`) is:

| Component | Shape | Source | Captures |
|---|---|---|---|
| `innov_mean_magnitude` | (B, 1) | `‖innovation[t]‖²` mean over valid T | Average surprise rate |
| `innov_max_magnitude` | (B, 1) | `‖innovation[t]‖²` max over valid T | Peak surprise event |
| `innov_std_magnitude` | (B, 1) | std over valid T | Surprise heterogeneity |
| `innov_peak_idx_norm` | (B, 1) | `argmax(t) / (T-1)` | WHEN the big surprise happens |
| `final_state` | (B, state_dim) | `x_mkm` at t=T | Posterior latent state (phenophase-like) |
| `trace_P_mean` | (B, 1) | mean of trace(P) over T | Avg. Kalman uncertainty |
| `trace_P_max` | (B, 1) | max of trace(P) over T | Peak uncertainty |
| `trace_P_range` | (B, 1) | max − min | Uncertainty dynamism |
| `hurst` | (B, 1) | Regional/temporal/spectral Hurst | Persistence regime |

Total: **8 scalars + state_dim** → for state_dim=7, a (B, 15) physics vector.

## Why each feature matters

### Innovation peak timing (WHEN)

Different crops have the biggest Kalman "surprise" at different phenophases:
- **Rice** — peak innovation early (Dormancy→Greenup), when the paddy floods and NDWI spikes unexpectedly.
- **Corn** — peak innovation at Greenup (fastest canopy onset).
- **Soybean** — peak innovation at Peak (smooth growth, so the biggest residual comes from noisy reflectance variation at max biomass).

`argmax_t ‖innovation[t]‖²` captures this directly.

### trace(P) dynamism (HOW CONFIDENT)

If the model is overconfident, `trace_P_range` will be near zero. If uncertainty genuinely tracks difficulty, `range` grows on edge cases. This is structurally impossible to get from an LSTM.

### Final state + Hurst (WHAT regime)

The MKM posterior after 7-stage rollout IS a learnt representation of the crop-cycle trajectory. Hurst adds the persistence regime as a scalar prior.

## Implementation

See [src/models/dynamis_crop_classifier.py](../../../../src/models/dynamis_crop_classifier.py):

```python
def _extract_physics(
    self,
    innovations: torch.Tensor,   # (B, T, state_dim)
    P_trajectory: torch.Tensor,  # (B, T, state_dim) — diagonals at each step
    final_state: torch.Tensor,   # (B, state_dim)
    mask: torch.Tensor | None,
    hurst_vec: torch.Tensor,
) -> torch.Tensor: ...

physics_vec = self._extract_physics(...)
physics_vec_n = self.physics_norm(physics_vec)  # LayerNorm for scale stability
crop_logits = self.head_crop(torch.cat([pooled, physics_vec_n], dim=-1))
```

`P_trajectory` is NEW — we collect the P diagonal at every timestep (not just final) so the head can see how uncertainty evolves across the crop cycle.

The head itself is now a 2-layer MLP with GELU + dropout (regularisation for small datasets):

```python
self.head_crop = nn.Sequential(
    nn.Linear(hidden_dim + physics_dim, hidden_dim),
    nn.GELU(),
    nn.Dropout(0.3),
    nn.Linear(hidden_dim, n_crops),
)
```

## Verification

1. Unit test `test_dynamis_exposes_physics_vector_to_crop_head` asserts the dict contains `physics_vector` with shape `(B, 8 + state_dim)`.
2. Unit test `test_dynamis_crop_head_gradient_flows_through_physics` perturbs the input and asserts both the physics vector AND crop logits change (catches silent disconnections).
3. End-to-end Colab run (v3) should show Dynamis ≥ baseline on F1m; acceptance gate is "within 1pp" (we still do better than baseline on calibration/uncertainty regardless).

## Relation to the Finance POC

In `dynamis-finance-ai/backend/core/pi_attention.py`, the physics vector was `[chaos, gravity, hurst, innovation]` but fed ONLY to the attention bias — the same mistake we made initially. The Finance POC got away with it because it used DRL+ensemble on top; we don't have that luxury with 778 points.

Our solution is more explicit: **the physics is a FIRST-CLASS feature of the classifier, not a hidden modulator**.

## Anti-patterns

- Computing innovations/P but only using them in the loss (what v2 did).
- Adding physics features to the head WITHOUT LayerNorm → raw innovation magnitudes can dominate the MLP.
- Using only `final_state` and skipping the trajectory statistics — the temporal pattern of surprise is what discriminates crops.
- Making the head too wide (>1 hidden layer of hidden_dim>64) — small-data overfitting.
