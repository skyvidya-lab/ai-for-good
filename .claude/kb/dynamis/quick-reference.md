# Dynamis Quick Reference

## Imports

```python
from src.dynamis import (
    MarkovKalmanModule, HRM_MKM, Executor,
    calculate_hurst, is_predictable_regime,
    hurst_temporal, hurst_spectral, hurst_features,
    ChaosAttention,
    innovation_loss, expected_calibration_error, dynamis_loss,
    PHENOPHASES, N_PHENOPHASES,
    build_phenology_transition_matrix, build_phenology_prior_tensor,
    phenophase_name_to_index, phenophase_index_to_name,
)
```

## Hurst interpretation cheat-sheet

| H value | Regime | Crop analogy |
|---|---|---|
| H < 0.4 | Anti-persistent | Pixel-level noise; clouds; stable mature canopy |
| 0.4 ≤ H ≤ 0.55 | Random walk | Insufficient temporal structure / skip or low confidence |
| H > 0.55 | Persistent | Active growth or active stress trajectory |

## MKM shapes

- `x_post`: `(B, state_dim)`
- `P_post`: `(B, state_dim, state_dim)`
- `measurement`: `(B, state_dim)`
- `innovation`: `(B, state_dim)`

## Loss recipe

```python
loss_dict = dynamis_loss(
    crop_logits, crop_labels,
    pheno_logits, pheno_labels,
    innovations,
    lambda_innovation=0.10,
    lambda_ece=0.05,
)
loss_dict["total"].backward()
```

## Uncertainty for OOD / background detection

```python
out = model(x, mask=mask, hurst=hurst)
trace_P = out["uncertainty"]              # (B,)
is_background = trace_P > trace_P.quantile(0.90)  # top 10% most uncertain
```
