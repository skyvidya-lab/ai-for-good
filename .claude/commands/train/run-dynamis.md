---
name: run-dynamis
description: Execute the Dynamis classifier (MKM + ChaosAttention + dual head).
---

# /run-dynamis

Runs the full Dynamis training loop using GPU.

## Usage

```
/run-dynamis                    # Default config
/run-dynamis --sample 5         # Sample mode
/run-dynamis --state-dim 7      # Override MKM state dimension
/run-dynamis --lambda-innov 0.1 # Innovation loss weight
/run-dynamis --epochs 20
/run-dynamis --seeds 42 123 456 # Ensemble across seeds
```

## Pipeline

1. Load consolidated temporal features (output of data pipeline).
2. Compute per-point Hurst exponent (conditional on date count ≥ 6).
3. GroupKFold(5) by `point_id`.
4. Train `DynamisCropClassifier` with `dynamis_loss` (CE + innovation + ECE).
5. Collect metrics + Kalman uncertainty per point.

## Outputs

- `results/dynamis_v1.json`
- `models/dynamis_v1.pt` (state_dict + config + phenology prior + scaler)
- `results/dynamis_v1_uncertainty.csv`

## Success Criteria

- Dynamis >= Baseline on crop_type F1 macro.
- ECE < 0.10 (model is calibrated).
- Passes Transfer Proof.
- Mean uncertainty on errors > mean uncertainty on correct predictions (uncertainty correlates with errors).
