---
name: run-baseline
description: Execute the LightGBM baseline pipeline (no Dynamis) on the current dataset.
---

# /run-baseline

Runs the baseline classifier:
- LightGBM on flattened temporal features + per-band statistics (mean, std, slope).
- Two heads: `crop_clf` (3 classes), `pheno_clf` (7 classes) conditional on crop.
- `GroupKFold(5)` by `point_id`.

## Usage

```
/run-baseline                   # Full 778 points (requires data on Drive)
/run-baseline --sample 5        # Sample mode: 5 regions only
/run-baseline --output results/baseline_v0.json
```

## Outputs

- `results/baseline_v0.json`: per-fold metrics (OA, Kappa, F1 macro).
- `results/baseline_v0_preds.csv`: OOF predictions.
- `models/baseline_v0.pkl`: pickled LightGBM models.

## Success Criteria

- Passes Transfer Proof (shuffled labels ≈ 33% for crop, ≈ 14% for pheno).
- OA > 60% for crop_type on real labels.
- Runs in < 10 min on CPU or T4.
