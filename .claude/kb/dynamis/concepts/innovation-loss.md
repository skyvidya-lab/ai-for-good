# Innovation Loss

The "surprise" minimisation objective that makes the MKM **learn true dynamics** rather than memorise outputs. Mathematically simple (mean-squared residual), conceptually central.

Implementation: [src/dynamis/innovation_loss.py](../../../../src/dynamis/innovation_loss.py).

## Definition

Given the innovation (residual) at each timestep:

```
y_t = z_t - H · x_pred_t
```

the innovation loss is:

```
L_inn = (1/BT) · sum_{b,t}  ‖y_t^(b)‖²
```

Simple MSE of the Kalman filter's "surprises" across the batch × sequence.

## Why this makes Dynamis work

Kalman filter innovations are **supposed to be white** (zero-mean, temporally uncorrelated, with covariance `S = H P H^T + R`) if the model `A`, `H`, `Q`, `R` correctly describes the dynamics. If innovations are large or structured, the filter is **mis-specified** — the dynamics don't match the data.

By minimising `L_inn` via backprop:

- `A` is pushed to truly explain state transitions (otherwise `x_pred` is far from the actual measurement).
- `H` is pushed to correctly map latent → observed.
- `Q`, `R` are pushed toward values that make `S` actually cover the observed `y`.

This is the **inductive bias** of Dynamis: minimising surprise = learning the right physics.

## Combined loss

In practice we use [src/dynamis/innovation_loss.py::dynamis_loss](../../../../src/dynamis/innovation_loss.py):

```python
total = CE_crop (weighted)
      + CE_pheno
      + λ_inn · L_inn              # surprise
      + λ_ece · (ECE_crop + ECE_pheno)   # calibration
```

Weights calibrated across v1-v3:

| Version | λ_inn | λ_ece | Rationale |
|---|---:|---:|---|
| v1-v2 | 0.10 | 0.05 | First-pass defaults |
| v3-v4 | **0.05** | **0.02** | Classification loss dominates; innovation is a regulariser not a co-equal objective |

Too much innovation weight → the model fights the classifier for gradient and underfits crop labels. Too little → the MKM becomes a glorified recurrent network with no physics constraint.

## Innovation as a feature, not just a loss

v3 introduced **physics-vector injection** — the innovation statistics (mean, max, std, peak index) are stacked with the pooled attention and fed into the crop head directly:

```python
physics_vec = concat([
    innov_mean_mag, innov_max_mag, innov_std_mag, innov_peak_idx,
    final_state, trace_P_stats, hurst,
])
crop_logits = head_crop(concat[pooled, physics_vec])
```

This is what let Dynamis beat the LightGBM baseline. The innovation PATTERN over time is a crop-discriminative signature:

- **Rice**: peak innovation early (Dormancy→Greenup — paddy flooding is not predicted by the default phenology `A`).
- **Corn**: peak innovation at Greenup (aggressive canopy growth).
- **Soybean**: smoother, more distributed innovations.

See [patterns/physics-vector-injection.md](../patterns/physics-vector-injection.md).

## Calibration: ECE (Expected Calibration Error)

`dynamis_loss` also includes `expected_calibration_error` on both heads:

```python
ece = sum_bins  (|bin_accuracy - bin_confidence|) * (bin_size / N)
```

It doesn't change accuracy; it just pressures the model to produce well-calibrated probabilities — useful because Dynamis is explicitly a **confidence-aware** model. See [src/dynamis/innovation_loss.py::expected_calibration_error](../../../../src/dynamis/innovation_loss.py).

For post-hoc calibration (v4 addition), we additionally fit a single scalar temperature `T` on out-of-fold logits — [src/training/calibration.py](../../../../src/training/calibration.py).

## Relation to HRM parent

Wang et al. 2025's HRM paper ([arXiv:2506.21734](https://arxiv.org/abs/2506.21734)) does not use an explicit innovation loss — it trains end-to-end on task labels only. Adding `L_inn` is our **Dynamis-specific augmentation**: it forces the latent-state dynamics (the "slow planner" analogue of HRM's L2 module) to actually explain the data, not just be convenient for the classifier.

In signal-processing terms: the HRM without `L_inn` is like a Kalman filter without innovation diagnostics — it runs, but nothing prevents it from being quietly wrong.

## Diagnostics

At training time:

```python
# Innovation magnitude distribution
innov_mag = innovations.pow(2).sum(-1)    # (B, T)
print(f'mean={innov_mag.mean():.3f}  max={innov_mag.max():.3f}')
# NIS test (over a validation batch)
nis = (y @ torch.linalg.solve(S, y)).mean().item()
print(f'NIS: {nis:.3f}  (target ≈ state_dim = 7)')
```

If mean innovation magnitude plateaus early in training at a non-trivial value → filter is under-specified. If NIS >> state_dim → filter is over-confident (Q/R too small). See [kalman-bayesian/concepts/innovation-and-gain.md](../../kalman-bayesian/concepts/innovation-and-gain.md).

## References

- HRM paper: <https://arxiv.org/abs/2506.21734>
- Innovation whiteness test — Bar-Shalom, Li & Kirubarajan, *Estimation with Applications to Tracking and Navigation* — canonical reference.
- ECE — Guo et al. 2017, *On Calibration of Modern Neural Networks* ([arXiv:1706.04599](https://arxiv.org/abs/1706.04599)).
- Our tests — [tests/test_dynamis_modules.py](../../../../tests/test_dynamis_modules.py): innovation shape sanity, calibration reduction.
