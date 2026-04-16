# Phenology Transition Prior

## Intent

Initialise the Kalman transition matrix `A` of the MKM with agronomic knowledge of crop phenology cycles, rather than random weights.

## The 7-stage cycle

```
Dormancy → Greenup → MidGreenup → Peak → Maturity → MidSenescence → Senescence → (Dormancy)
```

Each phenophase is a latent state. Valid transitions:
- **Self-loop** (stage persists across nearby dates): weight ~0.70
- **Forward step** (advance to next stage): weight ~0.30
- **Wrap** (Senescence → Dormancy, next year): weight ~0.10

## Implementation

See [src/dynamis/phenology_prior.py](../../../../src/dynamis/phenology_prior.py):

```python
from src.dynamis import build_phenology_prior_tensor
A = build_phenology_prior_tensor()  # (7, 7) row-stochastic
```

Used in `DynamisCropClassifier.__init__`:

```python
blended = (1 - lambda_prior) * self.mkm.A + lambda_prior * A_prior
self.mkm.A.copy_(blended)
```

`lambda_prior=0.5` is a soft blend — the model can override the prior via backprop but starts from a physically plausible state.

## Why it matters

- **Inductive bias**: nobody else in the competition is likely to encode this.
- **Sample efficiency**: 778 points is small — priors reduce the function space the model must search.
- **Interpretability**: the learned `A` can be visualised post-training; deviations from the prior reveal unexpected dynamics.

## Anti-patterns

- Making `A` a hard constraint (non-differentiable): loses the learning advantage.
- Setting `lambda_prior_strength=1.0`: behaves like a fixed HMM with no adaptation.
- Applying the prior without blending: erases any prior training from checkpoint loads.
