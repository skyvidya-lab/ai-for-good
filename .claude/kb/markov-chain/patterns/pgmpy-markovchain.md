# Pattern: pgmpy.models.MarkovChain for Phenology

Use pgmpy when you need to simulate, validate, or probe a discrete-time Markov chain without writing the algebra yourself.

## Install

```bash
pip install pgmpy
```

MIT license. Active project with scikit-learn-like API.

## Minimal example — simulate our phenology chain

```python
from pgmpy.models import MarkovChain as MC
from pgmpy.factors.discrete import State

PHENOPHASES = ["Dormancy", "Greenup", "MidGreenup", "Peak",
               "Maturity", "MidSenescence", "Senescence"]

model = MC(["phenophase"], [7])                   # one variable, cardinality 7

# Transition dict: {from_state_idx: {to_state_idx: probability}}
transition = {
    0: {0: 0.7, 1: 0.3},                          # Dormancy → Dormancy (0.7), Greenup (0.3)
    1: {1: 0.7, 2: 0.3},
    2: {2: 0.7, 3: 0.3},
    3: {3: 0.7, 4: 0.3},
    4: {4: 0.7, 5: 0.3},
    5: {5: 0.7, 6: 0.3},
    6: {6: 0.9, 0: 0.1},                          # Senescence → Senescence (0.9), wrap to Dormancy (0.1)
}
model.add_transition_model("phenophase", transition)

# Start in Dormancy, sample 50 time steps
model.set_start_state([State("phenophase", 0)])
df = model.sample(size=50)
print(df["phenophase"].map(lambda k: PHENOPHASES[k]).tolist())
```

## Validate our learned `A` matches the prior

After training, project the learned `A` back to a valid transition matrix and check its statistics:

```python
import numpy as np
import torch
from src.models import DynamisCropClassifier
from pgmpy.models import MarkovChain as MC
from pgmpy.factors.discrete import State

# Load checkpoint
ckpt = torch.load("models/dynamis_terra_v3.pt")
A_learned = ckpt["model_state_dict"]["mkm.A"].detach().numpy()

# Project to row-stochastic (handle learned A drifting off the simplex)
A_proj = np.clip(A_learned, 0, None)
A_proj = A_proj / A_proj.sum(axis=1, keepdims=True)

# Build pgmpy model with the projected transitions
model = MC(["phenophase"], [7])
trans = {i: {j: float(A_proj[i, j]) for j in range(7) if A_proj[i, j] > 1e-6}
         for i in range(7)}
model.add_transition_model("phenophase", trans)

# Stationary check — does it converge?
model.set_start_state([State("phenophase", 0)])
df = model.sample(size=10000)
empirical_pi = df["phenophase"].value_counts(normalize=True).sort_index()
print("Empirical stationary from learned A:", empirical_pi.values)

# Compare against original prior's stationary
from src.dynamis import build_phenology_transition_matrix
P_prior = build_phenology_transition_matrix()
pi_prior = np.ones(7) / 7
for _ in range(1000):
    pi_prior = pi_prior @ P_prior
print("Stationary from prior A:            ", pi_prior)
```

Interpretation:

- If the learned `A` gives similar stationary distribution → prior dominates; model learned weak deviations.
- If very different → model discovered real data-driven transitions (good or bad, worth investigating).

## Stationarity check with pgmpy

```python
# Is a given distribution stationary for our chain?
candidate = np.array([0.32, 0.10, 0.10, 0.10, 0.10, 0.10, 0.18])
# pgmpy accepts named-tuple State list for start state
sample = [State("phenophase", k) for k in np.random.choice(7, size=2000, p=candidate)]
is_stationary = model.is_stationarity(tolerance=0.01, sample=sample)
```

## Compute empirical transition frequencies from training data

Useful for "ground-truth" prior calibration — what transitions actually occur in `points_train_label.csv`?

```python
import pandas as pd
from src.dynamis import phenophase_name_to_index

df = pd.read_csv("data/points_train_label.csv")
# For each point, sort observations by date and count transitions
transitions = np.zeros((7, 7), dtype=int)
for pid, group in df.groupby("point_id"):
    seq = (group.sort_values("phenophase_date")
                 ["phenophase_name"]
                 .map(phenophase_name_to_index)
                 .tolist())
    for a, b in zip(seq[:-1], seq[1:]):
        transitions[a, b] += 1

# Row-normalise to get empirical P
P_empirical = transitions / transitions.sum(axis=1, keepdims=True).clip(min=1)
print(P_empirical.round(3))
```

**Why do this**: the 778 labelled points give us the empirical transition matrix. We can compare our hand-crafted prior `P_prior` to `P_empirical` and either adopt `P_empirical` directly or use it to inform `lambda_prior_strength` in `DynamisModelConfig`.

## What pgmpy does NOT do (for our use case)

- ❌ Continuous-state chains (that's our Kalman filter's job).
- ❌ Learnable-parameter chains (pgmpy `P` is static once set).
- ❌ Time-inhomogeneous chains (transitions must be constant across time).
- ❌ Gradient-based training.

So pgmpy is a **validation / simulation tool**, not a replacement for our MKM.

## When pgmpy is the right tool

| Task | Use pgmpy |
|---|---|
| Simulate synthetic phenophase trajectories | ✓ |
| Test stationary distribution hypotheses | ✓ |
| Validate that learned `A` behaves sensibly | ✓ |
| Estimate empirical `P` from labelled data | ✓ (or numpy) |
| Train a learnable transition matrix | ✗ (use our MKM) |
| Run inference on noisy observations | ✗ (use our KF) |

## References

- pgmpy API: <https://pgmpy.org/models/markovchain.html>
- pgmpy GitHub: <https://github.com/pgmpy/pgmpy>
- Tutorial: the pgmpy docs' Markov Chain chapter has a worked example mixing two variables (intel, diff) — good template.
