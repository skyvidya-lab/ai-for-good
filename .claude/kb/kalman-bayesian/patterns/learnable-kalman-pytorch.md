# Pattern: Learnable Kalman Filter in PyTorch

The core idea behind our `MarkovKalmanModule`: promote the KF matrices to `nn.Parameter` and let backprop tune them.

## Why this is interesting

Classical KF:
- You design `A`, `H`, `Q`, `R` by hand or measure them.
- The filter is optimal **given** those matrices.
- Garbage in, garbage out.

Learnable KF:
- Initialise `A`, `H`, `Q`, `R` reasonably (or with a domain prior like our phenology transition).
- Define a differentiable loss (innovation loss, downstream task loss).
- Backprop tunes all four matrices jointly on training data.
- You lose the optimality proof but gain the ability to **learn the dynamics** from a representative dataset.

This is how we go from "KF as hand-crafted state estimator" to "KF as neural-net-compatible physics layer".

## Implementation (simplified from our MKM)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableKalman(nn.Module):
    """Differentiable Kalman filter with learnable A, H, Q, R."""

    def __init__(self, state_dim: int, noise_spread: float = 0.5):
        super().__init__()
        self.state_dim = state_dim

        self.A = nn.Parameter(torch.eye(state_dim))
        self.H = nn.Parameter(torch.eye(state_dim))

        # Spread log-diagonals so different dims get different starting dynamics
        q_init = -2.0 + noise_spread * (2 * torch.rand(state_dim) - 1)
        r_init = -1.0 + noise_spread * (2 * torch.rand(state_dim) - 1)
        self.log_Q_diag = nn.Parameter(q_init)
        self.log_R_diag = nn.Parameter(r_init)

    def predict(self, x_post, P_post):
        Q = torch.diag_embed(torch.exp(self.log_Q_diag))
        B = x_post.shape[0]
        x_pred = F.linear(x_post, self.A)
        A_batch = self.A.unsqueeze(0).expand(B, -1, -1)
        P_pred = A_batch @ P_post @ A_batch.transpose(1, 2) + Q
        return x_pred, P_pred

    def update(self, x_pred, P_pred, measurement):
        R = torch.diag_embed(torch.exp(self.log_R_diag))
        B = x_pred.shape[0]
        H_batch = self.H.unsqueeze(0).expand(B, -1, -1)

        y = measurement - F.linear(x_pred, self.H)              # innovation
        S = H_batch @ P_pred @ H_batch.transpose(1, 2) + R
        S = S + 1e-6 * torch.eye(self.state_dim, device=S.device)  # jitter

        # Solve instead of invert — more stable
        HP_T = P_pred @ H_batch.transpose(1, 2)
        K = torch.linalg.solve(S, HP_T.transpose(1, 2)).transpose(1, 2)

        x_post = x_pred + (K @ y.unsqueeze(-1)).squeeze(-1)
        I = torch.eye(self.state_dim, device=x_pred.device).unsqueeze(0)
        P_post = (I - K @ H_batch) @ P_pred

        return x_post, P_post, y
```

## Training loop

```python
kf = LearnableKalman(state_dim=7)
optimiser = torch.optim.AdamW(kf.parameters(), lr=1e-3)

for epoch in range(n_epochs):
    x = torch.zeros(B, 7)
    P = 0.1 * torch.eye(7).unsqueeze(0).expand(B, -1, -1).clone()
    innovations = []

    for t in range(T):
        x_pred, P_pred = kf.predict(x, P)
        x, P, y = kf.update(x_pred, P_pred, observations[:, t])
        innovations.append(y)

    innov = torch.stack(innovations, dim=1)            # (B, T, state_dim)
    loss = (innov ** 2).mean()                         # innovation loss
    # + downstream task losses

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
```

## Key design choices from our MKM

1. **Spread in `log_Q_diag` / `log_R_diag` init**. Homogeneous init caused the v2 `trace(P) = 1.07` saturation bug. Non-zero `noise_spread` gives each state dim a distinct starting dynamic.

2. **`p_init_scale = 0.1`** instead of identity. Starting with a smaller `P_0` prevents immediate saturation on short sequences.

3. **Blend with a domain prior**. In `DynamisCropClassifier.__init__`:
   ```python
   A_prior = build_phenology_prior_tensor()
   blended = (1 - lambda_prior) * self.mkm.A + lambda_prior * A_prior
   self.mkm.A.copy_(blended)
   ```
   Initialise `A` near a physically-sensible structure, then let backprop refine.

4. **Innovation loss as regulariser**. Our [dynamis_loss](../../../../src/dynamis/innovation_loss.py) combines classification CE with `lambda_innovation * mean(innov**2)`. This keeps the filter honest — the state trajectory has to actually explain the observations, not just produce useful classifier features.

5. **Expose physics to the classifier**. The v3 insight: don't just USE `A`, `H`, `Q`, `R` inside the filter — also EXPOSE the innovation pattern, final state and `trace(P)` as features of the classification head. See [physics-vector-injection.md](../../dynamis/patterns/physics-vector-injection.md).

## When this pattern wins over pure neural nets

- **Small data** (778 points like us) — the KF structure acts as a strong prior, reducing overfitting.
- **Interpretable state** — you can literally plot `A` and read off the transition probabilities.
- **Uncertainty estimates** — `trace(P)` gives per-sample uncertainty for free.
- **Domain priors available** — if you know something about the dynamics (like our phenology cycle), you can bake it into `A`'s init.

## When it loses to pure neural nets

- **Non-linear dynamics** — hard to make work without EKF/UKF extensions.
- **High-dimensional state** — `P` is `state_dim × state_dim`; scales poorly beyond 50-100.
- **No domain prior** — if you'd initialise `A` randomly, you might as well use an LSTM.

## References

- Labbe ch. 06 (linear multivariate) + our implementation.
- Krishnan, Shalit, Sontag 2015, "Deep Kalman Filters" — first clean "Kalman inside a neural net" paper.
- Haarnoja et al. 2016, "Backprop KF" — end-to-end trained KF for tracking.
- Karl et al. 2017, "Deep Variational Bayes Filters" — the Bayesian extension (cf. our Pyro pattern).
