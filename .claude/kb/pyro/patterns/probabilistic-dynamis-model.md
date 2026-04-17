# Pattern: Probabilistic Dynamis Model (v5+ sketch)

## Intent

Re-express the current `MarkovKalmanModule` as a Pyro generative model so that:

1. `A`, `H`, `Q`, `R` are **Bayesian random variables** with priors (not point-estimated `nn.Parameter`s).
2. SVI produces a **posterior distribution over each matrix entry**, not a single value.
3. Inference returns not only `trace(P)` (Kalman uncertainty) but also **parameter uncertainty**.

This is speculative — not yet implemented. Documented here so we can evaluate after v3 results.

## Architecture sketch

```python
import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from src.dynamis.phenology_prior import build_phenology_prior_tensor

N_STATE = 7  # phenophases


class BayesianMKM(PyroModule):
    """Probabilistic MKM with priors on A, H, Q, R."""

    def __init__(self, state_dim: int = N_STATE, prior_strength: float = 0.3):
        super().__init__()
        self.state_dim = state_dim

        # Phenology prior as mean of A
        A_prior_mean = build_phenology_prior_tensor(dtype=torch.float32)
        self.A = PyroSample(
            dist.Normal(A_prior_mean, prior_strength).to_event(2)
        )

        # Observation matrix — weak prior toward identity
        H_prior_mean = torch.eye(state_dim)
        self.H = PyroSample(
            dist.Normal(H_prior_mean, prior_strength).to_event(2)
        )

        # Log-diagonal noise priors — heterogeneous start
        self.log_Q_diag = PyroSample(
            dist.Normal(-2.0 * torch.ones(state_dim), 0.5).to_event(1)
        )
        self.log_R_diag = PyroSample(
            dist.Normal(-1.0 * torch.ones(state_dim), 0.5).to_event(1)
        )

    def forward(self, measurements: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            measurements: (B, T, state_dim)
            mask: (B, T) bool

        Returns:
            (B, state_dim) final posterior mean
        """
        B, T, _ = measurements.shape
        A = self.A         # (state_dim, state_dim)
        H = self.H
        Q = torch.diag(torch.exp(self.log_Q_diag))
        R = torch.diag(torch.exp(self.log_R_diag))

        x = torch.zeros(B, self.state_dim)
        P = 0.1 * torch.eye(self.state_dim).expand(B, -1, -1).clone()

        with pyro.plate("points", B):
            for t in range(T):
                # Predict
                x_pred = x @ A.T
                P_pred = A @ P @ A.T + Q

                # Observation as a likelihood
                z_mean = x_pred @ H.T  # (B, state_dim)
                S = H @ P_pred @ H.T + R
                pyro.sample(
                    f"obs_{t}",
                    dist.MultivariateNormal(z_mean, S),
                    obs=torch.where(mask[:, t:t + 1],
                                    measurements[:, t, :],
                                    torch.zeros_like(measurements[:, t, :])),
                )

                # Update (standard Kalman)
                K = P_pred @ H.T @ torch.linalg.inv(S)
                innovation = measurements[:, t] - z_mean
                x = x_pred + (K @ innovation.unsqueeze(-1)).squeeze(-1)
                I_KH = torch.eye(self.state_dim) - K @ H
                P = I_KH @ P_pred

        return x


def dynamis_model(x, mask, hurst, crop_label=None, pheno_label=None):
    """Full Dynamis generative model — MKM + heads as random variables."""
    mkm = BayesianMKM()
    state = mkm(x, mask)   # (B, state_dim)

    # Classification heads with priors on weights
    crop_logits = pyro.sample(
        "crop_logits",
        dist.Normal(torch.zeros(3), 1.0).expand([x.shape[0], 3]).to_event(2),
    )
    pyro.sample("crop", dist.Categorical(logits=crop_logits), obs=crop_label)
```

## Training

```python
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal

pyro.clear_param_store()
guide = AutoNormal(dynamis_model)
svi = SVI(dynamis_model, guide, Adam({"lr": 5e-4}), Trace_ELBO())

for epoch in range(40):
    for xb, mb, hb, cb, pb in train_loader:
        loss = svi.step(xb, mb, hb, cb, pb)
```

## What we gain

1. **Posterior on A**: credible intervals on transition probabilities. Lets us say "there is an 80% chance that the Greenup→MidGreenup transition rate is in [0.25, 0.35]".
2. **Posterior on Q, R**: honest uncertainty on the noise assumption, which is currently just learned via backprop.
3. **Posterior predictive**: draw 100 trajectories per point, compute entropy over classes. Much richer than a single softmax.
4. **Easier OOD detection**: if the guide's posterior entropy on `A` blows up for a region, the physics itself is novel — useful for "background" class.

## What we trade

- **Speed**: SVI with a richer guide is 2-3× slower than point estimation.
- **Implementation effort**: ~2 days of refactoring vs pure PyTorch.
- **Sample size**: 778 points may be too few to actually learn a posterior on 7×7 A without the prior dominating.

## Decision gate

Implement only if:

1. Current v3 architecture saturates on the sample (no more baseline-beating gains available).
2. The uncertainty diagnostics from the sample run show that `trace(P)` alone is not enough to discriminate errors.
3. We have budget for a 2-day refactor.

Otherwise defer to post-challenge (the J-FET paper could focus entirely on this Bayesian extension as future work).

## References

- Pyro tutorial: [Hidden Markov Models in Pyro](http://pyro.ai/examples/hmm.html) — directly analogous to our phenophase HMM.
- [Gaussian Process](http://pyro.ai/examples/gp.html) tutorial shows `PyroModule` patterns.
