# Pattern: SVI with ELBO

## Intent

Fit an approximate posterior `q(z)` over latent variables by maximising the ELBO. Standard variational inference recipe.

## When to use

- Large datasets where MCMC is too slow.
- Neural-net-parameterised guides (amortised inference).
- Need training-time uncertainty quantification without deep rewrites.

## Minimal recipe

```python
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam


def model(x, y=None):
    """Generative model: p(y, theta | x)."""
    # Priors
    weight = pyro.sample("weight",
                          dist.Normal(torch.zeros(x.shape[-1]), 1.0).to_event(1))
    bias = pyro.sample("bias", dist.Normal(0.0, 1.0))
    sigma = pyro.sample("sigma", dist.HalfCauchy(1.0))

    # Likelihood plate
    with pyro.plate("data", x.shape[0]):
        mean = x @ weight + bias
        pyro.sample("obs", dist.Normal(mean, sigma), obs=y)


pyro.clear_param_store()
guide = AutoNormal(model)
svi = SVI(model, guide, Adam({"lr": 1e-2}), Trace_ELBO())

for step in range(2000):
    loss = svi.step(x_train, y_train)
    if step % 200 == 0:
        print(f"step {step} | elbo={-loss:.1f}")
```

## Training loop idioms

**Mini-batch SVI** (our sample size is 778, not batched by default but can be):

```python
from torch.utils.data import DataLoader, TensorDataset

dl = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)
for xb, yb in dl:
    loss = svi.step(xb, yb)
```

Note: when batching, multiply the likelihood contribution by `N/batch_size` inside `plate(..., subsample_size=batch_size)` — Pyro handles this automatically if you use `pyro.plate` with `subsample_size`.

**LR decay**:

```python
from pyro.optim import ExponentialLR
scheduler = ExponentialLR({"optimizer": torch.optim.Adam,
                           "optim_args": {"lr": 1e-2},
                           "gamma": 0.995})
svi = SVI(model, guide, scheduler, Trace_ELBO())
# scheduler.step() called automatically
```

**Early stopping**: track ELBO on a held-out set, stop when no improvement for N steps.

## Getting the posterior

```python
from pyro.infer import Predictive

# Extract fitted guide parameters
for name, value in pyro.get_param_store().items():
    print(name, value.shape)

# Posterior predictive
pred = Predictive(model, guide=guide, num_samples=500,
                   return_sites=["weight", "bias", "sigma", "obs"])
samples = pred(x_test)
y_pred_mean = samples["obs"].mean(0)
y_pred_std = samples["obs"].std(0)
```

## Translating our MKM

Current (point estimate):

```python
# src/dynamis/dynamis_core.py
self.A = nn.Parameter(torch.eye(state_dim))
```

Bayesian equivalent:

```python
from pyro.nn import PyroSample

class BayesianMKM(pyro.nn.PyroModule):
    def __init__(self, state_dim, A_prior_mean):
        super().__init__()
        self.A = PyroSample(
            dist.Normal(A_prior_mean, 0.3).to_event(2)  # prior
        )
        self.log_Q_diag = PyroSample(
            dist.Normal(-2.0 * torch.ones(state_dim), 0.5).to_event(1)
        )
        ...
```

With `AutoNormal` guide, each matrix entry gets its own Gaussian posterior. After training, `trace(P)` is no longer the only uncertainty — you also have credible intervals on every transition probability.

## Gotchas

- **Forget `.to_event(n)`**: without it, a multivariate prior is treated as N independent scalars — wrong shape handling.
- **Mean-field is restrictive**: `AutoNormal` assumes independent dims. For correlated params (like A's rows), use `AutoMultivariateNormal` or normalising flow guides.
- **`pyro.clear_param_store()`**: call between experiments; otherwise old params leak.
- **Numerical issues**: scale `HalfCauchy(1.0)` priors if likelihoods are very peaked — switch to `HalfNormal(0.1)`.

## References

- [SVI tutorial Part I](http://pyro.ai/examples/svi_part_i.html)
- [Auto-guides](http://docs.pyro.ai/en/stable/infer.autoguide.html)
- [Understanding ELBO](https://xyang35.github.io/2017/04/14/variational-lower-bound/)
