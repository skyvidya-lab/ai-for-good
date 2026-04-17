# Pattern: MCMC with NUTS

## Intent

Get gold-standard posterior samples via the No-U-Turn Sampler. Use when accuracy matters more than speed (e.g. calibration studies, posterior analysis on small validation sets).

## When

- Model has < 10k parameters.
- Dataset small enough that one NUTS chain finishes in minutes (≤ 1k points).
- Need posterior samples for credible-interval reporting, not just point estimates.
- Validating SVI results: if SVI and NUTS disagree, SVI is wrong.

For Dynamis Terra: suitable for **hyperparameter posterior studies** (e.g. posterior over `lambda_innovation` given 5-fold metrics) on tabular summaries, not on the full spectral pipeline.

## Recipe

```python
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS


def model(fold_losses):
    # Prior on the innovation weight
    lam = pyro.sample("lambda_innov", dist.HalfNormal(0.2))
    sigma = pyro.sample("sigma", dist.HalfCauchy(0.05))
    with pyro.plate("folds", len(fold_losses)):
        pyro.sample("loss", dist.Normal(lam, sigma), obs=fold_losses)


kernel = NUTS(model, jit_compile=True, target_accept_prob=0.85)
mcmc = MCMC(kernel, num_samples=1000, warmup_steps=500, num_chains=2)
mcmc.run(fold_losses_tensor)

mcmc.summary()          # prints mean, std, r_hat, n_eff per site
samples = mcmc.get_samples()
```

## Diagnostics (non-negotiable)

Every NUTS run MUST inspect:

| Diagnostic | Healthy | Action if bad |
|---|---|---|
| `r_hat` | < 1.01 | > 1.1 → need more warmup or re-parameterise |
| `n_eff` | > 200 | low → too much correlation, raise `target_accept_prob` to 0.9 |
| Divergences | 0 | any → reparameterise (non-centred priors, scaling) |
| Tree depth saturation | rare | frequent → increase `max_tree_depth` from 10 |

```python
print(mcmc.diagnostics())
```

## NumPyro speedup (when to jump backends)

If Pyro NUTS takes > 10 minutes, port JUST the inference script to NumPyro:

```python
import numpyro
import numpyro.distributions as dist_jax
from numpyro.infer import MCMC, NUTS
from jax import random

def model(fold_losses):
    lam = numpyro.sample("lambda_innov", dist_jax.HalfNormal(0.2))
    sigma = numpyro.sample("sigma", dist_jax.HalfCauchy(0.05))
    with numpyro.plate("folds", len(fold_losses)):
        numpyro.sample("loss", dist_jax.Normal(lam, sigma), obs=fold_losses)

kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=2)
mcmc.run(random.key(0), fold_losses_jnp)
```

~100× speedup for small-to-medium models. Worth the extra JAX dep only for MCMC-bound studies.

## Common reparameterisations

**Non-centred for hierarchical models**:

```python
# Centred (bad for NUTS with small n)
mu = pyro.sample("mu", dist.Normal(0, 1))
theta = pyro.sample("theta", dist.Normal(mu, sigma))   # correlated with mu

# Non-centred (NUTS-friendly)
mu = pyro.sample("mu", dist.Normal(0, 1))
theta_raw = pyro.sample("theta_raw", dist.Normal(0, 1))
theta = pyro.deterministic("theta", mu + sigma * theta_raw)
```

**Log-transform positive params**:

```python
log_sigma = pyro.sample("log_sigma", dist.Normal(0, 1))
sigma = pyro.deterministic("sigma", torch.exp(log_sigma))
```

## Gotchas

- **`jit_compile=True`**: faster but fails on dynamic shapes. Use only after validating without it.
- **`obs` shape**: must match the `plate` shape exactly — easy to get off-by-one.
- **Fresh RNG**: in NumPyro, `mcmc.run(rng_key, ...)` — reusing keys gives identical chains.

## References

- [Pyro MCMC docs](http://docs.pyro.ai/en/stable/mcmc.html)
- [NumPyro examples](https://num.pyro.ai/en/stable/examples/)
- Neal 2011, "MCMC using Hamiltonian dynamics" — the theory
