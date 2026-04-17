# Inference Algorithms

Pyro supports three families of inference. Choose based on model size and what kind of posterior approximation you need.

## 1. Stochastic Variational Inference (SVI)

Optimises a parameterised *guide* `q(z|x)` to approximate the true posterior `p(z|x)` by maximising the ELBO:

```
ELBO(q) = E_{q(z)} [ log p(x, z) - log q(z) ]
       = E_{q(z)} [ log p(x|z) ] - KL( q(z) || p(z) )
```

**When to use**: large datasets, neural-net-parameterised posteriors, when you need training speed and approximate uncertainty.

```python
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam

guide = AutoNormal(model)                  # mean-field Gaussian
svi = SVI(model, guide, Adam({"lr": 1e-3}), Trace_ELBO())
for step in range(n_steps):
    loss = svi.step(x, y)
```

**ELBO variants**:

| ELBO | When to use |
|---|---|
| `Trace_ELBO` | Default; continuous latents |
| `TraceMeanField_ELBO` | Known fully-factorised guide; more stable |
| `TraceGraph_ELBO` | Discrete latents; uses rao-blackwellisation |
| `TraceEnum_ELBO` | Discrete latents enumerated analytically (ideal for HMM-like 7-phenophase models) |
| `RenyiELBO` | Alpha-divergence variants |

**Auto-guides** (no manual guide writing):

- `AutoNormal` — mean-field Gaussian (independent dims).
- `AutoMultivariateNormal` — full-covariance Gaussian.
- `AutoDelta` — MAP estimation (point estimate).
- `AutoDiagonalNormal` — diagonal Gaussian.
- `AutoIAFNormal` — normalising flow via IAF for rich posteriors.

## 2. Markov Chain Monte Carlo (MCMC)

Samples directly from the posterior. Gold standard for accuracy, slow for large models.

**NUTS (No U-Turn Sampler)** — adaptive HMC, default choice for continuous latents.

```python
from pyro.infer import MCMC, NUTS

kernel = NUTS(model, jit_compile=True)        # PyTorch jit
mcmc = MCMC(kernel, num_samples=1000, warmup_steps=500)
mcmc.run(x, y)
samples = mcmc.get_samples()                   # dict of tensors
```

**When to use**: small to medium models, when posterior accuracy matters (e.g. calibration studies, hyperparameter inference). NumPyro gives ~100× speedup via JAX JIT.

## 3. Discrete Latent Enumeration

For models with discrete latents (like our 7 phenophases), Pyro can **enumerate** the discrete space instead of sampling — exact marginalisation.

```python
@config_enumerate
def model(x):
    # Discrete phenophase latent — enumerate all 7 values
    pheno = pyro.sample("pheno", dist.Categorical(torch.ones(7) / 7))
    # Continuous observation depends on the chosen pheno
    crop = pyro.sample("crop", dist.Categorical(logits_for_pheno[pheno]),
                       obs=crop_label)
```

Used with `TraceEnum_ELBO` in SVI or with `infer_discrete` in MCMC, this gives exact likelihoods without the variance of sampling. Critical if we ever make phenophase a random latent rather than an observed label.

## Predictive (sample from fitted posterior)

```python
from pyro.infer import Predictive

# Prior predictive (no guide)
pred_prior = Predictive(model, num_samples=500)

# Posterior predictive (uses fitted guide)
pred_post = Predictive(model, guide=guide, num_samples=500)

samples = pred_post(x_test)        # {"z": (500, ...), "y": (500, ...)}
```

Returns Monte Carlo samples of all sites. Compute any summary statistic you need — mean, quantiles, credible intervals.

## Diagnostics

- **ELBO curve**: should decrease monotonically (bumps = high variance gradients).
- **NUTS**: inspect `r_hat` per parameter (should be < 1.01) and `n_eff` (effective sample size).
- **ECE**: our `expected_calibration_error` from [src/dynamis/innovation_loss.py](../../../../src/dynamis/innovation_loss.py) works on any probabilistic prediction.

## For Dynamis Terra

| Use case | Algorithm |
|---|---|
| Replace current MKM training with Bayesian MKM | **SVI with `AutoNormal`** on `A`, `log_Q`, `log_R` |
| Marginalise over phenophase labels (unsupervised mode) | **SVI with `TraceEnum_ELBO`** |
| Calibration: posterior over `lambda_innovation` given 5 folds | **NUTS** (small model, worth the wait) |
| Inference-time uncertainty | **Predictive** with fitted guide |
