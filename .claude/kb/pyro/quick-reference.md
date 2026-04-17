# Pyro / NumPyro Quick Reference

## Install

```bash
# Pyro (PyTorch backend, matches our torch-based Dynamis stack)
pip install pyro-ppl

# NumPyro (JAX backend, 100x faster HMC/NUTS)
pip install numpyro
# GPU:
pip install 'numpyro[cuda12]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Decision: Pyro vs NumPyro

| Criterion | Pyro | NumPyro |
|---|---|---|
| Backend | PyTorch | JAX |
| Best for | Neural + probabilistic models, deep learning integration | Fast MCMC, pure Bayesian models |
| Fits our stack | ✓ (we are PyTorch-native) | ✗ (extra JAX dep) |
| Speed of MCMC | Baseline | ~100× faster |
| Use in Dynamis Terra | **Preferred** for SVI on the MKM | Only for standalone calibration studies |

## The 3 primitives

```python
import pyro
import pyro.distributions as dist

# 1. Sample a random variable (latent OR observed via obs=...)
z = pyro.sample("z", dist.Normal(0., 1.))

# 2. Declare a learnable parameter (point estimate, not random)
scale = pyro.param("scale", torch.tensor(1.0), constraint=dist.constraints.positive)

# 3. Vectorise over independent batch dimensions
with pyro.plate("data", N):
    pyro.sample("obs", dist.Normal(z, scale), obs=data)
```

## The 3 inference patterns

### SVI (optimisation-based, deterministic)

```python
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

guide = pyro.infer.autoguide.AutoNormal(model)
svi = SVI(model, guide, Adam({"lr": 1e-3}), Trace_ELBO())
for step in range(1000):
    loss = svi.step(x, y)
```

### MCMC / NUTS (sampling-based, gold-standard but slow)

```python
from pyro.infer import MCMC, NUTS
kernel = NUTS(model)
mcmc = MCMC(kernel, num_samples=1000, warmup_steps=500)
mcmc.run(x, y)
posterior = mcmc.get_samples()
```

### Predictive (sample from prior or posterior)

```python
from pyro.infer import Predictive
pred = Predictive(model, guide=guide, num_samples=500)
samples = pred(x_test)  # dict of {"z": (500, ...), ...}
```

## Effect handlers (advanced)

| Handler | Purpose |
|---|---|
| `pyro.poutine.trace` | Record the execution trace (all sampled values) |
| `pyro.poutine.replay` | Replay sampled values from a previous trace |
| `pyro.poutine.condition` | Fix certain latent variables to given values |
| `pyro.poutine.mask` | Mask observations (useful for missing data) |
| `pyro.poutine.do` | Causal do-intervention |

## Common gotchas

- Forgetting `pyro.clear_param_store()` between experiments → stale params.
- `obs=` argument on `pyro.sample` is what distinguishes latent from observed.
- `AutoNormal` guide assumes mean-field (independent) posteriors — for correlations use `AutoMultivariateNormal`.
- NumPyro: always pass a `jax.random.key(seed)` to `mcmc.run(rng_key, ...)`.

## Within our repo

```python
# Proposed integration (v5+): wrap the MKM as a Pyro module
from src.dynamis.dynamis_core import MarkovKalmanModule
import pyro, pyro.nn

mkm = MarkovKalmanModule(state_dim=7)
mkm_pyro = pyro.nn.PyroModule[type(mkm)](mkm)
# Priors on A, H, log_Q, log_R become first-class Bayesian parameters
```
