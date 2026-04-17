# Core Primitives

Four primitives define every Pyro/NumPyro model. Master these and you can read any Pyro code.

## 1. `pyro.sample(name, dist, obs=None)`

Draws from a distribution. The same call plays two roles:

- **Latent variable** (when `obs=None`) — a random variable the inference engine will learn a posterior over.
- **Observed variable** (when `obs=data`) — a likelihood term, conditions the model on known values.

```python
z = pyro.sample("z", dist.Normal(0, 1))               # latent
pyro.sample("y", dist.Normal(z, 0.1), obs=y_data)     # observed
```

The `name` must be unique per execution. Inference handlers identify variables by name.

## 2. `pyro.param(name, init, constraint=...)`

Declares a **point-estimate** parameter (not a random variable). Learned by maximum likelihood or MAP, not by Bayesian inference.

```python
bias = pyro.param("bias", torch.zeros(7))
scale = pyro.param("scale", torch.ones(1),
                   constraint=dist.constraints.positive)
```

In our MKM, `A`, `log_Q_diag`, `log_R_diag` are morally `param`s (we train them by backprop). Promoting them to `sample` statements would make them Bayesian — the payoff is uncertainty on the matrix estimates themselves.

## 3. `pyro.plate(name, size)`

Declares a dimension along which variables are conditionally **independent**. Critical for correct ELBO computation (without `plate`, Pyro assumes full dependence → wrong likelihood).

```python
with pyro.plate("points", N):
    # Each of the N samples is iid
    pyro.sample("obs", dist.Normal(mu, sigma), obs=data)
```

For multi-temporal sequences (our case): `plate` over points, inner sequential structure for time.

Nested plates:

```python
with pyro.plate("points", N, dim=-2):
    with pyro.plate("time", T, dim=-1):
        pyro.sample("x", dist.Normal(0, 1))  # shape (N, T)
```

## 4. `pyro.module(name, nn_module)` / `PyroModule`

Promotes a `torch.nn.Module` into a Pyro-registered module. Its parameters live in the Pyro param store and can be individually converted to random variables.

```python
from pyro.nn import PyroModule, PyroSample

class BayesianMKM(PyroModule):
    def __init__(self, state_dim):
        super().__init__()
        # Prior on A — mean-identity (weak prior toward random walk)
        self.A = PyroSample(
            dist.Normal(torch.eye(state_dim), 0.5).to_event(2)
        )
        ...
```

## Effect Handlers (the meta-primitive)

Primitives don't execute directly — they emit messages that handlers interpret. This is why the SAME `def model():` function can be used for prior sampling, SVI, MCMC, or trace recording.

Common handlers:

| Handler | What it does |
|---|---|
| `poutine.trace(model)` | Record every `sample`/`param` call into a dict |
| `poutine.replay(model, trace)` | Re-run but substitute values from a trace |
| `poutine.condition(model, data)` | Fix latent variables to given values |
| `poutine.mask(model, mask)` | Disable likelihoods where `mask==False` |
| `poutine.do(model, data)` | Causal intervention — break upstream deps |

Used together, they give you powerful introspection:

```python
tr = poutine.trace(model).get_trace(x)
for name, node in tr.nodes.items():
    if node["type"] == "sample":
        print(name, node["value"].shape, node.get("log_prob", None))
```

## Relation to our codebase

In [src/dynamis/dynamis_core.py](../../../../src/dynamis/dynamis_core.py), the `MarkovKalmanModule` currently has:

```python
self.A = nn.Parameter(torch.eye(state_dim))       # → pyro.param equivalent
self.log_Q_diag = nn.Parameter(...)               # → pyro.param equivalent
# Innovation computation → becomes pyro.sample with obs=measurement
```

A Bayesian upgrade would replace `nn.Parameter` with `PyroSample(prior)` and the innovation computation with explicit `pyro.sample("innovation_t", dist.Normal(..., R))`.
