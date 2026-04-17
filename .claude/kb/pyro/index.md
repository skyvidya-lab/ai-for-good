# Pyro / NumPyro KB

Universal probabilistic programming on PyTorch (Pyro) and JAX (NumPyro). Both share the same core primitives — `sample`, `param`, `plate` — and the same mental model: a stochastic function *is* a probabilistic model, inference is done by reinterpreting primitives (via guides, MCMC kernels, or effect handlers).

## Why it matters for Dynamis Terra

Our MKM in [src/dynamis/dynamis_core.py](../../../src/dynamis/dynamis_core.py) is a hand-written differentiable Kalman filter. Pyro/NumPyro would let us:

- Express the MKM as a generative model with explicit priors on `A`, `H`, `Q`, `R`.
- Run **Stochastic Variational Inference (SVI)** over the MKM parameters instead of point MLE via backprop — proper uncertainty on matrix estimates, not just `P`.
- Swap the dual classification head for a **probabilistic head** that marginalises over phenophase trajectories (via `TraceEnum_ELBO`).
- Use **MCMC/NUTS** for calibration studies on small validation folds.

Adoption is optional — we have a working model. Considered for v5+ if we want to quantify structural uncertainty on `A` and the phenology prior.

## Concepts

- [pyro-vs-numpyro.md](concepts/pyro-vs-numpyro.md) — When to choose each
- [core-primitives.md](concepts/core-primitives.md) — `sample`, `param`, `plate`, effect handlers
- [inference-algorithms.md](concepts/inference-algorithms.md) — SVI, MCMC (NUTS/HMC), enumeration
- [distributions.md](concepts/distributions.md) — PyTorch-compatible API, constraints, transforms

## Patterns

- [svi-elbo.md](patterns/svi-elbo.md) — Minimal SVI training loop
- [mcmc-nuts.md](patterns/mcmc-nuts.md) — NUTS sampler for posterior calibration
- [probabilistic-dynamis-model.md](patterns/probabilistic-dynamis-model.md) — Wrapping our MKM as a `numpyro.module`

## External References

- Landing: https://pyro.ai
- Pyro (PyTorch): https://github.com/pyro-ppl/pyro · docs: http://docs.pyro.ai
- NumPyro (JAX): https://github.com/pyro-ppl/numpyro · docs: http://num.pyro.ai
- Paper (Pyro): Bingham et al. 2019 · Paper (NumPyro): Phan et al. 2019
- License: Apache 2.0
