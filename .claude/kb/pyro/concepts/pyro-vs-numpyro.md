# Pyro vs NumPyro

Same mental model, two backends. The differences matter when choosing.

## Same

- Same primitives (`sample`, `param`, `plate`, effect handlers).
- Same distribution API (PyTorch-style, constraints, transforms).
- Same inference algorithms (SVI with ELBO variants, MCMC with HMC/NUTS, Predictive).
- Same effect-handler-based inference design.

## Different

| Aspect | Pyro (PyTorch) | NumPyro (JAX) |
|---|---|---|
| Tensor library | `torch.Tensor` | `jax.numpy` arrays |
| Autograd | PyTorch autograd | JAX `grad`, `vmap`, `jit` |
| State | Global `pyro.get_param_store()` | Functional — pass `rng_key` explicitly |
| GPU | `.to(device)` on tensors | `jax.device_put` / implicit via JAX config |
| Deep learning integration | `pyro.nn.PyroModule` wraps `nn.Module` | Thin wrapper via `flax` or `haiku` |
| MCMC speed | Baseline | **~100× faster** (JIT compilation of kernels) |
| SVI speed | Comparable | Slightly faster |
| Discrete latents | Enumeration via `TraceEnum_ELBO` | Enumeration via same, plus `infer_discrete` |
| Compile time | None (eager) | First-call JIT compile (seconds) |

## Which for Dynamis Terra

**Pyro.** Reasoning:

1. Our entire stack is PyTorch (MKM, ChaosAttention, training loop). NumPyro would require porting to JAX or maintaining two code paths.
2. We care more about integration than MCMC speed. Our models train via SVI-like objectives, not NUTS.
3. `PyroModule` wrapping lets us promote existing `nn.Module` parameters to Bayesian random variables incrementally.

**NumPyro** — only if we ever do a standalone Bayesian study (e.g. "given 5 fold checkpoints, what is the posterior distribution over `lambda_innovation`?") on small tabular data where MCMC is viable.

## Migration cost

If we started in Pyro and later wanted NumPyro speedups:
- Distribution calls: mostly drop-in (`dist.Normal` etc. identical).
- Primitives: identical signatures.
- Models: rewrite tensor ops from `torch` to `jnp`.
- Guides: rewrite neural components with `flax` / `haiku`.
- `rng_key` threading: non-trivial refactor.

Rule of thumb: writing Pyro-style code with no PyTorch-specific tricks keeps the door open.

## References

- Pyro paper: Bingham et al. 2019 ([JMLR](https://jmlr.org/papers/v20/18-403.html))
- NumPyro paper: Phan et al. 2019 ([arXiv:1912.11554](https://arxiv.org/abs/1912.11554))
