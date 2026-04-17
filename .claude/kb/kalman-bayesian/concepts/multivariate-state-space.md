# Multivariate State-Space Design

Moving from Labbe's 1-D toy (ch. 04) to the real thing (ch. 06). This is where practical design choices actually matter.

## Choosing the state dimension

The state `x` must capture **everything the filter needs to roll forward deterministically** if there were no noise. Two extremes:

- **Too small** — filter can't express the dynamics. Innovations won't be white; biases accumulate.
- **Too large** — unobservable states drift around, `P` becomes ill-conditioned, training is unstable.

Guideline: start with the minimal state that makes `A` linear and the observation model `H` plausible.

For Dynamis Terra our state is **7-dimensional** (one per phenophase). Each dimension represents the "mass" (unnormalised probability) of being in that stage.

## Transition matrix `A` — structure matters

For an unconstrained KF, `A` is a free `state_dim × state_dim` matrix. For physically-meaningful systems, impose structure:

### Sparse (band-diagonal)

Most systems have local dependencies. Phenology: stage `k` only depends on stages `{k-1, k, k+1}`.

```python
# 7-phenophase sparse A (tridiagonal)
A = torch.zeros(7, 7)
for i in range(7):
    A[i, i] = 0.7                # self-loop
    if i + 1 < 7:
        A[i, i+1] = 0.3          # forward
```

### Row-stochastic (HMM-like)

Rows sum to 1 — preserves "mass". Our [phenology_prior.py](../../../../src/dynamis/phenology_prior.py) builds this.

### Block-diagonal

For multi-subsystem problems (e.g. independent sensors observing different latent processes).

### Learnable with a prior

Our approach: `A` is a `nn.Parameter` initialised by blending a phenology prior with the identity:

```python
A_prior = build_phenology_prior_tensor()
A_init = (1 - lambda_prior) * torch.eye(7) + lambda_prior * A_prior
self.A = nn.Parameter(A_init)
```

The prior gives the filter a reasonable starting point; backprop adjusts specifics.

## Observation matrix `H`

Maps latent state to measurement. Three common designs:

### Identity
`z = x` — the measurement *is* the state (rare in practice).

### Pick-off rows
`H` is a 0/1 matrix that selects some state components:

```
H = [[1, 0, 0, 0, 0, 0, 0],   # observe "Peak"
     [0, 0, 0, 1, 0, 0, 0]]
```

### Full learnable
`H` is an unconstrained `obs_dim × state_dim` matrix, learned via backprop (our setup for the MKM — `H` is just another `nn.Parameter`).

## Initial state `x_0`, `P_0`

Two camps:

- **Informative `x_0`**: if you know the initial state (e.g. always starts in Dormancy), set `x_0` accordingly with small `P_0`.
- **Uninformative**: set `x_0 = 0`, `P_0 = large * I`. The filter converges after a few observations.

Our default is `x_0 = 0`, `P_0 = 0.1 * I` — small because (a) we don't want `P` to saturate before t=1, (b) the input-projection neural layer already encodes useful prior information.

## Observability and identifiability

A multivariate KF is only well-posed if the state is **observable** — informally, if the measurement history contains enough information to pin down the state.

**Observability test**: the matrix `[H; HA; HA²; ...; HA^(n-1)]` must have rank `state_dim`.

If not observable, some components of `x` drift freely without the measurements constraining them. Symptoms:
- `P` doesn't decrease along some eigenvectors over time.
- Filter estimates are not reproducible — they depend on `x_0`.

For our 7-state phenology KF, observability depends on both `H` and the temporal structure. With 10+ observations per region, we're well within the observable regime.

## Multi-sample batching

Our KF is **batched**: every call processes `(B, T, state_dim)` for B different points simultaneously. Critical implementation details:

- Matrices `A`, `H` are **shared** across the batch (one set of parameters).
- `x`, `P` have a leading batch dim: `x: (B, state_dim)`, `P: (B, state_dim, state_dim)`.
- Operations use `torch.bmm` (batched matmul), not `@`.

This is NOT standard Labbe material — it's a deep-learning generalisation. Labbe runs one filter; we run B filters sharing parameters.

## References

- Labbe ch. 06 — multivariate KF in Python.
- Kalman & Bucy 1961, "New results in linear filtering and prediction theory."
- Anderson & Moore, *Optimal Filtering* — the rigorous reference on observability.
