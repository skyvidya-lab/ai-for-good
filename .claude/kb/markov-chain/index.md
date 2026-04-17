# Markov Chain KB

Discrete-state, discrete-time stochastic processes. The mathematical scaffolding underneath our phenology transition prior.

## Why it matters for Dynamis Terra

Our 7-phenophase cycle (`Dormancy → Greenup → ... → Senescence → Dormancy`) is precisely a **Markov chain**: a sequence where the next state depends only on the current state, encoded by a row-stochastic transition matrix `A` of shape `7 × 7`.

Our [phenology_prior.py](../../../src/dynamis/phenology_prior.py) builds this matrix explicitly. The MKM then uses it to initialise the learnable Kalman transition, blending hard Markov structure with gradient-based refinement.

Understanding Markov chain theory lets us:
- Reason rigorously about *what* the prior encodes (what's reachable, what's absorbing).
- Check properties (irreducibility, stationarity) of our learned `A`.
- Validate synthetic data and simulate phenophase trajectories.
- Extend to hidden Markov models / continuous-time chains if needed.

## Sources

- Primary reference: **Markov Chains** lecture notes (University of Copenhagen, Math. Dept.)
  <https://www.math.ku.dk/bibliotek/arkivet/noter/stoknoter.pdf>
- **pgmpy**: Python library for graphical models, includes `MarkovChain` class.
  Code: <https://github.com/pgmpy/pgmpy>
  MarkovChain API: <https://pgmpy.org/models/markovchain.html>

## Concepts

- [markov-chain-basics.md](concepts/markov-chain-basics.md) — State space, transition matrix, Markov property
- [transition-matrix.md](concepts/transition-matrix.md) — Row-stochastic matrices, powers, n-step transitions
- [stationary-distribution.md](concepts/stationary-distribution.md) — Steady-state, irreducibility, ergodicity
- [classification-of-states.md](concepts/classification-of-states.md) — Absorbing / transient / recurrent / periodic states

## Patterns

- [pgmpy-markovchain.md](patterns/pgmpy-markovchain.md) — Using `pgmpy.models.MarkovChain` to simulate/validate our phenology prior
- [numpy-transition-matrix.md](patterns/numpy-transition-matrix.md) — Hand-rolled Markov chain in ~10 lines (what we actually use)

## External references

- pgmpy License: MIT
- pgmpy install: `pip install pgmpy`
- pgmpy capabilities: Bayesian Networks, Markov Networks, Dynamic BNs, causal inference, structure learning, inference — a superset that includes Markov chains
- Classical textbooks: Norris *Markov Chains* (1998); Grimmett & Stirzaker *Probability and Random Processes*
