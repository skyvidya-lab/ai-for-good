# Dynamis KB

Physics-informed modules for chaos inference on dynamic systems.

## Core reference — the parent theory

Our `HRM_MKM` architecture extends the **Hierarchical Reasoning Model** (HRM) of:

> Wang, G., Li, J., Sun, Y., Chen, X., Liu, C., Wu, Y., Lu, M., Song, S., & Abbasi Yadkori, Y. (2025). *Hierarchical Reasoning Model*. arXiv:2506.21734. <https://arxiv.org/abs/2506.21734>
> Local copy: `C:\Users\eluzq\workspace\00_benchmarking\skyvidya_core\skyvidya_dynamis\dynamis-teoria\2506.21734v1.pdf`

The HRM paper introduces the two-module recurrent architecture (slow abstract planner + fast tactical computer) that we extend with an explicit Markov-Kalman Module for uncertainty-aware latent state tracking.

## Concepts

- [markov-kalman-module.md](concepts/markov-kalman-module.md) — Differentiable Kalman Filter (A, H, Q, R learnable)
- [hurst-exponent.md](concepts/hurst-exponent.md) — R/S analysis for persistence detection
- [innovation-loss.md](concepts/innovation-loss.md) — "Surprise" minimisation objective
- [chaos-attention.md](concepts/chaos-attention.md) — Physics-modulated attention bias

## Patterns

- [phenology-transition-prior.md](patterns/phenology-transition-prior.md) — 7×7 matrix for MKM initialisation
- [agriculture-domain-mapping.md](patterns/agriculture-domain-mapping.md) — Finance → Agriculture concept map
- [physics-vector-injection.md](patterns/physics-vector-injection.md) — **v3 fix**: expose MKM/innovation/P signals as explicit features to the crop head

## Quick Reference

- [quick-reference.md](quick-reference.md)

## Related KB Domains (external theory)

- [../kalman-bayesian/](../kalman-bayesian/index.md) — Foundational KF theory (Labbe book). The math underneath our MKM.
- [../markov-chain/](../markov-chain/index.md) — Discrete Markov chain theory. Underpins the phenology transition prior.
- [../pyro/](../pyro/index.md) — Probabilistic programming (Pyro / NumPyro). Candidate for v5+ Bayesian MKM.

## Core Philosophy

> "LLMs are Cerebrums (semantic); Dynamis is the Cerebellum (physics/motor)."

Dynamis treats data as a chaotic dynamical system. It does not predict tokens — it predicts **latent state evolution** with explicit uncertainty (Kalman `P` matrix) and surprise quantification (innovation).
