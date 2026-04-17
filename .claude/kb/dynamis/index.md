# Dynamis KB

Physics-informed modules for chaos inference on dynamic systems.

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

## Core Philosophy

> "LLMs are Cerebrums (semantic); Dynamis is the Cerebellum (physics/motor)."

Dynamis treats data as a chaotic dynamical system. It does not predict tokens — it predicts **latent state evolution** with explicit uncertainty (Kalman `P` matrix) and surprise quantification (innovation).
