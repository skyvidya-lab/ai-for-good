# Dynamis: Finance → Agriculture Domain Mapping

The generic Dynamis core (`skyvidya_dynamis/dynamis-core-gsm-poc/dynamis_core.py`) is ported verbatim. What changes are the **interpretations** of the physical quantities.

## Mapping table

| Generic Dynamis concept | Finance analogue (V6 Gravitational) | Agriculture analogue (Terra) |
|---|---|---|
| Time series | Price candlesticks | NDVI/EVI per observation date |
| Feature 1 (Energy) | RSI | NDVI (vegetation vigor) |
| Feature 2 (Temperature) | Volatility σ₂₀ | Spectral entropy across 13 bands |
| Feature 3 (Velocity) | Momentum Δ₅ | ΔNDVI between dates |
| Feature 4 (Pressure/Mass) | Relative Volume | SWIR moisture ratio (B11/B12) |
| Latent state | Market regime | Phenophase (7 canonical) |
| Kalman A matrix | Regime transitions | **Phenology transition prior** (see `phenology_prior.py`) |
| Innovation | Price surprise | Spectral surprise at a date |
| Hurst H > 0.5 | Trending market | Sustained canopy change (growth or stress) |
| Hurst H < 0.5 | Mean-reverting | Noisy / stable vegetation |
| Dragon Kings | Crash precursors | Drought / pest onset |
| Chaos score | Innovation magnitude | Mean‖innovation‖² over observation window |
| Gravity | SELIC / macro force | Seasonal solar forcing (implicit via date) |

## Why this matters

- The **maths is identical**; only the data contract changes.
- The Innovation Loss objective generalises: minimise surprise → learn true dynamics.
- Uncertainty (Kalman `P`) is domain-agnostic — the same `trace(P)` that spotted market crashes can spot background/unknown crops.

## Keep constant across domains

- `HilbertEmbedding`, `MarkovKalmanModule`, `HRM_MKM`, `HierarchicalHPR` — all reusable as-is.
- `innovation_loss()` — unchanged.
- `calculate_hurst()` — generic on any 1D series.

## Replace per domain

- Feature extractor (finance `MarketBrain` → agriculture `vegetation_indices`).
- Transition prior (finance regime → phenology 7-state).
- Attention physics vector (finance {chaos, gravity, hurst, innovation} → agriculture {chaos, hurst}).
