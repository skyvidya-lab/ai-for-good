# Hurst Exponent

A single scalar that classifies a time series as **mean-reverting** (`H < 0.5`), **random walk** (`H ≈ 0.5`), or **persistent / trending** (`H > 0.5`). In Dynamis we use it as a **regime feature** — different crops leave different persistence fingerprints on NDVI.

Implementation: [src/dynamis/dynamis_core.py::calculate_hurst](../../../../src/dynamis/dynamis_core.py) (generic R/S) and [src/dynamis/hurst_geo.py](../../../../src/dynamis/hurst_geo.py) (geospatial cascade).

## Intuition

> "Does the series remember where it's been?"

- **Brownian motion** (random coin flips): H = 0.5. No memory.
- **Mean-reverting** (stock with bounded oscillation): H < 0.5. High → low → high.
- **Trending** (NDVI during Greenup): H > 0.5. Low → medium → high, monotonically.

## Computation: R/S analysis

The classical Hurst exponent is computed by fitting `log(R/S) = H · log(n) + c`:

```python
# For window sizes n_1, n_2, ...:
#   1. Split the series into chunks of size n_k.
#   2. For each chunk: compute cumulative deviation, range R, std S.
#   3. Average R/S across chunks → one point (n_k, R/S_k).
# Slope of log(n) vs log(R/S) = H.
```

See the full code in [calculate_hurst](../../../../src/dynamis/dynamis_core.py).

## Why it's useful for Dynamis Terra

Crop-specific persistence signatures:

| Crop | Hurst on NDVI during Greenup→Peak | Why |
|---|---|---|
| **Rice** | Paddy flooding creates a non-monotonic pulse → H lower (~0.6) | NDWI spikes before NDVI rises |
| **Corn** | Fastest, most monotonic canopy closure → H highest (~0.85) | Aggressive C4 growth |
| **Soybean** | Slower, smoother rise → H moderate (~0.7) | Legume, less rapid expansion |

A single Hurst value per point is already a discriminative feature. Our classifier receives it directly as part of the physics vector — see [patterns/physics-vector-injection.md](../patterns/physics-vector-injection.md).

## The saturation problem (and our v4 fix)

**Pathology**: on a strongly-monotonic series (Dormancy ≈ 0 → Peak ≈ 0.9 → Senescence ≈ 0.1, few timesteps), the log(R/S) vs log(n) regression fit produces slope ≥ 1.0, which is clipped to 1.0. Every such point gets H = 1.00 → feature has zero discriminative power.

v2 diagnostic: 100% of points had H ≈ 1.0.
v3 diagnostic: ~66% of points still saturated (despite better data).
v4 fix: introduce **DFA** (Detrended Fluctuation Analysis) and **diff-regional** variants, and use a priority cascade:

```
cascade: hurst_dfa → hurst_diff_regional → hurst_regional → hurst_temporal → hurst_spectral → 0.5
```

- **DFA** ([hurst_geo.py::hurst_dfa](../../../../src/dynamis/hurst_geo.py)): integrates the series, detrends each window linearly, fits log(F) vs log(n). Naturally bounded; does not blow up on monotone inputs.
- **diff-regional** ([hurst_diff_regional](../../../../src/dynamis/hurst_geo.py)): classical R/S on the **first differences** — removes the dominant trend that causes saturation.

Target: `saturation fraction < 10%`, `std > 0.1`.

## Temporal vs regional vs spectral

Three ways to compute Hurst from a PointSeries:

1. **Temporal** ([hurst_temporal](../../../../src/dynamis/hurst_geo.py)): on the NDVI values AT THE POINT across its own observation dates (typically 7 in our labels). Short series → noisy.

2. **Regional** ([hurst_regional](../../../../src/dynamis/hurst_geo.py)): on the NDVI values AT THE POINT'S COORDINATE across ALL consolidated dates of its REGION (up to ~18 dates after merging the 4 folders). Longer series → more stable.

3. **Spectral** ([hurst_spectral](../../../../src/dynamis/hurst_geo.py)): treats the 13-band reflectance AT A SINGLE DATE as a "series" ordered by wavelength. Always available (no date-count requirement) but weaker signal. Used as last-resort fallback.

The cascade prefers DFA/diff on regional series, falling back to temporal, then spectral, then a neutral 0.5.

## Usage in the pipeline

Cell 8 of [notebooks/02_baseline_vs_dynamis_v4.ipynb](../../../../notebooks/02_baseline_vs_dynamis_v4.ipynb) runs the cascade per point and stores:

- `hurst_vec: (N,)` — one scalar per point.
- `hurst_source: (N,)` int — which cascade level produced each value (for diagnostics).

`hurst_vec` flows into:

- The Dynamis model's `hurst` input — modulates ChaosAttention bias and is part of the physics vector.
- The LightGBM baseline's `phenology_features` (as `hurst_ndvi_anomaly = |H - 0.5|`).

## References

- Hurst 1951 (original hydrology paper, long-term Nile reservoir storage).
- Peng et al. 1994, "Mosaic organization of DNA nucleotides" — introduces DFA.
- Our empirical tests: [tests/test_dynamis_modules.py](../../../../tests/test_dynamis_modules.py) — DFA non-saturation, RW sanity.
- Theory reference: [kalman-bayesian/concepts/kalman-basics.md](../../kalman-bayesian/concepts/kalman-basics.md) — related dynamics concepts.
