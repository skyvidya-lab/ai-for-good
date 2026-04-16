"""
Hurst Exponent for geospatial / vegetation time-series.

H < 0.5  : anti-persistent (mean-reverting) — unstable vegetation
H ~ 0.5  : random walk / no memory
H > 0.5  : persistent (trending) — sustained growth or decline

Two modes:
    - temporal: H over NDVI time-series of a point (requires >= min_dates)
    - spectral: H over the 13-band spectral response at a single date (always available)

The conditional design means models can always compute *some* Hurst feature and use
a mask to indicate provenance (temporal vs spectral vs missing).
"""
from __future__ import annotations

import numpy as np

from .dynamis_core import calculate_hurst  # re-export the core implementation


def hurst_temporal(
    series: np.ndarray,
    min_dates: int = 6,
    min_window: int = 2,
) -> float:
    """
    Hurst on a temporal series (e.g. NDVI over observation dates).

    Returns NaN if fewer than `min_dates` valid points — caller should mask.
    """
    valid = np.asarray(series, dtype=np.float64)
    valid = valid[~np.isnan(valid)]
    if valid.size < min_dates:
        return float("nan")
    return calculate_hurst(valid, min_window=min_window, max_window=valid.size)


def hurst_spectral(bands_at_date: np.ndarray) -> float:
    """
    Hurst on the 13-band spectral response at a single observation date.

    Treats the bands as a pseudo-series (ordered by wavelength).
    Always returns a value (falls back to 0.5 if degenerate).
    """
    x = np.asarray(bands_at_date, dtype=np.float64)
    x = x[~np.isnan(x)]
    if x.size < 4:
        return 0.5
    return calculate_hurst(x, min_window=2, max_window=x.size)


def hurst_features(
    ndvi_series: np.ndarray,
    bands_per_date: np.ndarray,
    min_temporal_dates: int = 6,
) -> dict[str, float]:
    """
    Compute both Hurst flavours plus a provenance mask.

    Args:
        ndvi_series: (T,) NDVI values per observation date (may contain NaN).
        bands_per_date: (T, 13) raw bands per date.
        min_temporal_dates: threshold to trust temporal Hurst.

    Returns:
        dict with keys: hurst_temporal, hurst_spectral_mean, hurst_spectral_std,
        hurst_temporal_valid (1.0/0.0 mask).
    """
    h_temp = hurst_temporal(ndvi_series, min_dates=min_temporal_dates)
    spectrals = []
    for t in range(bands_per_date.shape[0]):
        spectrals.append(hurst_spectral(bands_per_date[t]))
    spectrals = np.asarray(spectrals, dtype=np.float64)
    return {
        "hurst_temporal": h_temp if not np.isnan(h_temp) else 0.5,
        "hurst_spectral_mean": float(np.nanmean(spectrals)) if spectrals.size else 0.5,
        "hurst_spectral_std": float(np.nanstd(spectrals)) if spectrals.size else 0.0,
        "hurst_temporal_valid": 0.0 if np.isnan(h_temp) else 1.0,
    }


__all__ = [
    "calculate_hurst",
    "hurst_temporal",
    "hurst_spectral",
    "hurst_features",
]
