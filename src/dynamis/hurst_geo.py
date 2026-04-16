"""
Hurst Exponent for geospatial / vegetation time-series.

H < 0.5  : anti-persistent (mean-reverting) — unstable vegetation
H ~ 0.5  : random walk / no memory
H > 0.5  : persistent (trending) — sustained growth or decline

Three modes (in order of preference for the agricultural task):

1. `hurst_regional(region_view, band)`: the BEST option when folder
   consolidation is used. Collects the band value for the region's centroid
   across ALL dates of ALL 4 folders and computes a single Hurst per region
   using ~18 datapoints. Reliable.

2. `hurst_temporal(ndvi_series)`: per-point Hurst on the point's own NDVI
   trajectory. Only meaningful when `min_dates >= 8`; returns NaN otherwise
   so the caller can mask instead of injecting a degenerate 0.5.

3. `hurst_spectral(bands_at_date)`: 13-band pseudo-series at a single date
   — fallback that is always available but has weaker discriminative power.

The conditional design means models can always compute *some* Hurst feature
and use a mask to indicate provenance (regional vs temporal vs spectral).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .dynamis_core import calculate_hurst  # re-export the core implementation


def hurst_temporal(
    series: np.ndarray,
    min_dates: int = 8,
    min_window: int = 3,
) -> float:
    """
    Hurst on a temporal series (e.g. NDVI over observation dates).

    Returns NaN if fewer than `min_dates` valid points — callers should mask.
    """
    valid = np.asarray(series, dtype=np.float64)
    valid = valid[~np.isnan(valid)]
    if valid.size < min_dates:
        return float("nan")
    return calculate_hurst(valid, min_window=min_window, max_window=max(min_window, valid.size // 2))


def hurst_spectral(bands_at_date: np.ndarray) -> float:
    """
    Hurst on the 13-band spectral response at a single observation date.

    Treats the bands as a pseudo-series (ordered by wavelength).
    Returns NaN on degenerate input (caller should fall back to 0.5 or similar).
    """
    x = np.asarray(bands_at_date, dtype=np.float64)
    x = x[~np.isnan(x)]
    if x.size < 6:
        return float("nan")
    return calculate_hurst(x, min_window=2, max_window=x.size)


def hurst_regional(
    region_view: dict[str, dict[str, "Path"]],  # {date: {band: path}}
    extract_fn,
    lon: float,
    lat: float,
    band_code: str = "ndvi_from_bands",
    min_dates: int = 8,
) -> float:
    """
    Hurst exponent over the ALL-dates NDVI series at a single (lon, lat) for a region.

    This is the preferred Hurst variant when the 4 folders have been consolidated:
    a region typically has 10-20 dates across folders, which is enough for R/S
    to stabilise.

    Args:
        region_view: {date: {band: path}} for ONE region (from `consolidate_regions`).
        extract_fn: callable `(band_paths, lon, lat, bands_order) -> np.ndarray`
            (typically `src.data.point_extractor.extract_bands_at_point`).
        lon, lat: WGS84 coordinates of the target point.
        band_code: `"ndvi_from_bands"` uses (B08-B04)/(B08+B04); otherwise pass a
            raw band name like "B08".
        min_dates: minimum consolidated dates required to trust the result.

    Returns:
        Hurst value, or NaN if fewer than `min_dates` valid dates.
    """
    from ..data.sentinel2_loader import MODEL_BANDS  # local import to avoid cycle

    dates = sorted(region_view.keys())
    vals: list[float] = []
    for d in dates:
        band_paths = region_view[d]
        if band_code == "ndvi_from_bands":
            bands_vec = extract_fn(band_paths, lon, lat, list(MODEL_BANDS))
            # Compute NDVI from B08 and B04 (indices 7 and 3 in MODEL_BANDS)
            nir = bands_vec[7]
            red = bands_vec[3]
            if np.isnan(nir) or np.isnan(red):
                continue
            ndvi = (nir - red) / (nir + red + 1e-6)
            vals.append(float(ndvi))
        else:
            bands_vec = extract_fn(band_paths, lon, lat, [band_code])
            if not np.isnan(bands_vec[0]):
                vals.append(float(bands_vec[0]))

    arr = np.asarray(vals, dtype=np.float64)
    if arr.size < min_dates:
        return float("nan")
    return calculate_hurst(arr, min_window=3, max_window=max(3, arr.size // 2))


def hurst_features(
    ndvi_series: np.ndarray,
    bands_per_date: np.ndarray,
    min_temporal_dates: int = 8,
) -> dict[str, float]:
    """
    Compute both Hurst flavours plus a provenance mask.

    Args:
        ndvi_series: (T,) NDVI values per observation date (may contain NaN).
        bands_per_date: (T, 13) raw bands per date.
        min_temporal_dates: threshold to trust temporal Hurst.

    Returns:
        dict with keys: hurst_temporal, hurst_spectral_mean, hurst_spectral_std,
        hurst_temporal_valid (1.0/0.0 mask), hurst_fallback.
    """
    h_temp = hurst_temporal(ndvi_series, min_dates=min_temporal_dates)
    spectrals: list[float] = []
    for t in range(bands_per_date.shape[0]):
        hs = hurst_spectral(bands_per_date[t])
        if not np.isnan(hs):
            spectrals.append(hs)
    sp_arr = np.asarray(spectrals, dtype=np.float64) if spectrals else np.array([0.5])

    temporal_valid = 0.0 if np.isnan(h_temp) else 1.0
    # Fallback cascade: temporal → spectral mean → 0.5
    if not np.isnan(h_temp):
        fallback = h_temp
    elif sp_arr.size > 0 and not np.all(np.isnan(sp_arr)):
        fallback = float(np.nanmean(sp_arr))
    else:
        fallback = 0.5

    return {
        "hurst_temporal": float(h_temp) if not np.isnan(h_temp) else 0.5,
        "hurst_spectral_mean": float(np.nanmean(sp_arr)) if sp_arr.size else 0.5,
        "hurst_spectral_std": float(np.nanstd(sp_arr)) if sp_arr.size else 0.0,
        "hurst_temporal_valid": temporal_valid,
        "hurst_fallback": float(fallback),
    }


__all__ = [
    "calculate_hurst",
    "hurst_temporal",
    "hurst_spectral",
    "hurst_regional",
    "hurst_features",
]
