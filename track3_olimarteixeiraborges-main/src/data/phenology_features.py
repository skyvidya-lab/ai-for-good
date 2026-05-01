"""
Phenology-aware per-point features for the LightGBM baseline.

The plain "mean/std/max/min/slope across 17 features" baseline collapses to a
constant predictor because it sees no structure that distinguishes rice, corn
and soybean. The agronomic literature says the discriminative signal lives in
the *shape* of the NDVI/NDWI trajectory across the 7 phenophases (see
`.claude/kb/crop-science/concepts/crop-signatures.md`).

This module derives features anchored to the labelled phenophase dates rather
than raw temporal position, so the same "Peak" comparison is made across all
points regardless of how many total observations they have.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .temporal_builder import FEATURE_NAMES, PointSeries
from .vegetation_indices import INDEX_NAMES

N_BANDS = 12  # MODEL_BANDS count
N_PHENO_FEATURES = 10  # must match the output of `compute_phenology_features`


def _iso_to_slash(date_iso: str) -> str:
    """YYYY-MM-DD → YYYY/M/D (matches labels CSV formatting)."""
    y, m, d = date_iso.split("-")
    return f"{int(y)}/{int(m)}/{int(d)}"


def _day_of_year(date_iso: str) -> int:
    return pd.Timestamp(date_iso).dayofyear


def _feature_idx(name: str) -> int:
    return FEATURE_NAMES.index(name)


@dataclass
class PhenologyFeatures:
    """Named container so downstream code doesn't juggle magic indices."""

    ndvi_peak_value: float
    ndvi_peak_doy: float
    ndvi_greenup_slope: float
    ndvi_senescence_slope: float
    ndwi_max_before_greenup: float
    ndwi_greenup_anomaly: float
    lswi_peak_minus_dormancy: float
    peak_to_senescence_days: float
    evi_peak_value: float
    hurst_ndvi_anomaly: float  # |H - 0.5| — deviation from random walk

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.ndvi_peak_value,
                self.ndvi_peak_doy,
                self.ndvi_greenup_slope,
                self.ndvi_senescence_slope,
                self.ndwi_max_before_greenup,
                self.ndwi_greenup_anomaly,
                self.lswi_peak_minus_dormancy,
                self.peak_to_senescence_days,
                self.evi_peak_value,
                self.hurst_ndvi_anomaly,
            ],
            dtype=np.float32,
        )


PHENO_FEATURE_NAMES: tuple[str, ...] = (
    "ndvi_peak_value",
    "ndvi_peak_doy",
    "ndvi_greenup_slope",
    "ndvi_senescence_slope",
    "ndwi_max_before_greenup",
    "ndwi_greenup_anomaly",
    "lswi_peak_minus_dormancy",
    "peak_to_senescence_days",
    "evi_peak_value",
    "hurst_ndvi_anomaly",
)


def compute_phenology_features(ps: PointSeries, hurst: float = 0.5) -> PhenologyFeatures:
    """
    Derive agronomic shape features from a PointSeries + its Hurst value.

    Uses `ps.phenophase_by_date` (expected format "YYYY/M/D") to anchor the
    features. Falls back to temporal extremes when phenophase labels are
    unavailable (inference time on unlabelled points).
    """
    X = ps.features  # (T, 17) where columns are bands + indices
    dates = ps.dates
    mask = ps.mask
    if X.size == 0 or mask.sum() == 0:
        return _empty_features(hurst)

    valid_t = np.where(mask)[0]
    ndvi = X[:, _feature_idx("ndvi")]
    ndwi = X[:, _feature_idx("ndwi")]
    lswi = X[:, _feature_idx("lswi")]
    evi = X[:, _feature_idx("evi")]

    ndvi_peak_t = int(valid_t[np.nanargmax(ndvi[valid_t])])
    ndvi_peak_value = float(ndvi[ndvi_peak_t])
    ndvi_peak_doy = float(_day_of_year(dates[ndvi_peak_t]))
    evi_peak_value = float(evi[ndvi_peak_t])

    # Phenophase-anchored features (when labels available)
    pheno_map = ps.phenophase_by_date or {}
    pheno_to_t: dict[str, int] = {}
    for t in valid_t:
        key = _iso_to_slash(dates[t])
        name = pheno_map.get(key)
        if name:
            pheno_to_t[name] = int(t)

    t_dormancy = pheno_to_t.get("Dormancy")
    t_greenup = pheno_to_t.get("Greenup")
    t_peak = pheno_to_t.get("Peak", ndvi_peak_t)
    t_senescence = pheno_to_t.get("Senescence")

    # Greenup slope = ΔNDVI / Δdays from Dormancy → Peak (or Greenup → Peak)
    if t_dormancy is not None and t_peak is not None and t_peak > t_dormancy:
        dd = _day_of_year(dates[t_peak]) - _day_of_year(dates[t_dormancy])
        ndvi_greenup_slope = float((ndvi[t_peak] - ndvi[t_dormancy]) / max(dd, 1))
    elif t_greenup is not None and t_peak is not None and t_peak > t_greenup:
        dd = _day_of_year(dates[t_peak]) - _day_of_year(dates[t_greenup])
        ndvi_greenup_slope = float((ndvi[t_peak] - ndvi[t_greenup]) / max(dd, 1))
    else:
        ndvi_greenup_slope = float(np.nanmax(ndvi[valid_t]) - np.nanmin(ndvi[valid_t]))

    # Senescence slope = ΔNDVI / Δdays from Peak → Senescence
    if t_peak is not None and t_senescence is not None and t_senescence > t_peak:
        dd = _day_of_year(dates[t_senescence]) - _day_of_year(dates[t_peak])
        ndvi_senescence_slope = float((ndvi[t_senescence] - ndvi[t_peak]) / max(dd, 1))
        peak_to_senescence_days = float(dd)
    else:
        ndvi_senescence_slope = 0.0
        peak_to_senescence_days = 0.0

    # Rice flooding signature: NDWI spike at Dormancy or Greenup
    flood_ts = [t for k, t in pheno_to_t.items() if k in ("Dormancy", "Greenup")]
    if flood_ts:
        ndwi_max_before_greenup = float(np.nanmax(ndwi[flood_ts]))
    else:
        # Fallback: first third of the season
        n_valid = len(valid_t)
        early = valid_t[: max(1, n_valid // 3)]
        ndwi_max_before_greenup = float(np.nanmax(ndwi[early]))

    # NDWI anomaly: deviation of pre-Greenup NDWI from the seasonal mean
    ndwi_season_mean = float(np.nanmean(ndwi[valid_t]))
    ndwi_greenup_anomaly = ndwi_max_before_greenup - ndwi_season_mean

    # LSWI: peak minus dormancy (water content shift)
    if t_dormancy is not None:
        lswi_peak_minus_dormancy = float(lswi[t_peak] - lswi[t_dormancy])
    else:
        lswi_peak_minus_dormancy = float(lswi[t_peak] - np.nanmin(lswi[valid_t]))

    return PhenologyFeatures(
        ndvi_peak_value=ndvi_peak_value,
        ndvi_peak_doy=ndvi_peak_doy,
        ndvi_greenup_slope=ndvi_greenup_slope,
        ndvi_senescence_slope=ndvi_senescence_slope,
        ndwi_max_before_greenup=ndwi_max_before_greenup,
        ndwi_greenup_anomaly=ndwi_greenup_anomaly,
        lswi_peak_minus_dormancy=lswi_peak_minus_dormancy,
        peak_to_senescence_days=peak_to_senescence_days,
        evi_peak_value=evi_peak_value,
        hurst_ndvi_anomaly=float(abs(hurst - 0.5)),
    )


def _empty_features(hurst: float) -> PhenologyFeatures:
    return PhenologyFeatures(
        ndvi_peak_value=0.0,
        ndvi_peak_doy=0.0,
        ndvi_greenup_slope=0.0,
        ndvi_senescence_slope=0.0,
        ndwi_max_before_greenup=0.0,
        ndwi_greenup_anomaly=0.0,
        lswi_peak_minus_dormancy=0.0,
        peak_to_senescence_days=0.0,
        evi_peak_value=0.0,
        hurst_ndvi_anomaly=float(abs(hurst - 0.5)),
    )


def batch_phenology_features(
    series_list: list[PointSeries],
    hurst_per_point: np.ndarray,
) -> np.ndarray:
    """Return (N, 10) matrix of phenology features for a list of points."""
    out = np.zeros((len(series_list), N_PHENO_FEATURES), dtype=np.float32)
    for i, ps in enumerate(series_list):
        feats = compute_phenology_features(ps, hurst=float(hurst_per_point[i]))
        out[i] = feats.to_array()
    return out


__all__ = [
    "PhenologyFeatures",
    "PHENO_FEATURE_NAMES",
    "N_PHENO_FEATURES",
    "compute_phenology_features",
    "batch_phenology_features",
]
