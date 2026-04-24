"""Feature preparation shared by baseline and dynamis stages.

Converts a list of PointSeries into the arrays that downstream training
code consumes: X (padded features), mask (validity), hurst_vec, and labels.

All preparation is deterministic given the same series_list, so it's not
cached to disk — we compute it fresh each stage. It's fast (~a few seconds).
"""
from __future__ import annotations

import re
from datetime import datetime
from dataclasses import dataclass

import numpy as np


CROPS = ['rice', 'corn', 'soybean']


# ---------------------------------------------------------------------------
# Date canonicalisation (v6 FIX #7)
# ---------------------------------------------------------------------------
def canonical_date(s: str) -> str:
    """Parse any of 'YYYY-MM-DD', 'YYYY/MM/DD', 'YYYY/M/D' to 'YYYY-MM-DD'."""
    s = str(s).strip()
    for fmt in ('%Y-%m-%d', '%Y/%m/%d'):
        try:
            return datetime.strptime(s, fmt).strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            continue
    parts = re.split(r'[-/]', s)
    if len(parts) == 3:
        y, m, d = parts
        try:
            return f'{int(y):04d}-{int(m):02d}-{int(d):02d}'
        except ValueError:
            pass
    raise ValueError(f'Unrecognised date format: {s!r}')


# ---------------------------------------------------------------------------
# Main pack
# ---------------------------------------------------------------------------
@dataclass
class PackedFeatures:
    X: np.ndarray                  # (n, T_max, F)
    mask: np.ndarray               # (n, T_max) bool
    hurst_vec: np.ndarray          # (n,)
    hurst_source: np.ndarray       # (n,) int8
    crop_labels: np.ndarray        # (n,) int64
    pheno_labels: np.ndarray       # (n, T_max) int64 (-100 = ignore)
    view: dict                     # consolidated view (for downstream refs)
    series_list: list              # passthrough for slope computation


def _regional_ndvi_series(region_view, lon, lat, extract_bands_at_point, model_bands):
    dates = sorted(region_view.keys())
    vals = []
    for d in dates:
        bands_vec = extract_bands_at_point(region_view[d], lon, lat, list(model_bands))
        nir, red = bands_vec[7], bands_vec[3]
        if np.isnan(nir) or np.isnan(red):
            continue
        vals.append(float((nir - red) / (nir + red + 1e-6)))
    return np.asarray(vals, dtype=np.float64)


def pack(series_list: list) -> PackedFeatures:
    """Turn a list of PointSeries into padded training tensors.

    This does:
      - Padding each series to T_max
      - Computing per-point Hurst via v4 cascade
      - Canonicalising dates for phenophase label lookup
      - Filling NaNs with 0 after a diagnostic print
    """
    from src.data import (
        consolidate_regions, MODEL_BANDS, FEATURE_NAMES, N_FEATURES,
        extract_bands_at_point,
    )
    from src.dynamis import (
        hurst_features, hurst_regional, hurst_dfa, hurst_diff_regional,
        phenophase_name_to_index,
    )
    from . import config

    folders = [str(config.EXTRACTED_DIR / f) for f in config.ZIPS]
    regions_in_series = {ps.region for ps in series_list}
    view = consolidate_regions(folders, regions_filter=regions_in_series)

    T_max = max(ps.features.shape[0] for ps in series_list)
    n_points = len(series_list)
    X = np.full((n_points, T_max, N_FEATURES), np.nan, dtype=np.float32)
    mask = np.zeros((n_points, T_max), dtype=bool)
    hurst_vec = np.full(n_points, 0.5, dtype=np.float32)
    hurst_source = np.zeros(n_points, dtype=np.int8)
    crop_labels = np.zeros(n_points, dtype=np.int64)
    pheno_labels = np.zeros((n_points, T_max), dtype=np.int64)

    for i, ps in enumerate(series_list):
        T = ps.features.shape[0]
        X[i, :T] = ps.features.astype(np.float32)
        mask[i, :T] = ps.mask

        region_view = view.get(ps.region, {})
        ndvi_series = (_regional_ndvi_series(region_view, ps.lon, ps.lat,
                                              extract_bands_at_point, MODEL_BANDS)
                       if len(region_view) >= 8 else np.array([]))

        # Hurst cascade (DFA → diff-regional → R/S-regional → temporal → spectral)
        h, src = float('nan'), 0
        if ndvi_series.size >= 8:
            h_dfa = hurst_dfa(ndvi_series)
            if not np.isnan(h_dfa):
                h, src = h_dfa, 1
        if np.isnan(h) and ndvi_series.size >= 8:
            h_diff = hurst_diff_regional(
                region_view, extract_bands_at_point, ps.lon, ps.lat, min_dates=8)
            if not np.isnan(h_diff):
                h, src = h_diff, 2
        if np.isnan(h) and ndvi_series.size >= 8:
            h_rs = hurst_regional(
                region_view, extract_bands_at_point, ps.lon, ps.lat, min_dates=8)
            if not np.isnan(h_rs):
                h, src = h_rs, 3
        if np.isnan(h):
            ndvi_col = FEATURE_NAMES.index('ndvi')
            hf = hurst_features(ps.features[:, ndvi_col], ps.features[:, :12],
                                min_temporal_dates=8)
            if hf['hurst_temporal_valid']:
                h, src = hf['hurst_temporal'], 4
            elif hf['hurst_spectral_mean'] != 0.5:
                h, src = hf['hurst_spectral_mean'], 5
            else:
                h, src = 0.5, 0
        hurst_vec[i] = h
        hurst_source[i] = src

        crop_labels[i] = CROPS.index(ps.crop_type) if ps.crop_type in CROPS else 0

        pheno_map_canon = {}
        for k, v in (ps.phenophase_by_date or {}).items():
            try:
                pheno_map_canon[canonical_date(k)] = v
            except Exception:
                continue
        for t, date in enumerate(ps.dates):
            try:
                key = canonical_date(date)
            except Exception:
                pheno_labels[i, t] = -100
                continue
            pheno_name = pheno_map_canon.get(key)
            pheno_labels[i, t] = phenophase_name_to_index(pheno_name) if pheno_name else -100

    # Sanity diagnostics
    nan_in_valid = int(np.isnan(X[mask]).sum())
    total_valid = int(mask.sum()) * X.shape[-1]
    print(
        f'[features] NaNs in valid timesteps: {nan_in_valid}/{total_valid} '
        f'= {nan_in_valid / max(total_valid, 1):.2%} (will be zeroed)'
    )
    X = np.nan_to_num(X, nan=0.0)

    n_matched = int((pheno_labels != -100).sum())
    n_total = int(mask.sum())
    match_rate = n_matched / max(n_total, 1)
    print(f'[features] phenophase match rate: {n_matched}/{n_total} = {match_rate:.1%}')
    if match_rate < 0.20:
        raise AssertionError(
            f'Phenophase match rate only {match_rate:.1%} — date canonicaliser '
            f'did not recognise the CSV format. Check series_list[0].dates vs '
            f'series_list[0].phenophase_by_date.keys().'
        )
    if match_rate < 0.50:
        print(f'[features] [warn] match rate {match_rate:.1%} below 50% — weak supervision')

    print(f'[features] X={X.shape}, mask={mask.shape}, hurst={hurst_vec.shape}')
    crop_dist = dict(zip(CROPS, np.bincount(crop_labels, minlength=3).tolist()))
    print(f'[features] crop distribution: {crop_dist}')

    source_names = {1: 'DFA', 2: 'diff-regional', 3: 'R/S-regional',
                    4: 'temporal', 5: 'spectral', 0: 'fallback'}
    for code, name in source_names.items():
        n = int((hurst_source == code).sum())
        if n:
            print(f'[features]   Hurst source {name:15s}: {n}')
    sat_frac = float((hurst_vec >= 0.98).mean())
    print(
        f'[features] Hurst: min={hurst_vec.min():.3f} '
        f'median={np.median(hurst_vec):.3f} max={hurst_vec.max():.3f} '
        f'saturation≥0.98: {sat_frac:.1%}'
    )

    return PackedFeatures(
        X=X, mask=mask, hurst_vec=hurst_vec, hurst_source=hurst_source,
        crop_labels=crop_labels, pheno_labels=pheno_labels,
        view=view, series_list=series_list,
    )


def flatten(X: np.ndarray, mask: np.ndarray, series_list: list | None = None) -> np.ndarray:
    """Per-point statistics per feature: mean, std, max, min, time-aware slope.

    v6 FIX #5: slope is ∂feature/∂day (real calendar time), not per-index.
    """
    n, T, F = X.shape
    out = np.zeros((n, F * 5), dtype=np.float32)
    for i in range(n):
        valid = mask[i]
        if valid.sum() < 1:
            continue
        Xi = X[i, valid]
        out[i, 0*F:1*F] = Xi.mean(axis=0)
        out[i, 1*F:2*F] = Xi.std(axis=0)
        out[i, 2*F:3*F] = Xi.max(axis=0)
        out[i, 3*F:4*F] = Xi.min(axis=0)
        if Xi.shape[0] >= 2:
            # Time-aware slope when we have the series dates
            if series_list is not None:
                try:
                    ps = series_list[i]
                    dates = (list(ps.dates) if len(ps.dates) >= valid.sum()
                             else [ps.dates[t] for t in range(Xi.shape[0]) if valid[t]])
                    d0 = canonical_date(dates[0])
                    d1 = canonical_date(dates[-1])
                    span = (datetime.strptime(d1, '%Y-%m-%d') -
                            datetime.strptime(d0, '%Y-%m-%d')).days
                    if span > 0:
                        out[i, 4*F:5*F] = (Xi[-1] - Xi[0]) / span
                        continue
                except Exception:
                    pass
            out[i, 4*F:5*F] = (Xi[-1] - Xi[0]) / max(Xi.shape[0] - 1, 1)
    return out
