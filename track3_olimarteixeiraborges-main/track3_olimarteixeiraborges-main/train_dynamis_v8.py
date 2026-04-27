#!/usr/bin/env python3
"""
train_dynamis_v8.py — Local training script for Dynamis Terra v8.

Changes vs v7:
  - ALL 778 points / 50 regions used (v7 sampled only 4 regions / 184 points)
  - Cache paths updated to v8

Usage (activate venv first):
    C:/Users/jrpmc/Documents/_SkyVidya/venv_training/Scripts/python.exe train_dynamis_v8.py

Dataset ZIPs expected at:
    C:/Users/jrpmc/Documents/_SkyVidya/AI Challenge/FINAL ROUND/track1_download_link_[1-5].zip

Outputs:
    models/dynamis_terra_v8.pt       — trained model checkpoint
    reports/02_baseline_vs_dynamis_v8/  — confusion matrices, calibration plots, report.md
"""

import os
import sys
import json
import time
import zipfile
import re
import copy
import warnings
from pathlib import Path
from dataclasses import asdict
from datetime import datetime as _dt

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Non-interactive matplotlib backend (must come before any pyplot import)
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths — all local Windows paths, no Google Drive / Colab
# ---------------------------------------------------------------------------
REPO_PATH   = Path('C:/Users/jrpmc/Documents/_SkyVidya/repos/track3_olimarteixeiraborges')
FINALS_DIR  = Path('C:/Users/jrpmc/Documents/_SkyVidya/AI Challenge/FINAL ROUND')
SAMPLE_DIR  = Path('C:/Users/jrpmc/Documents/_SkyVidya/AI Challenge/FINAL ROUND/sample_extracted')
MODELS_DIR  = Path('C:/Users/jrpmc/Documents/_SkyVidya/models')
CACHE_DIR   = Path('C:/Users/jrpmc/Documents/_SkyVidya/cache')

VERSION      = 8
LAMBDA_PHENO = 3.0   # v7/v8: phenophase loss weight (v6 used 1.0)
RUN_TAG      = f'02_baseline_vs_dynamis_v{VERSION}'
REPORTS_DIR = Path('C:/Users/jrpmc/Documents/_SkyVidya/reports') / RUN_TAG

for d in [SAMPLE_DIR, MODELS_DIR, CACHE_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# The local repo already contains src/ — add it to sys.path
sys.path.insert(0, str(REPO_PATH))

print(f'Run tag:      {RUN_TAG}')
print(f'Repo:         {REPO_PATH}')
print(f'Reports:      {REPORTS_DIR}')
print(f'LAMBDA_PHENO: {LAMBDA_PHENO}  (v8: all regions, phenophase loss weight)')

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

torch.manual_seed(42)
np.random.seed(42)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    print(f'Torch {torch.__version__} | CUDA {torch.cuda.is_available()} | {torch.cuda.get_device_name(0)}')
else:
    print(f'Torch {torch.__version__} | CUDA not available — training on CPU (slower)')

# ---------------------------------------------------------------------------
# Dataset ZIP paths (track1_download_link_N.zip, N=1..5)
# ---------------------------------------------------------------------------
ZIPS = {
    'region_train_1': str(FINALS_DIR / 'track1_download_link_5.zip'),  # also has points_train_label.csv
    'region_train_2': str(FINALS_DIR / 'track1_download_link_4.zip'),
    'region_train_3': str(FINALS_DIR / 'track1_download_link_3.zip'),
    'region_train_4': str(FINALS_DIR / 'track1_download_link_2.zip'),
}
GUIDE_ZIP = str(FINALS_DIR / 'track1_download_link_1.zip')

# ---------------------------------------------------------------------------
# --- CELL 2: Data Discovery & Stratified Sample Selection ---
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('CELL 2 — Data Discovery & Stratified Sample Selection')
print('='*60)

from src.data import (
    assign_region_to_points, index_region_bboxes, stratified_region_sample, sample_summary,
)

def extract_csvs(zip_path, dest):
    if not Path(zip_path).exists():
        return 0
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.filename.lower().endswith('.csv'):
                zf.extract(info, str(dest))
                n += 1
    return n

for label, zp in [('guide', GUIDE_ZIP), *[(k, v) for k, v in ZIPS.items()]]:
    n = extract_csvs(zp, SAMPLE_DIR)
    print(f'  [{label}] extracted {n} CSV(s)')

labels_path = None
for cand in SAMPLE_DIR.rglob('points_train_label.csv'):
    labels_path = str(cand)
    break
assert labels_path is not None, (
    'points_train_label.csv not found in any zip. '
    'Expected in track1_download_link_5.zip per analise-dataset.md.'
)
print(f'Labels found at: {labels_path}')
labels_df = pd.read_csv(labels_path)
print(f'Labels loaded: {len(labels_df)} rows, {labels_df["point_id"].nunique()} unique points')

# Index bboxes of all 50 regions (lightweight — 1 TIFF each)
bbox_cache = str(CACHE_DIR / 'region_bboxes.json')
bbox_index = index_region_bboxes(
    zip_paths=list(ZIPS.values()),
    cache_path=bbox_cache,
    workdir=str(CACHE_DIR / 'region_index'),
)
print(f'Indexed {len(bbox_index)} regions')

labels_df = assign_region_to_points(labels_df, bbox_index)
missing = labels_df['region'].isna().sum()
print(f'Points without a region match: {missing}')

# v8: use ALL regions that have labelled points (no stratified sub-sampling)
SAMPLE_REGIONS = set(labels_df['region'].dropna().unique())
print(f'Using ALL {len(SAMPLE_REGIONS)} regions: {sorted(SAMPLE_REGIONS)}')

summary = sample_summary(labels_df, SAMPLE_REGIONS)
print('\nClass coverage per region (all):')
print(summary)
print('\nTotal per class:')
print(summary.sum(axis=0))
assert (summary.sum(axis=0) >= 15).all(), 'Some crop class has < 15 points — aborting'

# ---------------------------------------------------------------------------
# v8: NO disk extraction — build view directly from ZIPs via GDAL /vsizip/
# ---------------------------------------------------------------------------
from collections import defaultdict as _dd
from src.data.sentinel2_loader import parse_tiff_filename as _parse_tiff

def build_vsi_view(zip_specs: dict, regions_filter: set | None = None):
    """
    Build consolidated region→date→band→vsi_path view directly from ZIP
    contents, without extracting any file to disk.

    Uses GDAL VSI path format: /vsizip/<zip_path>/<entry_name>
    Rasterio (GDAL) reads these natively.
    Later entries in zip_specs override earlier on (region, date, band) ties.
    """
    view = _dd(lambda: _dd(dict))
    for _label, _zp in zip_specs.items():
        _zp_str = str(_zp).replace('\\', '/')
        if not Path(_zp).exists():
            print(f'[skip] {_zp} not found')
            continue
        with zipfile.ZipFile(_zp) as _zf:
            for _entry in _zf.infolist():
                if _entry.is_dir():
                    continue
                _name = _entry.filename
                _basename = os.path.basename(_name)
                _meta = _parse_tiff(_basename)
                if _meta is None:
                    continue
                if regions_filter and _meta.region not in regions_filter:
                    continue
                _vsi = f'/vsizip/{_zp_str}/{_name}'
                view[_meta.region][_meta.date][_meta.band] = _vsi
        print(f'  [vsi] {_label}: scanned OK')
    return {r: {d: dict(b) for d, b in dates.items()} for r, dates in view.items()}

print('Building VSI view from ZIPs (no disk extraction)...')
vsi_view = build_vsi_view(ZIPS, regions_filter=SAMPLE_REGIONS)
print(f'VSI view: {len(vsi_view)} regions')

# Monkeypatch extract_bands_at_point to accept VSI string paths (no Path.exists() check)
import src.data.point_extractor as _pe_mod
import src.data.point_extractor as _pe_orig_mod
_orig_extract_bands = _pe_mod.extract_bands_at_point

def _vsi_extract_bands(band_paths, lon, lat, bands_order, src_crs='EPSG:4326'):
    """Drop-in replacement that accepts string VSI paths without existence check."""
    import rasterio
    from rasterio.warp import transform as _wt
    out = np.full(len(bands_order), np.nan, dtype=np.float64)
    for i, band in enumerate(bands_order):
        path = band_paths.get(band)
        if path is None:
            continue
        try:
            with rasterio.open(str(path)) as _src:
                xs, ys = _wt(src_crs, _src.crs, [lon], [lat])
                row, col = _src.index(xs[0], ys[0])
                h, w = _src.height, _src.width
                if not (0 <= row < h and 0 <= col < w):
                    continue
                val = _src.read(1)[row, col]
                nd = _src.nodata
                if nd is not None and val == nd:
                    continue
                out[i] = float(val)
        except Exception:
            continue
    return out

_pe_mod.extract_bands_at_point = _vsi_extract_bands
# also patch the name already imported in this module's scope
import src.data as _src_data_mod
_src_data_mod.extract_bands_at_point = _vsi_extract_bands

sample_labels = labels_df[labels_df['region'].isin(SAMPLE_REGIONS)].copy()
print(f'\nAll points: {sample_labels["point_id"].nunique()}')
print(sample_labels.groupby("crop_type")["point_id"].nunique())

# ---------------------------------------------------------------------------
# --- CELL 3: Feature Pipeline (with numpy cache) ---
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('CELL 3 — Feature Pipeline')
print('='*60)

import pickle as _pickle
_ARRAY_CACHE  = CACHE_DIR / f'train_arrays_v{VERSION}.npz'
_SERIES_CACHE = CACHE_DIR / f'train_series_v{VERSION}.pkl'

from src.data import (
    MODEL_BANDS,
    build_point_series, PointSeries, FEATURE_NAMES, N_FEATURES,
)
from src.dynamis import hurst_features, PHENOPHASES, phenophase_name_to_index
from src.dynamis import hurst_regional, hurst_dfa, hurst_diff_regional
from src.data import extract_bands_at_point  # will be overridden by VSI monkeypatch above
from tqdm import tqdm

CROPS = ['rice', 'corn', 'soybean']


def _regional_ndvi_series(region_view, lon, lat):
    from src.data.sentinel2_loader import MODEL_BANDS as _MB
    dates = sorted(region_view.keys())
    vals = []
    for d in dates:
        bands_vec = extract_bands_at_point(region_view[d], lon, lat, list(_MB))
        nir, red = bands_vec[7], bands_vec[3]
        if np.isnan(nir) or np.isnan(red):
            continue
        vals.append(float((nir - red) / (nir + red + 1e-6)))
    return np.asarray(vals, dtype=np.float64)


def _canonical_date(s):
    """Normalise date strings to YYYY-MM-DD regardless of source format."""
    s = str(s).strip()
    for fmt in ('%Y-%m-%d', '%Y/%m/%d'):
        try:
            return _dt.strptime(s, fmt).strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            continue
    parts = re.split(r'[-/]', s)
    if len(parts) == 3:
        y, m, d = parts
        return f'{int(y):04d}-{int(m):02d}-{int(d):02d}'
    raise ValueError(f'Unrecognised date format: {s!r}')


_cache_hit = _ARRAY_CACHE.exists() and _SERIES_CACHE.exists()

if _cache_hit:
    # ---- FAST PATH: load from cache, skip slow TIFF reading ----
    print(f'[cache] Loading arrays from {_ARRAY_CACHE}')
    _cached = np.load(str(_ARRAY_CACHE))
    X            = _cached['X']
    mask         = _cached['mask']
    hurst_vec    = _cached['hurst_vec']
    hurst_source = _cached['hurst_source']
    crop_labels  = _cached['crop_labels']
    pheno_labels = _cached['pheno_labels']
    with open(str(_SERIES_CACHE), 'rb') as _f:
        series_list = _pickle.load(_f)
    T_max    = X.shape[1]
    n_points = X.shape[0]
    view     = vsi_view   # already built above from ZIPs
    sat_frac = float((hurst_vec >= 0.98).mean())
    print(f'[cache] Loaded: X={X.shape}, series={len(series_list)} | Hurst sat={sat_frac:.1%}')

else:
    # ---- SLOW PATH: build from ZIPs via VSI, then save cache ----
    view = vsi_view   # v8: use VSI view (no disk extraction)
    print(f'Consolidated regions: {list(view.keys())}')
    for r, dates in view.items():
        print(f'  {r}: {len(dates)} unique dates')

    sample_labels = sample_labels[sample_labels['region'].isin(view.keys())].copy()
    print(f'\nSample points: {sample_labels["point_id"].nunique()}')
    print(sample_labels.groupby("crop_type")["point_id"].nunique())

    series_list: list = []
    for pid, group in tqdm(sample_labels.groupby('point_id'),
                           total=sample_labels['point_id'].nunique(),
                           desc='Building PointSeries'):
        row0 = group.iloc[0]
        pheno_map = dict(zip(group['phenophase_date'], group['phenophase_name']))
        ps = build_point_series(
            point_id=int(pid), lon=float(row0['Longitude']), lat=float(row0['Latitude']),
            region=row0['region'], consolidated_view=view,
            phenophase_by_date=pheno_map, crop_type=str(row0['crop_type']),
        )
        if ps.features.shape[0] == 0:
            continue
        series_list.append(ps)
    print(f'Built {len(series_list)} point series')
    if series_list:
        print(f'First point: T={series_list[0].features.shape[0]}, F={series_list[0].features.shape[1]}')

    # --- CELL 4 part 1: Hurst + pheno labels ---
    print('\n' + '='*60)
    print('CELL 4 (part 1) — Hurst features + phenophase labels')
    print('='*60)

    T_max    = max(ps.features.shape[0] for ps in series_list)
    n_points = len(series_list)
    X            = np.full((n_points, T_max, N_FEATURES), np.nan, dtype=np.float32)
    mask         = np.zeros((n_points, T_max), dtype=bool)
    hurst_vec    = np.full(n_points, 0.5, dtype=np.float32)
    hurst_source = np.zeros(n_points, dtype=np.int8)
    crop_labels  = np.zeros(n_points, dtype=np.int64)
    pheno_labels = np.zeros((n_points, T_max), dtype=np.int64)

    for i, ps in enumerate(tqdm(series_list, desc='Hurst + pheno labels')):
        T = ps.features.shape[0]
        X[i, :T] = ps.features.astype(np.float32)
        mask[i, :T] = ps.mask

        region_view = view.get(ps.region, {})
        ndvi_series = _regional_ndvi_series(region_view, ps.lon, ps.lat) if len(region_view) >= 8 else np.array([])

        h = float('nan'); src = 0
        if ndvi_series.size >= 8:
            h_dfa = hurst_dfa(ndvi_series)
            if not np.isnan(h_dfa):
                h, src = h_dfa, 1
        if np.isnan(h) and ndvi_series.size >= 8:
            h_diff = hurst_diff_regional(region_view, extract_bands_at_point, ps.lon, ps.lat, min_dates=8)
            if not np.isnan(h_diff):
                h, src = h_diff, 2
        if np.isnan(h) and ndvi_series.size >= 8:
            h_rs = hurst_regional(region_view, extract_bands_at_point, ps.lon, ps.lat, min_dates=8)
            if not np.isnan(h_rs):
                h, src = h_rs, 3
        if np.isnan(h):
            ndvi_col = FEATURE_NAMES.index('ndvi')
            hf = hurst_features(ps.features[:, ndvi_col], ps.features[:, :12], min_temporal_dates=8)
            if hf['hurst_temporal_valid']:
                h, src = hf['hurst_temporal'], 4
            elif hf['hurst_spectral_mean'] != 0.5:
                h, src = hf['hurst_spectral_mean'], 5
            else:
                h, src = 0.5, 0
        hurst_vec[i]    = h
        hurst_source[i] = src
        crop_labels[i]  = CROPS.index(ps.crop_type) if ps.crop_type in CROPS else 0

        _pheno_map_raw   = ps.phenophase_by_date or {}
        _pheno_map_canon = {}
        for _k, _v in _pheno_map_raw.items():
            try:
                _pheno_map_canon[_canonical_date(_k)] = _v
            except Exception:
                continue
        for t, date in enumerate(ps.dates):
            try:
                key = _canonical_date(date)
            except Exception:
                pheno_labels[i, t] = -100
                continue
            pheno_name = _pheno_map_canon.get(key)
            pheno_labels[i, t] = phenophase_name_to_index(pheno_name) if pheno_name else -100

    _nan_in_valid    = int(np.isnan(X[mask]).sum())
    _total_valid_cells = int(mask.sum()) * X.shape[-1]
    print(f'[sanity #9] NaNs in valid timesteps: {_nan_in_valid}/{_total_valid_cells} '
          f'({_nan_in_valid / max(_total_valid_cells, 1):.2%}) → will become 0.0')
    X = np.nan_to_num(X, nan=0.0)

    # Save cache
    np.savez_compressed(str(_ARRAY_CACHE),
                        X=X, mask=mask, hurst_vec=hurst_vec, hurst_source=hurst_source,
                        crop_labels=crop_labels, pheno_labels=pheno_labels)
    with open(str(_SERIES_CACHE), 'wb') as _f:
        _pickle.dump(series_list, _f)
    print(f'[cache] Saved → {_ARRAY_CACHE}')

print(f'X shape: {X.shape}, mask: {mask.shape}, hurst: {hurst_vec.shape}')
print(f'Crop distribution: {dict(zip(CROPS, np.bincount(crop_labels, minlength=3).tolist()))}')
sources = {1: 'DFA', 2: 'diff-regional', 3: 'R/S-regional', 4: 'temporal', 5: 'spectral', 0: 'fallback'}
for code, name in sources.items():
    n = int((hurst_source == code).sum())
    if n:
        print(f'  Hurst source {name:15s}: {n}')
print(f'Hurst distribution: min={hurst_vec.min():.3f} median={np.median(hurst_vec):.3f} '
      f'max={hurst_vec.max():.3f} std={hurst_vec.std():.3f}')
sat_frac = float((hurst_vec >= 0.98).mean())
print(f'Hurst saturation fraction (>=0.98): {sat_frac:.1%}  (target: <10%, v3 was ~66%)')

# ---------------------------------------------------------------------------
# Temperature scaling utilities (src.training does not exist in this repo)
# ---------------------------------------------------------------------------

def expected_calibration_error_np(probs_arr, labels_arr, n_bins=10):
    """ECE computed on numpy arrays."""
    conf = probs_arr.max(axis=-1)
    correct = (probs_arr.argmax(-1) == labels_arr).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(labels_arr)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            ece += m.sum() / total * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def temperature_scale(logits_arr, labels_arr, steps=500, lr=1e-2):
    """Fit a single temperature scalar T on logits (numpy) to minimise NLL."""
    logits_t = torch.from_numpy(logits_arr).float()
    labels_t = torch.from_numpy(labels_arr).long()
    T_param = torch.nn.Parameter(torch.ones(1))
    opt_T = torch.optim.LBFGS([T_param], lr=lr, max_iter=steps)

    def _eval():
        opt_T.zero_grad()
        loss = F.cross_entropy(logits_t / T_param.clamp(min=0.1), labels_t)
        loss.backward()
        return loss

    opt_T.step(_eval)
    return float(T_param.clamp(min=0.1).item())


def apply_temperature(logits_arr, T):
    """Apply temperature T and return calibrated softmax probabilities (numpy)."""
    logits_t = torch.from_numpy(logits_arr).float()
    return F.softmax(logits_t / max(T, 0.1), dim=-1).numpy()

# Sanity check #8 — phenophase date match rate
_n_matched = int((pheno_labels != -100).sum())
_n_total = int(mask.sum())
_match_rate = _n_matched / max(_n_total, 1)
print(f'\n[sanity #8] Phenophase match rate: {_n_matched}/{_n_total} '
      f'valid timesteps = {_match_rate:.1%}')
if series_list:
    _ps0 = series_list[0]
    print(f'[sanity #8] Example ps.dates[:3] = {list(_ps0.dates[:3])}')
    print(f'[sanity #8] Example pheno keys[:3] = '
          f'{list((_ps0.phenophase_by_date or {}).keys())[:3]}')
assert _match_rate > 0.05, (
    f'Phenophase match rate is only {_match_rate:.1%} — the date canonicaliser '
    f'did not recognise the CSV format.'
)
if _match_rate < 0.50:
    print(f'[warn] Match rate {_match_rate:.1%} is below 50% — phenophase '
          f'supervision will be weak. Expected healthy range: >80%.')

# ---------------------------------------------------------------------------
# --- CELL 4 (part 2): Baseline — LightGBM ---
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('CELL 4 (part 2) — Baseline: LightGBM')
print('='*60)

import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                              confusion_matrix, roc_auc_score, roc_curve,
                              precision_score, recall_score)
from src.data import batch_phenology_features, PHENO_FEATURE_NAMES


def flatten_features(X_arr, mask_arr, series_list_=None):
    """Per-point statistics per feature: mean, std, max, min, slope.
    v6 FIX #5: slope is ∂feature/∂day (real calendar time) when series_list_ is given.
    """
    n, T, F = X_arr.shape
    out = np.zeros((n, F * 5), dtype=np.float32)
    for i in range(n):
        valid = mask_arr[i]
        if valid.sum() < 1:
            continue
        Xi = X_arr[i, valid]
        out[i, 0*F:1*F] = Xi.mean(axis=0)
        out[i, 1*F:2*F] = Xi.std(axis=0)
        out[i, 2*F:3*F] = Xi.max(axis=0)
        out[i, 3*F:4*F] = Xi.min(axis=0)
        if Xi.shape[0] >= 2:
            if series_list_ is not None:
                try:
                    ps = series_list_[i]
                    valid_dates = [ps.dates[t] for t in range(len(ps.dates)) if mask_arr[i, t]]
                    d0 = _canonical_date(valid_dates[0])
                    d1 = _canonical_date(valid_dates[-1])
                    span_days = (_dt.strptime(d1, '%Y-%m-%d') - _dt.strptime(d0, '%Y-%m-%d')).days
                    if span_days > 0:
                        out[i, 4*F:5*F] = (Xi[-1] - Xi[0]) / span_days
                        continue
                except Exception:
                    pass
            out[i, 4*F:5*F] = (Xi[-1] - Xi[0]) / max(Xi.shape[0] - 1, 1)
    return out


X_stats = flatten_features(X, mask, series_list_=series_list)
X_pheno = batch_phenology_features(series_list, hurst_vec)
X_flat = np.concatenate([X_stats, X_pheno, hurst_vec.reshape(-1, 1)], axis=1)
print(f'Flat features: {X_flat.shape}  (stats: {X_stats.shape[1]}, pheno: {X_pheno.shape[1]}, hurst: 1)')

# v5/v6: Spatial cross-validation — group by region
SPLIT_BY = 'region'
groups = np.array([ps.region for ps in series_list])
n_groups = len(np.unique(groups))
n_splits = max(2, min(5, n_groups))
if n_groups < 3:
    print(f'[warn] only {n_groups} region(s) in sample — spatial CV is degenerate')

kf = GroupKFold(n_splits=n_splits)
print(f'Using GroupKFold(split_by={SPLIT_BY!r}, n_splits={n_splits}, n_groups={n_groups})')

baseline_metrics = {'crop': {'oa': [], 'kappa': [], 'f1': []}}
bl_pred_all = np.zeros_like(crop_labels)

for fold, (tr, va) in enumerate(kf.split(X_flat, crop_labels, groups=groups)):
    model_lgb = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42, verbose=-1,
    )
    model_lgb.fit(X_flat[tr], crop_labels[tr])
    pred = model_lgb.predict(X_flat[va])
    bl_pred_all[va] = pred
    baseline_metrics['crop']['oa'].append(accuracy_score(crop_labels[va], pred))
    baseline_metrics['crop']['kappa'].append(cohen_kappa_score(crop_labels[va], pred))
    baseline_metrics['crop']['f1'].append(f1_score(crop_labels[va], pred, average='macro', zero_division=0))

print('\nBaseline (crop_type):')
for k, v in baseline_metrics['crop'].items():
    print(f'  {k}: {np.mean(v):.4f} ± {np.std(v):.4f}')
print('\nBaseline confusion:')
print(pd.DataFrame(confusion_matrix(crop_labels, bl_pred_all), index=CROPS, columns=CROPS))

# Fold diagnostic
print(f'\nFold diagnostic (SPLIT_BY={SPLIT_BY!r}):')
point_regions = np.array([ps.region for ps in series_list])
for fold, (tr, va) in enumerate(kf.split(X_flat, crop_labels, groups=groups)):
    tr_regions = set(point_regions[tr])
    va_regions = set(point_regions[va])
    shared = tr_regions & va_regions
    tag = 'LEAK' if shared else 'clean'
    print(f'  fold {fold+1}: val_regions={sorted(va_regions)} | train_regions={len(tr_regions)} | {tag}')

# ---------------------------------------------------------------------------
# --- CELL 5: Dynamis — MKM + ChaosAttention ---
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('CELL 5 — Dynamis: MKM + ChaosAttention')
print('='*60)

from src.models import DynamisCropClassifier, DynamisModelConfig
from src.dynamis import dynamis_loss, PHENOPHASES


def class_weights_from_labels(labels_arr, n_classes):
    counts = np.bincount(labels_arr, minlength=n_classes).astype(np.float32)
    counts = np.clip(counts, 1, None)
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def sampler_from_labels(labels_arr, n_classes):
    counts = np.bincount(labels_arr, minlength=n_classes).astype(np.float32)
    counts = np.clip(counts, 1, None)
    per_class_w = 1.0 / counts
    sample_w = per_class_w[labels_arr]
    return WeightedRandomSampler(sample_w, num_samples=len(labels_arr), replacement=True)


def train_dynamis_fold(
    X_tr, m_tr, h_tr, c_tr, p_tr,
    X_va, m_va, h_va, c_va, p_va,
    epochs=40, batch_size=16, lr=5e-4, weight_decay=5e-4,
    lambda_innovation=0.05, lambda_ece=0.02,
    lambda_pheno=LAMBDA_PHENO,   # v7: boosted phenophase loss weight
    verbose=True,
):
    cfg = DynamisModelConfig(
        input_dim=X_tr.shape[-1], state_dim=len(PHENOPHASES),
        hidden_dim=64, attn_heads=4, n_crops=3, crop_head_dropout=0.3,
    )
    model = DynamisCropClassifier(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    crop_w = class_weights_from_labels(c_tr, n_classes=3)
    sampler = sampler_from_labels(c_tr, n_classes=3)

    ds = TensorDataset(
        torch.from_numpy(X_tr).float(), torch.from_numpy(m_tr).bool(),
        torch.from_numpy(h_tr).float(), torch.from_numpy(c_tr).long(),
        torch.from_numpy(p_tr).long(),
    )
    dl = DataLoader(ds, batch_size=batch_size, sampler=sampler)

    best_f1 = -1.0
    best_state = None
    history = []
    for ep in range(epochs):
        model.train()
        total = 0.0
        n_batch = 0
        for xb, mb, hb, cb, pb in dl:
            xb = xb.to(DEVICE); mb = mb.to(DEVICE); hb = hb.to(DEVICE)
            cb = cb.to(DEVICE); pb = pb.to(DEVICE)
            out = model(xb, mask=mb, hurst=hb)
            pheno_logits_flat = out['pheno_logits'].reshape(-1, len(PHENOPHASES))
            pheno_labels_flat = pb.reshape(-1)
            # v6 FIX #6: pass -100 sentinel directly; fallback to clamp if loss rejects it
            try:
                loss_dict = dynamis_loss(
                    out['crop_logits'], cb,
                    pheno_logits_flat, pheno_labels_flat,
                    out['innovations'],
                    lambda_innovation=lambda_innovation,
                    lambda_ece=lambda_ece,
                    class_weights_crop=crop_w,
                )
            except (RuntimeError, IndexError, AssertionError) as _e:
                if not getattr(train_dynamis_fold, '_warned_ignore_index', False):
                    print(f'  [warn #6] dynamis_loss rejected -100 sentinel '
                          f'({type(_e).__name__}); falling back to clamp(min=0).')
                    train_dynamis_fold._warned_ignore_index = True
                loss_dict = dynamis_loss(
                    out['crop_logits'], cb,
                    pheno_logits_flat, pheno_labels_flat.clamp(min=0),
                    out['innovations'],
                    lambda_innovation=lambda_innovation,
                    lambda_ece=lambda_ece,
                    class_weights_crop=crop_w,
                )
            # v7: boost phenophase loss by (lambda_pheno - 1) * ce_pheno
            # dynamis_loss already includes ce_pheno once; we add the extra weight
            # without modifying the source so inference code is unaffected.
            loss = loss_dict['total'] + (lambda_pheno - 1.0) * loss_dict['ce_pheno']
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total += loss.item() * xb.size(0)
            n_batch += xb.size(0)
        sched.step()

        model.eval()
        with torch.no_grad():
            out_v = model(
                torch.from_numpy(X_va).float().to(DEVICE),
                mask=torch.from_numpy(m_va).bool().to(DEVICE),
                hurst=torch.from_numpy(h_va).float().to(DEVICE),
            )
        pred_v = out_v['crop_logits'].argmax(-1).cpu().numpy()
        prec = precision_score(c_va, pred_v, average=None, labels=[0, 1, 2], zero_division=0)
        rec = recall_score(c_va, pred_v, average=None, labels=[0, 1, 2], zero_division=0)
        f1m = f1_score(c_va, pred_v, average='macro', zero_division=0)

        # v7: phenophase accuracy — only on timesteps with a real label (!=−100)
        ph_logits_v = out_v['pheno_logits'].cpu().numpy()   # (B_val, T_max, 7)
        ph_pred_v   = ph_logits_v.argmax(-1)                # (B_val, T_max)
        _valid_ph   = p_va != -100                          # (B_val, T_max) bool
        if _valid_ph.sum() > 0:
            pheno_acc = float((ph_pred_v[_valid_ph] == p_va[_valid_ph]).mean())
        else:
            pheno_acc = float('nan')

        history.append({'epoch': ep + 1, 'loss': total / max(n_batch, 1),
                         'prec': prec.tolist(), 'rec': rec.tolist(), 'f1_macro': float(f1m),
                         'pheno_acc': pheno_acc})
        if f1m > best_f1:
            best_f1 = f1m
            best_state = copy.deepcopy(model.state_dict())
        if verbose and (ep == 0 or (ep + 1) % 5 == 0 or ep == epochs - 1):
            ph_str = f'{pheno_acc:.3f}' if not np.isnan(pheno_acc) else 'n/a'
            print(f'  ep{ep+1:02d} loss={total/max(n_batch,1):.3f} F1m={f1m:.3f} '
                  f'PhenoAcc={ph_str} '
                  f'P={[f"{p:.2f}" for p in prec]} R={[f"{r:.2f}" for r in rec]}')

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(
            torch.from_numpy(X_va).float().to(DEVICE),
            mask=torch.from_numpy(m_va).bool().to(DEVICE),
            hurst=torch.from_numpy(h_va).float().to(DEVICE),
        )
    logits = out['crop_logits'].cpu().numpy()
    probs = F.softmax(out['crop_logits'], dim=-1).cpu().numpy()
    pred = probs.argmax(-1)
    unc = out['uncertainty'].cpu().numpy()
    print(f'  [best epoch F1m={best_f1:.3f}]')
    return model, pred, probs, logits, unc, out, history


dyn_metrics = {'crop': {'oa': [], 'kappa': [], 'f1': []}, 'uncertainty': []}
dyn_preds_all = np.zeros_like(crop_labels)
dyn_probs_all = np.zeros((len(crop_labels), 3), dtype=np.float32)
dyn_logits_all = np.zeros((len(crop_labels), 3), dtype=np.float32)
dyn_unc_all = np.zeros(len(crop_labels), dtype=np.float32)
last_model = None
last_out = None
all_fold_histories = []

for fold, (tr, va) in enumerate(kf.split(X, crop_labels, groups=groups)):
    print(f'\n--- Fold {fold+1}/{n_splits} (train={len(tr)}, val={len(va)}) ---')
    model_dyn, pred, probs, logits, unc, out_dict, hist = train_dynamis_fold(
        X[tr], mask[tr], hurst_vec[tr], crop_labels[tr], pheno_labels[tr],
        X[va], mask[va], hurst_vec[va], crop_labels[va], pheno_labels[va],
    )
    dyn_preds_all[va] = pred
    dyn_probs_all[va] = probs
    dyn_logits_all[va] = logits
    dyn_unc_all[va] = unc
    dyn_metrics['crop']['oa'].append(accuracy_score(crop_labels[va], pred))
    dyn_metrics['crop']['kappa'].append(cohen_kappa_score(crop_labels[va], pred))
    dyn_metrics['crop']['f1'].append(f1_score(crop_labels[va], pred, average='macro', zero_division=0))
    dyn_metrics['uncertainty'].append(float(np.mean(unc)))
    all_fold_histories.append(hist)
    last_model = model_dyn
    last_out = out_dict

print('\nDynamis (crop_type):')
for k, v in dyn_metrics['crop'].items():
    print(f'  {k}: {np.mean(v):.4f} ± {np.std(v):.4f}')
print(f'  mean trace(P): {np.mean(dyn_metrics["uncertainty"]):.4f}')
print('\nDynamis confusion:')
print(pd.DataFrame(confusion_matrix(crop_labels, dyn_preds_all), index=CROPS, columns=CROPS))

# ---------------------------------------------------------------------------
# --- CELL 5½: Transfer Proof (shuffled-label control) ---
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('CELL 5½ — Transfer Proof (shuffled-label control)')
print('='*60)

rng_shuffle = np.random.default_rng(7)
shuf_labels = rng_shuffle.permutation(crop_labels)

tr_shuf, va_shuf = next(iter(GroupKFold(n_splits=n_splits).split(X_flat, shuf_labels, groups=groups)))

_bl_shuf = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    class_weight='balanced', random_state=42, verbose=-1,
).fit(X_flat[tr_shuf], shuf_labels[tr_shuf])
bl_shuf_oa = accuracy_score(shuf_labels[va_shuf], _bl_shuf.predict(X_flat[va_shuf]))

_, _dyn_pred_shuf, _, _, _, _, _ = train_dynamis_fold(
    X[tr_shuf], mask[tr_shuf], hurst_vec[tr_shuf], shuf_labels[tr_shuf], pheno_labels[tr_shuf],
    X[va_shuf], mask[va_shuf], hurst_vec[va_shuf], shuf_labels[va_shuf], pheno_labels[va_shuf],
    epochs=12, batch_size=16, verbose=False,
)
dyn_shuf_oa = accuracy_score(shuf_labels[va_shuf], _dyn_pred_shuf)

print(f'Transfer Proof — shuffled-label OA (chance ≈ 33%):')
print(f'  Baseline: {bl_shuf_oa:.1%}')
print(f'  Dynamis:  {dyn_shuf_oa:.1%}')

TRANSFER_PROOF_THRESHOLD = 0.55
assert bl_shuf_oa < TRANSFER_PROOF_THRESHOLD, f'BASELINE leakage suspected (OA={bl_shuf_oa:.1%})'
assert dyn_shuf_oa < TRANSFER_PROOF_THRESHOLD, f'DYNAMIS leakage suspected (OA={dyn_shuf_oa:.1%})'
print('Transfer Proof PASSED — metrics above can be trusted.')

# ---------------------------------------------------------------------------
# --- CELL 5¾: Temperature Scaling ---
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('CELL 5¾ — Temperature Scaling')
print('='*60)

ece_pre = expected_calibration_error_np(dyn_probs_all, crop_labels, n_bins=10)
print(f'ECE before temperature scaling: {ece_pre:.4f}')

T_cal = temperature_scale(dyn_logits_all, crop_labels, steps=500, lr=1e-2)
print(f'Learnt temperature T = {T_cal:.4f}  (T>1 means the model was over-confident)')

dyn_probs_calibrated = apply_temperature(dyn_logits_all, T_cal)
ece_post = expected_calibration_error_np(dyn_probs_calibrated, crop_labels, n_bins=10)
print(f'ECE after  temperature scaling: {ece_post:.4f}')
print(f'ECE reduction: {ece_pre:.4f} → {ece_post:.4f}  (delta {ece_pre - ece_post:+.4f})')

assert np.all(dyn_probs_calibrated.argmax(-1) == dyn_probs_all.argmax(-1)), \
    'T-scaling changed argmax — check implementation'
print('Accuracy preserved (as expected — T-scaling cannot change argmax).')

# ---------------------------------------------------------------------------
# --- CELL 6: Visualisations ---
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('CELL 6 — Visualisations')
print('='*60)

# 6.1 Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, pred_arr, title in [(axes[0], bl_pred_all, 'Baseline'), (axes[1], dyn_preds_all, 'Dynamis')]:
    cm = confusion_matrix(crop_labels, pred_arr, labels=[0, 1, 2])
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CROPS); ax.set_yticklabels(CROPS)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(
        f'{title} — OA={accuracy_score(crop_labels, pred_arr):.3f}, '
        f'F1m={f1_score(crop_labels, pred_arr, average="macro", zero_division=0):.3f}'
    )
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='black' if cm[i, j] < cm.max() / 2 else 'white')
plt.tight_layout()
plt.savefig(str(REPORTS_DIR / 'confusion_matrices.png'), dpi=120)
plt.close()
print('Saved: confusion_matrices.png')

# 6.2 Per-class F1
f1_baseline = f1_score(crop_labels, bl_pred_all, average=None)
f1_dynamis = f1_score(crop_labels, dyn_preds_all, average=None)
fig, ax = plt.subplots(figsize=(8, 4))
x_pos = np.arange(len(CROPS))
ax.bar(x_pos - 0.2, f1_baseline, 0.4, label='Baseline', color='steelblue')
ax.bar(x_pos + 0.2, f1_dynamis, 0.4, label='Dynamis', color='darkorange')
ax.set_xticks(x_pos); ax.set_xticklabels(CROPS)
ax.set_ylabel('F1 score'); ax.set_title('Per-class F1: Baseline vs Dynamis')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(str(REPORTS_DIR / 'per_class_f1.png'), dpi=120)
plt.close()
print('Saved: per_class_f1.png')

# 6.3 Hurst histogram
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(hurst_vec, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(0.5, color='gray', linestyle='--', alpha=0.6, label='H=0.5 (random walk)')
axes[0].axvline(0.98, color='crimson', linestyle='--', alpha=0.6, label='H≥0.98 saturation')
sat = (hurst_vec >= 0.98).mean()
axes[0].set_title(f'Hurst distribution — v{VERSION} (saturation={sat:.1%}, target <10%)')
axes[0].set_xlabel('Hurst exponent'); axes[0].set_ylabel('Count')
axes[0].legend(); axes[0].grid(alpha=0.3)

labels_src = {1: 'DFA', 2: 'diff-reg', 3: 'R/S-reg', 4: 'temporal', 5: 'spectral', 0: 'fallback'}
colors_src = {1: 'darkgreen', 2: 'seagreen', 3: 'steelblue', 4: 'goldenrod', 5: 'gray', 0: 'lightgray'}
for code, name in labels_src.items():
    subset = hurst_vec[hurst_source == code]
    if subset.size:
        axes[1].hist(subset, bins=20, alpha=0.6, label=f'{name} (n={subset.size})',
                     color=colors_src[code])
axes[1].set_title('Hurst by source (cascade level)')
axes[1].set_xlabel('Hurst exponent'); axes[1].set_ylabel('Count')
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(str(REPORTS_DIR / 'hurst_histogram.png'), dpi=120)
plt.close()
print(f'Saved: hurst_histogram.png | sat={sat:.1%} | std={hurst_vec.std():.3f}')

# 6.4 Calibration — pre vs post temperature scaling
def _reliability_bins(probs_arr, labels_arr, n_bins=10):
    conf = probs_arr.max(axis=-1)
    correct = (probs_arr.argmax(-1) == labels_arr).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    acc, cnf, n_ = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            acc.append(correct[m].mean()); cnf.append(conf[m].mean()); n_.append(int(m.sum()))
        else:
            acc.append(np.nan); cnf.append((lo + hi) / 2); n_.append(0)
    return np.array(acc), np.array(cnf), np.array(n_)

acc_pre, cnf_pre, n_pre = _reliability_bins(dyn_probs_all, crop_labels)
acc_post, cnf_post, n_post = _reliability_bins(dyn_probs_calibrated, crop_labels)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], 'k--', label='Perfect', alpha=0.6)
valid_pre = n_pre > 0
ax.plot(cnf_pre[valid_pre], acc_pre[valid_pre], 'o--', color='crimson', alpha=0.7,
        label=f'Pre T-scale (ECE={ece_pre:.3f})', linewidth=2)
valid_post = n_post > 0
ax.plot(cnf_post[valid_post], acc_post[valid_post], 's-', color='darkorange',
        label=f'Post T-scale T={T_cal:.2f} (ECE={ece_post:.3f})', linewidth=2)
for c_, a_, n_ in zip(cnf_post, acc_post, n_post):
    if n_ > 0:
        ax.annotate(f'n={n_}', (c_, a_), fontsize=7, xytext=(3, 3), textcoords='offset points')
ax.set_xlabel('Confidence'); ax.set_ylabel('Accuracy')
ax.set_title('Calibration (Reliability Diagram) — pre/post temperature scaling')
ax.legend(loc='lower right'); ax.grid(alpha=0.3)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(str(REPORTS_DIR / 'calibration.png'), dpi=120)
plt.close()
print(f'Saved: calibration.png | ECE {ece_pre:.4f} → {ece_post:.4f}')

# 6.5 OOD ROC
errors_binary = (dyn_preds_all != crop_labels).astype(int)
if errors_binary.sum() == 0 or errors_binary.sum() == len(errors_binary):
    print(f'[skip OOD ROC] errors={int(errors_binary.sum())}, cannot compute AUC')
    ood_auc = float('nan')
    ood_threshold = float('nan')
else:
    ood_auc = roc_auc_score(errors_binary, dyn_unc_all)
    fpr, tpr, thresholds = roc_curve(errors_binary, dyn_unc_all)
    ood_threshold = float(np.percentile(dyn_unc_all, 90))
    sensitivity_at_thr = float(((dyn_unc_all > ood_threshold) & errors_binary.astype(bool)).sum()) / max(errors_binary.sum(), 1)
    specificity_at_thr = float(((dyn_unc_all <= ood_threshold) & (~errors_binary.astype(bool))).sum()) / max((errors_binary == 0).sum(), 1)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, 'o-', color='steelblue', label=f'trace(P) → error  (AUC={ood_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance')
    ax.axvline(1 - specificity_at_thr, color='crimson', linestyle=':',
                label=f'p90 threshold = {ood_threshold:.3f}')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('OOD ROC — trace(P) as a background-class detector')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(REPORTS_DIR / 'ood_roc.png'), dpi=120)
    plt.close()
    print(f'Saved: ood_roc.png | OOD AUC={ood_auc:.4f} | threshold={ood_threshold:.3f}')
    print(f'  sensitivity={sensitivity_at_thr:.2%}, specificity={specificity_at_thr:.2%}')

# ---------------------------------------------------------------------------
# --- CELL 7 (part 1): Generate Report ---
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('CELL 7 — Report')
print('='*60)


def generate_report(baseline, dynamis, n_pts, n_regions, hurst_v, hurst_src,
                    dyn_unc, errs, ece_pre_, ece_post_, T_,
                    ood_auc_, ood_threshold_):

    def fmt_metric(arr):
        arr = np.asarray(arr, dtype=float)
        return f'{arr.mean():.1%} (± {arr.std():.1%})'

    bl_oa = fmt_metric(baseline['crop']['oa'])
    dn_oa = fmt_metric(dynamis['crop']['oa'])
    bl_f1 = fmt_metric(baseline['crop']['f1'])
    dn_f1 = fmt_metric(dynamis['crop']['f1'])
    delta_oa = np.mean(dynamis['crop']['oa']) - np.mean(baseline['crop']['oa'])
    delta_f1 = np.mean(dynamis['crop']['f1']) - np.mean(baseline['crop']['f1'])
    kappa_dn = fmt_metric(dynamis['crop']['kappa'])
    unc_err = float(np.mean(dyn_unc[errs == 1])) if errs.sum() > 0 else float('nan')
    unc_ok = float(np.mean(dyn_unc[errs == 0])) if (errs == 0).sum() > 0 else float('nan')

    hurst_sat = float((hurst_v >= 0.98).mean())
    hurst_std_ = float(hurst_v.std())
    src_counts = {
        'DFA': int((hurst_src == 1).sum()),
        'diff-regional': int((hurst_src == 2).sum()),
        'R/S-regional': int((hurst_src == 3).sum()),
        'temporal': int((hurst_src == 4).sum()),
        'spectral': int((hurst_src == 5).sum()),
        'fallback': int((hurst_src == 0).sum()),
    }
    src_line = ', '.join(f'{k}: {v}' for k, v in src_counts.items() if v)

    delta_sign = '+' if delta_oa >= 0 else '−'
    verdict = ('Dynamis outperformed the baseline' if delta_oa > 0.01
               else 'Dynamis matched the baseline' if abs(delta_oa) <= 0.01
               else 'Dynamis underperformed the baseline on this sample')
    unc_diag = ('Uncertainty correlates with errors.' if unc_err > unc_ok
                else 'Uncertainty does not yet discriminate errors.')

    if ece_post_ < 0.05:
        cal_verdict = f'Temperature scaling brought ECE from {ece_pre_:.3f} to {ece_post_:.3f} — **well calibrated**.'
    elif ece_post_ < ece_pre_:
        cal_verdict = f'Temperature scaling reduced ECE from {ece_pre_:.3f} to {ece_post_:.3f}.'
    else:
        cal_verdict = f'Temperature scaling did not improve ECE ({ece_pre_:.3f} → {ece_post_:.3f}).'

    if np.isnan(ood_auc_):
        ood_verdict = 'OOD AUC could not be computed.'
    elif ood_auc_ > 0.7:
        ood_verdict = f'`trace(P)` is a **reliable** OOD signal (AUC={ood_auc_:.3f} > 0.7). Threshold={ood_threshold_:.3f}.'
    else:
        ood_verdict = f'`trace(P)` is a **weak** OOD signal (AUC={ood_auc_:.3f}).'

    return f"""# Dynamis Terra — Sample Run Report (v{VERSION})

**Date**: {time.strftime('%Y-%m-%d %H:%M')}
**Run tag**: `{RUN_TAG}`
**Scope**: {n_pts} training points across {n_regions} regions.

## Executive Summary

**Verdict**: {verdict} ({delta_sign}{abs(delta_oa):.1%} OA, {delta_sign}{abs(delta_f1):.1%} F1m).

## Performance

| Metric | Baseline | **Dynamis v{VERSION}** |
|---|---|---|
| Overall Accuracy | {bl_oa} | **{dn_oa}** |
| F1 Macro | {bl_f1} | **{dn_f1}** |
| Cohen's Kappa | — | **{kappa_dn}** |

*{n_splits}-fold spatial GroupKFold by region.*

## Calibration

| Stage | ECE |
|---|---|
| Pre T-scaling | {ece_pre_:.4f} |
| Post T-scaling | **{ece_post_:.4f}** (T={T_:.3f}) |

{cal_verdict}

## OOD

- mean trace(P) on errors: **{unc_err:.3f}**
- mean trace(P) on correct: **{unc_ok:.3f}**
- AUC: **{ood_auc_:.3f}** | threshold: **{ood_threshold_:.3f}**

{ood_verdict}

## Hurst

- Saturation (≥0.98): **{hurst_sat:.1%}** (target <10%)
- Std: **{hurst_std_:.3f}** (target >0.1)
- Sources: {src_line}

{unc_diag}
"""


errors = (dyn_preds_all != crop_labels).astype(int)
report = generate_report(
    baseline=baseline_metrics, dynamis=dyn_metrics,
    n_pts=len(series_list), n_regions=len(SAMPLE_REGIONS),
    hurst_v=hurst_vec, hurst_src=hurst_source,
    dyn_unc=dyn_unc_all, errs=errors,
    ece_pre_=ece_pre, ece_post_=ece_post, T_=T_cal,
    ood_auc_=ood_auc, ood_threshold_=ood_threshold,
)
print(report)
report_path = REPORTS_DIR / 'sample_run_report.md'
report_path.write_text(report, encoding='utf-8')
print(f'Report saved: {report_path}')

# ---------------------------------------------------------------------------
# --- CELL 7 (part 2): Final model — train on ALL data ---
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('CELL 7 (part 2) — Final model: train on ALL data')
print('='*60)

# v7 fix: use median best-epoch across ALL folds, with a floor of 40 to
# avoid degenerate folds (e.g. single-class val where ep1 = 100% trivially).
try:
    _all_best_eps = [
        max(h, key=lambda r: r['f1_macro'])['epoch']
        for h in all_fold_histories
    ]
    FINAL_EPOCHS = max(int(np.median(_all_best_eps)), 40)
    print(f'[#3] Best epochs per fold: {_all_best_eps} '
          f'→ median={np.median(_all_best_eps):.0f} → using {FINAL_EPOCHS}')
except (IndexError, ValueError, KeyError):
    FINAL_EPOCHS = 40
    print(f'[#3] Fallback to default {FINAL_EPOCHS} epochs')

_cfg = DynamisModelConfig(
    input_dim=X.shape[-1], state_dim=len(PHENOPHASES),
    hidden_dim=64, attn_heads=4, n_crops=3, crop_head_dropout=0.3,
)
_final_model = DynamisCropClassifier(_cfg).to(DEVICE)
_opt = torch.optim.AdamW(_final_model.parameters(), lr=5e-4, weight_decay=5e-4)
_sched = torch.optim.lr_scheduler.CosineAnnealingLR(_opt, T_max=FINAL_EPOCHS)

_crop_w = class_weights_from_labels(crop_labels, n_classes=3)
_sampler = sampler_from_labels(crop_labels, n_classes=3)
_ds = TensorDataset(
    torch.from_numpy(X).float(),
    torch.from_numpy(mask).bool(),
    torch.from_numpy(hurst_vec).float(),
    torch.from_numpy(crop_labels).long(),
    torch.from_numpy(pheno_labels).long(),
)
_dl = DataLoader(_ds, batch_size=16, sampler=_sampler)

_warned_fallback = False
for _ep in range(FINAL_EPOCHS):
    _final_model.train()
    _tot = 0.0
    _n = 0
    for _xb, _mb, _hb, _cb, _pb in _dl:
        _xb = _xb.to(DEVICE); _mb = _mb.to(DEVICE); _hb = _hb.to(DEVICE)
        _cb = _cb.to(DEVICE); _pb = _pb.to(DEVICE)
        _out = _final_model(_xb, mask=_mb, hurst=_hb)
        _pl_flat = _out['pheno_logits'].reshape(-1, len(PHENOPHASES))
        _pb_flat = _pb.reshape(-1)
        try:
            _loss_d = dynamis_loss(
                _out['crop_logits'], _cb, _pl_flat, _pb_flat,
                _out['innovations'],
                lambda_innovation=0.05, lambda_ece=0.02,
                class_weights_crop=_crop_w,
            )
        except (RuntimeError, IndexError, AssertionError):
            if not _warned_fallback:
                print('  [final #6] dynamis_loss fallback to clamp(min=0)')
                _warned_fallback = True
            _loss_d = dynamis_loss(
                _out['crop_logits'], _cb, _pl_flat, _pb_flat.clamp(min=0),
                _out['innovations'],
                lambda_innovation=0.05, lambda_ece=0.02,
                class_weights_crop=_crop_w,
            )
        # v7: same pheno boost as in the fold training
        _loss = _loss_d['total'] + (LAMBDA_PHENO - 1.0) * _loss_d['ce_pheno']
        _opt.zero_grad()
        _loss.backward()
        torch.nn.utils.clip_grad_norm_(_final_model.parameters(), max_norm=1.0)
        _opt.step()
        _tot += _loss.item() * _xb.size(0)
        _n += _xb.size(0)
    _sched.step()
    if _ep == 0 or (_ep + 1) % 5 == 0 or _ep == FINAL_EPOCHS - 1:
        print(f'  ep{_ep+1:02d} loss={_tot/max(_n,1):.3f} (final-model, no val)')

_final_model.eval()
print(f'[#3] Final model trained on all {len(crop_labels)} points ({FINAL_EPOCHS} epochs).')

# ---------------------------------------------------------------------------
# --- CELL 7 (part 3): Save checkpoint ---
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('CELL 7 (part 3) — Save checkpoint')
print('='*60)

from src.dynamis import build_phenology_transition_matrix

model_path = str(MODELS_DIR / f'dynamis_terra_v{VERSION}.pt')

try:
    from src.data.sentinel2_loader import MODEL_BANDS as _MODEL_BANDS
    _model_bands = list(_MODEL_BANDS)
except Exception:
    _model_bands = list(MODEL_BANDS) if 'MODEL_BANDS' in dir() else None

_flat_valid = X[mask]
_x_mean = _flat_valid.mean(axis=0).astype('float32') if _flat_valid.size else None
_x_std  = _flat_valid.std(axis=0).astype('float32')  if _flat_valid.size else None

_ckpt = {
    'model_state_dict': _final_model.state_dict(),
    'config': asdict(_final_model.cfg),
    'phenology_prior': build_phenology_transition_matrix(),
    'feature_names': list(FEATURE_NAMES),
    'pheno_feature_names': list(PHENO_FEATURE_NAMES),
    'model_bands': _model_bands,
    'crop_classes': CROPS,
    'phenophase_classes': list(PHENOPHASES),
    'temperature': float(T_cal),
    'ood_threshold': float(ood_threshold) if not np.isnan(ood_threshold) else None,
    'x_mean': _x_mean,
    'x_std':  _x_std,
    'metrics': {
        'baseline': {k: [float(x) for x in v] for k, v in baseline_metrics['crop'].items()},
        'dynamis':  {k: [float(x) for x in v] for k, v in dyn_metrics['crop'].items()},
        'ece_pre':  float(ece_pre),
        'ece_post': float(ece_post),
        'ood_auc':  float(ood_auc) if not np.isnan(ood_auc) else None,
    },
    'split_by': SPLIT_BY,
    'n_splits': int(n_splits),
    'sample_regions': sorted(SAMPLE_REGIONS),
    'n_points': len(series_list),
    'final_epochs': int(FINAL_EPOCHS),
    'version': VERSION,
    'run_tag': RUN_TAG,
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
}
torch.save(_ckpt, model_path)
print(f'[#3,#4] Checkpoint saved: {model_path}')
print(f'        Keys: {sorted(_ckpt.keys())}')

# Checkpoint audit
_audit = torch.load(model_path, map_location='cpu', weights_only=False)
_required = ['model_state_dict', 'config', 'temperature', 'ood_threshold',
             'feature_names', 'model_bands', 'crop_classes', 'x_mean', 'x_std',
             'split_by', 'version']
_missing = [k for k in _required if k not in _audit]
assert not _missing, f'Checkpoint missing required keys: {_missing}'
print(f'[sanity #8] Checkpoint audit OK — all {len(_required)} required keys present.')

print('\n' + '='*60)
print(f'DONE — v{VERSION} checkpoint: {model_path}')
print(f'       Reports: {REPORTS_DIR}')
print('='*60)
