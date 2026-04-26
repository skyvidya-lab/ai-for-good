"""Inference script for ITU AI/Space Computing Challenge 2026, Track 1.

Reads test_point.csv + region_test/ TIFFs from /input,
writes result.json to /output in the official format:

    {
      "124.703696_48.543523_2018/9/1": ["corn", "Senescence"],
      ...
    }

All 7 known bugs from inference_fixed.py are addressed here:
  1. Output format: result.json with {lon_lat_date: [crop, pheno]} dict
  2. MODEL_PATH: uses v7 checkpoint (configurable via env)
  3. Temperature scaling from checkpoint
  4. x_mean/x_std normalization from checkpoint
  5. Updated Hurst cascade (DFA → diff → RS → temporal → spectral)
  6. Uses state_trajectory for phenophase (with pheno_logits ensemble)
  7. Proper fallback for unmatched regions
"""
import os
import sys
import json
import argparse
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import rasterio
from rasterio.warp import transform_bounds

sys.path.insert(0, '/workspace')

from src.data import (
    consolidate_regions,
    build_point_series,
    extract_bands_at_point,
    FEATURE_NAMES,
    N_FEATURES,
    MODEL_BANDS,
)
from src.data.temporal_builder import point_region_from_coords
from src.dynamis import (
    hurst_regional, hurst_features, hurst_dfa, hurst_diff_regional,
    PHENOPHASES, phenophase_index_to_name,
)
from src.models import DynamisCropClassifier, DynamisModelConfig


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_VERSION = int(os.environ.get('DYNAMIS_VERSION', '7'))
MODEL_PATH = os.environ.get(
    'DYNAMIS_MODEL_PATH',
    f'/workspace/models/dynamis_terra_v{MODEL_VERSION}.pt',
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_date(d: str):
    """Parse a date string in various formats; returns None on failure."""
    if d is None:
        return None
    d = str(d).strip()
    for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%Y-%m-%d-%H-%M'):
        try:
            return datetime.strptime(d, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(d)
    except Exception:
        return None


def _canonical_date(s: str) -> str:
    """Normalise any date format to YYYY-MM-DD for matching."""
    s = str(s).strip()
    for fmt in ('%Y-%m-%d', '%Y/%m/%d'):
        try:
            return datetime.strptime(s, fmt).strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            continue
    import re
    parts = re.split(r'[-/]', s)
    if len(parts) == 3:
        y, m, d = parts
        try:
            return f'{int(y):04d}-{int(m):02d}-{int(d):02d}'
        except ValueError:
            pass
    return s  # return as-is if nothing works


def build_bbox_index(view):
    """Build {region: (min_lon, min_lat, max_lon, max_lat)} from consolidated view."""
    bbox_index = {}
    for region_id, dates in view.items():
        for _date, bands in dates.items():
            if not bands:
                continue
            first_path = list(bands.values())[0]
            try:
                with rasterio.open(str(first_path)) as src:
                    bounds = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
                    bbox_index[region_id] = (bounds[0], bounds[1], bounds[2], bounds[3])
                    break
            except Exception as e:
                print(f'  [warn] bbox read failed for {region_id}: {e}')
    return bbox_index


def _apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to logits before softmax."""
    if temperature is None or temperature <= 0:
        return logits
    return logits / temperature


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(data_dir='/input', result_dir='/output'):
    print('=== Loading test points ===')
    points_df = pd.read_csv(os.path.join(data_dir, 'test_point.csv'))
    print(f'{len(points_df)} rows | columns: {list(points_df.columns)}')

    unique_pids = points_df['point_id'].unique()
    print(f'{len(unique_pids)} unique point_ids | {len(points_df)} total rows')

    tiff_dir = os.path.join(data_dir, 'region_test')
    print(f'\n=== Consolidating regions from {tiff_dir} ===')
    view = consolidate_regions([tiff_dir])
    for r, dates in sorted(view.items()):
        print(f'  {r}: {len(dates)} dates')

    print('\n=== Building bbox index ===')
    bbox_index = build_bbox_index(view)
    print(f'{len(bbox_index)} regions indexed')

    # ---------------------------------------------------------------
    # Step 1: Build ONE PointSeries per unique location
    # ---------------------------------------------------------------
    print('\n=== Building PointSeries cache by location ===')
    pid_to_series = {}
    pid_to_region = {}
    fallback_pids = set()
    ndvi_col = FEATURE_NAMES.index('ndvi') if 'ndvi' in FEATURE_NAMES else 0

    for pid in unique_pids:
        row0 = points_df[points_df['point_id'] == pid].iloc[0]
        lon = float(row0['Longitude'])
        lat = float(row0['Latitude'])

        region = point_region_from_coords(lon, lat, bbox_index)
        if region is None:
            print(f'  [warn] point {pid} ({lon:.4f},{lat:.4f}): no region match')
            fallback_pids.add(pid)
            continue

        ps = build_point_series(
            point_id=int(pid), lon=lon, lat=lat,
            region=region, consolidated_view=view,
            phenophase_by_date=None, crop_type=None,
        )
        if ps.features.shape[0] == 0:
            print(f'  [warn] point {pid}: empty series in {region}')
            fallback_pids.add(pid)
            continue

        pid_to_series[int(pid)] = ps
        pid_to_region[int(pid)] = region

    valid_pids = sorted(pid_to_series.keys())
    print(f'{len(valid_pids)} unique locations with data | {len(fallback_pids)} fallback')

    # ---------------------------------------------------------------
    # Step 2: Load model + checkpoint artifacts
    # ---------------------------------------------------------------
    print('\n=== Loading model ===')
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    config = DynamisModelConfig(**checkpoint['config'])
    model = DynamisCropClassifier(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    CROPS = checkpoint.get('crop_classes', ['rice', 'corn', 'soybean'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # FIX #3: Load temperature from checkpoint
    temperature = checkpoint.get('temperature', 1.0)
    # FIX #4: Load normalisation stats
    x_mean = checkpoint.get('x_mean')
    x_std = checkpoint.get('x_std')
    ood_threshold = checkpoint.get('ood_threshold')

    print(f'classes={CROPS} | device={device} | temperature={temperature:.4f}')
    if x_mean is not None:
        print(f'normalisation: x_mean shape={x_mean.shape}, x_std shape={x_std.shape}')
    else:
        print('normalisation: not available in checkpoint (raw features)')

    # ---------------------------------------------------------------
    # Step 3: Batch inference
    # ---------------------------------------------------------------
    if valid_pids:
        T_max = max(ps.features.shape[0] for ps in pid_to_series.values())
        n = len(valid_pids)
        X = np.zeros((n, T_max, N_FEATURES), dtype=np.float32)
        mask = np.zeros((n, T_max), dtype=bool)
        hurst_vec = np.full(n, 0.5, dtype=np.float32)

        print('\n=== Computing Hurst features (v4/v6 cascade) ===')
        regional_cache = {}

        for i, pid in enumerate(valid_pids):
            ps = pid_to_series[pid]
            T = ps.features.shape[0]
            X[i, :T] = ps.features.astype(np.float32)
            mask[i, :T] = ps.mask

            # FIX #5: Updated Hurst cascade (DFA → diff-regional → R/S-regional → temporal → spectral)
            region_view = view.get(ps.region, {})
            ndvi_ts = ps.features[:T, ndvi_col]
            bands_ts = ps.features[:T, :12]
            h = float('nan')

            # Stage 1: DFA (best quality)
            if len(region_view) >= 8:
                cache_key = (ps.region, round(ps.lon, 5), round(ps.lat, 5))
                if cache_key not in regional_cache:
                    # Build regional NDVI series for DFA
                    try:
                        ndvi_regional = []
                        for d_key in sorted(region_view.keys()):
                            bands_vec = extract_bands_at_point(
                                region_view[d_key], ps.lon, ps.lat, list(MODEL_BANDS))
                            nir, red = bands_vec[7], bands_vec[3]
                            if not np.isnan(nir) and not np.isnan(red):
                                ndvi_regional.append(float((nir - red) / (nir + red + 1e-6)))
                        ndvi_regional = np.asarray(ndvi_regional, dtype=np.float64)
                        if ndvi_regional.size >= 8:
                            h_dfa = hurst_dfa(ndvi_regional)
                            if not np.isnan(h_dfa):
                                h = h_dfa
                                regional_cache[cache_key] = h
                    except Exception:
                        pass

                if np.isnan(h) and cache_key in regional_cache:
                    h = regional_cache[cache_key]

            # Stage 2: diff-regional
            if np.isnan(h) and len(region_view) >= 8:
                try:
                    h_diff = hurst_diff_regional(
                        region_view, extract_bands_at_point, ps.lon, ps.lat, min_dates=8)
                    if not np.isnan(h_diff):
                        h = h_diff
                except Exception:
                    pass

            # Stage 3: R/S-regional
            if np.isnan(h) and len(region_view) >= 8:
                try:
                    h_rs = hurst_regional(
                        region_view, extract_bands_at_point, ps.lon, ps.lat, min_dates=8)
                    if not np.isnan(h_rs):
                        h = h_rs
                except Exception:
                    pass

            # Stage 4: temporal
            if np.isnan(h):
                hf = hurst_features(ndvi_ts, bands_ts, min_temporal_dates=8)
                if hf['hurst_temporal_valid']:
                    h = hf['hurst_temporal']
                # Stage 5: spectral fallback
                elif hf['hurst_spectral_mean'] != 0.5:
                    h = hf['hurst_spectral_mean']
                else:
                    h = 0.5

            hurst_vec[i] = h

        # FIX #4: Apply normalisation if available
        if x_mean is not None and x_std is not None:
            x_mean_np = np.asarray(x_mean, dtype=np.float32)
            x_std_np = np.asarray(x_std, dtype=np.float32)
            x_std_np = np.where(x_std_np < 1e-8, 1.0, x_std_np)
            X = (X - x_mean_np[np.newaxis, np.newaxis, :]) / x_std_np[np.newaxis, np.newaxis, :]

        X = np.nan_to_num(X, nan=0.0)

        print('\n=== Running inference ===')
        with torch.no_grad():
            out = model(
                torch.from_numpy(X).float().to(device),
                mask=torch.from_numpy(mask).bool().to(device),
                hurst=torch.from_numpy(hurst_vec).float().to(device),
            )

        # FIX #3: Apply temperature scaling to crop logits
        crop_logits_np = out['crop_logits'].cpu().numpy()
        crop_logits_scaled = _apply_temperature(crop_logits_np, temperature)
        crop_preds = crop_logits_scaled.argmax(axis=-1)  # (n,)

        pheno_logits_np = out['pheno_logits'].cpu().numpy()   # (n, T_max, 7)

        # FIX #6: Also extract state_trajectory for phenophase
        state_traj_np = out['state_trajectory'].cpu().numpy()  # (n, T_max, 7)
        uncertainty_np = out['uncertainty'].cpu().numpy()       # (n,)
    else:
        crop_preds = np.array([])
        pheno_logits_np = np.array([])
        state_traj_np = np.array([])

    # ---------------------------------------------------------------
    # Step 4: Build results — one entry per row of test_point.csv
    #         in the official JSON format
    # ---------------------------------------------------------------
    print('\n=== Building per-row results ===')
    pid_to_idx = {pid: i for i, pid in enumerate(valid_pids)}

    # Determine fallback values from most common predictions
    fallback_crop = CROPS[0]
    fallback_pheno = PHENOPHASES[0]
    if len(valid_pids) > 0:
        crop_counts = Counter(int(c) for c in crop_preds)
        most_common_crop_idx = crop_counts.most_common(1)[0][0]
        fallback_crop = CROPS[most_common_crop_idx]

    result_json = {}
    row_count = 0

    for _, row in points_df.iterrows():
        pid = int(row['point_id'])
        lon = float(row['Longitude'])
        lat = float(row['Latitude'])
        target_date_str = str(row.get('phenophase_date', '')).strip()

        # Build the JSON key in the exact format the platform expects
        # Key format: "Longitude_Latitude_Date" with date as YYYY/M/D (from test_point.csv)
        json_key = f"{lon}_{lat}_{target_date_str}"

        if pid in pid_to_idx:
            idx = pid_to_idx[pid]
            ps = pid_to_series[pid]

            # Crop prediction (same for all dates of this point)
            crop_pred = CROPS[int(crop_preds[idx])]

            # Phenophase prediction — match target_date to nearest available TIFF date
            target_dt = _parse_date(target_date_str)
            if target_dt is not None and ps.dates:
                diffs = []
                for d in ps.dates:
                    dt = _parse_date(d)
                    diffs.append(abs((dt - target_dt).days) if dt is not None else 99999)
                t_idx = int(np.argmin(diffs))
            else:
                valid_t = np.where(ps.mask)[0]
                t_idx = int(valid_t[-1]) if len(valid_t) > 0 else 0

            # Safety clamp
            T_max_actual = pheno_logits_np.shape[1]
            t_idx = min(t_idx, T_max_actual - 1)

            # FIX #6: Ensemble pheno_logits + state_trajectory for phenophase
            pheno_scores = pheno_logits_np[idx, t_idx, :]       # (7,)
            state_scores = state_traj_np[idx, t_idx, :]         # (7,)

            # Softmax both and average
            pheno_probs = np.exp(pheno_scores - pheno_scores.max())
            pheno_probs /= pheno_probs.sum()
            state_probs = np.exp(state_scores - state_scores.max())
            state_probs /= state_probs.sum()

            # Weight: 0.5 pheno_logits + 0.5 state_trajectory
            combined_probs = 0.5 * pheno_probs + 0.5 * state_probs
            pheno_pred = PHENOPHASES[int(combined_probs.argmax())]
        else:
            # FIX #7: Fallback — use most common predictions
            crop_pred = fallback_crop
            pheno_pred = fallback_pheno

        result_json[json_key] = [crop_pred, pheno_pred]
        row_count += 1

    # ---------------------------------------------------------------
    # Step 5: Validate row count then write output
    # ---------------------------------------------------------------
    total_expected = len(points_df)
    if row_count != total_expected:
        print(f'[WARN] Expected {total_expected} rows but got {row_count}!')
        # Note: if test_point.csv has duplicate (lon, lat, date), dict keys
        # will deduplicate. Check for this.
        n_unique_keys = len(result_json)
        print(f'[INFO] Unique JSON keys: {n_unique_keys} (duplicates collapsed: {row_count - n_unique_keys})')
    else:
        print(f'[OK] All {total_expected} rows processed.')

    # Report duplicate key collision
    n_unique_keys = len(result_json)
    if n_unique_keys != row_count:
        print(f'[INFO] {row_count} rows → {n_unique_keys} unique JSON keys')
        print(f'  ({row_count - n_unique_keys} rows had same (lon, lat, date) — last prediction wins)')

    os.makedirs(result_dir, exist_ok=True)

    # FIX #1: Primary output — result.json in official format
    json_path = os.path.join(result_dir, 'result.json')
    with open(json_path, 'w') as f:
        json.dump(result_json, f, indent=2)
    print(f'\n=== Done: {n_unique_keys} predictions → {json_path} ===')

    # Summary stats
    crop_dist = Counter(v[0] for v in result_json.values())
    pheno_dist = Counter(v[1] for v in result_json.values())
    print(f'\nCrop distribution: {dict(crop_dist)}')
    print(f'Phenophase distribution: {dict(pheno_dist)}')

    # OOD report (if threshold available)
    if ood_threshold is not None and len(valid_pids) > 0:
        n_ood = int((uncertainty_np > ood_threshold).sum())
        print(f'OOD points (uncertainty > {ood_threshold:.3f}): {n_ood}/{len(valid_pids)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/input')
    parser.add_argument('--result_dir', default='/output')
    args = parser.parse_args()
    main(args.data_dir, args.result_dir)
