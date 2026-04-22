import os
import sys
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import rasterio
from rasterio.warp import transform_bounds

sys.path.insert(0, '/workspace')

from src.data import (
    consolidate_regions,
    build_point_series,
    extract_bands_at_point,
    FEATURE_NAMES,
    N_FEATURES,
)
from src.data.temporal_builder import point_region_from_coords
from src.dynamis import hurst_regional, hurst_features, PHENOPHASES
from src.models import DynamisCropClassifier, DynamisModelConfig

MODEL_PATH = '/workspace/models/dynamis_terra_v6.pt'


def _parse_date(d: str):
    """Parse a date string in various formats; returns None on failure."""
    if d is None:
        return None
    d = str(d).strip()
    # Try multiple formats: ISO (YYYY-MM-DD), slash (YYYY/M/D)
    for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%Y-%m-%d-%H-%M'):
        try:
            return datetime.strptime(d, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(d)
    except Exception:
        return None


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
                    # transform_bounds returns (west, south, east, north)
                    bbox_index[region_id] = (bounds[0], bounds[1], bounds[2], bounds[3])
                    break
            except Exception as e:
                print(f'  [warn] bbox read failed for {region_id}: {e}')
    return bbox_index


def main(data_dir='/input', result_dir='/output'):
    print('=== Loading test points ===')
    points_df = pd.read_csv(os.path.join(data_dir, 'test_point.csv'))
    print(f'{len(points_df)} rows | columns: {list(points_df.columns)}')

    # Count unique point_ids vs total rows
    unique_pids = points_df['point_id'].unique()
    print(f'{len(unique_pids)} unique point_ids | {len(points_df)} total rows (multiple phenophase_dates per point)')

    tiff_dir = os.path.join(data_dir, 'region_test')
    print(f'\n=== Consolidating regions from {tiff_dir} ===')
    view = consolidate_regions([tiff_dir])
    for r, dates in sorted(view.items()):
        print(f'  {r}: {len(dates)} dates')

    print('\n=== Building bbox index ===')
    bbox_index = build_bbox_index(view)
    print(f'{len(bbox_index)} regions indexed')

    # ---------------------------------------------------------------
    # Step 1: Build ONE PointSeries per unique location (heavy work)
    # ---------------------------------------------------------------
    print('\n=== Building PointSeries cache by location ===')
    pid_to_series = {}   # {point_id: PointSeries}
    pid_to_region = {}   # {point_id: str}
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
    # Step 2: Load model
    # ---------------------------------------------------------------
    print('\n=== Loading model ===')
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    config = DynamisModelConfig(**checkpoint['config'])
    model = DynamisCropClassifier(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    CROPS = checkpoint.get('crop_classes', ['rice', 'corn', 'soybean'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f'classes={CROPS} | device={device}')

    # ---------------------------------------------------------------
    # Step 3: Batch inference — one forward pass for all unique points
    # ---------------------------------------------------------------
    # Build arrays indexed by position in valid_pids
    pid_to_idx = {pid: i for i, pid in enumerate(valid_pids)}

    if valid_pids:
        T_max = max(ps.features.shape[0] for ps in pid_to_series.values())
        n = len(valid_pids)
        X = np.zeros((n, T_max, N_FEATURES), dtype=np.float32)
        mask = np.zeros((n, T_max), dtype=bool)
        hurst_vec = np.full(n, 0.5, dtype=np.float32)

        print('\n=== Computing Hurst features ===')
        regional_cache = {}

        for i, pid in enumerate(valid_pids):
            ps = pid_to_series[pid]
            T = ps.features.shape[0]
            X[i, :T] = ps.features.astype(np.float32)
            mask[i, :T] = ps.mask

            region_view = view.get(ps.region, {})
            cache_key = (ps.region, round(ps.lon, 5), round(ps.lat, 5))

            if cache_key in regional_cache:
                h_reg = regional_cache[cache_key]
            elif len(region_view) >= 8:
                h_reg = hurst_regional(
                    region_view, extract_bands_at_point, ps.lon, ps.lat, min_dates=8
                )
                regional_cache[cache_key] = h_reg
            else:
                h_reg = float('nan')

            ndvi_ts = ps.features[:T, ndvi_col]
            bands_ts = ps.features[:T, :12]
            hf = hurst_features(ndvi_ts, bands_ts, min_temporal_dates=8)

            if not np.isnan(h_reg):
                hurst_vec[i] = h_reg
            elif hf['hurst_temporal_valid']:
                hurst_vec[i] = hf['hurst_temporal']
            else:
                hurst_vec[i] = hf['hurst_fallback']

        X = np.nan_to_num(X, nan=0.0)

        print('\n=== Running inference ===')
        with torch.no_grad():
            out = model(
                torch.from_numpy(X).float().to(device),
                mask=torch.from_numpy(mask).bool().to(device),
                hurst=torch.from_numpy(hurst_vec).float().to(device),
            )
        crop_preds = out['crop_logits'].argmax(dim=-1).cpu().numpy()       # (n,)
        pheno_logits_np = out['pheno_logits'].cpu().numpy()                # (n, T_max, 7)
    else:
        crop_preds = np.array([])
        pheno_logits_np = np.array([])

    # ---------------------------------------------------------------
    # Step 4: Build results — ONE row per (point_id, phenophase_date)
    #         matching the exact structure of test_point.csv
    # ---------------------------------------------------------------
    print('\n=== Building per-row results ===')
    results = []

    # Determine fallback values
    fallback_crop = CROPS[0]
    fallback_pheno = PHENOPHASES[0]
    if len(valid_pids) > 0:
        # Use the most common prediction as fallback
        from collections import Counter
        crop_counts = Counter(int(c) for c in crop_preds)
        most_common_crop_idx = crop_counts.most_common(1)[0][0]
        fallback_crop = CROPS[most_common_crop_idx]

    for _, row in points_df.iterrows():
        pid = int(row['point_id'])
        lon = float(row['Longitude'])
        lat = float(row['Latitude'])
        target_date_str = str(row.get('phenophase_date', ''))

        if pid in pid_to_idx:
            idx = pid_to_idx[pid]
            ps = pid_to_series[pid]

            # Crop prediction (same for all dates of this point)
            crop_pred = CROPS[int(crop_preds[idx])]

            # Phenophase prediction (specific to this phenophase_date)
            target_dt = _parse_date(target_date_str)
            if target_dt is not None and ps.dates:
                diffs = []
                for d in ps.dates:
                    dt = _parse_date(d)
                    diffs.append(abs((dt - target_dt).days) if dt is not None else 99999)
                t_idx = int(np.argmin(diffs))
            else:
                # Fallback: use the last valid (unmasked) timestep
                valid_t = np.where(ps.mask)[0]
                t_idx = int(valid_t[-1]) if len(valid_t) > 0 else 0

            # Safety clamp: t_idx must be within the padded T_max window
            T_max_actual = pheno_logits_np.shape[1]
            t_idx = min(t_idx, T_max_actual - 1)

            pheno_pred = PHENOPHASES[int(pheno_logits_np[idx, t_idx, :].argmax())]
        else:
            # Fallback for points without region match
            crop_pred = fallback_crop
            pheno_pred = fallback_pheno

        results.append({
            'point_id': pid,
            'Longitude': lon,
            'Latitude': lat,
            'phenophase_date': target_date_str,
            'Pre_crop_type': crop_pred,       # ← FIXED: was 'crop_type'
            'Pre_phenophase': pheno_pred,      # ← FIXED: was 'phenophase'
        })

    # ---------------------------------------------------------------
    # Step 5: Validate row count then write output
    # ---------------------------------------------------------------
    total_expected = len(points_df)
    if len(results) != total_expected:
        print(f'[WARN] Expected {total_expected} rows but got {len(results)} — check fallback logic!')
    else:
        print(f'[OK] All {total_expected} rows accounted for.')

    os.makedirs(result_dir, exist_ok=True)

    # Primary output: CSV (platform requirement)
    csv_path = os.path.join(result_dir, 'result.csv')
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f'\n=== Done: {len(results)} predictions → {csv_path} ===')

    # Secondary output: JSON (backup / compatibility)
    out_path = os.path.join(result_dir, 'result.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'=== Also saved JSON: {out_path} ===')

    # Summary stats
    from collections import Counter
    crop_dist = Counter(r['Pre_crop_type'] for r in results)
    pheno_dist = Counter(r['Pre_phenophase'] for r in results)
    print(f'\nCrop distribution: {dict(crop_dist)}')
    print(f'Phenophase distribution: {dict(pheno_dist)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/input')
    parser.add_argument('--result_dir', default='/output')
    args = parser.parse_args()
    main(args.data_dir, args.result_dir)
