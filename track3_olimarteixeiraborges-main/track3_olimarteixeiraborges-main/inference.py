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

MODEL_PATH = '/workspace/models/dynamis_terra_v5.pt'


def _parse_date(d: str):
    """Parse slash (2018/9/1) or ISO (2018-09-01) date strings; returns None on failure."""
    for fmt in ('%Y/%m/%d', '%Y-%m-%d'):
        try:
            return datetime.strptime(d.strip(), fmt)
        except Exception:
            pass
    try:
        return datetime.fromisoformat(d.strip())
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
                    bbox_index[region_id] = (bounds[0], bounds[1], bounds[2], bounds[3])
                    break
            except Exception as e:
                print(f'  [warn] bbox read failed for {region_id}: {e}')
    return bbox_index


def main(data_dir='/input', result_dir='/output'):
    print('=== Loading test points ===')
    # Read all columns as strings to preserve exact Longitude/Latitude/phenophase_date
    # values for use as output dict keys (e.g. "124.703696_48.543523_2018/9/1")
    points_df = pd.read_csv(os.path.join(data_dir, 'test_point.csv'), dtype=str)
    print(f'  {len(points_df)} rows | columns: {list(points_df.columns)}')

    tiff_dir = os.path.join(data_dir, 'region_test')
    print(f'\n=== Consolidating regions from {tiff_dir} ===')
    view = consolidate_regions([tiff_dir])
    for r, dates in sorted(view.items()):
        print(f'  {r}: {len(dates)} dates')

    print('\n=== Building bbox index ===')
    bbox_index = build_bbox_index(view)
    print(f'  {len(bbox_index)} regions indexed')

    print('\n=== Building PointSeries cache by location ===')
    # Cache by (lon_str, lat_str) — each unique spatial location is processed once
    location_cache: dict[tuple, object] = {}
    ndvi_col = FEATURE_NAMES.index('ndvi') if 'ndvi' in FEATURE_NAMES else 0

    for _, row in points_df.iterrows():
        lon_str = row['Longitude']
        lat_str = row['Latitude']
        loc_key = (lon_str, lat_str)
        if loc_key in location_cache:
            continue
        lon = float(lon_str)
        lat = float(lat_str)
        region = point_region_from_coords(lon, lat, bbox_index)
        if region is None:
            print(f'  [warn] ({lon:.4f},{lat:.4f}): no region match')
            location_cache[loc_key] = None
            continue
        ps = build_point_series(
            point_id=0, lon=lon, lat=lat,
            region=region, consolidated_view=view,
            phenophase_by_date=None, crop_type=None,
        )
        if ps.features.shape[0] == 0:
            print(f'  [warn] ({lon:.4f},{lat:.4f}): empty series in {region}')
            location_cache[loc_key] = None
        else:
            location_cache[loc_key] = ps

    unique_locs = [(k, ps) for k, ps in location_cache.items() if ps is not None]
    n_fallback = sum(1 for ps in location_cache.values() if ps is None)
    print(f'  {len(unique_locs)} unique locations with data | {n_fallback} fallback')

    print('\n=== Loading model ===')
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    config = DynamisModelConfig(**checkpoint['config'])
    model = DynamisCropClassifier(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    CROPS = checkpoint.get('crop_classes', ['rice', 'corn', 'soybean'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f'  classes={CROPS} | device={device}')

    # loc_results: (lon_str, lat_str) -> (crop_str, pheno_logits[T_max,7], state_traj[T_max,7], PointSeries)
    loc_results = {}

    if unique_locs:
        T_max = max(ps.features.shape[0] for _, ps in unique_locs)
        n = len(unique_locs)
        X = np.zeros((n, T_max, N_FEATURES), dtype=np.float32)
        mask_arr = np.zeros((n, T_max), dtype=bool)
        hurst_vec = np.full(n, 0.5, dtype=np.float32)
        regional_cache = {}

        print('\n=== Computing Hurst features ===')
        for i, (loc_key, ps) in enumerate(unique_locs):
            T = ps.features.shape[0]
            X[i, :T] = ps.features.astype(np.float32)
            mask_arr[i, :T] = ps.mask

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
                mask=torch.from_numpy(mask_arr).bool().to(device),
                hurst=torch.from_numpy(hurst_vec).float().to(device),
            )
        preds = out['crop_logits'].argmax(dim=-1).cpu().numpy()
        pheno_logits_np = out['pheno_logits'].cpu().numpy()    # (n, T_max, 7) — kept but not used for prediction
        state_traj_np = out['state_trajectory'].cpu().numpy()  # (n, T_max, 7) — Kalman posterior state

        for i, (loc_key, ps) in enumerate(unique_locs):
            loc_results[loc_key] = (CROPS[int(preds[i])], pheno_logits_np[i], state_traj_np[i], ps)

    # Compute fallback prediction from the first successful location
    fallback_crop = CROPS[0]
    fallback_pheno = PHENOPHASES[0]
    if loc_results:
        first_crop, first_pheno_logits, first_state_traj, first_ps = next(iter(loc_results.values()))
        valid = np.where(first_ps.mask)[0]
        ft_idx = int(valid[-1]) if len(valid) > 0 else 0
        fallback_crop = first_crop
        fallback_pheno = PHENOPHASES[int(first_state_traj[ft_idx].argmax())]

    print('\n=== Building output dict ===')
    # Required format: {"Longitude_Latitude_phenophase_date": ["crop_type", "PhenophaseStage"]}
    results: dict[str, list] = {}
    for _, row in points_df.iterrows():
        lon_str = row['Longitude']
        lat_str = row['Latitude']
        date_str = row['phenophase_date']
        key_str = f"{lon_str}_{lat_str}_{date_str}"
        loc_key = (lon_str, lat_str)

        if loc_key not in loc_results:
            results[key_str] = [fallback_crop, fallback_pheno]
            continue

        crop_pred, pheno_logits_i, state_traj_i, ps = loc_results[loc_key]

        # Find the timestep closest to the requested phenophase_date.
        # Prefer valid (mask=True) timesteps: masked ones have zero-filled features
        # and produce unreliable predictions. Add a large penalty so masked dates
        # are only chosen when every timestep is invalid.
        target_dt = _parse_date(date_str)
        if target_dt is not None and ps.dates:
            diffs = []
            for t, d in enumerate(ps.dates):
                dt = _parse_date(d)
                base = abs((dt - target_dt).days) if dt is not None else 99999
                if not ps.mask[t]:
                    base += 10000  # penalise cloudy / missing timesteps
                diffs.append(base)
            t_idx = int(np.argmin(diffs))
        else:
            valid = np.where(ps.mask)[0]
            t_idx = int(valid[-1]) if len(valid) > 0 else 0

        # Use Kalman posterior state (state_trajectory) for phenophase.
        # The MKM state_dim = 7 = N_PHENOPHASES; transition matrix A is
        # initialised with the phenology prior so the 7 dims directly
        # correspond to the 7 phenophase stages.  The neural head_pheno
        # (pheno_logits) collapsed to near-random in the previous submission.
        pheno_pred = PHENOPHASES[int(state_traj_i[t_idx].argmax())]
        results[key_str] = [crop_pred, pheno_pred]

    os.makedirs(result_dir, exist_ok=True)
    out_path = os.path.join(result_dir, 'result.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n=== Done: {len(results)} predictions → {out_path} ===')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/input')
    parser.add_argument('--result_dir', default='/output')
    args = parser.parse_args()
    main(args.data_dir, args.result_dir)
