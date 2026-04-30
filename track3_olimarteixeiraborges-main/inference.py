import os
import sys
import json
import pickle
import argparse
import re
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as _F
import rasterio
from rasterio.warp import transform_bounds

sys.path.insert(0, '/workspace')

from src.data import (
    consolidate_regions,
    build_point_series,
    extract_bands_at_point,
    FEATURE_NAMES,
    N_FEATURES,
    batch_phenology_features,
)
from src.data.temporal_builder import point_region_from_coords
from src.dynamis import hurst_regional, hurst_features, PHENOPHASES

MODEL_PATH = '/workspace/models/dynamis_terra_v11.pt'


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


def _canonical_date(s):
    """Normalise date to YYYY-MM-DD; raises ValueError on failure."""
    s = str(s).strip()
    for fmt in ('%Y-%m-%d', '%Y/%m/%d'):
        try:
            return datetime.strptime(s, fmt).strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            continue
    parts = re.split(r'[-/]', s)
    if len(parts) == 3:
        y, m, d = parts
        return f'{int(y):04d}-{int(m):02d}-{int(d):02d}'
    raise ValueError(f'Unrecognised date format: {s!r}')


def flatten_features_v10(ps_list, mask_arr, hurst_arr):
    """
    Compute v10 flat features from a list of PointSeries objects.
    Matches train_dynamis_v10.py: 9 stats × N_FEATURES + 1 (n_valid) + pheno + hurst
    """
    n   = len(ps_list)
    F   = N_FEATURES
    n_stats = 9
    base = np.zeros((n, F * n_stats + 1), dtype=np.float32)

    for i, ps in enumerate(ps_list):
        T_ps  = ps.features.shape[0]
        # mask_arr[i] has T_max entries; only the first T_ps correspond to this point
        valid = mask_arr[i, :T_ps]
        n_v   = int(valid.sum())
        base[i, F * n_stats] = n_v

        if n_v < 1:
            continue

        Xi = ps.features[valid].astype(np.float32)

        base[i, 0*F:1*F] = Xi.mean(axis=0)
        base[i, 1*F:2*F] = Xi.std(axis=0)
        base[i, 2*F:3*F] = Xi.max(axis=0)
        base[i, 3*F:4*F] = Xi.min(axis=0)

        if n_v >= 2:
            try:
                valid_dates = [ps.dates[t] for t in range(len(ps.dates)) if mask_arr[i, t]]
                d0   = _canonical_date(valid_dates[0])
                d1   = _canonical_date(valid_dates[-1])
                span = (datetime.strptime(d1, '%Y-%m-%d') -
                        datetime.strptime(d0, '%Y-%m-%d')).days
                if span > 0:
                    base[i, 4*F:5*F] = (Xi[-1] - Xi[0]) / span
                else:
                    base[i, 4*F:5*F] = (Xi[-1] - Xi[0]) / max(n_v - 1, 1)
            except Exception:
                base[i, 4*F:5*F] = (Xi[-1] - Xi[0]) / max(n_v - 1, 1)

        if n_v >= 4:
            base[i, 5*F:6*F] = np.percentile(Xi, 10, axis=0)
            base[i, 6*F:7*F] = np.percentile(Xi, 25, axis=0)
            base[i, 7*F:8*F] = np.percentile(Xi, 75, axis=0)
            base[i, 8*F:9*F] = np.percentile(Xi, 90, axis=0)

    pheno_feats = batch_phenology_features(ps_list, hurst_arr)
    X_flat = np.concatenate([base, pheno_feats, hurst_arr.reshape(-1, 1)], axis=1)
    return X_flat


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


def _load_model_ensemble(checkpoint, device):
    """
    Load DynamisV9 model(s) from checkpoint.  Returns list of eval-mode models.
    Supports v10_stacked, v9_dual_timescale, and legacy v8 checkpoints.
    """
    version = checkpoint.get('model_version', 'v8')
    config  = checkpoint['config']

    if version in ('v10_stacked', 'v9_dual_timescale') or 'ensemble_state_dicts' in checkpoint:
        from src.models import DynamisTerraV9, DynamisV9Config
        cfg = DynamisV9Config(**config)
        models = []
        for sd in checkpoint['ensemble_state_dicts']:
            m = DynamisTerraV9(cfg)
            m.load_state_dict(sd)
            m.eval()
            models.append(m.to(device))
        print(f'  [{version}] {len(models)} DynamisV9 models loaded | device={device}')
    else:
        from src.models import DynamisCropClassifier, DynamisModelConfig
        cfg = DynamisModelConfig(**config)
        m = DynamisCropClassifier(cfg)
        m.load_state_dict(checkpoint['model_state_dict'])
        m.eval()
        models = [m.to(device)]
        print(f'  [v8 single] 1 model loaded | device={device}')

    return models


def _load_stacking_models(checkpoint):
    """
    Load LightGBM + meta-LightGBM from a v10_stacked checkpoint.
    Returns (lgb_model, meta_model) or (None, None) if not a v10 checkpoint.
    """
    if checkpoint.get('model_version') != 'v10_stacked':
        return None, None
    lgb_model  = pickle.loads(checkpoint['lgb_model_bytes'])
    meta_model = pickle.loads(checkpoint['meta_model_bytes'])
    print(f'  [v10] LGB + meta models loaded from checkpoint')
    return lgb_model, meta_model


def _run_ensemble(models, X_t, mask_t, hurst_t):
    """
    Run all models and average crop_logits and state_trajectory.
    Returns dict with averaged tensors.
    """
    all_crop   = []
    all_pheno  = []
    all_state  = []
    all_innov  = []
    all_unc    = []
    all_P      = []

    with torch.no_grad():
        for m in models:
            out = m(X_t, mask=mask_t, hurst=hurst_t)
            all_crop.append(out['crop_logits'])
            all_pheno.append(out['pheno_logits'])
            all_state.append(out['state_trajectory'])
            all_innov.append(out['innovations'])
            all_unc.append(out['uncertainty'])
            all_P.append(out['P_trajectory'])

    return {
        'crop_logits':     torch.stack(all_crop,  0).mean(0),
        'pheno_logits':    torch.stack(all_pheno, 0).mean(0),
        'state_trajectory': torch.stack(all_state, 0).mean(0),
        'innovations':     torch.stack(all_innov, 0).mean(0),
        'uncertainty':     torch.stack(all_unc,   0).mean(0),
        'P_trajectory':    torch.stack(all_P,     0).mean(0),
    }


def main(data_dir='/input', result_dir='/output'):
    print('=== Loading test points ===')
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
    n_fallback  = sum(1 for ps in location_cache.values() if ps is None)
    print(f'  {len(unique_locs)} unique locations with data | {n_fallback} fallback')

    print('\n=== Loading model(s) ===')
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model_version = checkpoint.get('model_version', 'v8')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = _load_model_ensemble(checkpoint, device)
    CROPS  = checkpoint.get('crop_classes', ['rice', 'corn', 'soybean'])

    # For v10: also load LightGBM + meta stacking models
    lgb_model, meta_model = _load_stacking_models(checkpoint)
    use_stacking = (lgb_model is not None and meta_model is not None)
    print(f'  Stacking: {"ENABLED (v10)" if use_stacking else "disabled"}')

    # loc_results: key -> (crop_str, pheno_logits[T,7], _, PointSeries)
    loc_results = {}

    if unique_locs:
        T_max = max(ps.features.shape[0] for _, ps in unique_locs)
        n     = len(unique_locs)
        X     = np.zeros((n, T_max, N_FEATURES), dtype=np.float32)
        mask_arr  = np.zeros((n, T_max), dtype=bool)
        hurst_vec = np.full(n, 0.5, dtype=np.float32)
        regional_cache = {}

        print('\n=== Computing Hurst features ===')
        for i, (loc_key, ps) in enumerate(unique_locs):
            T = ps.features.shape[0]
            X[i, :T]        = ps.features.astype(np.float32)
            mask_arr[i, :T] = ps.mask

            region_view = view.get(ps.region, {})
            cache_key   = (ps.region, round(ps.lon, 5), round(ps.lat, 5))
            if cache_key in regional_cache:
                h_reg = regional_cache[cache_key]
            elif len(region_view) >= 8:
                h_reg = hurst_regional(
                    region_view, extract_bands_at_point, ps.lon, ps.lat, min_dates=8
                )
                regional_cache[cache_key] = h_reg
            else:
                h_reg = float('nan')

            ndvi_ts  = ps.features[:T, ndvi_col]
            bands_ts = ps.features[:T, :12]
            hf = hurst_features(ndvi_ts, bands_ts, min_temporal_dates=8)
            if not np.isnan(h_reg):
                hurst_vec[i] = h_reg
            elif hf['hurst_temporal_valid']:
                hurst_vec[i] = hf['hurst_temporal']
            else:
                hurst_vec[i] = hf['hurst_fallback']

        X = np.nan_to_num(X, nan=0.0)

        print('\n=== Running DynamisV9 ensemble ===')
        X_t     = torch.from_numpy(X).float().to(device)
        mask_t  = torch.from_numpy(mask_arr).bool().to(device)
        hurst_t = torch.from_numpy(hurst_vec).float().to(device)

        out = _run_ensemble(models, X_t, mask_t, hurst_t)

        crop_logits_np  = out['crop_logits'].cpu().numpy()      # (n, 3)
        pheno_logits_np = out['pheno_logits'].cpu().numpy()      # (n, T_max, 7)

        if use_stacking:
            print('\n=== Running v10 stacking ===')
            ps_list = [ps for _, ps in unique_locs]
            X_flat  = flatten_features_v10(ps_list, mask_arr, hurst_vec)
            print(f'  Flat features shape: {X_flat.shape}')

            lgb_proba   = lgb_model.predict_proba(X_flat).astype(np.float32)  # (n, 3)
            dyn_softmax = _F.softmax(
                torch.from_numpy(crop_logits_np).float(), dim=-1
            ).numpy()  # (n, 3)

            meta_X = np.concatenate([
                lgb_proba,       # (n, 3)
                dyn_softmax,     # (n, 3)
                crop_logits_np,  # (n, 3) raw logits
            ], axis=1).astype(np.float32)

            preds = meta_model.predict(meta_X).astype(int)          # (n,)
            print(f'  Meta predictions: {np.bincount(preds, minlength=3).tolist()}')
        else:
            preds = crop_logits_np.argmax(axis=-1)

        for i, (loc_key, ps) in enumerate(unique_locs):
            loc_results[loc_key] = (
                CROPS[int(preds[i])], pheno_logits_np[i], None, ps
            )

    # Fallback prediction from first successful location
    fallback_crop  = CROPS[0]
    fallback_pheno = PHENOPHASES[0]
    if loc_results:
        first_crop, first_pheno_logits, _, first_ps = next(iter(loc_results.values()))
        valid  = np.where(first_ps.mask)[0]
        ft_idx = int(valid[-1]) if len(valid) > 0 else 0
        fallback_crop  = first_crop
        fallback_pheno = PHENOPHASES[int(first_pheno_logits[ft_idx].argmax())]

    print('\n=== Building output dict ===')
    results: dict[str, list] = {}
    for _, row in points_df.iterrows():
        lon_str  = row['Longitude']
        lat_str  = row['Latitude']
        date_str = row['phenophase_date']
        key_str  = f"{lon_str}_{lat_str}_{date_str}"
        loc_key  = (lon_str, lat_str)

        if loc_key not in loc_results:
            results[key_str] = [fallback_crop, fallback_pheno]
            continue

        crop_pred, pheno_logits_i, _, ps = loc_results[loc_key]

        target_dt = _parse_date(date_str)
        if target_dt is not None and ps.dates:
            diffs = []
            for t, d in enumerate(ps.dates):
                dt   = _parse_date(d)
                base = abs((dt - target_dt).days) if dt is not None else 99999
                if not ps.mask[t]:
                    base += 10000  # penalise cloudy timesteps
                diffs.append(base)
            t_idx = int(np.argmin(diffs))
        else:
            valid = np.where(ps.mask)[0]
            t_idx = int(valid[-1]) if len(valid) > 0 else 0

        # Use head_pheno logits (explicitly trained, PhenoAcc=0.994 locally)
        # pheno_logits_i shape: (T_max, 7) — pick timestep closest to target date
        pheno_pred = PHENOPHASES[int(pheno_logits_i[t_idx].argmax())]
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
