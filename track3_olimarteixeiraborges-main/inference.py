import os
import sys
import json
import pickle
import argparse
import re
import math
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
    batch_phenology_features,
)
from src.data.temporal_builder import point_region_from_coords
from src.dynamis import hurst_regional, hurst_features, PHENOPHASES

MODEL_PATH = '/workspace/models/ltae_ensemble.pt'
CROPS = ['rice', 'corn', 'soybean']


def _parse_date(d: str):
    for fmt in ('%Y/%m/%d', '%Y-%m-%d'):
        try:
            return datetime.strptime(d.strip(), fmt)
        except Exception:
            pass
    try:
        return datetime.fromisoformat(d.strip())
    except Exception:
        return None

def get_doy(date_str):
    try: return pd.to_datetime(str(date_str)).dayofyear
    except: return 1

def build_bbox_index(view):
    bbox_index = {}
    for region_id, dates in view.items():
        for _date, bands in dates.items():
            if not bands: continue
            first_path = list(bands.values())[0]
            try:
                with rasterio.open(str(first_path)) as src:
                    bounds = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
                    bbox_index[region_id] = (bounds[0], bounds[1], bounds[2], bounds[3])
                    break
            except Exception as e:
                print(f'  [warn] bbox read failed for {region_id}: {e}')
    return bbox_index

# ==========================================
# L-TAE V4 Architecture Definitions
# ==========================================

class PositionalEncodingDOY(nn.Module):
    def __init__(self, d_model, max_len=367):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x, doy):
        return x + self.pe[torch.clamp(doy, 0, 366)]

class LTAECore(nn.Module):
    def __init__(self, input_dim, d_model=128, n_heads=4, n_layers=3, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.doy_encoding = PositionalEncodingDOY(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=n_heads, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
    def forward(self, x, doy, pad_mask=None):
        h = self.input_proj(x)
        h = self.doy_encoding(h, doy)
        return self.transformer(h, src_key_padding_mask=pad_mask)

class LTAECrop(nn.Module):
    def __init__(self, input_dim, n_crops=3, d_model=128):
        super().__init__()
        self.core = LTAECore(input_dim + 1, d_model=d_model, n_layers=2, dropout=0.3)
        self.crop_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(d_model, n_crops)
        )
    def forward(self, x, doy, hurst, pad_mask=None):
        B, T, _ = x.shape
        h_exp = hurst.unsqueeze(1).unsqueeze(2).expand(-1, T, 1)
        x_in = torch.cat([x, h_exp], dim=-1)
        h = self.core(x_in, doy, pad_mask)
        valid_mask = ~pad_mask
        h_pool = (h * valid_mask.unsqueeze(-1)).sum(1) / valid_mask.sum(1, keepdim=True).clamp(min=1)
        return self.crop_head(h_pool)

class LTAERicePhenology(nn.Module):
    def __init__(self, input_dim, n_pheno, d_model=128):
        super().__init__()
        self.core = LTAECore(input_dim, d_model=d_model, n_layers=3, dropout=0.2)
        self.pheno_head = nn.Linear(d_model, n_pheno)
    def forward(self, x, doy, pad_mask=None):
        h = self.core(x, doy, pad_mask)
        return self.pheno_head(h)


def main(data_dir='/input', result_dir='/output'):
    print('=== Loading test points ===')
    points_df = pd.read_csv(os.path.join(data_dir, 'test_point.csv'), dtype=str)
    print(f'  {len(points_df)} rows | columns: {list(points_df.columns)}')

    tiff_dir = os.path.join(data_dir, 'region_test')
    print(f'\\n=== Consolidating regions from {tiff_dir} ===')
    view = consolidate_regions([tiff_dir])

    print('\\n=== Building bbox index ===')
    bbox_index = build_bbox_index(view)

    print('\\n=== Building PointSeries cache by location ===')
    location_cache = {}
    ndvi_col = FEATURE_NAMES.index('ndvi') if 'ndvi' in FEATURE_NAMES else 0

    for _, row in points_df.iterrows():
        lon_str = row['Longitude']
        lat_str = row['Latitude']
        loc_key = (lon_str, lat_str)
        if loc_key in location_cache: continue
        lon = float(lon_str)
        lat = float(lat_str)
        region = point_region_from_coords(lon, lat, bbox_index)
        if region is None:
            location_cache[loc_key] = None
            continue
        ps = build_point_series(0, lon, lat, region, view, None, None)
        if ps.features.shape[0] == 0:
            location_cache[loc_key] = None
        else:
            location_cache[loc_key] = ps

    unique_locs = [(k, ps) for k, ps in location_cache.items() if ps is not None]

    print('\\n=== Loading L-TAE Ensemble ===')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    
    crop_models = []
    pheno_models = []
    for fold_data in checkpoint['folds']:
        c_mod = LTAECrop(N_FEATURES, n_crops=3)
        c_mod.load_state_dict(fold_data['crop_state'])
        c_mod.to(device).eval()
        
        p_mod = LTAERicePhenology(N_FEATURES, len(PHENOPHASES))
        p_mod.load_state_dict(fold_data['pheno_state'])
        p_mod.to(device).eval()
        
        crop_models.append((c_mod, fold_data))
        pheno_models.append((p_mod, fold_data))
        
    print(f"  Loaded {len(crop_models)} folds for Ensemble")

    loc_results = {}

    if unique_locs:
        T_max = max(ps.features.shape[0] for _, ps in unique_locs)
        n = len(unique_locs)
        X = np.zeros((n, T_max, N_FEATURES), dtype=np.float32)
        mask_arr = np.zeros((n, T_max), dtype=bool)
        doy_arr = np.zeros((n, T_max), dtype=np.int64)
        hurst_vec = np.full(n, 0.5, dtype=np.float32)
        regional_cache = {}

        print('\\n=== Computing Hurst & DOY features ===')
        for i, (loc_key, ps) in enumerate(unique_locs):
            T = ps.features.shape[0]
            X[i, :T] = ps.features.astype(np.float32)
            mask_arr[i, :T] = ps.mask
            doy_arr[i, :T] = [get_doy(d) for d in ps.dates]

            region_view = view.get(ps.region, {})
            cache_key = (ps.region, round(ps.lon, 5), round(ps.lat, 5))
            if cache_key in regional_cache:
                h_reg = regional_cache[cache_key]
            elif len(region_view) >= 8:
                h_reg = hurst_regional(region_view, extract_bands_at_point, ps.lon, ps.lat, min_dates=8)
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

        print('\\n=== Running L-TAE Ensemble Inference ===')
        crop_probs = np.zeros((n, 3))
        pheno_logits_sum = np.zeros((n, T_max, len(PHENOPHASES)))
        
        doy_t = torch.tensor(doy_arr, dtype=torch.long).to(device)
        pad_t = ~torch.tensor(mask_arr, dtype=torch.bool).to(device)

        for (c_mod, fd), (p_mod, _) in zip(crop_models, pheno_models):
            X_n = np.where(mask_arr[..., None], (X - fd['mu']) / fd['sd'], 0.0)
            h_n = (hurst_vec - fd['hm']) / fd['hsd']
            
            X_t = torch.tensor(X_n, dtype=torch.float32).to(device)
            h_t = torch.tensor(h_n, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                c_out = c_mod(X_t, doy_t, h_t, pad_t)
                crop_probs += torch.softmax(c_out, dim=-1).cpu().numpy()
                
                p_out = p_mod(X_t, doy_t, pad_t)
                pheno_logits_sum += p_out.cpu().numpy()
                
        preds = crop_probs.argmax(axis=-1)

        for i, (loc_key, ps) in enumerate(unique_locs):
            loc_results[loc_key] = (
                CROPS[int(preds[i])], pheno_logits_sum[i], None, ps
            )

    fallback_crop = CROPS[0]
    fallback_pheno = PHENOPHASES[0]
    if loc_results:
        first_crop, first_pheno_logits, _, first_ps = next(iter(loc_results.values()))
        valid = np.where(first_ps.mask)[0]
        ft_idx = int(valid[-1]) if len(valid) > 0 else 0
        fallback_crop = first_crop
        fallback_pheno = PHENOPHASES[int(first_pheno_logits[ft_idx].argmax())]

    print('\\n=== Building output dict ===')
    results = {}
    for _, row in points_df.iterrows():
        lon_str = row['Longitude']
        lat_str = row['Latitude']
        date_str = row['phenophase_date']
        key_str = f"{lon_str}_{lat_str}_{date_str}"
        loc_key = (lon_str, lat_str)

        if loc_key not in loc_results:
            results[key_str] = [fallback_crop, fallback_pheno]
            continue

        crop_pred, pheno_logits_i, _, ps = loc_results[loc_key]

        target_dt = _parse_date(date_str)
        if target_dt is not None and ps.dates:
            diffs = []
            for t, d in enumerate(ps.dates):
                dt = _parse_date(d)
                base = abs((dt - target_dt).days) if dt is not None else 99999
                if not ps.mask[t]:
                    base += 10000
                diffs.append(base)
            t_idx = int(np.argmin(diffs))
        else:
            valid = np.where(ps.mask)[0]
            t_idx = int(valid[-1]) if len(valid) > 0 else 0

        pheno_pred = PHENOPHASES[int(pheno_logits_i[t_idx].argmax())]
        results[key_str] = [crop_pred, pheno_pred]

    os.makedirs(result_dir, exist_ok=True)
    out_path = os.path.join(result_dir, 'result.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\\n=== Done: {len(results)} predictions → {out_path} ===')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/input')
    parser.add_argument('--result_dir', default='/output')
    args = parser.parse_args()
    main(args.data_dir, args.result_dir)
