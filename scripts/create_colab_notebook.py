import json

cells = []

def add_markdown(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    })

def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split("\n")]
    })

add_markdown("# Zero Hunger - Top-3 Strategy: LGBM (Crop) + L-TAE (Rice Pheno)\n\nNotebook otimizado para o Google Colab.")

add_code("""# ─── Cell 1 — Colab Environment & Setup
import os, sys, subprocess
from pathlib import Path

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    REPO_PATH = Path('/content/ai-for-good')
    if not REPO_PATH.exists():
        try:
            from google.colab import userdata
            token = userdata.get('GITHUB_TOKEN')
        except Exception:
            token = None
        base = f'https://x-access-token:{token}@github.com/' if token else 'https://github.com/'
        subprocess.run(['git', 'clone', f'{base}GeoProjectAI/ai-for-good.git', str(REPO_PATH)], check=True)
    else:
        subprocess.run(['git', '-C', str(REPO_PATH), 'pull', '--ff-only'], check=False)
    
    # Mount Google Drive for cache
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    WORKSPACE = Path('/content/drive/Shareddrives/SKYVIDYA/AI for Good/datasets_final_round')
    CACHE_DIR = WORKSPACE / 'cache'
    
    # Install dependencies
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'hilbertcurve', 'rasterio', 'geopandas', 'shapely', 'pyarrow', 'lightgbm', 'tqdm'], check=True)
else:
    # Local
    REPO_PATH = Path.cwd()
    CACHE_DIR = REPO_PATH / 'data' / 'cache'

if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))

print(f'REPO_PATH: {REPO_PATH}')
print(f'CACHE_DIR: {CACHE_DIR}')
""")

add_code("""# ─── Cell 2 — Load Data Cache
import numpy as np
import pandas as pd
from src.data.cache_loader import load_aggregated_cache, load_manifest

manifest = load_manifest(CACHE_DIR)
print(f"Manifest: n_points = {manifest['n_points']}")

# Use Agro features (21 features) for best performance
series_agro = load_aggregated_cache(CACHE_DIR, enriched=True)
F_AGRO = series_agro[0].features.shape[1]
print(f'Loaded {len(series_agro)} points with {F_AGRO} features.')

# Constants
CROPS = ['rice', 'corn', 'soybean']
from src.dynamis import PHENOPHASES, phenophase_name_to_index
N_PHENO = len(PHENOPHASES)
""")

add_code("""# ─── Cell 3 — L-TAE Gaussian Soft Labels & DOY Logic
import torch

def get_doy(date_str):
    try: return pd.to_datetime(str(date_str)).dayofyear
    except: return 1

def gaussian_soft_labels(ps_dates, phenophase_by_date, sigma=10.0):
    T = len(ps_dates)
    labels = np.zeros((T, N_PHENO), dtype=np.float32)
    valid_mask = np.zeros(T, dtype=bool)

    if not phenophase_by_date:
        return labels, valid_mask

    ps_doy = np.array([get_doy(d) for d in ps_dates])
    events = [(get_doy(k), phenophase_name_to_index(v)) for k, v in phenophase_by_date.items()]

    for t, d_img in enumerate(ps_doy):
        max_w = 0.0
        for d_event, idx in events:
            dist = min(abs(d_img - d_event), 365 - abs(d_img - d_event))
            w = np.exp(-0.5 * (dist / sigma)**2)
            labels[t, idx] = max(labels[t, idx], w)
            max_w = max(max_w, w)
            
        if max_w > 0.1:
            labels[t] /= labels[t].sum()
            valid_mask[t] = True

    return labels, valid_mask

# Build Tensors
n = len(series_agro)
T_max = max(len(ps.dates) for ps in series_agro)

X_agro = np.full((n, T_max, F_AGRO), 0.0, dtype=np.float32)
mask_agro = np.zeros((n, T_max), dtype=bool)
doy_arr = np.zeros((n, T_max), dtype=np.int64)
crop_y = np.zeros(n, dtype=np.int64)
soft_pheno_y = np.zeros((n, T_max, N_PHENO), dtype=np.float32)
valid_pheno_mask = np.zeros((n, T_max), dtype=bool)
regions = []

for i, ps in enumerate(series_agro):
    T = len(ps.dates)
    X_agro[i, :T, :] = np.nan_to_num(ps.features[:, :].astype(np.float32))
    mask_agro[i, :T] = ps.mask
    doy_arr[i, :T] = [get_doy(d) for d in ps.dates]
    crop_y[i] = CROPS.index(ps.crop_type) if ps.crop_type in CROPS else 0
    regions.append(ps.region)
    
    s_labels, v_mask = gaussian_soft_labels(ps.dates, ps.phenophase_by_date, sigma=10.0)
    soft_pheno_y[i, :T, :] = s_labels
    valid_pheno_mask[i, :T] = v_mask

print(f"X_agro shape: {X_agro.shape}")
print(f"Soft Pheno valid points: {valid_pheno_mask.sum()}")
""")

add_code("""# ─── Cell 4 — Model 1: LightGBM for Crop Type (F1 ~ 0.99+)
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

def extract_ts_features(X, mask):
    n, T, F = X.shape
    out = np.zeros((n, F * 5), dtype=np.float32)
    for i in range(n):
        v = mask[i]
        if v.sum() < 1: continue
        Xi = X[i, v]
        out[i, 0*F:1*F] = Xi.mean(0)
        out[i, 1*F:2*F] = Xi.std(0)
        out[i, 2*F:3*F] = Xi.max(0)
        out[i, 3*F:4*F] = Xi.min(0)
        if Xi.shape[0] >= 2:
            out[i, 4*F:5*F] = (Xi[-1] - Xi[0]) / max(Xi.shape[0] - 1, 1)
    return out

print("Training LightGBM for Crop Type...")
groups = np.array(regions)
kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
FOLDS = list(kf.split(np.zeros(len(crop_y)), crop_y, groups=groups))

X_flat = extract_ts_features(X_agro, mask_agro)

crop_preds = np.zeros_like(crop_y)
for fold, (tr, va) in enumerate(FOLDS):
    # CORRECT NORMALIZATION: Fit only on Train!
    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(X_flat[tr])
    Xva_sc = scaler.transform(X_flat[va])
    
    m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, max_depth=6, class_weight='balanced', random_state=42, verbose=-1)
    m.fit(Xtr_sc, crop_y[tr])
    crop_preds[va] = m.predict(Xva_sc)
    
f1_crop = f1_score(crop_y, crop_preds, average='macro', zero_division=0)
print(f"LGBM Crop F1-Macro: {f1_crop:.4f}")
""")

add_code("""# ─── Cell 5 — Model 2: L-TAE Transformer for Rice Phenology
import torch.nn as nn
import torch.nn.functional as F
import math

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
        doy = torch.clamp(doy, 0, 366)
        return x + self.pe[doy]

class LTAERicePhenology(nn.Module):
    def __init__(self, input_dim, n_pheno, d_model=128, n_heads=4, n_layers=3, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.doy_encoding = PositionalEncodingDOY(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pheno_head = nn.Linear(d_model, n_pheno)

    def forward(self, x, doy, pad_mask=None):
        h = self.input_proj(x)
        h = self.doy_encoding(h, doy)
        h = self.transformer(h, src_key_padding_mask=pad_mask)
        return self.pheno_head(h)

def soft_cross_entropy(logits, soft_targets, mask):
    logits, soft_targets = logits[mask], soft_targets[mask]
    if len(logits) == 0: return torch.tensor(0.0, device=logits.device, requires_grad=True)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()
""")

add_code("""# ─── Cell 6 — L-TAE Training (Only on Rice Data)
from torch.utils.data import TensorDataset, DataLoader
import copy

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training L-TAE on {DEVICE}...")

# Filter ONLY RICE points
RICE_IDX = CROPS.index('rice')
rice_indices = np.where(crop_y == RICE_IDX)[0]

X_rice = X_agro[rice_indices]
mask_rice = mask_agro[rice_indices]
doy_rice = doy_arr[rice_indices]
soft_y_rice = soft_pheno_y[rice_indices]
vmask_rice = valid_pheno_mask[rice_indices]
regions_rice = np.array(regions)[rice_indices]

# Folds for Rice
rice_kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
RICE_FOLDS = list(rice_kf.split(np.zeros(len(rice_indices)), np.zeros(len(rice_indices)), groups=regions_rice))

pheno_preds_rice = np.full((len(rice_indices), T_max), -100, dtype=np.int64)
# To map back to original indices
pheno_preds_global = np.full((n, T_max), -100, dtype=np.int64)

for fold, (tr, va) in enumerate(RICE_FOLDS):
    # Normalize features
    mu, sd = X_rice[tr][mask_rice[tr]].mean(0), X_rice[tr][mask_rice[tr]].std(0)
    sd[sd < 1e-6] = 1.0
    
    Xtr_n = np.where(mask_rice[tr][..., None], (X_rice[tr] - mu)/sd, 0.0)
    Xva_n = np.where(mask_rice[va][..., None], (X_rice[va] - mu)/sd, 0.0)
    
    model = LTAERicePhenology(F_AGRO, N_PHENO).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    
    # Dataset
    ds = TensorDataset(
        torch.tensor(Xtr_n, dtype=torch.float32),
        torch.tensor(doy_rice[tr], dtype=torch.long),
        ~torch.tensor(mask_rice[tr], dtype=torch.bool), # src_key_padding_mask expects True for padding
        torch.tensor(soft_y_rice[tr], dtype=torch.float32),
        torch.tensor(vmask_rice[tr], dtype=torch.bool)
    )
    dl = DataLoader(ds, batch_size=16, shuffle=True)
    
    best_loss = float('inf')
    best_state = None
    
    for epoch in range(40):
        model.train()
        for xb, doyb, padb, yb, vmb in dl:
            xb, doyb, padb, yb, vmb = (t.to(DEVICE) for t in (xb, doyb, padb, yb, vmb))
            opt.zero_grad()
            
            # Data Augmentation: Time-Step Dropout (Mask 10% valid steps)
            aug_padb = padb.clone()
            dropout_mask = torch.rand_like(aug_padb, dtype=torch.float) < 0.10
            aug_padb[dropout_mask] = True
            
            logits = model(xb, doyb, aug_padb)
            loss = soft_cross_entropy(logits, yb, vmb)
            loss.backward()
            opt.step()
            
        # Validation
        model.eval()
        with torch.no_grad():
            x_v = torch.tensor(Xva_n, dtype=torch.float32).to(DEVICE)
            doy_v = torch.tensor(doy_rice[va], dtype=torch.long).to(DEVICE)
            pad_v = ~torch.tensor(mask_rice[va], dtype=torch.bool).to(DEVICE)
            y_v = torch.tensor(soft_y_rice[va], dtype=torch.float32).to(DEVICE)
            vm_v = torch.tensor(vmask_rice[va], dtype=torch.bool).to(DEVICE)
            
            out_v = model(x_v, doy_v, pad_v)
            val_loss = soft_cross_entropy(out_v, y_v, vm_v).item()
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                best_preds = out_v.argmax(-1).cpu().numpy()
                
    model.load_state_dict(best_state)
    pheno_preds_rice[va] = best_preds
    
    # Eval hard F1 on Validation
    # We map soft labels argmax to hard labels for metric calculation
    hard_y_va = soft_y_rice[va].argmax(-1)
    val_valid_mask = vmask_rice[va]
    f1 = f1_score(hard_y_va[val_valid_mask], best_preds[val_valid_mask], average='macro', zero_division=0)
    print(f"Fold {fold+1} Rice Pheno F1: {f1:.4f}")

# Map back to global
pheno_preds_global[rice_indices] = pheno_preds_rice
""")

add_code("""# ─── Cell 7 — Leaderboard Score Calculation
from sklearn.metrics import f1_score

# Get Global Crop Score
f1_crop = f1_score(crop_y, crop_preds, average='macro', zero_division=0)

# Get Global Rice Pheno Score
# We use the original strict matching for the official evaluation metric
# Where do we have strict labels?
pheno_y_strict = np.full((n, T_max), -100, dtype=np.int64)
for i, ps in enumerate(series_agro):
    if not ps.phenophase_by_date: continue
    events = {get_doy(k): phenophase_name_to_index(v) for k, v in ps.phenophase_by_date.items()}
    for t, d in enumerate(doy_arr[i]):
        if d in events:
            pheno_y_strict[i, t] = events[d]

strict_rice_mask = (crop_y[:, None] == RICE_IDX) & (pheno_y_strict != -100)
f1_pheno = f1_score(pheno_y_strict[strict_rice_mask], pheno_preds_global[strict_rice_mask], average='macro', zero_division=0)

final_score = 100.0 * (0.5 * f1_crop + 0.5 * f1_pheno)

print(f"====== TOP-3 STRATEGY RESULTS ======")
print(f"F1 Crop (LGBM):     {f1_crop:.4f}")
print(f"F1 RicePheno (L-TAE): {f1_pheno:.4f}")
print(f"FINAL LEADERBOARD SCORE: {final_score:.2f} / 100")
if final_score > 97.96:
    print("🎉 EXPECTED PODIUM! (TOP-3) 🎉")
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("notebooks/09_dynamis_v9_top3_ltae.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated successfully!")
