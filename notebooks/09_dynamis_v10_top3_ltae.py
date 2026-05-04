# -*- coding: utf-8 -*-
"""09_dynamis_v10_top3_ltae.ipynb

# Zero Hunger - Top-1 Strategy V10: Pipeline Cientificamente Mitigado
"""

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
        subprocess.run(['git', 'clone', f'{base}skyvidya-lab/ai-for-good.git', str(REPO_PATH)], check=True)
    else:
        subprocess.run(['git', '-C', str(REPO_PATH), 'pull', '--ff-only'], check=False)

    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    WORKSPACE = Path('/content/drive/Shareddrives/SKYVIDYA/AI for Good/datasets_final_round')
    CACHE_DIR = REPO_PATH / 'data' / 'cache'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'hilbertcurve', 'rasterio', 'geopandas', 'shapely', 'pyarrow', 'lightgbm', 'tqdm', 'seaborn'], check=True)
else:
    REPO_PATH = Path('C:/Users/eluzq/workspace/ai-for-good')
    CACHE_DIR = REPO_PATH / 'data' / 'cache'
    WORKSPACE = REPO_PATH

if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))

import numpy as np
import pandas as pd
from src.data.cache_loader import load_aggregated_cache

print("Searching for the most enriched cache variant available...")
try:
    series_agro = load_aggregated_cache(CACHE_DIR, variant="full_plus")
    print("Loaded FULL_PLUS variant (36 features).")
except Exception:
    try:
        series_agro = load_aggregated_cache(CACHE_DIR, variant="full")
        print("Loaded FULL variant.")
    except Exception:
        series_agro = load_aggregated_cache(CACHE_DIR, enriched=True)
        print("Loaded ENRICHED variant.")

F_AGRO = series_agro[0].features.shape[1]
# MITIGATION 1: Include background class explicitly
CROPS = ['rice', 'corn', 'soybean', 'background']
from src.dynamis import PHENOPHASES, phenophase_name_to_index
N_PHENO = len(PHENOPHASES)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import copy
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.temporal_builder import FEATURE_NAMES as BASE_FEATURE_NAMES
from src.dynamis import hurst_features

def get_doy(date_str):
    try: return pd.to_datetime(str(date_str)).dayofyear
    except: return 1

def gaussian_soft_labels(ps_dates, phenophase_by_date, sigma=10.0):
    T = len(ps_dates)
    labels = np.zeros((T, N_PHENO), dtype=np.float32)
    valid_mask = np.zeros(T, dtype=bool)
    if not phenophase_by_date: return labels, valid_mask
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

n = len(series_agro)
T_max = max(len(ps.dates) for ps in series_agro)
X_agro = np.full((n, T_max, F_AGRO), 0.0, dtype=np.float32)
mask_agro = np.zeros((n, T_max), dtype=bool)
doy_arr = np.zeros((n, T_max), dtype=np.int64)
crop_y = np.zeros(n, dtype=np.int64)
soft_pheno_y = np.zeros((n, T_max, N_PHENO), dtype=np.float32)
valid_pheno_mask = np.zeros((n, T_max), dtype=bool)

regions = []
hurst_vec = np.zeros(n, dtype=np.float32)
ndvi_idx = list(BASE_FEATURE_NAMES).index('ndvi')

for i, ps in enumerate(series_agro):
    T = len(ps.dates)
    X_agro[i, :T, :] = np.nan_to_num(ps.features[:, :].astype(np.float32))
    mask_agro[i, :T] = ps.mask
    doy_arr[i, :T] = [get_doy(d) for d in ps.dates]
    # MITIGATION 1: Map unknown to background (3) instead of rice (0)
    crop_y[i] = CROPS.index(ps.crop_type) if ps.crop_type in CROPS[:3] else 3
    regions.append(ps.region)

    hf = hurst_features(ps.features[:, ndvi_idx], ps.features[:, :12], min_temporal_dates=8)
    hurst_val = hf['hurst_temporal'] if hf['hurst_temporal_valid'] else hf['hurst_spectral_mean']
    hurst_vec[i] = np.clip(hurst_val, 0.1, 0.95)

    s_labels, v_mask = gaussian_soft_labels(ps.dates, ps.phenophase_by_date, sigma=10.0)
    soft_pheno_y[i, :T, :] = s_labels
    valid_pheno_mask[i, :T] = v_mask

import math

# MITIGATION 2: Standard Positional Encoding (Relative to sequence, NOT Absolute DOY)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=150):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class LTAECore(nn.Module):
    def __init__(self, input_dim, d_model=128, n_heads=4, n_layers=3, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=n_heads, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
    def forward(self, x, pad_mask=None):
        h = self.input_proj(x)
        h = self.pos_encoding(h)
        return self.transformer(h, src_key_padding_mask=pad_mask)

class LTAECrop(nn.Module):
    def __init__(self, input_dim, n_crops=4, d_model=128):
        super().__init__()
        self.core = LTAECore(input_dim + 1, d_model=d_model, n_layers=2, dropout=0.3)
        self.crop_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(d_model, n_crops)
        )
    def forward(self, x, hurst, pad_mask=None):
        B, T, _ = x.shape
        h_exp = hurst.unsqueeze(1).unsqueeze(2).expand(-1, T, 1)
        x_in = torch.cat([x, h_exp], dim=-1)
        h = self.core(x_in, pad_mask)
        valid_mask = ~pad_mask
        h_pool = (h * valid_mask.unsqueeze(-1)).sum(1) / valid_mask.sum(1, keepdim=True).clamp(min=1)
        return self.crop_head(h_pool)

class LTAERicePhenology(nn.Module):
    def __init__(self, input_dim, n_pheno, d_model=128):
        super().__init__()
        self.core = LTAECore(input_dim, d_model=d_model, n_layers=3, dropout=0.2)
        self.pheno_head = nn.Linear(d_model, n_pheno)
    def forward(self, x, pad_mask=None):
        h = self.core(x, pad_mask)
        return self.pheno_head(h)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training L-TAE Crop on {DEVICE}...")

groups = np.array(regions)
kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
FOLDS = list(kf.split(np.zeros(len(crop_y)), crop_y, groups=groups))

crop_preds_global = np.zeros_like(crop_y)
EPOCHS = 40
crop_folds_data = []

for fold, (tr, va) in enumerate(FOLDS):
    mu, sd = X_agro[tr][mask_agro[tr]].mean(0), X_agro[tr][mask_agro[tr]].std(0)
    sd[sd < 1e-6] = 1.0
    Xtr_n = np.where(mask_agro[tr][..., None], (X_agro[tr] - mu)/sd, 0.0)
    Xva_n = np.where(mask_agro[va][..., None], (X_agro[va] - mu)/sd, 0.0)

    hm, hsd = hurst_vec[tr].mean(), max(hurst_vec[tr].std(), 1e-6)
    h_tr_n = (hurst_vec[tr] - hm) / hsd
    h_va_n = (hurst_vec[va] - hm) / hsd

    model = LTAECrop(F_AGRO, n_crops=4).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)

    cw = torch.tensor(1.0 / np.maximum(np.bincount(crop_y[tr], minlength=4), 1), dtype=torch.float32).to(DEVICE)
    cw = cw / cw.sum() * 4
    crit = nn.CrossEntropyLoss(weight=cw)

    ds = TensorDataset(
        torch.tensor(Xtr_n, dtype=torch.float32),
        torch.tensor(h_tr_n, dtype=torch.float32),
        ~torch.tensor(mask_agro[tr], dtype=torch.bool),
        torch.tensor(crop_y[tr], dtype=torch.long)
    )
    dl = DataLoader(ds, batch_size=32, shuffle=True)

    best_loss = float('inf')
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        for xb, hb, padb, yb in dl:
            xb, hb, padb, yb = (t.to(DEVICE) for t in (xb, hb, padb, yb))
            opt.zero_grad()
            aug_padb = padb.clone()
            dropout_mask = torch.rand_like(aug_padb, dtype=torch.float) < 0.15
            aug_padb[dropout_mask] = True

            logits = model(xb, hb, aug_padb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            x_v = torch.tensor(Xva_n, dtype=torch.float32).to(DEVICE)
            h_v = torch.tensor(h_va_n, dtype=torch.float32).to(DEVICE)
            pad_v = ~torch.tensor(mask_agro[va], dtype=torch.bool).to(DEVICE)
            y_v = torch.tensor(crop_y[va], dtype=torch.long).to(DEVICE)

            out_v = model(x_v, h_v, pad_v)
            val_loss = crit(out_v, y_v).item()

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
            
            # MITIGATION 3: Use last epoch for OOF to avoid Double-Dipping Selection Bias
            if epoch == EPOCHS - 1:
                last_preds = out_v.argmax(-1).cpu().numpy()

    crop_preds_global[va] = last_preds
    crop_folds_data.append({'mu': mu, 'sd': sd, 'hm': hm, 'hsd': hsd, 'crop_state': best_state})
    print(f"Fold {fold+1} Crop F1: {f1_score(crop_y[va], last_preds, average='macro', zero_division=0):.4f}")

f1_crop_global = f1_score(crop_y, crop_preds_global, average='macro', zero_division=0)
print(f"GLOBAL L-TAE Crop F1-Macro (OOF): {f1_crop_global:.4f}")

def soft_cross_entropy(logits, soft_targets, mask):
    logits, soft_targets = logits[mask], soft_targets[mask]
    if len(logits) == 0: return torch.tensor(0.0, device=logits.device, requires_grad=True)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()

print(f"\nTraining L-TAE Rice Pheno on {DEVICE}...")

RICE_IDX = CROPS.index('rice')
rice_indices = np.where(crop_y == RICE_IDX)[0]
X_rice, mask_rice = X_agro[rice_indices], mask_agro[rice_indices]
soft_y_rice, vmask_rice = soft_pheno_y[rice_indices], valid_pheno_mask[rice_indices]
regions_rice = np.array(regions)[rice_indices]

rice_kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
RICE_FOLDS = list(rice_kf.split(np.zeros(len(rice_indices)), np.zeros(len(rice_indices)), groups=regions_rice))

pheno_preds_global = np.full((n, T_max), -100, dtype=np.int64)
pheno_preds_rice = np.full((len(rice_indices), T_max), -100, dtype=np.int64)
pheno_folds_data = []

for fold, (tr, va) in enumerate(RICE_FOLDS):
    mu, sd = X_rice[tr][mask_rice[tr]].mean(0), X_rice[tr][mask_rice[tr]].std(0)
    sd[sd < 1e-6] = 1.0
    Xtr_n = np.where(mask_rice[tr][..., None], (X_rice[tr] - mu)/sd, 0.0)
    Xva_n = np.where(mask_rice[va][..., None], (X_rice[va] - mu)/sd, 0.0)

    model = LTAERicePhenology(F_AGRO, N_PHENO).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    ds = TensorDataset(
        torch.tensor(Xtr_n, dtype=torch.float32),
        ~torch.tensor(mask_rice[tr], dtype=torch.bool),
        torch.tensor(soft_y_rice[tr], dtype=torch.float32),
        torch.tensor(vmask_rice[tr], dtype=torch.bool)
    )
    dl = DataLoader(ds, batch_size=16, shuffle=True)

    best_loss = float('inf')
    best_state = None

    for epoch in range(40):
        model.train()
        for xb, padb, yb, vmb in dl:
            xb, padb, yb, vmb = (t.to(DEVICE) for t in (xb, padb, yb, vmb))
            opt.zero_grad()
            aug_padb = padb.clone()
            dropout_mask = torch.rand_like(aug_padb, dtype=torch.float) < 0.10
            aug_padb[dropout_mask] = True

            logits = model(xb, aug_padb)
            loss = soft_cross_entropy(logits, yb, vmb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            x_v = torch.tensor(Xva_n, dtype=torch.float32).to(DEVICE)
            pad_v = ~torch.tensor(mask_rice[va], dtype=torch.bool).to(DEVICE)
            y_v = torch.tensor(soft_y_rice[va], dtype=torch.float32).to(DEVICE)
            vm_v = torch.tensor(vmask_rice[va], dtype=torch.bool).to(DEVICE)

            out_v = model(x_v, pad_v)
            val_loss = soft_cross_entropy(out_v, y_v, vm_v).item()

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
            
            # MITIGATION 3: Use last epoch for OOF
            if epoch == 39:
                last_preds = out_v.argmax(-1).cpu().numpy()

    pheno_preds_rice[va] = last_preds
    pheno_folds_data.append({'pheno_state': best_state})

    hard_y_va = soft_y_rice[va].argmax(-1)
    val_valid_mask = vmask_rice[va]
    f1 = f1_score(hard_y_va[val_valid_mask], last_preds[val_valid_mask], average='macro', zero_division=0)
    print(f"Fold {fold+1} Rice Pheno F1 (Soft Labels): {f1:.4f}")

pheno_preds_global[rice_indices] = pheno_preds_rice

print("\n====== TOP-3 STRATEGY V10 (Cientificamente Mitigado) ======")

# ─── Métrica de Fenologia HONESTA e INTEGRADA (Pipeline Leakage + Strict Leakage Removidos)
pheno_y_strict = np.full((n, T_max), -100, dtype=np.int64)
for i, ps in enumerate(series_agro):
    if not ps.phenophase_by_date: continue
    events = {get_doy(k): phenophase_name_to_index(v) for k, v in ps.phenophase_by_date.items()}
    for t, d in enumerate(doy_arr[i]):
        if d in events:
            pheno_y_strict[i, t] = events[d]

# MITIGATION 4: PIPELINE LEAKAGE CORRECTION
# Usamos predições (crop_preds_global) em vez do rótulo real (crop_y)
pipeline_rice_mask_strict = (crop_preds_global[:, None] == RICE_IDX) & (pheno_y_strict != -100)
f1_pheno_pipeline_strict = f1_score(
    pheno_y_strict[pipeline_rice_mask_strict], 
    pheno_preds_global[pipeline_rice_mask_strict], 
    average='macro', zero_division=0
)

# MITIGATION 5: AVALIAÇÃO EM TODOS OS DIAS VÁLIDOS (Soft-Mask Transition Support)
pipeline_rice_mask_all = (crop_preds_global[:, None] == RICE_IDX) & valid_pheno_mask
y_true_soft = soft_pheno_y[pipeline_rice_mask_all].argmax(-1)
f1_pheno_pipeline_all = f1_score(
    y_true_soft, 
    pheno_preds_global[pipeline_rice_mask_all], 
    average='macro', zero_division=0
)

final_score_strict = 100.0 * (0.5 * f1_crop_global + 0.5 * f1_pheno_pipeline_strict)
final_score_honest = 100.0 * (0.5 * f1_crop_global + 0.5 * f1_pheno_pipeline_all)

print(f"F1 Crop (Sem Double-Dipping):       {f1_crop_global:.4f}")
print(f"F1 Pheno (Estrito + Pipeline):      {f1_pheno_pipeline_strict:.4f} (Métrica Oficial de Competição se as regras mantiverem o viés)")
print(f"F1 Pheno (Honesto + Pipeline):      {f1_pheno_pipeline_all:.4f} (Real capacidade em todos os dias úteis)")
print(f"--> FINAL LEADERBOARD SCORE (Estrito):  {final_score_strict:.2f} / 100")
print(f"--> SCORE REALISTA DE GENERALIZAÇÃO:    {final_score_honest:.2f} / 100")

# Save Models
model_dir = WORKSPACE / 'models'
model_dir.mkdir(exist_ok=True, parents=True)
ensemble_data = {'folds': []}
for i in range(5):
    fd = crop_folds_data[i].copy()
    fd.update(pheno_folds_data[i])
    ensemble_data['folds'].append(fd)
torch.save(ensemble_data, model_dir / 'ltae_ensemble_v10.pt')
print("\n✅ V10 Training Completed. Weights saved to ltae_ensemble_v10.pt")
