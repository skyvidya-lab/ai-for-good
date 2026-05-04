import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

# Fix paths
REPO_PATH = Path('C:/Users/eluzq/workspace/ai-for-good')
if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))

from src.data.cache_loader import load_aggregated_cache
CACHE_DIR = REPO_PATH / 'data' / 'cache'
from src.dynamis import PHENOPHASES, phenophase_name_to_index

print("Loading cache...")
series_agro = load_aggregated_cache(CACHE_DIR, variant="full_plus")
F_AGRO = series_agro[0].features.shape[1]
CROPS = ['rice', 'corn', 'soybean']
N_PHENO = len(PHENOPHASES)

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

class LTAERicePhenology(nn.Module):
    def __init__(self, input_dim, n_pheno, d_model=128):
        super().__init__()
        self.core = LTAECore(input_dim, d_model=d_model, n_layers=3, dropout=0.2)
        self.pheno_head = nn.Linear(d_model, n_pheno)
    def forward(self, x, doy, pad_mask=None):
        h = self.core(x, doy, pad_mask)
        return self.pheno_head(h)

def soft_cross_entropy(logits, soft_targets, mask):
    logits, soft_targets = logits[mask], soft_targets[mask]
    if len(logits) == 0: return torch.tensor(0.0, device=logits.device, requires_grad=True)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Let's train DOY ONLY model
print("\n--- TEST 5: DOY-Only Shortcut (Zeroing Spectral Features) ---")
RICE_IDX = CROPS.index('rice')
rice_indices = np.where(crop_y == RICE_IDX)[0]
# ZEROING X!
X_rice = np.zeros_like(X_agro[rice_indices])
mask_rice, doy_rice = mask_agro[rice_indices], doy_arr[rice_indices]
soft_y_rice, vmask_rice = soft_pheno_y[rice_indices], valid_pheno_mask[rice_indices]
regions_rice = np.array(regions)[rice_indices]

rice_kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
tr, va = next(rice_kf.split(np.zeros(len(rice_indices)), np.zeros(len(rice_indices)), groups=regions_rice))

Xtr_n = np.zeros_like(X_rice[tr])
Xva_n = np.zeros_like(X_rice[va])

model = LTAERicePhenology(F_AGRO, N_PHENO).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

ds = TensorDataset(
    torch.tensor(Xtr_n, dtype=torch.float32),
    torch.tensor(doy_rice[tr], dtype=torch.long),
    ~torch.tensor(mask_rice[tr], dtype=torch.bool),
    torch.tensor(soft_y_rice[tr], dtype=torch.float32),
    torch.tensor(vmask_rice[tr], dtype=torch.bool)
)
dl = DataLoader(ds, batch_size=16, shuffle=True)

for epoch in range(15):
    model.train()
    for xb, doyb, padb, yb, vmb in dl:
        xb, doyb, padb, yb, vmb = (t.to(DEVICE) for t in (xb, doyb, padb, yb, vmb))
        opt.zero_grad()
        logits = model(xb, doyb, padb)
        loss = soft_cross_entropy(logits, yb, vmb)
        loss.backward()
        opt.step()

model.eval()
with torch.no_grad():
    x_v = torch.tensor(Xva_n, dtype=torch.float32).to(DEVICE)
    doy_v = torch.tensor(doy_rice[va], dtype=torch.long).to(DEVICE)
    pad_v = ~torch.tensor(mask_rice[va], dtype=torch.bool).to(DEVICE)
    out_v = model(x_v, doy_v, pad_v)
    best_preds = out_v.argmax(-1).cpu().numpy()

hard_y_va = soft_y_rice[va].argmax(-1)
val_valid_mask = vmask_rice[va]
f1 = f1_score(hard_y_va[val_valid_mask], best_preds[val_valid_mask], average='macro', zero_division=0)
print(f"Fold 1 Rice Pheno F1 with ZERO Spectral Input (Only DOY): {f1:.4f}")

# Strict evaluation test
pheno_y_strict = np.full((len(va), T_max), -100, dtype=np.int64)
for idx, i in enumerate(va):
    ps = series_agro[rice_indices[i]]
    if not ps.phenophase_by_date: continue
    events = {get_doy(k): phenophase_name_to_index(v) for k, v in ps.phenophase_by_date.items()}
    for t, d in enumerate(doy_arr[rice_indices[i]]):
        if d in events:
            pheno_y_strict[idx, t] = events[d]

strict_rice_mask_va = (pheno_y_strict != -100)
f1_strict = f1_score(pheno_y_strict[strict_rice_mask_va], best_preds[strict_rice_mask_va], average='macro', zero_division=0)
print(f"Fold 1 Strict Rice Pheno F1 with ZERO Spectral Input (Only DOY): {f1_strict:.4f}")

