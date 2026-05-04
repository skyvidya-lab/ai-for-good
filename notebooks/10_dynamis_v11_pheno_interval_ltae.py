# -*- coding: utf-8 -*-
"""10_dynamis_v11_pheno_interval_ltae.py

# Zero Hunger - V11: Phenophase Interval Temporal Anchor (Crop-Aware)

Key innovations over V10:
    1. CORRECTED phenophase order: Greenup→MidGreenup→Maturity→Peak→Senescence→MidSenescence→Dormancy
       (Maturity BEFORE Peak is systematic in 100% of 778 points — grain-fill onset convention)
    2. PhenoIntervalEmbedding: injects per-point observed intervals as temporal anchor
    3. Crop-specific interval priors as fallback (soybean senescence ~8.5d shorter than rice!)
       Between-region variation (SNR<1.0) is captured by using each point's observed intervals.
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
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    'hilbertcurve', 'rasterio', 'geopandas', 'shapely',
                    'pyarrow', 'lightgbm', 'tqdm', 'seaborn'], check=True)
else:
    REPO_PATH = Path('C:/Users/eluzq/workspace/ai-for-good')
    CACHE_DIR = REPO_PATH / 'data' / 'cache'
    WORKSPACE = REPO_PATH

if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))

# ─── Imports
import numpy as np
import pandas as pd
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.cache_loader import load_aggregated_cache
from src.data.temporal_builder import FEATURE_NAMES as BASE_FEATURE_NAMES
from src.dynamis import (
    PHENOPHASES, PHENOPHASE_INTERVALS_MEAN, PHENOPHASE_CUMULATIVE_DAYS,
    phenophase_name_to_index, hurst_features, build_phenology_interval_embedding,
    get_crop_interval_prior,
)

# ─── Load Cache
print("Loading most enriched cache variant...")
for variant in ("full_plus", "full", "extended", "enriched"):
    try:
        series_agro = load_aggregated_cache(CACHE_DIR, variant=variant)
        print(f"Loaded '{variant}' variant ({series_agro[0].features.shape[1]} features).")
        break
    except Exception:
        continue

F_AGRO = series_agro[0].features.shape[1]
N_PHENO = len(PHENOPHASES)

# CORRECTED: include background class (V10 mitigation retained)
CROPS = ['rice', 'corn', 'soybean', 'background']

# ─── Empirical Interval Constants (from EDA 2026-05-03, 778 points, 100% consistent)
# Canonical sequence: Greenup(0)→MidGreenup(1)→Maturity(2)→Peak(3)→Senescence(4)→MidSenescence(5)→Dormancy(6)
INTERVAL_MEAN = np.array(PHENOPHASE_INTERVALS_MEAN, dtype=np.float32)  # days between phases
CUMULATIVE_DAYS = np.array(PHENOPHASE_CUMULATIVE_DAYS, dtype=np.float32)  # cumulative from Greenup


def get_doy(date_str):
    try:
        return pd.to_datetime(str(date_str)).dayofyear
    except:
        return 1


def compute_phenophase_intervals(
    phenophase_by_date: dict | None,
    crop_type: str | None = None,
) -> np.ndarray:
    """
    Compute actual inter-phase intervals (days) for a point, using the CORRECTED
    canonical sequence. Falls back to the **crop-specific** empirical prior (not global)
    for missing phases.

    EDA result (2026-05-03, 778 pts, 49 regions):
      - Between-crop differences ARE significant: soybean senescence ~8.5d shorter than rice.
      - Between-region variation (SNR<1.0) is smaller than intra-point noise — no region prior needed.
      - Per-point observed intervals are the best estimate; crop prior is the fallback.

    Returns:
        (7,) float32 normalized array (divided by 30d).
    """
    crop_means, _ = get_crop_interval_prior(crop_type)
    intervals = list(crop_means)   # fallback: crop-specific empirical mean

    if phenophase_by_date:
        dated = sorted(phenophase_by_date.items(), key=lambda kv: pd.to_datetime(kv[0]))
        date_by_phase = {v: pd.to_datetime(k) for k, v in dated}
        intervals[0] = 0.0  # Greenup is always the anchor
        for i, phase in enumerate(PHENOPHASES[1:], start=1):
            prev_phase = PHENOPHASES[i - 1]
            if phase in date_by_phase and prev_phase in date_by_phase:
                delta = (date_by_phase[phase] - date_by_phase[prev_phase]).days
                intervals[i] = float(max(delta, 0))

    return np.array(intervals, dtype=np.float32) / 30.0  # normalize: ~30d max


def gaussian_soft_labels(ps_dates, phenophase_by_date, sigma=10.0):
    """Generate soft label distribution per timestep using the CORRECTED sequence."""
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
            w = np.exp(-0.5 * (dist / sigma) ** 2)
            labels[t, idx] = max(labels[t, idx], w)
            max_w = max(max_w, w)
        if max_w > 0.1:
            labels[t] /= labels[t].sum()
            valid_mask[t] = True
    return labels, valid_mask


# ─── Build Data Arrays
n = len(series_agro)
T_max = max(len(ps.dates) for ps in series_agro)

X_agro          = np.full((n, T_max, F_AGRO), 0.0, dtype=np.float32)
mask_agro       = np.zeros((n, T_max), dtype=bool)
doy_arr         = np.zeros((n, T_max), dtype=np.int64)
crop_y          = np.zeros(n, dtype=np.int64)
soft_pheno_y    = np.zeros((n, T_max, N_PHENO), dtype=np.float32)
valid_pheno_mask = np.zeros((n, T_max), dtype=bool)

# NEW: per-point phenophase interval embedding (n, 7) normalized
pheno_intervals = np.zeros((n, N_PHENO), dtype=np.float32)

regions = []
hurst_vec = np.zeros(n, dtype=np.float32)
ndvi_idx = list(BASE_FEATURE_NAMES).index('ndvi')

for i, ps in enumerate(series_agro):
    T = len(ps.dates)
    X_agro[i, :T, :]   = np.nan_to_num(ps.features[:, :].astype(np.float32))
    mask_agro[i, :T]   = ps.mask
    doy_arr[i, :T]     = [get_doy(d) for d in ps.dates]
    crop_y[i]          = CROPS.index(ps.crop_type) if ps.crop_type in CROPS[:3] else 3
    regions.append(ps.region)

    hf = hurst_features(ps.features[:, ndvi_idx], ps.features[:, :12], min_temporal_dates=8)
    hurst_val = hf['hurst_temporal'] if hf['hurst_temporal_valid'] else hf['hurst_spectral_mean']
    hurst_vec[i] = np.clip(hurst_val, 0.1, 0.95)

    # Compute actual phenophase intervals (normalized), fallback to crop-specific prior
    raw_intervals = compute_phenophase_intervals(ps.phenophase_by_date, crop_type=ps.crop_type)
    pheno_intervals[i] = raw_intervals  # already normalized by /30.0 inside function

    s_labels, v_mask = gaussian_soft_labels(ps.dates, ps.phenophase_by_date, sigma=10.0)
    soft_pheno_y[i, :T, :]    = s_labels
    valid_pheno_mask[i, :T]   = v_mask

print(f"Data built: {n} points, T_max={T_max}, F_AGRO={F_AGRO}")
print(f"Phenophase intervals shape: {pheno_intervals.shape}")
print(f"Sample interval row (point 0, normalized): {pheno_intervals[0]}")


# ─── Architecture

class PositionalEncoding(nn.Module):
    """Standard relative sequence positional encoding (V10 mitigation: no absolute DOY bias)."""
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


class PhenoIntervalEmbedding(nn.Module):
    """
    NEW in V11: Projects the per-point phenophase interval vector (7,) to a scalar
    bias added to the CLS token before the Transformer encoder.

    Rationale: intervals are ~100% consistent (std 1-4 days), so the model should
    use them as a strong anchor. A small MLP maps the interval signature to a
    d_model-dimensional bias injected at the sequence level.
    """
    def __init__(self, n_pheno: int = 7, d_model: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_pheno, 64),
            nn.GELU(),
            nn.Linear(64, d_model),
        )

    def forward(self, intervals: torch.Tensor) -> torch.Tensor:
        """
        Args:
            intervals: (B, N_PHENO) normalized interval tensor
        Returns:
            (B, 1, d_model) bias to add to first token (CLS-like)
        """
        return self.proj(intervals).unsqueeze(1)


class LTAECore(nn.Module):
    def __init__(self, input_dim, d_model=128, n_heads=4, n_layers=3, dropout=0.2, use_interval=True):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.use_interval = use_interval
        if use_interval:
            self.interval_emb = PhenoIntervalEmbedding(N_PHENO, d_model)

    def forward(self, x, pad_mask=None, intervals=None):
        h = self.input_proj(x)
        h = self.pos_encoding(h)
        # Inject interval embedding as additive bias to all tokens
        if self.use_interval and intervals is not None:
            interval_bias = self.interval_emb(intervals)  # (B, 1, d_model)
            h = h + interval_bias
        return self.transformer(h, src_key_padding_mask=pad_mask)


class LTAECrop(nn.Module):
    def __init__(self, input_dim, n_crops=4, d_model=128):
        super().__init__()
        self.core = LTAECore(input_dim + 1, d_model=d_model, n_layers=2, dropout=0.3, use_interval=True)
        self.crop_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(d_model, n_crops)
        )

    def forward(self, x, hurst, intervals, pad_mask=None):
        B, T, _ = x.shape
        h_exp = hurst.unsqueeze(1).unsqueeze(2).expand(-1, T, 1)
        x_in = torch.cat([x, h_exp], dim=-1)
        h = self.core(x_in, pad_mask, intervals)
        valid_mask = ~pad_mask
        h_pool = (h * valid_mask.unsqueeze(-1)).sum(1) / valid_mask.sum(1, keepdim=True).clamp(min=1)
        return self.crop_head(h_pool)


class LTAERicePhenology(nn.Module):
    def __init__(self, input_dim, n_pheno, d_model=128):
        super().__init__()
        self.core = LTAECore(input_dim, d_model=d_model, n_layers=3, dropout=0.2, use_interval=True)
        self.pheno_head = nn.Linear(d_model, n_pheno)

    def forward(self, x, intervals, pad_mask=None):
        h = self.core(x, pad_mask, intervals)
        return self.pheno_head(h)


# ─── Soft Cross-Entropy
def soft_cross_entropy(logits, soft_targets, mask):
    logits, soft_targets = logits[mask], soft_targets[mask]
    if len(logits) == 0:
        return torch.tensor(0.0, device=logits.device if len(logits) > 0 else 'cpu', requires_grad=True)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()


# ─── Train Crop Model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nTraining L-TAE Crop (V11 with PhenoIntervalEmbedding) on {DEVICE}...")

groups = np.array(regions)
kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
FOLDS = list(kf.split(np.zeros(len(crop_y)), crop_y, groups=groups))

crop_preds_global = np.zeros_like(crop_y)
EPOCHS = 40
crop_folds_data = []

for fold, (tr, va) in enumerate(FOLDS):
    mu, sd = X_agro[tr][mask_agro[tr]].mean(0), X_agro[tr][mask_agro[tr]].std(0)
    sd[sd < 1e-6] = 1.0
    Xtr_n = np.where(mask_agro[tr][..., None], (X_agro[tr] - mu) / sd, 0.0)
    Xva_n = np.where(mask_agro[va][..., None], (X_agro[va] - mu) / sd, 0.0)

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
        torch.tensor(pheno_intervals[tr], dtype=torch.float32),   # NEW
        ~torch.tensor(mask_agro[tr], dtype=torch.bool),
        torch.tensor(crop_y[tr], dtype=torch.long)
    )
    dl = DataLoader(ds, batch_size=32, shuffle=True)

    best_loss = float('inf')
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        for xb, hb, intv_b, padb, yb in dl:
            xb, hb, intv_b, padb, yb = (t.to(DEVICE) for t in (xb, hb, intv_b, padb, yb))
            opt.zero_grad()
            aug_padb = padb.clone()
            aug_padb[torch.rand_like(aug_padb, dtype=torch.float) < 0.15] = True
            logits = model(xb, hb, intv_b, aug_padb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            x_v = torch.tensor(Xva_n, dtype=torch.float32).to(DEVICE)
            h_v = torch.tensor(h_va_n, dtype=torch.float32).to(DEVICE)
            intv_v = torch.tensor(pheno_intervals[va], dtype=torch.float32).to(DEVICE)
            pad_v = ~torch.tensor(mask_agro[va], dtype=torch.bool).to(DEVICE)
            y_v = torch.tensor(crop_y[va], dtype=torch.long).to(DEVICE)

            out_v = model(x_v, h_v, intv_v, pad_v)
            val_loss = crit(out_v, y_v).item()
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
            # MITIGATION: use last epoch for OOF (no double-dipping)
            if epoch == EPOCHS - 1:
                last_preds = out_v.argmax(-1).cpu().numpy()

    crop_preds_global[va] = last_preds
    crop_folds_data.append({'mu': mu, 'sd': sd, 'hm': hm, 'hsd': hsd, 'crop_state': best_state})
    f1 = f1_score(crop_y[va], last_preds, average='macro', zero_division=0)
    print(f"  Fold {fold + 1} Crop F1: {f1:.4f}")

f1_crop_global = f1_score(crop_y, crop_preds_global, average='macro', zero_division=0)
print(f"GLOBAL Crop F1-Macro (OOF): {f1_crop_global:.4f}")


# ─── Train Rice Phenology Model
print(f"\nTraining L-TAE Rice Phenology (V11) on {DEVICE}...")

RICE_IDX = CROPS.index('rice')
rice_indices = np.where(crop_y == RICE_IDX)[0]
X_rice      = X_agro[rice_indices]
mask_rice   = mask_agro[rice_indices]
soft_y_rice = soft_pheno_y[rice_indices]
vmask_rice  = valid_pheno_mask[rice_indices]
intv_rice   = pheno_intervals[rice_indices]  # NEW
regions_rice = np.array(regions)[rice_indices]

rice_kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
RICE_FOLDS = list(rice_kf.split(np.zeros(len(rice_indices)), np.zeros(len(rice_indices)), groups=regions_rice))

pheno_preds_global = np.full((n, T_max), -100, dtype=np.int64)
pheno_preds_rice   = np.full((len(rice_indices), T_max), -100, dtype=np.int64)
pheno_folds_data   = []

for fold, (tr, va) in enumerate(RICE_FOLDS):
    mu, sd = X_rice[tr][mask_rice[tr]].mean(0), X_rice[tr][mask_rice[tr]].std(0)
    sd[sd < 1e-6] = 1.0
    Xtr_n = np.where(mask_rice[tr][..., None], (X_rice[tr] - mu) / sd, 0.0)
    Xva_n = np.where(mask_rice[va][..., None], (X_rice[va] - mu) / sd, 0.0)

    model = LTAERicePhenology(F_AGRO, N_PHENO).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    ds = TensorDataset(
        torch.tensor(Xtr_n, dtype=torch.float32),
        torch.tensor(intv_rice[tr], dtype=torch.float32),   # NEW
        ~torch.tensor(mask_rice[tr], dtype=torch.bool),
        torch.tensor(soft_y_rice[tr], dtype=torch.float32),
        torch.tensor(vmask_rice[tr], dtype=torch.bool)
    )
    dl = DataLoader(ds, batch_size=16, shuffle=True)

    best_loss = float('inf')
    best_state = None

    for epoch in range(40):
        model.train()
        for xb, intv_b, padb, yb, vmb in dl:
            xb, intv_b, padb, yb, vmb = (t.to(DEVICE) for t in (xb, intv_b, padb, yb, vmb))
            opt.zero_grad()
            aug_padb = padb.clone()
            aug_padb[torch.rand_like(aug_padb, dtype=torch.float) < 0.10] = True
            logits = model(xb, intv_b, aug_padb)
            loss = soft_cross_entropy(logits, yb, vmb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            x_v   = torch.tensor(Xva_n, dtype=torch.float32).to(DEVICE)
            intv_v = torch.tensor(intv_rice[va], dtype=torch.float32).to(DEVICE)
            pad_v = ~torch.tensor(mask_rice[va], dtype=torch.bool).to(DEVICE)
            y_v   = torch.tensor(soft_y_rice[va], dtype=torch.float32).to(DEVICE)
            vm_v  = torch.tensor(vmask_rice[va], dtype=torch.bool).to(DEVICE)

            out_v = model(x_v, intv_v, pad_v)
            val_loss = soft_cross_entropy(out_v, y_v, vm_v).item()
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
            if epoch == 39:
                last_preds = out_v.argmax(-1).cpu().numpy()

    pheno_preds_rice[va] = last_preds
    pheno_folds_data.append({'pheno_state': best_state})

    hard_y_va = soft_y_rice[va].argmax(-1)
    val_vm = vmask_rice[va]
    f1 = f1_score(hard_y_va[val_vm], last_preds[val_vm], average='macro', zero_division=0)
    print(f"  Fold {fold + 1} Rice Pheno F1: {f1:.4f}")

pheno_preds_global[rice_indices] = pheno_preds_rice


# ─── Evaluation (same honest pipeline as V10)
print("\n====== V11 — PHENOPHASE INTERVAL ANCHOR RESULTS ======")

pheno_y_strict = np.full((n, T_max), -100, dtype=np.int64)
for i, ps in enumerate(series_agro):
    if not ps.phenophase_by_date:
        continue
    events = {get_doy(k): phenophase_name_to_index(v) for k, v in ps.phenophase_by_date.items()}
    for t, d in enumerate(doy_arr[i]):
        if d in events:
            pheno_y_strict[i, t] = events[d]

# Pipeline-corrected mask (use predicted crop, not ground truth)
pipeline_rice_mask_strict = (crop_preds_global[:, None] == RICE_IDX) & (pheno_y_strict != -100)
f1_pheno_strict = f1_score(
    pheno_y_strict[pipeline_rice_mask_strict],
    pheno_preds_global[pipeline_rice_mask_strict],
    average='macro', zero_division=0
)

pipeline_rice_mask_all = (crop_preds_global[:, None] == RICE_IDX) & valid_pheno_mask
y_true_soft = soft_pheno_y[pipeline_rice_mask_all].argmax(-1)
f1_pheno_honest = f1_score(
    y_true_soft,
    pheno_preds_global[pipeline_rice_mask_all],
    average='macro', zero_division=0
)

final_strict  = 100.0 * (0.5 * f1_crop_global + 0.5 * f1_pheno_strict)
final_honest  = 100.0 * (0.5 * f1_crop_global + 0.5 * f1_pheno_honest)

print(f"F1 Crop (OOF):                   {f1_crop_global:.4f}")
print(f"F1 Pheno (Strict + Pipeline):    {f1_pheno_strict:.4f}")
print(f"F1 Pheno (Honest + Pipeline):    {f1_pheno_honest:.4f}")
print(f"--> LEADERBOARD SCORE (Strict):  {final_strict:.2f} / 100")
print(f"--> REALISTIC SCORE (Honest):    {final_honest:.2f} / 100")


# ─── Save Ensemble
model_dir = WORKSPACE / 'models'
model_dir.mkdir(exist_ok=True, parents=True)
ensemble_data = {'folds': [], 'version': 'v11_pheno_interval'}
for i in range(5):
    fd = crop_folds_data[i].copy()
    fd.update(pheno_folds_data[i])
    ensemble_data['folds'].append(fd)
torch.save(ensemble_data, model_dir / 'ltae_ensemble_v11.pt')
print("\n✅ V11 Training Completed. Weights saved to ltae_ensemble_v11.pt")
