#!/usr/bin/env python3
"""
train_dynamis_v9.py — Dynamis Terra v9: dual-timescale HRM + ensemble.

Changes vs v8:
  1. DynamisTerraV9 — H-module (MKM, slow T_slow=4) + L-module (GRU, fast)
  2. Rice-specific A-prior: self_loop=0.85, forward=0.15
  3. hidden_dim=128 (vs 64 in v8)
  4. Ensemble of 5 models with different seeds → average crop_logits at inference
  5. Reuses v8 data cache (same 764 points / 47 regions)

Usage (activate venv first):
    C:/Users/jrpmc/Documents/_SkyVidya/venv_training/Scripts/python.exe train_dynamis_v9.py

Outputs:
    models/dynamis_terra_v9.pt     — ensemble checkpoint
    reports/02_baseline_vs_dynamis_v9/
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_PATH   = Path('C:/Users/jrpmc/Documents/_SkyVidya/repos/track3_olimarteixeiraborges')
FINALS_DIR  = Path('C:/Users/jrpmc/Documents/_SkyVidya/AI Challenge/FINAL ROUND')
SAMPLE_DIR  = Path('C:/Users/jrpmc/Documents/_SkyVidya/AI Challenge/FINAL ROUND/sample_extracted')
MODELS_DIR  = Path('C:/Users/jrpmc/Documents/_SkyVidya/models')
CACHE_DIR   = Path('C:/Users/jrpmc/Documents/_SkyVidya/cache')

VERSION      = 9
LAMBDA_PHENO = 4.0   # v9: heavier phenophase loss (v8 used 3.0)
RUN_TAG      = f'02_baseline_vs_dynamis_v{VERSION}'
REPORTS_DIR  = Path('C:/Users/jrpmc/Documents/_SkyVidya/reports') / RUN_TAG

ENSEMBLE_SEEDS = [42, 137, 271, 503, 777]  # 5-model ensemble

for d in [SAMPLE_DIR, MODELS_DIR, CACHE_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_PATH))

print(f'Run tag:      {RUN_TAG}')
print(f'Repo:         {REPO_PATH}')
print(f'Reports:      {REPORTS_DIR}')
print(f'LAMBDA_PHENO: {LAMBDA_PHENO}  (v9: dual-timescale + ensemble)')
print(f'Ensemble seeds: {ENSEMBLE_SEEDS}')

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    print(f'Torch {torch.__version__} | CUDA | {torch.cuda.get_device_name(0)}')
else:
    print(f'Torch {torch.__version__} | CPU')

# ---------------------------------------------------------------------------
# CELL 2 — Load v8 cache (same data, no re-extraction needed)
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('CELL 2 — Load v8 data cache')
print('='*60)

import pickle as _pickle
_ARRAY_CACHE  = CACHE_DIR / 'train_arrays_v8.npz'   # reuse v8 cache
_SERIES_CACHE = CACHE_DIR / 'train_series_v8.pkl'

assert _ARRAY_CACHE.exists(), (
    f'v8 cache not found: {_ARRAY_CACHE}\n'
    f'Run train_dynamis_v8.py first to build the cache.'
)
assert _SERIES_CACHE.exists(), f'v8 series cache not found: {_SERIES_CACHE}'

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

CROPS      = ['rice', 'corn', 'soybean']
SAMPLE_REGIONS = set(ps.region for ps in series_list)
N_FEATURES = X.shape[2]

from src.dynamis import PHENOPHASES, phenophase_name_to_index
from src.data import FEATURE_NAMES, N_FEATURES as _NF

sat_frac = float((hurst_vec >= 0.98).mean())
print(f'Loaded v8 cache: X={X.shape}, mask={mask.shape}, '
      f'series={len(series_list)}')
print(f'Hurst saturation: {sat_frac:.1%}  |  mask True: {mask.mean():.1%}')
print(f'Crop distribution: {dict(zip(CROPS, np.bincount(crop_labels, minlength=3).tolist()))}')

def _canonical_date(s):
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

# ---------------------------------------------------------------------------
# CELL 3 — Utilities
# ---------------------------------------------------------------------------
def expected_calibration_error_np(probs_arr, labels_arr, n_bins=10):
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
    return F.softmax(torch.from_numpy(logits_arr).float() / max(T, 0.1), dim=-1).numpy()


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


# ---------------------------------------------------------------------------
# CELL 4 — Baseline LightGBM
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('CELL 4 — Baseline: LightGBM')
print('='*60)

import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                              confusion_matrix, roc_auc_score, roc_curve,
                              precision_score, recall_score)
from src.data import batch_phenology_features, PHENO_FEATURE_NAMES


def flatten_features(X_arr, mask_arr, series_list_=None):
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
                    span = (_dt.strptime(d1, '%Y-%m-%d') - _dt.strptime(d0, '%Y-%m-%d')).days
                    if span > 0:
                        out[i, 4*F:5*F] = (Xi[-1] - Xi[0]) / span
                        continue
                except Exception:
                    pass
            out[i, 4*F:5*F] = (Xi[-1] - Xi[0]) / max(Xi.shape[0] - 1, 1)
    return out


X_stats = flatten_features(X, mask, series_list_=series_list)
X_pheno = batch_phenology_features(series_list, hurst_vec)
X_flat  = np.concatenate([X_stats, X_pheno, hurst_vec.reshape(-1, 1)], axis=1)
print(f'Flat features: {X_flat.shape}')

groups  = np.array([ps.region for ps in series_list])
n_groups = len(np.unique(groups))
n_splits = max(2, min(5, n_groups))
kf = GroupKFold(n_splits=n_splits)
print(f'GroupKFold(n_splits={n_splits}, n_groups={n_groups})')

SPLIT_BY = 'region'
baseline_metrics = {'crop': {'oa': [], 'kappa': [], 'f1': []}}
bl_pred_all = np.zeros_like(crop_labels)

for fold, (tr, va) in enumerate(kf.split(X_flat, crop_labels, groups=groups)):
    model_lgb = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        class_weight='balanced', random_state=42, verbose=-1,
    )
    model_lgb.fit(X_flat[tr], crop_labels[tr])
    pred = model_lgb.predict(X_flat[va])
    bl_pred_all[va] = pred
    baseline_metrics['crop']['oa'].append(accuracy_score(crop_labels[va], pred))
    baseline_metrics['crop']['kappa'].append(cohen_kappa_score(crop_labels[va], pred))
    baseline_metrics['crop']['f1'].append(f1_score(crop_labels[va], pred, average='macro', zero_division=0))

print('Baseline (crop_type):')
for k, v in baseline_metrics['crop'].items():
    print(f'  {k}: {np.mean(v):.4f} ± {np.std(v):.4f}')
print('Baseline confusion:')
print(pd.DataFrame(confusion_matrix(crop_labels, bl_pred_all), index=CROPS, columns=CROPS))

# ---------------------------------------------------------------------------
# CELL 5 — Dynamis Terra v9 cross-validation
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('CELL 5 — Dynamis Terra v9: cross-validation')
print('='*60)

from src.models import DynamisTerraV9, DynamisV9Config
from src.dynamis import dynamis_loss, PHENOPHASES


def build_v9_model(seed: int | None = None) -> DynamisTerraV9:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    cfg = DynamisV9Config(
        input_dim=N_FEATURES,
        state_dim=len(PHENOPHASES),
        hidden_dim=128,
        attn_heads=4,
        n_crops=3,
        T_slow=4,
        crop_head_dropout=0.3,
        lambda_prior_strength=0.7,
        prior_self_loop=0.85,
        prior_forward=0.15,
    )
    return DynamisTerraV9(cfg).to(DEVICE)


def train_v9_fold(
    X_tr, m_tr, h_tr, c_tr, p_tr,
    X_va, m_va, h_va, c_va, p_va,
    seed=42,
    epochs=50, batch_size=16, lr=3e-4, weight_decay=5e-4,
    lambda_innovation=0.05, lambda_ece=0.02,
    lambda_pheno=LAMBDA_PHENO,
    verbose=True,
):
    model = build_v9_model(seed=seed)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    crop_w  = class_weights_from_labels(c_tr, n_classes=3)
    sampler = sampler_from_labels(c_tr, n_classes=3)

    ds = TensorDataset(
        torch.from_numpy(X_tr).float(), torch.from_numpy(m_tr).bool(),
        torch.from_numpy(h_tr).float(), torch.from_numpy(c_tr).long(),
        torch.from_numpy(p_tr).long(),
    )
    dl = DataLoader(ds, batch_size=batch_size, sampler=sampler)

    best_f1    = -1.0
    best_state = None
    history    = []
    _warned_fallback = False

    for ep in range(epochs):
        model.train()
        total = 0.0; n_batch = 0
        for xb, mb, hb, cb, pb in dl:
            xb = xb.to(DEVICE); mb = mb.to(DEVICE); hb = hb.to(DEVICE)
            cb = cb.to(DEVICE); pb = pb.to(DEVICE)
            out = model(xb, mask=mb, hurst=hb)
            # pheno_logits: (B, T, 7) → flatten to (B*T, 7)
            # pheno_labels in batch: (B, T) → flatten to (B*T,); -100 = ignore_index
            pheno_flat  = out['pheno_logits'].reshape(-1, len(PHENOPHASES))  # (B*T, 7)
            plabel_flat = pb.reshape(-1)                                       # (B*T,)
            loss_dict = dynamis_loss(
                out['crop_logits'], cb,
                pheno_flat, plabel_flat,
                out['innovations'],
                lambda_innovation=lambda_innovation,
                lambda_ece=lambda_ece,
                class_weights_crop=crop_w,
            )
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
        prec   = precision_score(c_va, pred_v, average=None, labels=[0, 1, 2], zero_division=0)
        rec    = recall_score(c_va, pred_v, average=None, labels=[0, 1, 2], zero_division=0)
        f1m    = f1_score(c_va, pred_v, average='macro', zero_division=0)

        # pheno_logits is (n_val, T, 7) → flatten for per-timestep accuracy
        ph_logits_v = out_v['pheno_logits'].cpu().numpy()            # (n_val, T, 7)
        ph_pred_v   = ph_logits_v.argmax(-1)                         # (n_val, T)
        # compare only where label != -100; flatten both
        _ph_pred_flat = ph_pred_v.reshape(-1)
        _ph_true_flat = p_va.reshape(-1)
        _valid_ph     = _ph_true_flat != -100
        pheno_acc   = float((_ph_pred_flat[_valid_ph] == _ph_true_flat[_valid_ph]).mean()) if _valid_ph.sum() > 0 else float('nan')

        history.append({
            'epoch': ep + 1,
            'loss': total / max(n_batch, 1),
            'prec': prec.tolist(),
            'rec': rec.tolist(),
            'f1_macro': float(f1m),
            'pheno_acc': pheno_acc,
        })
        if f1m > best_f1:
            best_f1    = f1m
            best_state = copy.deepcopy(model.state_dict())

        if verbose and (ep == 0 or (ep + 1) % 5 == 0 or ep == epochs - 1):
            ph_str = f'{pheno_acc:.3f}' if not np.isnan(pheno_acc) else 'n/a'
            print(f'  ep{ep+1:02d} loss={total/max(n_batch,1):.3f} '
                  f'F1m={f1m:.3f} PhenoAcc={ph_str} '
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
    probs  = F.softmax(out['crop_logits'], dim=-1).cpu().numpy()
    pred   = probs.argmax(-1)
    unc    = out['uncertainty'].cpu().numpy()
    print(f'  [best F1m={best_f1:.3f}]')
    return model, pred, probs, logits, unc, out, history


# Cross-validation loop (seed=42 for all folds — reproducible)
torch.manual_seed(42)
np.random.seed(42)

dyn_metrics   = {'crop': {'oa': [], 'kappa': [], 'f1': []}, 'uncertainty': []}
dyn_preds_all = np.zeros_like(crop_labels)
dyn_probs_all = np.zeros((len(crop_labels), 3), dtype=np.float32)
dyn_logits_all = np.zeros((len(crop_labels), 3), dtype=np.float32)
dyn_unc_all   = np.zeros(len(crop_labels), dtype=np.float32)
all_fold_histories = []
last_out = None

for fold, (tr, va) in enumerate(kf.split(X, crop_labels, groups=groups)):
    print(f'\n--- Fold {fold+1}/{n_splits} (train={len(tr)}, val={len(va)}) ---')
    model_dyn, pred, probs, logits, unc, out_dict, hist = train_v9_fold(
        X[tr], mask[tr], hurst_vec[tr], crop_labels[tr], pheno_labels[tr],
        X[va], mask[va], hurst_vec[va], crop_labels[va], pheno_labels[va],
        seed=42, epochs=50,
    )
    dyn_preds_all[va]  = pred
    dyn_probs_all[va]  = probs
    dyn_logits_all[va] = logits
    dyn_unc_all[va]    = unc
    dyn_metrics['crop']['oa'].append(accuracy_score(crop_labels[va], pred))
    dyn_metrics['crop']['kappa'].append(cohen_kappa_score(crop_labels[va], pred))
    dyn_metrics['crop']['f1'].append(f1_score(crop_labels[va], pred, average='macro', zero_division=0))
    dyn_metrics['uncertainty'].append(float(np.mean(unc)))
    all_fold_histories.append(hist)
    last_out = out_dict

print('\nDynamis v9 (crop_type):')
for k, v in dyn_metrics['crop'].items():
    print(f'  {k}: {np.mean(v):.4f} ± {np.std(v):.4f}')
print(f'  mean trace(P): {np.mean(dyn_metrics["uncertainty"]):.4f}')
print('\nDynamis v9 confusion:')
print(pd.DataFrame(confusion_matrix(crop_labels, dyn_preds_all), index=CROPS, columns=CROPS))

# ---------------------------------------------------------------------------
# CELL 5¾ — Temperature Scaling
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('CELL 5¾ — Temperature Scaling')
print('='*60)

ece_pre = expected_calibration_error_np(dyn_probs_all, crop_labels, n_bins=10)
print(f'ECE before T-scaling: {ece_pre:.4f}')

T_cal = temperature_scale(dyn_logits_all, crop_labels, steps=500, lr=1e-2)
print(f'Learnt temperature T = {T_cal:.4f}')

dyn_probs_calibrated = apply_temperature(dyn_logits_all, T_cal)
ece_post = expected_calibration_error_np(dyn_probs_calibrated, crop_labels, n_bins=10)
print(f'ECE after  T-scaling: {ece_post:.4f}  (delta {ece_pre - ece_post:+.4f})')
assert np.all(dyn_probs_calibrated.argmax(-1) == dyn_probs_all.argmax(-1))
print('Accuracy preserved (as expected).')

# ---------------------------------------------------------------------------
# CELL 6 — Visualisations
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('CELL 6 — Visualisations')
print('='*60)

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, pred_arr, title in [
    (axes[0], bl_pred_all, 'Baseline (LightGBM)'),
    (axes[1], dyn_preds_all, 'Dynamis v9'),
]:
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

# Hurst histogram
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(hurst_vec, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
ax.axvline(0.5, color='gray', linestyle='--', alpha=0.6)
sat = (hurst_vec >= 0.98).mean()
ax.set_title(f'Hurst — v{VERSION} (sat={sat:.1%})')
ax.set_xlabel('Hurst exponent'); ax.set_ylabel('Count')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(str(REPORTS_DIR / 'hurst_histogram.png'), dpi=120)
plt.close()
print(f'Saved: hurst_histogram.png | sat={sat:.1%}')

# OOD ROC
errors_binary = (dyn_preds_all != crop_labels).astype(int)
if errors_binary.sum() == 0 or errors_binary.sum() == len(errors_binary):
    print(f'[skip OOD ROC] errors={int(errors_binary.sum())}')
    ood_auc = float('nan')
    ood_threshold = float('nan')
else:
    ood_auc = roc_auc_score(errors_binary, dyn_unc_all)
    ood_threshold = float(np.percentile(dyn_unc_all, 90))
    print(f'OOD AUC={ood_auc:.4f} | threshold={ood_threshold:.3f}')

# ---------------------------------------------------------------------------
# CELL 7 (part 1) — Determine best epoch count from CV
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('CELL 7 — Final ensemble: train 5 models on ALL data')
print('='*60)

try:
    _all_best_eps = [
        max(h, key=lambda r: r['f1_macro'])['epoch']
        for h in all_fold_histories
    ]
    FINAL_EPOCHS = max(int(np.median(_all_best_eps)), 50)
    print(f'Best epochs per fold: {_all_best_eps} → using {FINAL_EPOCHS}')
except (IndexError, ValueError, KeyError):
    FINAL_EPOCHS = 50
    print(f'Fallback to {FINAL_EPOCHS} epochs')

# ---------------------------------------------------------------------------
# CELL 7 (part 2) — Train ensemble (5 seeds × all data)
# ---------------------------------------------------------------------------
ensemble_state_dicts = []
_crop_w = class_weights_from_labels(crop_labels, n_classes=3)

for seed in ENSEMBLE_SEEDS:
    print(f'\n--- Ensemble seed={seed} ({FINAL_EPOCHS} epochs) ---')
    torch.manual_seed(seed)
    np.random.seed(seed)

    _model = build_v9_model(seed=seed)
    _opt   = torch.optim.AdamW(_model.parameters(), lr=3e-4, weight_decay=5e-4)
    _sched = torch.optim.lr_scheduler.CosineAnnealingLR(_opt, T_max=FINAL_EPOCHS)
    _sampler = sampler_from_labels(crop_labels, n_classes=3)
    _ds = TensorDataset(
        torch.from_numpy(X).float(), torch.from_numpy(mask).bool(),
        torch.from_numpy(hurst_vec).float(), torch.from_numpy(crop_labels).long(),
        torch.from_numpy(pheno_labels).long(),
    )
    _dl = DataLoader(_ds, batch_size=16, sampler=_sampler)
    _warned_fb = False

    for _ep in range(FINAL_EPOCHS):
        _model.train()
        _tot = 0.0; _n = 0
        for _xb, _mb, _hb, _cb, _pb in _dl:
            _xb = _xb.to(DEVICE); _mb = _mb.to(DEVICE); _hb = _hb.to(DEVICE)
            _cb = _cb.to(DEVICE); _pb = _pb.to(DEVICE)
            _out = _model(_xb, mask=_mb, hurst=_hb)
            # pheno_logits: (B, T, 7) → (B*T, 7); labels: (B, T) → (B*T,)
            _pl_flat  = _out['pheno_logits'].reshape(-1, len(PHENOPHASES))
            _pb_flat  = _pb.reshape(-1)
            _ld = dynamis_loss(
                _out['crop_logits'], _cb, _pl_flat, _pb_flat,
                _out['innovations'],
                lambda_innovation=0.05, lambda_ece=0.02,
                class_weights_crop=_crop_w,
            )
            _loss = _ld['total'] + (LAMBDA_PHENO - 1.0) * _ld['ce_pheno']
            _opt.zero_grad()
            _loss.backward()
            torch.nn.utils.clip_grad_norm_(_model.parameters(), max_norm=1.0)
            _opt.step()
            _tot += _loss.item() * _xb.size(0)
            _n += _xb.size(0)
        _sched.step()
        if _ep == 0 or (_ep + 1) % 10 == 0 or _ep == FINAL_EPOCHS - 1:
            print(f'  ep{_ep+1:02d} loss={_tot/max(_n,1):.3f}')

    _model.eval()
    ensemble_state_dicts.append(copy.deepcopy(_model.state_dict()))
    print(f'  Seed {seed} done.')

print(f'\nEnsemble: {len(ensemble_state_dicts)} models trained.')

# ---------------------------------------------------------------------------
# CELL 7 (part 3) — Quick ensemble evaluation on full dataset
# ---------------------------------------------------------------------------
print('\nEnsemble evaluation on full dataset (train set — for reference):')
_ens_cfg = DynamisV9Config(
    input_dim=N_FEATURES, state_dim=len(PHENOPHASES),
    hidden_dim=128, attn_heads=4, n_crops=3, T_slow=4, crop_head_dropout=0.3,
    lambda_prior_strength=0.7, prior_self_loop=0.85, prior_forward=0.15,
)

_X_t = torch.from_numpy(X).float().to(DEVICE)
_m_t = torch.from_numpy(mask).bool().to(DEVICE)
_h_t = torch.from_numpy(hurst_vec).float().to(DEVICE)

ens_crop_logits_list = []
ens_state_traj_list  = []
with torch.no_grad():
    for sd in ensemble_state_dicts:
        _em = DynamisTerraV9(_ens_cfg).to(DEVICE)
        _em.load_state_dict(sd)
        _em.eval()
        _eo = _em(_X_t, mask=_m_t, hurst=_h_t)
        ens_crop_logits_list.append(_eo['crop_logits'].cpu())
        ens_state_traj_list.append(_eo['state_trajectory'].cpu())

ens_crop_logits = torch.stack(ens_crop_logits_list, 0).mean(0)
ens_preds = ens_crop_logits.argmax(-1).numpy()
ens_f1 = f1_score(crop_labels, ens_preds, average='macro', zero_division=0)
ens_oa = accuracy_score(crop_labels, ens_preds)
print(f'Ensemble (train set) — OA={ens_oa:.4f}, F1m={ens_f1:.4f}')
print(pd.DataFrame(confusion_matrix(crop_labels, ens_preds), index=CROPS, columns=CROPS))

# ---------------------------------------------------------------------------
# CELL 7 (part 4) — Save checkpoint
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('CELL 7 (part 4) — Save checkpoint')
print('='*60)

from src.dynamis import build_phenology_transition_matrix
from src.data.sentinel2_loader import MODEL_BANDS as _MODEL_BANDS

model_path = str(MODELS_DIR / f'dynamis_terra_v{VERSION}.pt')

_flat_valid = X[mask]
_x_mean = _flat_valid.mean(axis=0).astype('float32') if _flat_valid.size else None
_x_std  = _flat_valid.std(axis=0).astype('float32')  if _flat_valid.size else None

_cfg_dict = asdict(_ens_cfg)

_ckpt = {
    # ── Core (required by inference.py) ────────────────────────────────
    'model_state_dict': ensemble_state_dicts[0],   # primary model (fallback)
    'ensemble_state_dicts': ensemble_state_dicts,   # all 5 for averaging
    'config': _cfg_dict,
    'model_version': 'v9_dual_timescale',
    # ── Data metadata ───────────────────────────────────────────────────
    'crop_classes': CROPS,
    'phenophase_classes': list(PHENOPHASES),
    'feature_names': list(FEATURE_NAMES),
    'pheno_feature_names': list(PHENO_FEATURE_NAMES),
    'model_bands': list(_MODEL_BANDS),
    # ── Calibration ─────────────────────────────────────────────────────
    'temperature': float(T_cal),
    'ood_threshold': float(ood_threshold) if not np.isnan(ood_threshold) else None,
    'x_mean': _x_mean,
    'x_std':  _x_std,
    # ── Physics prior ────────────────────────────────────────────────────
    'phenology_prior': build_phenology_transition_matrix(
        self_loop=0.85, forward=0.15, wrap_to_dormancy=0.05
    ),
    # ── Training metadata ────────────────────────────────────────────────
    'metrics': {
        'baseline': {k: [float(x) for x in v] for k, v in baseline_metrics['crop'].items()},
        'dynamis':  {k: [float(x) for x in v] for k, v in dyn_metrics['crop'].items()},
        'ensemble_oa': float(ens_oa),
        'ensemble_f1': float(ens_f1),
        'ece_pre':  float(ece_pre),
        'ece_post': float(ece_post),
        'ood_auc':  float(ood_auc) if not np.isnan(ood_auc) else None,
    },
    'ensemble_seeds': ENSEMBLE_SEEDS,
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
sz = Path(model_path).stat().st_size / 1e6
print(f'[saved] {model_path}  ({sz:.1f} MB)')

# Audit
_audit = torch.load(model_path, map_location='cpu', weights_only=False)
_required = ['model_state_dict', 'config', 'temperature', 'ood_threshold',
             'feature_names', 'model_bands', 'crop_classes', 'x_mean', 'x_std',
             'split_by', 'version', 'ensemble_state_dicts']
_missing = [k for k in _required if k not in _audit]
assert not _missing, f'Checkpoint missing: {_missing}'
print(f'Checkpoint audit OK — {len(ensemble_state_dicts)} ensemble models, '
      f'{len(_required)} required keys present.')

print('\n' + '='*60)
print(f'DONE — v{VERSION} checkpoint: {model_path}')
print(f'       Reports:              {REPORTS_DIR}')
print(f'       CV F1m={np.mean(dyn_metrics["crop"]["f1"]):.4f} | '
      f'Ensemble F1m={ens_f1:.4f}')
print('='*60)
print('\nNext step: copy checkpoint to repo and push to GitLab.')
print(f'  cp "{model_path}" '
      f'"C:/Users/jrpmc/Documents/_SkyVidya/repos/track3_olimarteixeiraborges/models/dynamis_terra_v9.pt"')
