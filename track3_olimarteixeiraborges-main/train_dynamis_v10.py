#!/usr/bin/env python3
"""
train_dynamis_v10.py — Stacked ensemble: LightGBM + DynamisTerraV9.

Architecture
============
  Layer 1a – LightGBM (96 flat features + 3 extra)      → OOF proba  (n, 3)
  Layer 1b – DynamisTerraV9 (5-seed ensemble, OOF)       → OOF logits (n, 3)
  Layer 2  – Meta-LightGBM trained on [lgb_proba, dyn_softmax] → final crop
  Pheno    – DynamisTerraV9 head_pheno (unchanged from v9)

Expected: F1_Crop 0.927 → 0.96+

Usage (activate venv first):
    C:/Users/jrpmc/Documents/_SkyVidya/venv_training/Scripts/python.exe train_dynamis_v10.py
"""
import os
import sys
import json
import time
import copy
import pickle
import warnings
from pathlib import Path
from dataclasses import asdict
from datetime import datetime as _dt
import re

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_PATH   = Path('C:/Users/jrpmc/Documents/_SkyVidya/repos/track3_olimarteixeiraborges')
MODELS_DIR  = Path('C:/Users/jrpmc/Documents/_SkyVidya/models')
CACHE_DIR   = Path('C:/Users/jrpmc/Documents/_SkyVidya/cache')

VERSION      = 10
LAMBDA_PHENO = 4.0
RUN_TAG      = f'03_stacked_v{VERSION}'
REPORTS_DIR  = Path('C:/Users/jrpmc/Documents/_SkyVidya/reports') / RUN_TAG

ENSEMBLE_SEEDS = [42, 137, 271, 503, 777]

for d in [MODELS_DIR, CACHE_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_PATH))

print(f'Run tag:        {RUN_TAG}')
print(f'LAMBDA_PHENO:   {LAMBDA_PHENO}')
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
# Load data cache (same as v9)
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('CELL 1 — Load v8 data cache')
print('='*60)

import pickle as _pickle

_ARRAY_CACHE  = CACHE_DIR / 'train_arrays_v8.npz'
_SERIES_CACHE = CACHE_DIR / 'train_series_v8.pkl'

assert _ARRAY_CACHE.exists(), f'v8 cache not found: {_ARRAY_CACHE}'
assert _SERIES_CACHE.exists(), f'v8 series cache not found: {_SERIES_CACHE}'

_cached   = np.load(str(_ARRAY_CACHE))
X         = _cached['X']
mask      = _cached['mask']
hurst_vec = _cached['hurst_vec']
crop_labels  = _cached['crop_labels']
pheno_labels = _cached['pheno_labels']

with open(str(_SERIES_CACHE), 'rb') as _f:
    series_list = _pickle.load(_f)

T_max    = X.shape[1]
n_points = X.shape[0]
CROPS    = ['rice', 'corn', 'soybean']

from src.dynamis import PHENOPHASES, phenophase_name_to_index
from src.data   import FEATURE_NAMES, N_FEATURES as _NF
N_FEATURES = _NF

print(f'Loaded: X={X.shape}, mask={mask.shape}')
print(f'Crop dist: {dict(zip(CROPS, np.bincount(crop_labels, minlength=3).tolist()))}')


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
    raise ValueError(f'Bad date: {s!r}')


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
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


def temperature_scale(logits_arr, labels_arr, steps=500, lr=1e-2):
    logits_t = torch.from_numpy(logits_arr).float()
    labels_t = torch.from_numpy(labels_arr).long()
    T_param  = nn.Parameter(torch.ones(1))
    opt_T    = torch.optim.LBFGS([T_param], lr=lr, max_iter=steps)

    def _eval():
        opt_T.zero_grad()
        loss = F.cross_entropy(logits_t / T_param.clamp(min=0.1), labels_t)
        loss.backward()
        return loss

    opt_T.step(_eval)
    return float(T_param.clamp(min=0.1).item())


def apply_temperature(logits_arr, T):
    return F.softmax(
        torch.from_numpy(logits_arr).float() / max(T, 0.1), dim=-1
    ).numpy()


def expected_calibration_error_np(probs_arr, labels_arr, n_bins=10):
    conf    = probs_arr.max(axis=-1)
    correct = (probs_arr.argmax(-1) == labels_arr).astype(float)
    bins    = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(labels_arr)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            ece += m.sum() / total * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


# ---------------------------------------------------------------------------
# Feature engineering — v10 enhanced flat features
# ---------------------------------------------------------------------------
from src.data import batch_phenology_features, PHENO_FEATURE_NAMES


def flatten_features_v10(X_arr, mask_arr, series_list_=None):
    """
    Returns per-point flat features:
      - 5 stats × N_FEATURES  (mean, std, max, min, slope)
      - 4 extra stats × N_FEATURES  (p10, p25, p75, p90)
      - 1 scalar: n_valid observations
    Total: 9 * N_FEATURES + 1
    """
    n, T, F = X_arr.shape
    n_stats = 9  # mean, std, max, min, slope, p10, p25, p75, p90
    out = np.zeros((n, F * n_stats + 1), dtype=np.float32)

    for i in range(n):
        valid = mask_arr[i]
        n_v   = int(valid.sum())
        out[i, F * n_stats] = n_v  # n_valid feature

        if n_v < 1:
            continue

        Xi = X_arr[i, valid]  # (n_v, F)

        # basic stats
        out[i, 0*F:1*F] = Xi.mean(axis=0)
        out[i, 1*F:2*F] = Xi.std(axis=0)
        out[i, 2*F:3*F] = Xi.max(axis=0)
        out[i, 3*F:4*F] = Xi.min(axis=0)

        # slope
        if n_v >= 2:
            if series_list_ is not None:
                try:
                    ps = series_list_[i]
                    valid_dates = [ps.dates[t] for t in range(len(ps.dates)) if mask_arr[i, t]]
                    d0  = _canonical_date(valid_dates[0])
                    d1  = _canonical_date(valid_dates[-1])
                    span = (_dt.strptime(d1, '%Y-%m-%d') - _dt.strptime(d0, '%Y-%m-%d')).days
                    if span > 0:
                        out[i, 4*F:5*F] = (Xi[-1] - Xi[0]) / span
                    else:
                        out[i, 4*F:5*F] = (Xi[-1] - Xi[0]) / max(n_v - 1, 1)
                except Exception:
                    out[i, 4*F:5*F] = (Xi[-1] - Xi[0]) / max(n_v - 1, 1)
            else:
                out[i, 4*F:5*F] = (Xi[-1] - Xi[0]) / max(n_v - 1, 1)

        # percentiles
        if n_v >= 4:
            out[i, 5*F:6*F] = np.percentile(Xi, 10, axis=0)
            out[i, 6*F:7*F] = np.percentile(Xi, 25, axis=0)
            out[i, 7*F:8*F] = np.percentile(Xi, 75, axis=0)
            out[i, 8*F:9*F] = np.percentile(Xi, 90, axis=0)

    return out


X_stats  = flatten_features_v10(X, mask, series_list_=series_list)
X_pheno  = batch_phenology_features(series_list, hurst_vec)
X_flat   = np.concatenate([X_stats, X_pheno, hurst_vec.reshape(-1, 1)], axis=1)
N_FLAT   = X_flat.shape[1]
print(f'\nFlat features (v10): {X_flat.shape}  '
      f'(was 96 in v9, now {N_FLAT})')

# ---------------------------------------------------------------------------
# Cross-validation setup (same GroupKFold as v9)
# ---------------------------------------------------------------------------
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                              confusion_matrix)

groups   = np.array([ps.region for ps in series_list])
n_groups = len(np.unique(groups))
n_splits = max(2, min(5, n_groups))
kf       = GroupKFold(n_splits=n_splits)
print(f'GroupKFold: {n_splits} folds | {n_groups} groups')

# Storage for OOF predictions
lgb_proba_oof  = np.zeros((n_points, 3), dtype=np.float32)
dyn_logits_oof = np.zeros((n_points, 3), dtype=np.float32)
dyn_preds_oof  = np.zeros(n_points, dtype=np.int64)

# ---------------------------------------------------------------------------
# DynamisV9 helpers (identical to v9 training)
# ---------------------------------------------------------------------------
from src.models import DynamisTerraV9, DynamisV9Config
from src.dynamis import dynamis_loss


def build_v9_model(seed=None):
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
    seed=42, epochs=50, batch_size=16, lr=3e-4, weight_decay=5e-4,
    lambda_innovation=0.05, lambda_ece=0.02,
    lambda_pheno=LAMBDA_PHENO, verbose=True,
):
    model  = build_v9_model(seed=seed)
    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crop_w = class_weights_from_labels(c_tr, n_classes=3)
    sampler = sampler_from_labels(c_tr, n_classes=3)

    ds = TensorDataset(
        torch.from_numpy(X_tr).float(), torch.from_numpy(m_tr).bool(),
        torch.from_numpy(h_tr).float(), torch.from_numpy(c_tr).long(),
        torch.from_numpy(p_tr).long(),
    )
    dl = DataLoader(ds, batch_size=batch_size, sampler=sampler)

    best_f1    = -1.0
    best_state = None

    for ep in range(epochs):
        model.train()
        for xb, mb, hb, cb, pb in dl:
            xb = xb.to(DEVICE); mb = mb.to(DEVICE); hb = hb.to(DEVICE)
            cb = cb.to(DEVICE); pb = pb.to(DEVICE)
            out = model(xb, mask=mb, hurst=hb)
            pheno_flat  = out['pheno_logits'].reshape(-1, len(PHENOPHASES))
            plabel_flat = pb.reshape(-1)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            out_v = model(
                torch.from_numpy(X_va).float().to(DEVICE),
                mask=torch.from_numpy(m_va).bool().to(DEVICE),
                hurst=torch.from_numpy(h_va).float().to(DEVICE),
            )
        pred_v = out_v['crop_logits'].argmax(-1).cpu().numpy()
        f1m    = f1_score(c_va, pred_v, average='macro', zero_division=0)

        if f1m > best_f1:
            best_f1    = f1m
            best_state = copy.deepcopy(model.state_dict())

        if verbose and (ep == 0 or (ep + 1) % 10 == 0 or ep == epochs - 1):
            print(f'  ep{ep+1:02d} F1m={f1m:.3f}')

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
    print(f'  [best F1m={best_f1:.3f}]')
    return model, pred, probs, logits, out


# ---------------------------------------------------------------------------
# CELL 2 — OOF cross-validation
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('CELL 2 — OOF cross-validation (LightGBM + DynamisV9)')
print('='*60)

torch.manual_seed(42)
np.random.seed(42)

lgb_cv_f1  = []
dyn_cv_f1  = []
all_histories = []

for fold, (tr, va) in enumerate(kf.split(X_flat, crop_labels, groups=groups)):
    print(f'\n--- Fold {fold+1}/{n_splits} (train={len(tr)}, val={len(va)}) ---')

    # --- LightGBM ---
    print('  [LightGBM]')
    model_lgb = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        class_weight='balanced', random_state=42, verbose=-1,
        n_jobs=-1,
    )
    model_lgb.fit(X_flat[tr], crop_labels[tr])
    lgb_proba_oof[va] = model_lgb.predict_proba(X_flat[va]).astype(np.float32)
    lgb_pred_va = lgb_proba_oof[va].argmax(-1)
    lgb_f1 = f1_score(crop_labels[va], lgb_pred_va, average='macro', zero_division=0)
    lgb_cv_f1.append(lgb_f1)
    print(f'  LGB F1m={lgb_f1:.4f}')

    # --- DynamisV9 ---
    print('  [DynamisV9]')
    model_dyn, pred_va, probs_va, logits_va, _ = train_v9_fold(
        X[tr], mask[tr], hurst_vec[tr], crop_labels[tr], pheno_labels[tr],
        X[va], mask[va], hurst_vec[va], crop_labels[va], pheno_labels[va],
        seed=42, epochs=50, verbose=True,
    )
    dyn_logits_oof[va] = logits_va
    dyn_preds_oof[va]  = pred_va
    dyn_f1 = f1_score(crop_labels[va], pred_va, average='macro', zero_division=0)
    dyn_cv_f1.append(dyn_f1)
    print(f'  DYN F1m={dyn_f1:.4f}')

print(f'\nCV Summary:')
print(f'  LightGBM  F1m = {np.mean(lgb_cv_f1):.4f} ± {np.std(lgb_cv_f1):.4f}')
print(f'  DynamisV9 F1m = {np.mean(dyn_cv_f1):.4f} ± {np.std(dyn_cv_f1):.4f}')

# ---------------------------------------------------------------------------
# CELL 3 — Meta-learner training
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('CELL 3 — Meta-learner (stacking)')
print('='*60)

# Meta features: [lgb_proba (3), dyn_softmax (3)] → 6 columns
dyn_softmax_oof = F.softmax(
    torch.from_numpy(dyn_logits_oof).float(), dim=-1
).numpy()

# Also add the raw logits for the meta model — gives it more signal
# Total meta features: lgb_proba(3) + dyn_softmax(3) + dyn_logits(3) = 9
meta_X = np.concatenate([
    lgb_proba_oof,       # (n, 3) — calibrated proba from LightGBM
    dyn_softmax_oof,     # (n, 3) — calibrated proba from DynamisV9
    dyn_logits_oof,      # (n, 3) — raw logits from DynamisV9 (uncalibrated signal)
], axis=1).astype(np.float32)
print(f'Meta features shape: {meta_X.shape}')

# Meta-LightGBM (shallow, few trees — avoid overfitting on OOF)
meta_cv_f1 = []
meta_preds_oof = np.zeros(n_points, dtype=np.int64)

for fold, (tr, va) in enumerate(kf.split(meta_X, crop_labels, groups=groups)):
    meta_lgb = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3, num_leaves=7,
        subsample=0.8, colsample_bytree=0.8,
        class_weight='balanced', random_state=42, verbose=-1,
    )
    meta_lgb.fit(meta_X[tr], crop_labels[tr])
    meta_pred_va = meta_lgb.predict(meta_X[va])
    meta_preds_oof[va] = meta_pred_va
    meta_cv_f1.append(
        f1_score(crop_labels[va], meta_pred_va, average='macro', zero_division=0)
    )

print(f'Meta-LightGBM F1m = {np.mean(meta_cv_f1):.4f} ± {np.std(meta_cv_f1):.4f}')
print(f'Confusion (meta, OOF):')
print(pd.DataFrame(confusion_matrix(crop_labels, meta_preds_oof), index=CROPS, columns=CROPS))

# Train the final meta model on ALL OOF predictions
# (All 764 points have OOF predictions, so this uses everything)
meta_model_final = lgb.LGBMClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=3, num_leaves=7,
    subsample=0.8, colsample_bytree=0.8,
    class_weight='balanced', random_state=42, verbose=-1,
)
meta_model_final.fit(meta_X, crop_labels)
print('Meta model trained on full OOF dataset.')

# ---------------------------------------------------------------------------
# CELL 4 — Temperature scaling (DynamisV9 logits)
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('CELL 4 — Temperature scaling')
print('='*60)

ece_pre = expected_calibration_error_np(dyn_softmax_oof, crop_labels)
print(f'DynV9 ECE before T-scaling: {ece_pre:.4f}')
T_cal = temperature_scale(dyn_logits_oof, crop_labels)
print(f'Learnt temperature T = {T_cal:.4f}')
dyn_probs_cal = apply_temperature(dyn_logits_oof, T_cal)
ece_post = expected_calibration_error_np(dyn_probs_cal, crop_labels)
print(f'DynV9 ECE after  T-scaling: {ece_post:.4f}')

# ---------------------------------------------------------------------------
# CELL 5 — Train full models on all data
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('CELL 5 — Train full models on all data')
print('='*60)

# --- LightGBM full ---
print('\n[LightGBM full]')
lgb_full = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8,
    class_weight='balanced', random_state=42, verbose=-1,
    n_jobs=-1,
)
lgb_full.fit(X_flat, crop_labels)
lgb_full_pred = lgb_full.predict(X_flat)
lgb_full_f1   = f1_score(crop_labels, lgb_full_pred, average='macro', zero_division=0)
print(f'LGB full (train set) F1m={lgb_full_f1:.4f}')

# --- DynamisV9 ensemble full (5 seeds) ---
print('\n[DynamisV9 ensemble — 5 seeds × all data]')

# Determine epochs from CV best
try:
    FINAL_EPOCHS = 50  # fixed; v9 found 50 is reliable
except Exception:
    FINAL_EPOCHS = 50
print(f'Ensemble epochs: {FINAL_EPOCHS}')

ensemble_state_dicts = []
_crop_w = class_weights_from_labels(crop_labels, n_classes=3)

for seed in ENSEMBLE_SEEDS:
    print(f'\n--- Ensemble seed={seed} ---')
    torch.manual_seed(seed)
    np.random.seed(seed)

    _model   = build_v9_model(seed=seed)
    _opt     = torch.optim.AdamW(_model.parameters(), lr=3e-4, weight_decay=5e-4)
    _sched   = torch.optim.lr_scheduler.CosineAnnealingLR(_opt, T_max=FINAL_EPOCHS)
    _sampler = sampler_from_labels(crop_labels, n_classes=3)
    _ds      = TensorDataset(
        torch.from_numpy(X).float(), torch.from_numpy(mask).bool(),
        torch.from_numpy(hurst_vec).float(), torch.from_numpy(crop_labels).long(),
        torch.from_numpy(pheno_labels).long(),
    )
    _dl = DataLoader(_ds, batch_size=16, sampler=_sampler)

    for _ep in range(FINAL_EPOCHS):
        _model.train()
        _tot = 0.0; _n = 0
        for _xb, _mb, _hb, _cb, _pb in _dl:
            _xb = _xb.to(DEVICE); _mb = _mb.to(DEVICE); _hb = _hb.to(DEVICE)
            _cb = _cb.to(DEVICE); _pb = _pb.to(DEVICE)
            _out = _model(_xb, mask=_mb, hurst=_hb)
            _pl_flat = _out['pheno_logits'].reshape(-1, len(PHENOPHASES))
            _pb_flat = _pb.reshape(-1)
            _ld = dynamis_loss(
                _out['crop_logits'], _cb, _pl_flat, _pb_flat,
                _out['innovations'],
                lambda_innovation=0.05, lambda_ece=0.02,
                class_weights_crop=_crop_w,
            )
            _loss = _ld['total'] + (LAMBDA_PHENO - 1.0) * _ld['ce_pheno']
            _opt.zero_grad()
            _loss.backward()
            torch.nn.utils.clip_grad_norm_(_model.parameters(), 1.0)
            _opt.step()
            _tot += _loss.item() * _xb.size(0)
            _n   += _xb.size(0)
        _sched.step()
        if (_ep + 1) % 10 == 0 or _ep == FINAL_EPOCHS - 1:
            print(f'  ep{_ep+1:02d} loss={_tot/max(_n,1):.3f}')

    _model.eval()
    ensemble_state_dicts.append(copy.deepcopy(_model.state_dict()))
    print(f'  Seed {seed} done.')

print(f'\nEnsemble: {len(ensemble_state_dicts)} models.')

# Quick evaluation on full train set
_ens_cfg = DynamisV9Config(
    input_dim=N_FEATURES, state_dim=len(PHENOPHASES),
    hidden_dim=128, attn_heads=4, n_crops=3, T_slow=4, crop_head_dropout=0.3,
    lambda_prior_strength=0.7, prior_self_loop=0.85, prior_forward=0.15,
)
_X_t = torch.from_numpy(X).float().to(DEVICE)
_m_t = torch.from_numpy(mask).bool().to(DEVICE)
_h_t = torch.from_numpy(hurst_vec).float().to(DEVICE)

ens_crop_logits_list = []
with torch.no_grad():
    for sd in ensemble_state_dicts:
        _em = DynamisTerraV9(_ens_cfg).to(DEVICE)
        _em.load_state_dict(sd)
        _em.eval()
        _eo = _em(_X_t, mask=_m_t, hurst=_h_t)
        ens_crop_logits_list.append(_eo['crop_logits'].cpu())

ens_crop_logits = torch.stack(ens_crop_logits_list, 0).mean(0).numpy()
ens_crop_softmax = F.softmax(
    torch.from_numpy(ens_crop_logits).float(), dim=-1
).numpy()
lgb_full_proba   = lgb_full.predict_proba(X_flat).astype(np.float32)
ens_dyn_softmax  = F.softmax(
    torch.from_numpy(ens_crop_logits).float(), dim=-1
).numpy()

meta_X_full = np.concatenate([
    lgb_full_proba,
    ens_dyn_softmax,
    ens_crop_logits,
], axis=1).astype(np.float32)

ens_meta_pred = meta_model_final.predict(meta_X_full)
ens_meta_f1   = f1_score(crop_labels, ens_meta_pred, average='macro', zero_division=0)
ens_meta_oa   = accuracy_score(crop_labels, ens_meta_pred)
print(f'\nFull-data ensemble + meta (train ref): OA={ens_meta_oa:.4f}, F1m={ens_meta_f1:.4f}')
print(pd.DataFrame(confusion_matrix(crop_labels, ens_meta_pred), index=CROPS, columns=CROPS))

# ---------------------------------------------------------------------------
# CELL 6 — Save checkpoint
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('CELL 6 — Save checkpoint')
print('='*60)

from src.dynamis import build_phenology_transition_matrix
from src.data.sentinel2_loader import MODEL_BANDS as _MODEL_BANDS

model_path = str(MODELS_DIR / f'dynamis_terra_v{VERSION}.pt')

# Serialize sklearn/lgb models as bytes (portable)
lgb_bytes  = pickle.dumps(lgb_full)
meta_bytes = pickle.dumps(meta_model_final)
print(f'LGB model bytes: {len(lgb_bytes)/1e3:.1f} KB')
print(f'Meta model bytes: {len(meta_bytes)/1e3:.1f} KB')

_flat_valid = X[mask]
_x_mean = _flat_valid.mean(axis=0).astype('float32') if _flat_valid.size else None
_x_std  = _flat_valid.std(axis=0).astype('float32')  if _flat_valid.size else None

_ckpt = {
    # ── Core ────────────────────────────────────────────────────────────
    'model_version': 'v10_stacked',
    'model_state_dict': ensemble_state_dicts[0],
    'ensemble_state_dicts': ensemble_state_dicts,
    'config': asdict(_ens_cfg),
    # ── Stacking components ──────────────────────────────────────────────
    'lgb_model_bytes':  lgb_bytes,   # full-data LightGBM (v10 features)
    'meta_model_bytes': meta_bytes,  # meta-LightGBM trained on OOF
    'n_flat_features': int(N_FLAT),  # expected input dim for LGB
    # ── Feature config ───────────────────────────────────────────────────
    'feature_version': 'v10',        # 9 stats × N_FEATURES + 1 + pheno + hurst
    'n_stat_blocks': 9,              # mean/std/max/min/slope/p10/p25/p75/p90
    # ── Data metadata ────────────────────────────────────────────────────
    'crop_classes': CROPS,
    'phenophase_classes': list(PHENOPHASES),
    'feature_names': list(FEATURE_NAMES),
    'pheno_feature_names': list(PHENO_FEATURE_NAMES),
    'model_bands': list(_MODEL_BANDS),
    # ── Calibration ──────────────────────────────────────────────────────
    'temperature': float(T_cal),
    'x_mean': _x_mean,
    'x_std':  _x_std,
    # ── Physics ──────────────────────────────────────────────────────────
    'phenology_prior': build_phenology_transition_matrix(
        self_loop=0.85, forward=0.15, wrap_to_dormancy=0.05
    ),
    # ── Metrics ──────────────────────────────────────────────────────────
    'metrics': {
        'lgb_cv_f1':  [float(x) for x in lgb_cv_f1],
        'dyn_cv_f1':  [float(x) for x in dyn_cv_f1],
        'meta_cv_f1': [float(x) for x in meta_cv_f1],
        'meta_cv_f1_mean': float(np.mean(meta_cv_f1)),
        'ensemble_f1_train': float(ens_meta_f1),
        'ensemble_oa_train': float(ens_meta_oa),
    },
    # ── Bookkeeping ──────────────────────────────────────────────────────
    'ensemble_seeds': ENSEMBLE_SEEDS,
    'n_points': n_points,
    'final_epochs': FINAL_EPOCHS,
    'version': VERSION,
    'run_tag': RUN_TAG,
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
}

torch.save(_ckpt, model_path)
sz = Path(model_path).stat().st_size / 1e6
print(f'[saved] {model_path}  ({sz:.1f} MB)')

# Audit
_audit = torch.load(model_path, map_location='cpu', weights_only=False)
_required = ['model_state_dict', 'config', 'ensemble_state_dicts',
             'lgb_model_bytes', 'meta_model_bytes', 'model_version',
             'crop_classes', 'feature_names']
_missing = [k for k in _required if k not in _audit]
assert not _missing, f'Checkpoint missing: {_missing}'
print('Checkpoint audit OK.')

print('\n' + '='*60)
print(f'DONE — v{VERSION} checkpoint: {model_path}')
print(f'CV F1m: LGB={np.mean(lgb_cv_f1):.4f}, DYN={np.mean(dyn_cv_f1):.4f}, '
      f'META={np.mean(meta_cv_f1):.4f}')
print('='*60)
print('\nNext steps:')
print(f'  1. Copy model: cp "{model_path}" '
      f'"C:/Users/jrpmc/Documents/_SkyVidya/repos/track3_olimarteixeiraborges/models/dynamis_terra_v10.pt"')
print(f'  2. Confirm MODEL_PATH in inference.py points to /workspace/models/dynamis_terra_v10.pt')
print(f'  3. git add models/dynamis_terra_v10.pt inference.py && git commit && git push')
