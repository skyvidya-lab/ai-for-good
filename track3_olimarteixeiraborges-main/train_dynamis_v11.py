#!/usr/bin/env python3
"""
train_dynamis_v11.py — Stacked ensemble v11: pheno class-balanced + stronger pheno signal.

Key fixes vs v10
================
  1. Pheno class weights — Dormancy=96.7% of labels → model always predicted it.
     Now each of the 7 phenophase classes gets inverse-frequency weighting so the
     model is forced to learn Greenup / Peak / Maturity etc.
  2. lambda_pheno = 8.0 (was 4.0) — heavier phenophase supervision.
  3. epochs = 80 (was 50) — more gradient steps on rare phenophases.
  4. Stacking architecture for crop unchanged (LightGBM + DynamisV9 + meta).

Paths — EDIT THESE before running on a different machine:
=========================================================
  REPO_PATH  — root of the cloned repository
  CACHE_DIR  — folder with train_arrays_v8.npz and train_series_v8.pkl
  MODELS_DIR — where to save the final .pt checkpoint
"""
import os, sys, re, copy, json, time, pickle, warnings, argparse
from pathlib import Path
from dataclasses import asdict
from datetime import datetime as _dt

warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths — resolved via CLI args, env vars, or auto-detect relative to script
# ---------------------------------------------------------------------------
_here = Path(__file__).resolve().parent   # folder where this script lives

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument('--repo',     default=None)
_ap.add_argument('--cache',    default=None)
_ap.add_argument('--models',   default=None)
_ap.add_argument('--reports',  default=None)
_args, _ = _ap.parse_known_args()

REPO_PATH   = Path(_args.repo)    if _args.repo    else Path(os.environ.get('REPO_PATH',   str(_here)))
CACHE_DIR   = Path(_args.cache)   if _args.cache   else Path(os.environ.get('CACHE_DIR',   str(_here / 'cache')))
MODELS_DIR  = Path(_args.models)  if _args.models  else Path(os.environ.get('MODELS_DIR',  str(_here / 'models')))
REPORTS_DIR = Path(_args.reports) if _args.reports else Path(os.environ.get('REPORTS_DIR', str(_here / 'reports' / 'v11')))

for _d in [CACHE_DIR, MODELS_DIR, REPORTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

print(f'REPO_PATH:  {REPO_PATH}')
print(f'CACHE_DIR:  {CACHE_DIR}')
print(f'MODELS_DIR: {MODELS_DIR}')

VERSION        = 11
LAMBDA_PHENO   = 8.0    # v11: doubled vs v10 (4.0) — forces pheno learning
FINAL_EPOCHS   = 80     # v11: more epochs for rare phenophases
ENSEMBLE_SEEDS = [42, 137, 271, 503, 777]

sys.path.insert(0, str(REPO_PATH))

print(f'Run tag:       v{VERSION}_stacked_pheno_balanced')
print(f'LAMBDA_PHENO:  {LAMBDA_PHENO}')
print(f'FINAL_EPOCHS:  {FINAL_EPOCHS}')
print(f'Seeds:         {ENSEMBLE_SEEDS}')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}' + (f' | {torch.cuda.get_device_name(0)}' if DEVICE == 'cuda' else ''))

# ---------------------------------------------------------------------------
# Load cache
# ---------------------------------------------------------------------------
print('\n=== Loading cache ===')
import pickle as _pkl

_c          = np.load(str(CACHE_DIR / 'train_arrays_v8.npz'))
X           = _c['X']
mask        = _c['mask']
hurst_vec   = _c['hurst_vec']
crop_labels  = _c['crop_labels']
pheno_labels = _c['pheno_labels']   # (764, 29)  — -100 = unknown

with open(str(CACHE_DIR / 'train_series_v8.pkl'), 'rb') as _f:
    series_list = _pkl.load(_f)

CROPS     = ['rice', 'corn', 'soybean']
n_points  = X.shape[0]

from src.dynamis import PHENOPHASES, phenophase_name_to_index
from src.data   import FEATURE_NAMES, N_FEATURES as _NF, batch_phenology_features, PHENO_FEATURE_NAMES
N_FEATURES = _NF
N_PHENO    = len(PHENOPHASES)

print(f'X={X.shape} | mask={mask.shape}')
print(f'Crops: {dict(zip(CROPS, np.bincount(crop_labels, minlength=3).tolist()))}')

# ---- Pheno class distribution (this is why F1_RicePheno collapsed) --------
_flat_ph  = pheno_labels.flatten()
_valid_ph = _flat_ph[_flat_ph != -100]
print('\nPheno class distribution (training labels):')
_pheno_counts = np.zeros(N_PHENO, dtype=np.int64)
for i, name in enumerate(PHENOPHASES):
    cnt = int((_valid_ph == i).sum())
    _pheno_counts[i] = cnt
    print(f'  {i} {name}: {cnt} ({cnt/len(_valid_ph)*100:.1f}%)')

# ---- Pheno class weights (inverse frequency, clipped) ----------------------
# This is the critical fix: Dormancy=96.7% → very low weight;
# Peak=0.3% → very high weight. Forces model to learn rare phenophases.
_total_ph = len(_valid_ph)
_pheno_w  = _total_ph / (N_PHENO * np.clip(_pheno_counts, 1, None).astype(np.float32))
_pheno_w  = np.clip(_pheno_w, 0.1, 50.0)   # cap to avoid exploding gradients
pheno_class_weights = torch.tensor(_pheno_w, dtype=torch.float32, device=DEVICE)
print('\nPheno class weights (clipped to [0.1, 50]):')
for i, (name, w) in enumerate(zip(PHENOPHASES, _pheno_w)):
    print(f'  {i} {name}: {w:.2f}')


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
    sample_w = (1.0 / counts)[labels_arr]
    return WeightedRandomSampler(sample_w, num_samples=len(labels_arr), replacement=True)


def temperature_scale(logits_arr, labels_arr, steps=500, lr=1e-2):
    logits_t = torch.from_numpy(logits_arr).float()
    labels_t = torch.from_numpy(labels_arr).long()
    T_param  = nn.Parameter(torch.ones(1))
    opt_T    = torch.optim.LBFGS([T_param], lr=lr, max_iter=steps)
    def _eval():
        opt_T.zero_grad()
        loss = F.cross_entropy(logits_t / T_param.clamp(min=0.1), labels_t)
        loss.backward(); return loss
    opt_T.step(_eval)
    return float(T_param.clamp(min=0.1).item())


def apply_temperature(logits_arr, T):
    return F.softmax(torch.from_numpy(logits_arr).float() / max(T, 0.1), dim=-1).numpy()


# ---------------------------------------------------------------------------
# v10 flat features (9 stat blocks × N_FEATURES + 1 + pheno + hurst = 165)
# ---------------------------------------------------------------------------
def flatten_features_v10(X_arr, mask_arr, series_list_=None):
    n, T, F_dim = X_arr.shape
    out = np.zeros((n, F_dim * 9 + 1), dtype=np.float32)
    for i in range(n):
        valid = mask_arr[i]; n_v = int(valid.sum())
        out[i, F_dim * 9] = n_v
        if n_v < 1: continue
        Xi = X_arr[i, valid]
        out[i, 0*F_dim:1*F_dim] = Xi.mean(axis=0)
        out[i, 1*F_dim:2*F_dim] = Xi.std(axis=0)
        out[i, 2*F_dim:3*F_dim] = Xi.max(axis=0)
        out[i, 3*F_dim:4*F_dim] = Xi.min(axis=0)
        if n_v >= 2:
            if series_list_ is not None:
                try:
                    ps = series_list_[i]
                    vd = [ps.dates[t] for t in range(len(ps.dates)) if mask_arr[i, t]]
                    span = (_dt.strptime(_canonical_date(vd[-1]), '%Y-%m-%d') -
                            _dt.strptime(_canonical_date(vd[0]), '%Y-%m-%d')).days
                    out[i, 4*F_dim:5*F_dim] = (Xi[-1]-Xi[0]) / span if span > 0 else (Xi[-1]-Xi[0]) / max(n_v-1,1)
                except Exception:
                    out[i, 4*F_dim:5*F_dim] = (Xi[-1]-Xi[0]) / max(n_v-1,1)
            else:
                out[i, 4*F_dim:5*F_dim] = (Xi[-1]-Xi[0]) / max(n_v-1,1)
        if n_v >= 4:
            out[i, 5*F_dim:6*F_dim] = np.percentile(Xi, 10, axis=0)
            out[i, 6*F_dim:7*F_dim] = np.percentile(Xi, 25, axis=0)
            out[i, 7*F_dim:8*F_dim] = np.percentile(Xi, 75, axis=0)
            out[i, 8*F_dim:9*F_dim] = np.percentile(Xi, 90, axis=0)
    return out


X_stats = flatten_features_v10(X, mask, series_list_=series_list)
X_pheno = batch_phenology_features(series_list, hurst_vec)
X_flat  = np.concatenate([X_stats, X_pheno, hurst_vec.reshape(-1, 1)], axis=1)
print(f'\nFlat features: {X_flat.shape}')

groups   = np.array([ps.region for ps in series_list])
n_groups = len(np.unique(groups))
n_splits = max(2, min(5, n_groups))
kf       = GroupKFold(n_splits=n_splits)
print(f'GroupKFold: {n_splits} folds | {n_groups} groups')

# ---------------------------------------------------------------------------
# DynamisV9 — training with pheno class weights
# ---------------------------------------------------------------------------
from src.models import DynamisTerraV9, DynamisV9Config
from src.dynamis import dynamis_loss


def build_v9_model(seed=None):
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed)
    cfg = DynamisV9Config(
        input_dim=N_FEATURES, state_dim=N_PHENO, hidden_dim=128,
        attn_heads=4, n_crops=3, T_slow=4, crop_head_dropout=0.3,
        lambda_prior_strength=0.7, prior_self_loop=0.85, prior_forward=0.15,
    )
    return DynamisTerraV9(cfg).to(DEVICE)


def train_v9_fold_v11(
    X_tr, m_tr, h_tr, c_tr, p_tr,
    X_va, m_va, h_va, c_va, p_va,
    seed=42, epochs=FINAL_EPOCHS, batch_size=16, lr=3e-4, weight_decay=5e-4,
    verbose=True,
):
    model   = build_v9_model(seed=seed)
    opt     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crop_w  = class_weights_from_labels(c_tr, n_classes=3)
    sampler = sampler_from_labels(c_tr, n_classes=3)

    # Compute per-fold pheno class weights from training split only
    _pfl = p_tr.flatten(); _pfl_v = _pfl[_pfl != -100]
    if len(_pfl_v) > 0:
        _pc = np.zeros(N_PHENO, dtype=np.int64)
        for _i in range(N_PHENO): _pc[_i] = int((_pfl_v == _i).sum())
        _pw = len(_pfl_v) / (N_PHENO * np.clip(_pc, 1, None).astype(np.float32))
        _pw = np.clip(_pw, 0.1, 50.0)
        fold_pheno_w = torch.tensor(_pw, dtype=torch.float32, device=DEVICE)
    else:
        fold_pheno_w = pheno_class_weights

    ds = TensorDataset(
        torch.from_numpy(X_tr).float(), torch.from_numpy(m_tr).bool(),
        torch.from_numpy(h_tr).float(), torch.from_numpy(c_tr).long(),
        torch.from_numpy(p_tr).long(),
    )
    dl = DataLoader(ds, batch_size=batch_size, sampler=sampler)

    best_f1 = -1.0; best_state = None

    for ep in range(epochs):
        model.train()
        for xb, mb, hb, cb, pb in dl:
            xb=xb.to(DEVICE); mb=mb.to(DEVICE); hb=hb.to(DEVICE)
            cb=cb.to(DEVICE); pb=pb.to(DEVICE)
            out = model(xb, mask=mb, hurst=hb)
            pheno_flat  = out['pheno_logits'].reshape(-1, N_PHENO)
            plabel_flat = pb.reshape(-1)
            loss_dict = dynamis_loss(
                out['crop_logits'], cb,
                pheno_flat, plabel_flat,
                out['innovations'],
                lambda_innovation=0.05, lambda_ece=0.02,
                class_weights_crop=crop_w,
                class_weights_pheno=fold_pheno_w,   # KEY FIX
            )
            loss = loss_dict['total'] + (LAMBDA_PHENO - 1.0) * loss_dict['ce_pheno']
            opt.zero_grad(); loss.backward()
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
        f1m = f1_score(c_va, pred_v, average='macro', zero_division=0)

        # Pheno accuracy on non-dormancy classes (the ones the server actually tests)
        ph_pred = out_v['pheno_logits'].cpu().numpy().argmax(-1).reshape(-1)
        ph_true = p_va.reshape(-1)
        _vm = ph_true != -100
        ph_acc_all = float((ph_pred[_vm] == ph_true[_vm]).mean()) if _vm.sum() > 0 else float('nan')
        _vm_nd = _vm & (ph_true != 0)   # exclude dormancy
        ph_acc_nd = float((ph_pred[_vm_nd] == ph_true[_vm_nd]).mean()) if _vm_nd.sum() > 0 else float('nan')

        if f1m > best_f1: best_f1 = f1m; best_state = copy.deepcopy(model.state_dict())

        if verbose and (ep == 0 or (ep+1) % 10 == 0 or ep == epochs-1):
            nd_str = f'{ph_acc_nd:.3f}' if not np.isnan(ph_acc_nd) else 'n/a'
            print(f'  ep{ep+1:02d} F1m={f1m:.3f} PhenoAll={ph_acc_all:.3f} PhenoNonDorm={nd_str}')

    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(
            torch.from_numpy(X_va).float().to(DEVICE),
            mask=torch.from_numpy(m_va).bool().to(DEVICE),
            hurst=torch.from_numpy(h_va).float().to(DEVICE),
        )
    logits = out['crop_logits'].cpu().numpy()
    pred   = logits.argmax(-1)
    print(f'  [best F1m={best_f1:.3f}]')
    return model, pred, logits, out


# ---------------------------------------------------------------------------
# OOF cross-validation
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('OOF cross-validation')
print('='*60)

import copy
lgb_proba_oof  = np.zeros((n_points, 3), dtype=np.float32)
dyn_logits_oof = np.zeros((n_points, 3), dtype=np.float32)

lgb_cv_f1 = []; dyn_cv_f1 = []

torch.manual_seed(42); np.random.seed(42)

for fold, (tr, va) in enumerate(kf.split(X_flat, crop_labels, groups=groups)):
    print(f'\n--- Fold {fold+1}/{n_splits} ---')

    # LightGBM
    model_lgb = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, class_weight='balanced',
        random_state=42, verbose=-1, n_jobs=-1,
    )
    model_lgb.fit(X_flat[tr], crop_labels[tr])
    lgb_proba_oof[va] = model_lgb.predict_proba(X_flat[va]).astype(np.float32)
    lgb_f1 = f1_score(crop_labels[va], lgb_proba_oof[va].argmax(-1), average='macro', zero_division=0)
    lgb_cv_f1.append(lgb_f1)
    print(f'  LGB F1m={lgb_f1:.4f}')

    # DynamisV9 with pheno class weights
    _, pred_va, logits_va, _ = train_v9_fold_v11(
        X[tr], mask[tr], hurst_vec[tr], crop_labels[tr], pheno_labels[tr],
        X[va], mask[va], hurst_vec[va], crop_labels[va], pheno_labels[va],
        seed=42, verbose=True,
    )
    dyn_logits_oof[va] = logits_va
    dyn_f1 = f1_score(crop_labels[va], pred_va, average='macro', zero_division=0)
    dyn_cv_f1.append(dyn_f1)
    print(f'  DYN F1m={dyn_f1:.4f}')

print(f'\nCV: LGB={np.mean(lgb_cv_f1):.4f} | DYN={np.mean(dyn_cv_f1):.4f}')

# Meta-learner
dyn_softmax_oof = F.softmax(torch.from_numpy(dyn_logits_oof).float(), dim=-1).numpy()
meta_X = np.concatenate([lgb_proba_oof, dyn_softmax_oof, dyn_logits_oof], axis=1).astype(np.float32)

meta_preds_oof = np.zeros(n_points, dtype=np.int64)
meta_cv_f1 = []
for fold, (tr, va) in enumerate(kf.split(meta_X, crop_labels, groups=groups)):
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                            num_leaves=7, class_weight='balanced', random_state=42, verbose=-1)
    m.fit(meta_X[tr], crop_labels[tr])
    meta_preds_oof[va] = m.predict(meta_X[va])
    meta_cv_f1.append(f1_score(crop_labels[va], meta_preds_oof[va], average='macro', zero_division=0))

print(f'Meta F1m = {np.mean(meta_cv_f1):.4f}')
print(pd.DataFrame(confusion_matrix(crop_labels, meta_preds_oof), index=CROPS, columns=CROPS))

meta_model_final = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                                        num_leaves=7, class_weight='balanced', random_state=42, verbose=-1)
meta_model_final.fit(meta_X, crop_labels)

T_cal = temperature_scale(dyn_logits_oof, crop_labels)
print(f'Temperature T={T_cal:.4f}')

# ---------------------------------------------------------------------------
# Full ensemble (all data, 5 seeds)
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print(f'Full ensemble — {len(ENSEMBLE_SEEDS)} seeds × {FINAL_EPOCHS} epochs')
print('='*60)

lgb_full = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=31,
                                subsample=0.8, colsample_bytree=0.8, class_weight='balanced',
                                random_state=42, verbose=-1, n_jobs=-1)
lgb_full.fit(X_flat, crop_labels)
print(f'LGB full F1={f1_score(crop_labels, lgb_full.predict(X_flat), average="macro", zero_division=0):.4f}')

ensemble_state_dicts = []
_crop_w = class_weights_from_labels(crop_labels, n_classes=3)

for seed in ENSEMBLE_SEEDS:
    print(f'\n--- Seed={seed} ---')
    torch.manual_seed(seed); np.random.seed(seed)
    _m   = build_v9_model(seed=seed)
    _opt = torch.optim.AdamW(_m.parameters(), lr=3e-4, weight_decay=5e-4)
    _sch = torch.optim.lr_scheduler.CosineAnnealingLR(_opt, T_max=FINAL_EPOCHS)
    _spl = sampler_from_labels(crop_labels, n_classes=3)
    _ds  = TensorDataset(
        torch.from_numpy(X).float(), torch.from_numpy(mask).bool(),
        torch.from_numpy(hurst_vec).float(), torch.from_numpy(crop_labels).long(),
        torch.from_numpy(pheno_labels).long(),
    )
    _dl = DataLoader(_ds, batch_size=16, sampler=_spl)

    for _ep in range(FINAL_EPOCHS):
        _m.train()
        _tot = 0.0; _n = 0
        for _xb, _mb, _hb, _cb, _pb in _dl:
            _xb=_xb.to(DEVICE); _mb=_mb.to(DEVICE); _hb=_hb.to(DEVICE)
            _cb=_cb.to(DEVICE); _pb=_pb.to(DEVICE)
            _out = _m(_xb, mask=_mb, hurst=_hb)
            _pf  = _out['pheno_logits'].reshape(-1, N_PHENO)
            _pbf = _pb.reshape(-1)
            _ld  = dynamis_loss(
                _out['crop_logits'], _cb, _pf, _pbf, _out['innovations'],
                lambda_innovation=0.05, lambda_ece=0.02,
                class_weights_crop=_crop_w,
                class_weights_pheno=pheno_class_weights,   # KEY FIX
            )
            _loss = _ld['total'] + (LAMBDA_PHENO - 1.0) * _ld['ce_pheno']
            _opt.zero_grad(); _loss.backward()
            torch.nn.utils.clip_grad_norm_(_m.parameters(), 1.0)
            _opt.step()
            _tot += _loss.item() * _xb.size(0); _n += _xb.size(0)
        _sch.step()
        if (_ep+1) % 10 == 0 or _ep == FINAL_EPOCHS-1:
            print(f'  ep{_ep+1:02d} loss={_tot/max(_n,1):.3f}')

    _m.eval()
    ensemble_state_dicts.append(copy.deepcopy(_m.state_dict()))
    print(f'  Seed {seed} done.')

# ---------------------------------------------------------------------------
# Save checkpoint
# ---------------------------------------------------------------------------
print('\n=== Saving checkpoint ===')
from src.dynamis import build_phenology_transition_matrix
from src.data.sentinel2_loader import MODEL_BANDS as _MB

_cfg = DynamisV9Config(input_dim=N_FEATURES, state_dim=N_PHENO, hidden_dim=128,
                        attn_heads=4, n_crops=3, T_slow=4, crop_head_dropout=0.3,
                        lambda_prior_strength=0.7, prior_self_loop=0.85, prior_forward=0.15)

_flat_valid = X[mask]
_ckpt = {
    'model_version': 'v10_stacked',   # same key — inference.py handles this version
    'model_state_dict': ensemble_state_dicts[0],
    'ensemble_state_dicts': ensemble_state_dicts,
    'config': asdict(_cfg),
    'lgb_model_bytes':  pickle.dumps(lgb_full),
    'meta_model_bytes': pickle.dumps(meta_model_final),
    'n_flat_features': int(X_flat.shape[1]),
    'feature_version': 'v10',
    'n_stat_blocks': 9,
    'crop_classes': CROPS,
    'phenophase_classes': list(PHENOPHASES),
    'feature_names': list(FEATURE_NAMES),
    'pheno_feature_names': list(PHENO_FEATURE_NAMES),
    'model_bands': list(_MB),
    'temperature': float(T_cal),
    'x_mean': _flat_valid.mean(axis=0).astype('float32') if _flat_valid.size else None,
    'x_std':  _flat_valid.std(axis=0).astype('float32')  if _flat_valid.size else None,
    'phenology_prior': build_phenology_transition_matrix(self_loop=0.85, forward=0.15, wrap_to_dormancy=0.05),
    'pheno_class_weights': _pheno_w.tolist(),
    'metrics': {
        'lgb_cv_f1':  [float(x) for x in lgb_cv_f1],
        'dyn_cv_f1':  [float(x) for x in dyn_cv_f1],
        'meta_cv_f1': [float(x) for x in meta_cv_f1],
    },
    'ensemble_seeds': ENSEMBLE_SEEDS,
    'n_points': n_points,
    'final_epochs': FINAL_EPOCHS,
    'lambda_pheno': LAMBDA_PHENO,
    'version': VERSION,
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
}

model_path = str(MODELS_DIR / f'dynamis_terra_v{VERSION}.pt')
torch.save(_ckpt, model_path)
sz = Path(model_path).stat().st_size / 1e6
print(f'[saved] {model_path}  ({sz:.1f} MB)')

print('\n' + '='*60)
print(f'DONE — v{VERSION}')
print(f'CV: LGB={np.mean(lgb_cv_f1):.4f} | DYN={np.mean(dyn_cv_f1):.4f} | META={np.mean(meta_cv_f1):.4f}')
print('='*60)
print(f'\nNext: copy {model_path}')
print(f'  to  <repo>/models/dynamis_terra_v11.pt')
print(f'  and update MODEL_PATH in inference.py to /workspace/models/dynamis_terra_v11.pt')
