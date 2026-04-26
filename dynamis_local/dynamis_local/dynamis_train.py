"""Stage 4 — Dynamis CV + calibration + OOD + final model + checkpoint.

Reads series_list cache from build stage. Outputs:
  REPORTS_DIR/dynamis_metrics.json  { CV folds, temperature, OOD threshold }
  REPORTS_DIR/dynamis_oof.npz       OOF logits/probs/preds for the report
  MODELS_DIR/dynamis_terra_v{V}.pt  checkpoint with everything inference needs

Idempotent: if checkpoint exists and --force isn't set, skip.

The v6 audit fixes are preserved:
  - #3: final model trained on 100% of data (no held-out val at the end)
  - #4: checkpoint carries T, ood_threshold, norm stats, bands, etc.
  - #6: pheno labels passed as -100 sentinel (fallback to clamp if loss rejects)
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from . import config, build, features

# torch is imported lazily inside run() so that CLI --help works in
# environments where torch is not installed.


# ---------------------------------------------------------------------------
# Fold training
# ---------------------------------------------------------------------------
def _class_weights_from_labels(labels: np.ndarray, n_classes: int, device: str):
    import torch
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    counts = np.clip(counts, 1, None)
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32, device=device)


def _sampler_from_labels(labels: np.ndarray, n_classes: int):
    from torch.utils.data import WeightedRandomSampler
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    counts = np.clip(counts, 1, None)
    per_class_w = 1.0 / counts
    sample_w = per_class_w[labels]
    return WeightedRandomSampler(sample_w, num_samples=len(labels), replacement=True)


def _train_fold(
    X_tr, m_tr, h_tr, c_tr, p_tr,
    X_va, m_va, h_va, c_va, p_va,
    *, device, epochs=40, batch_size=16, lr=5e-4, weight_decay=5e-4,
    lambda_innovation=0.05, lambda_ece=0.02, verbose=True,
):
    """Train one fold. Returns (best_model, pred, probs, logits, unc, out, history)."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import precision_score, recall_score, f1_score
    from src.models import DynamisCropClassifier, DynamisModelConfig
    from src.dynamis import dynamis_loss, PHENOPHASES

    cfg = DynamisModelConfig(
        input_dim=X_tr.shape[-1], state_dim=len(PHENOPHASES),
        hidden_dim=64, attn_heads=4, n_crops=3, crop_head_dropout=0.3,
    )
    model = DynamisCropClassifier(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    crop_w = _class_weights_from_labels(c_tr, n_classes=3, device=device)
    sampler = _sampler_from_labels(c_tr, n_classes=3)

    ds = TensorDataset(
        torch.from_numpy(X_tr).float(), torch.from_numpy(m_tr).bool(),
        torch.from_numpy(h_tr).float(), torch.from_numpy(c_tr).long(),
        torch.from_numpy(p_tr).long(),
    )
    dl = DataLoader(ds, batch_size=batch_size, sampler=sampler)

    best_f1, best_state = -1.0, None
    history = []
    warned_fallback = False
    for ep in range(epochs):
        model.train()
        total, n = 0.0, 0
        for xb, mb, hb, cb, pb in dl:
            xb = xb.to(device); mb = mb.to(device); hb = hb.to(device)
            cb = cb.to(device); pb = pb.to(device)
            out = model(xb, mask=mb, hurst=hb)
            pl_flat = out['pheno_logits'].reshape(-1, len(PHENOPHASES))
            pb_flat = pb.reshape(-1)
            # v6 FIX #6: prefer -100 sentinel, fallback to clamp(min=0)
            is_rice = (cb == 0).unsqueeze(1).expand(-1, pb.size(1)).reshape(-1)
            try:
                loss_d = dynamis_loss(
                    out['crop_logits'], cb, pl_flat, pb_flat, out['innovations'],
                    lambda_innovation=lambda_innovation, lambda_ece=lambda_ece,
                    class_weights_crop=crop_w, is_rice=is_rice
                )
            except (RuntimeError, IndexError, AssertionError):
                if not warned_fallback:
                    print('  [fold #6] dynamis_loss fallback to clamp(min=0)')
                    warned_fallback = True
                loss_d = dynamis_loss(
                    out['crop_logits'], cb, pl_flat, pb_flat.clamp(min=0),
                    out['innovations'],
                    lambda_innovation=lambda_innovation, lambda_ece=lambda_ece,
                    class_weights_crop=crop_w, is_rice=is_rice
                )
            loss = loss_d['total']
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total += loss.item() * xb.size(0); n += xb.size(0)
        sched.step()

        model.eval()
        with torch.no_grad():
            out_v = model(
                torch.from_numpy(X_va).float().to(device),
                mask=torch.from_numpy(m_va).bool().to(device),
                hurst=torch.from_numpy(h_va).float().to(device),
            )
        pred_v = out_v['crop_logits'].argmax(-1).cpu().numpy()
        prec = precision_score(c_va, pred_v, average=None, labels=[0, 1, 2], zero_division=0)
        rec = recall_score(c_va, pred_v, average=None, labels=[0, 1, 2], zero_division=0)
        f1m = f1_score(c_va, pred_v, average='macro', zero_division=0)
        history.append({
            'epoch': ep + 1, 'loss': total / max(n, 1),
            'prec': prec.tolist(), 'rec': rec.tolist(),
            'f1_macro': float(f1m),
        })
        if f1m > best_f1:
            best_f1 = f1m
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f'  ep{ep+1:02d} loss={total/max(n,1):.3f} F1m={f1m:.3f}')

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(
            torch.from_numpy(X_va).float().to(device),
            mask=torch.from_numpy(m_va).bool().to(device),
            hurst=torch.from_numpy(h_va).float().to(device),
        )
    logits = out['crop_logits'].cpu().numpy()
    probs = F.softmax(out['crop_logits'], dim=-1).cpu().numpy()
    pred = probs.argmax(-1)
    unc = out['uncertainty'].cpu().numpy()
    print(f'  [fold best F1m={best_f1:.3f}]')
    return model, pred, probs, logits, unc, out, history


# ---------------------------------------------------------------------------
# Final model (no held-out val — 100% of data)
# ---------------------------------------------------------------------------
def _train_final_model(
    X, mask, hurst_vec, crop_labels, pheno_labels,
    *, device, epochs, batch_size=16, lr=5e-4, weight_decay=5e-4,
    lambda_innovation=0.05, lambda_ece=0.02,
):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from src.models import DynamisCropClassifier, DynamisModelConfig
    from src.dynamis import dynamis_loss, PHENOPHASES

    cfg = DynamisModelConfig(
        input_dim=X.shape[-1], state_dim=len(PHENOPHASES),
        hidden_dim=64, attn_heads=4, n_crops=3, crop_head_dropout=0.3,
    )
    model = DynamisCropClassifier(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    crop_w = _class_weights_from_labels(crop_labels, n_classes=3, device=device)
    sampler = _sampler_from_labels(crop_labels, n_classes=3)
    ds = TensorDataset(
        torch.from_numpy(X).float(), torch.from_numpy(mask).bool(),
        torch.from_numpy(hurst_vec).float(), torch.from_numpy(crop_labels).long(),
        torch.from_numpy(pheno_labels).long(),
    )
    dl = DataLoader(ds, batch_size=batch_size, sampler=sampler)

    warned_fallback = False
    for ep in range(epochs):
        model.train()
        total, n = 0.0, 0
        for xb, mb, hb, cb, pb in dl:
            xb = xb.to(device); mb = mb.to(device); hb = hb.to(device)
            cb = cb.to(device); pb = pb.to(device)
            out = model(xb, mask=mb, hurst=hb)
            pl_flat = out['pheno_logits'].reshape(-1, len(PHENOPHASES))
            pb_flat = pb.reshape(-1)
            is_rice = (cb == 0).unsqueeze(1).expand(-1, pb.size(1)).reshape(-1)
            try:
                loss_d = dynamis_loss(
                    out['crop_logits'], cb, pl_flat, pb_flat, out['innovations'],
                    lambda_innovation=lambda_innovation, lambda_ece=lambda_ece,
                    class_weights_crop=crop_w, is_rice=is_rice
                )
            except (RuntimeError, IndexError, AssertionError):
                if not warned_fallback:
                    print('  [final #6] fallback to clamp(min=0)')
                    warned_fallback = True
                loss_d = dynamis_loss(
                    out['crop_logits'], cb, pl_flat, pb_flat.clamp(min=0),
                    out['innovations'],
                    lambda_innovation=lambda_innovation, lambda_ece=lambda_ece,
                    class_weights_crop=crop_w, is_rice=is_rice
                )
            loss = loss_d['total']
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total += loss.item() * xb.size(0); n += xb.size(0)
        sched.step()
        if ep == 0 or (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f'  final ep{ep+1:02d} loss={total/max(n,1):.3f}')
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def run(*, force: bool = False) -> None:
    import torch
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import (
        accuracy_score, cohen_kappa_score, f1_score, roc_auc_score, confusion_matrix,
    )
    from src.dynamis import build_phenology_transition_matrix, PHENOPHASES
    from src.training import (
        apply_temperature, expected_calibration_error_np, temperature_scale,
    )
    from src.data import FEATURE_NAMES, MODEL_BANDS

    print(f'[dynamis] version: v{config.VERSION}')
    ckpt_path = config.final_checkpoint_path()
    metrics_path = config.dynamis_metrics_path()

    if not force and ckpt_path.exists() and metrics_path.exists():
        print(f'[dynamis] already complete: {ckpt_path} (use --force to rebuild)')
        return

    device = config.pick_device()
    torch.manual_seed(42); np.random.seed(42)

    series_list = build.load_cache()
    print(f'[dynamis] loaded {len(series_list)} series')
    packed = features.pack(series_list)
    X, mask = packed.X, packed.mask
    hurst_vec = packed.hurst_vec
    crop_labels, pheno_labels = packed.crop_labels, packed.pheno_labels

    # Spatial CV setup
    SPLIT_BY = 'region'
    groups = np.array([ps.region for ps in series_list])
    n_groups = len(np.unique(groups))
    n_splits = max(2, min(5, n_groups))
    kf = GroupKFold(n_splits=n_splits)
    print(f'[dynamis] GroupKFold(split_by={SPLIT_BY!r}, n_splits={n_splits})')

    # CV loop
    dyn_metrics = {'crop': {'oa': [], 'kappa': [], 'f1': []}, 'uncertainty': []}
    dyn_preds_all = np.zeros_like(crop_labels)
    dyn_probs_all = np.zeros((len(crop_labels), 3), dtype=np.float32)
    dyn_logits_all = np.zeros((len(crop_labels), 3), dtype=np.float32)
    dyn_unc_all = np.zeros(len(crop_labels), dtype=np.float32)
    last_hist = None
    for fold, (tr, va) in enumerate(kf.split(X, crop_labels, groups=groups)):
        print(f'\n[dynamis] --- Fold {fold+1} (train={len(tr)}, val={len(va)}) ---')
        _, pred, probs, logits, unc, _, hist = _train_fold(
            X[tr], mask[tr], hurst_vec[tr], crop_labels[tr], pheno_labels[tr],
            X[va], mask[va], hurst_vec[va], crop_labels[va], pheno_labels[va],
            device=device,
        )
        dyn_preds_all[va] = pred
        dyn_probs_all[va] = probs
        dyn_logits_all[va] = logits
        dyn_unc_all[va] = unc
        dyn_metrics['crop']['oa'].append(accuracy_score(crop_labels[va], pred))
        dyn_metrics['crop']['kappa'].append(cohen_kappa_score(crop_labels[va], pred))
        dyn_metrics['crop']['f1'].append(
            f1_score(crop_labels[va], pred, average='macro', zero_division=0))
        dyn_metrics['uncertainty'].append(float(np.mean(unc)))
        last_hist = hist

    print('\n[dynamis] CV summary:')
    for k, v in dyn_metrics['crop'].items():
        print(f'  {k}: {np.mean(v):.4f} ± {np.std(v):.4f}')
    confusion = confusion_matrix(crop_labels, dyn_preds_all)
    print(f'[dynamis] confusion:')
    print(pd.DataFrame(confusion, index=features.CROPS, columns=features.CROPS))

    # Temperature scaling
    ece_pre = expected_calibration_error_np(dyn_probs_all, crop_labels, n_bins=10)
    T = temperature_scale(dyn_logits_all, crop_labels, steps=500, lr=1e-2)
    dyn_probs_calibrated = apply_temperature(dyn_logits_all, T)
    ece_post = expected_calibration_error_np(dyn_probs_calibrated, crop_labels, n_bins=10)
    print(f'[dynamis] ECE {ece_pre:.4f} → {ece_post:.4f}, T={T:.4f}')
    assert np.all(dyn_probs_calibrated.argmax(-1) == dyn_probs_all.argmax(-1))

    # OOD threshold via trace(P) vs error AUC
    errors_binary = (dyn_preds_all != crop_labels).astype(int)
    if errors_binary.sum() in (0, len(errors_binary)):
        ood_auc, ood_threshold = float('nan'), float('nan')
        print('[dynamis] [skip OOD] all predictions correct or all wrong')
    else:
        ood_auc = float(roc_auc_score(errors_binary, dyn_unc_all))
        ood_threshold = float(np.percentile(dyn_unc_all, 90))
        print(f'[dynamis] OOD AUC={ood_auc:.4f}, threshold(p90)={ood_threshold:.3f}')

    # Final model on 100% data
    try:
        final_epochs = max(last_hist, key=lambda r: r['f1_macro'])['epoch']
    except (TypeError, ValueError, KeyError):
        final_epochs = 40
    print(f'\n[dynamis] final model: {final_epochs} epochs on all {len(crop_labels)} points')
    final_model = _train_final_model(
        X, mask, hurst_vec, crop_labels, pheno_labels,
        device=device, epochs=final_epochs,
    )

    # Checkpoint
    try:
        from src.data.sentinel2_loader import MODEL_BANDS as _MODEL_BANDS
        model_bands = list(_MODEL_BANDS)
    except Exception:
        model_bands = list(MODEL_BANDS) if 'MODEL_BANDS' in dir() else None

    flat_valid = X[mask]
    x_mean = flat_valid.mean(axis=0).astype('float32') if flat_valid.size else None
    x_std = flat_valid.std(axis=0).astype('float32') if flat_valid.size else None

    ckpt = {
        'model_state_dict': final_model.state_dict(),
        'config': asdict(final_model.cfg),
        'phenology_prior': build_phenology_transition_matrix(),
        # inference-time artifacts
        'feature_names': list(FEATURE_NAMES),
        'model_bands': model_bands,
        'crop_classes': features.CROPS,
        'phenophase_classes': list(PHENOPHASES),
        'temperature': float(T),
        'ood_threshold': ood_threshold if not np.isnan(ood_threshold) else None,
        'x_mean': x_mean,
        'x_std': x_std,
        # provenance
        'metrics': {
            'dynamis_oa':    [float(x) for x in dyn_metrics['crop']['oa']],
            'dynamis_kappa': [float(x) for x in dyn_metrics['crop']['kappa']],
            'dynamis_f1':    [float(x) for x in dyn_metrics['crop']['f1']],
            'ece_pre': float(ece_pre),
            'ece_post': float(ece_post),
            'ood_auc': ood_auc if not np.isnan(ood_auc) else None,
        },
        'split_by': SPLIT_BY,
        'n_splits': int(n_splits),
        'final_epochs': int(final_epochs),
        'version': config.VERSION,
        'run_tag': config.RUN_TAG,
        'timestamp': pd.Timestamp.now().isoformat(),
    }
    torch.save(ckpt, ckpt_path)
    print(f'[dynamis] checkpoint saved: {ckpt_path}')

    # Audit — fail loud if anything critical missing
    required = ['model_state_dict', 'config', 'temperature', 'feature_names',
                'crop_classes', 'x_mean', 'x_std', 'split_by', 'version']
    reloaded = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    missing = [k for k in required if k not in reloaded]
    if missing:
        raise AssertionError(f'[dynamis] checkpoint missing keys: {missing}')
    print(f'[dynamis] checkpoint audit: {len(required)} required keys present ✓')

    # Persist CV metrics + OOF predictions for the report stage
    with open(metrics_path, 'w') as f:
        json.dump({
            'version': config.VERSION,
            'split_by': SPLIT_BY,
            'n_splits': int(n_splits),
            'metrics': {k: [float(x) for x in v] for k, v in dyn_metrics['crop'].items()},
            'uncertainty_mean': [float(x) for x in dyn_metrics['uncertainty']],
            'ece_pre': float(ece_pre),
            'ece_post': float(ece_post),
            'temperature': float(T),
            'ood_auc': ood_auc if not np.isnan(ood_auc) else None,
            'ood_threshold': ood_threshold if not np.isnan(ood_threshold) else None,
            'confusion': {
                'classes': features.CROPS,
                'matrix': confusion.tolist(),
            },
            'final_epochs': int(final_epochs),
        }, f, indent=2)
    print(f'[dynamis] metrics → {metrics_path}')

    np.savez(
        config.dynamis_oof_path(),
        preds=dyn_preds_all,
        probs=dyn_probs_all,
        logits=dyn_logits_all,
        unc=dyn_unc_all,
        labels=crop_labels,
        groups=groups,
    )
    print(f'[dynamis] OOF arrays → {config.dynamis_oof_path()}')


def is_complete() -> bool:
    return (config.final_checkpoint_path().exists()
            and config.dynamis_metrics_path().exists())
