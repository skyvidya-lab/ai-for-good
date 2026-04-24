"""Stage 3 — baseline LightGBM cross-validation.

Reads series_list cache from build stage. Outputs:
  REPORTS_DIR/baseline_metrics.json   { oa, kappa, f1 folds + confusion matrix }

Idempotent: if the metrics file exists and --force isn't set, skip.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config, build, features


def _save_metrics(out_path, baseline_metrics, confusion, y_true, y_pred, groups,
                  split_by, n_splits):
    """Persist baseline outputs as a single JSON for later stages to read."""
    data = {
        'version': config.VERSION,
        'split_by': split_by,
        'n_splits': int(n_splits),
        'metrics': {
            k: [float(x) for x in v] for k, v in baseline_metrics['crop'].items()
        },
        'confusion': {
            'classes': features.CROPS,
            'matrix': confusion.tolist(),
        },
        'predictions': {
            'crop_labels': y_true.tolist(),
            'crop_pred': y_pred.tolist(),
            'groups': [str(g) for g in groups.tolist()],
        },
    }
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'[baseline] saved metrics → {out_path}')


def run(*, force: bool = False) -> None:
    print(f'[baseline] version: v{config.VERSION}')

    out_path = config.baseline_metrics_path()
    if not force and out_path.exists():
        print(f'[baseline] already complete: {out_path} (use --force to rebuild)')
        return

    # Lazy imports (require src)
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import (
        accuracy_score, cohen_kappa_score, f1_score, confusion_matrix,
    )
    from src.data import batch_phenology_features

    series_list = build.load_cache()
    print(f'[baseline] loaded {len(series_list)} series from cache')

    packed = features.pack(series_list)
    X = packed.X
    mask = packed.mask
    hurst_vec = packed.hurst_vec
    crop_labels = packed.crop_labels

    # Flat features: stats + phenology + hurst
    X_stats = features.flatten(X, mask, series_list=series_list)
    X_pheno = batch_phenology_features(series_list, hurst_vec)
    X_flat = np.concatenate(
        [X_stats, X_pheno, hurst_vec.reshape(-1, 1)], axis=1,
    )
    print(
        f'[baseline] flat features: {X_flat.shape} '
        f'(stats: {X_stats.shape[1]}, pheno: {X_pheno.shape[1]}, hurst: 1)'
    )

    # Spatial CV by region (v5 fix — never revert to point-level grouping)
    SPLIT_BY = 'region'
    groups = np.array([ps.region for ps in series_list])
    n_groups = len(np.unique(groups))
    n_splits = max(2, min(5, n_groups))
    if n_groups < 3:
        print(f'[baseline] [warn] only {n_groups} groups — spatial CV degenerate')
    kf = GroupKFold(n_splits=n_splits)
    print(f'[baseline] GroupKFold(split_by={SPLIT_BY!r}, n_splits={n_splits}, '
          f'n_groups={n_groups})')

    # Spatial-CV diagnostic
    for fold, (tr, va) in enumerate(kf.split(X_flat, crop_labels, groups=groups)):
        tr_r, va_r = set(groups[tr]), set(groups[va])
        overlap = '❌ LEAK' if (tr_r & va_r) else '✓ clean'
        print(f'[baseline]   fold {fold+1}: val={sorted(va_r)} | {overlap}')

    # CV loop
    baseline_metrics = {'crop': {'oa': [], 'kappa': [], 'f1': []}}
    bl_pred_all = np.zeros_like(crop_labels)
    for fold, (tr, va) in enumerate(kf.split(X_flat, crop_labels, groups=groups)):
        model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8,
            class_weight='balanced',
            random_state=42, verbose=-1,
        )
        model.fit(X_flat[tr], crop_labels[tr])
        pred = model.predict(X_flat[va])
        bl_pred_all[va] = pred
        baseline_metrics['crop']['oa'].append(
            accuracy_score(crop_labels[va], pred))
        baseline_metrics['crop']['kappa'].append(
            cohen_kappa_score(crop_labels[va], pred))
        baseline_metrics['crop']['f1'].append(
            f1_score(crop_labels[va], pred, average='macro', zero_division=0))

    print('\n[baseline] (crop_type) summary:')
    for k, v in baseline_metrics['crop'].items():
        print(f'  {k}: {np.mean(v):.4f} ± {np.std(v):.4f}')

    confusion = confusion_matrix(crop_labels, bl_pred_all)
    print(f'\n[baseline] confusion:')
    print(pd.DataFrame(confusion, index=features.CROPS, columns=features.CROPS))

    _save_metrics(out_path, baseline_metrics, confusion, crop_labels, bl_pred_all,
                  groups, SPLIT_BY, n_splits)


def is_complete() -> bool:
    return config.baseline_metrics_path().exists()
