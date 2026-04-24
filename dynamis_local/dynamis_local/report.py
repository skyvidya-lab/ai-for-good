"""Stage 5 — generate a markdown report from baseline + dynamis outputs.

Reads:
  REPORTS_DIR/baseline_metrics.json
  REPORTS_DIR/dynamis_metrics.json
  REPORTS_DIR/dynamis_oof.npz
  MODELS_DIR/dynamis_terra_v{V}.pt (for timestamp + provenance)

Writes:
  REPORTS_DIR/report.md

No plotting libraries needed — markdown tables only. If you want figures,
extend this with matplotlib and save PNGs alongside the MD.
"""
from __future__ import annotations

import json

import numpy as np

from . import config


def _fmt_mean_std(arr):
    return f'{np.mean(arr):.4f} ± {np.std(arr):.4f}'


def _confusion_md(matrix, classes):
    lines = ['| | ' + ' | '.join(classes) + ' |']
    lines.append('|---|' + '---|' * len(classes))
    for i, row in enumerate(matrix):
        lines.append(f'| **{classes[i]}** | ' + ' | '.join(str(v) for v in row) + ' |')
    return '\n'.join(lines)


def run(*, force: bool = False) -> None:
    out_path = config.report_md_path()
    if not force and out_path.exists():
        print(f'[report] already exists: {out_path} (use --force to rebuild)')
        return

    baseline_path = config.baseline_metrics_path()
    dynamis_path = config.dynamis_metrics_path()
    if not baseline_path.exists() or not dynamis_path.exists():
        raise FileNotFoundError(
            '[report] need baseline + dynamis metrics. Run those stages first.'
        )

    with open(baseline_path) as f:
        b = json.load(f)
    with open(dynamis_path) as f:
        d = json.load(f)

    lines = []
    lines.append(f'# Dynamis Terra Report — v{config.VERSION}')
    lines.append('')
    lines.append(f'- **Run tag:** `{config.RUN_TAG}`')
    lines.append(f'- **Split strategy:** `{d["split_by"]}` ({d["n_splits"]} folds)')
    lines.append('')

    # Baseline
    lines.append('## Baseline (LightGBM)')
    lines.append('')
    lines.append('| Metric | Mean ± Std |')
    lines.append('|---|---|')
    for k, v in b['metrics'].items():
        lines.append(f'| {k} | {_fmt_mean_std(v)} |')
    lines.append('')
    lines.append('**Confusion matrix:**')
    lines.append('')
    lines.append(_confusion_md(b['confusion']['matrix'], b['confusion']['classes']))
    lines.append('')

    # Dynamis
    lines.append('## Dynamis')
    lines.append('')
    lines.append('| Metric | Mean ± Std |')
    lines.append('|---|---|')
    for k, v in d['metrics'].items():
        lines.append(f'| {k} | {_fmt_mean_std(v)} |')
    lines.append(f'| uncertainty (trace P) | {_fmt_mean_std(d["uncertainty_mean"])} |')
    lines.append('')
    lines.append('**Confusion matrix:**')
    lines.append('')
    lines.append(_confusion_md(d['confusion']['matrix'], d['confusion']['classes']))
    lines.append('')
    lines.append('## Calibration & OOD')
    lines.append('')
    lines.append(f'- **Temperature T:** {d["temperature"]:.4f}')
    lines.append(f'- **ECE (pre):**  {d["ece_pre"]:.4f}')
    lines.append(f'- **ECE (post):** {d["ece_post"]:.4f}')
    lines.append(f'- **ECE reduction:** {d["ece_pre"] - d["ece_post"]:+.4f}')
    if d.get('ood_auc') is not None:
        lines.append(f'- **OOD AUC:** {d["ood_auc"]:.4f}')
        lines.append(f'- **OOD threshold (p90 of trace P):** {d["ood_threshold"]:.3f}')
    else:
        lines.append('- **OOD:** not computed (all folds correct or all wrong)')
    lines.append('')

    # Delta Dynamis vs baseline
    lines.append('## Baseline ↔ Dynamis delta')
    lines.append('')
    lines.append('| Metric | Baseline | Dynamis | Δ |')
    lines.append('|---|---|---|---|')
    for k in ('oa', 'kappa', 'f1'):
        b_mean = np.mean(b['metrics'][k])
        d_mean = np.mean(d['metrics'][k])
        lines.append(f'| {k} | {b_mean:.4f} | {d_mean:.4f} | {d_mean - b_mean:+.4f} |')
    lines.append('')

    # Provenance
    lines.append('## Provenance')
    lines.append('')
    lines.append(f'- Checkpoint: `{config.final_checkpoint_path().name}`')
    lines.append(f'- Reports dir: `{config.REPORTS_DIR}`')
    lines.append(f'- Models dir: `{config.MODELS_DIR}`')
    lines.append(f'- Final-model epochs: {d["final_epochs"]}')
    lines.append('')

    out_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'[report] written → {out_path}')


def is_complete() -> bool:
    return config.report_md_path().exists()
