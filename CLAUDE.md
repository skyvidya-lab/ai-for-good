# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Dynamis Terra** — Physics-informed crop type + phenophase classification from multi-temporal Sentinel-2 imagery. Built for the **ITU AI and Space Computing Challenge 2026, Track 1 Final Round** (Space Intelligence Empowering Zero Hunger, SDG 2), organised by Zhejiang Lab / Zero2x. Team: Bonanza / Dynamis.

**Deadlines:**
- Validation environment: until **2026-05-07 UTC**
- Solution submission: until **2026-05-10 UTC**

**Scoring:** 60% algorithm + 40% solution design document + presentation video

## Dataset Anatomy

| Fact | Value |
|---|---|
| Training points | 778 (rice: 367, corn: 229, soybean: 182) |
| Labels | `points_train_label.csv`, 5 447 rows (7 per point) |
| Phenophases | 7: Dormancy, Greenup, MidGreenup, Peak, Maturity, MidSenescence, Senescence |
| Regions | 50 unique (region00–region57 with gaps) |
| Dates/region | 1–15 (mean ~4.5) |
| Folders | 4 overlapping `region_train_1..4` (same spatial set, different batches) |
| Bands | 13 (B01-B12 + B8A); B10 is atypical for L2A and is excluded from `MODEL_BANDS` |
| Volume | ~70 GB (4 × 17.5 GB zips + 129 MB labels/guide) |
| Storage | Google Drive: `/content/drive/Shareddrives/SKYVIDYA/AI for Good/datasets_final_round` |
| Test points | 171 unlabeled in `test_point.csv` |
| Platform | Kaggle-like (upload model → validate → submit) |

Full details in [.claude/kb/challenge/dataset-anatomy.md](.claude/kb/challenge/dataset-anatomy.md).

## Strategic Insights

1. **Consolidate the 4 folders by region** — each folder is a temporal batch, not a split; merging gives ~18 dates/region for rich time-series.
2. **7 phenophases = canonical state machine** — Dynamis MKM uses a phenology transition prior (Dormancy → Greenup → ... → Senescence) to initialise A.
3. **778 points is small for deep learning** — MKM (~200 learnable params) beats heavy LSTM/Transformer here.
4. **GroupKFold by point_id is mandatory** — avoid leakage across the 7 phenophases of a single point.
5. **"background" class may appear in test** — use Kalman covariance P as OOD detector (threshold on trace(P)).

## Dynamis Component Map

| Module | Status | File |
|---|---|---|
| MKM (Markov-Kalman) | **CORE** — with phenology prior | [src/dynamis/dynamis_core.py](src/dynamis/dynamis_core.py) |
| Phenology transition prior | **CORE** | [src/dynamis/phenology_prior.py](src/dynamis/phenology_prior.py) |
| Innovation Loss + ECE | **CORE** | [src/dynamis/innovation_loss.py](src/dynamis/innovation_loss.py) |
| Hurst Exponent (conditional) | Active feature | [src/dynamis/hurst_geo.py](src/dynamis/hurst_geo.py) |
| ChaosAttention (PIA lite) | Active attention | [src/dynamis/chaos_attention.py](src/dynamis/chaos_attention.py) |
| Hilbert Embedding | **REMOVED** from critical path | (available in `dynamis_core.py`) |
| HierarchicalHPR | Deferred | (available in `dynamis_core.py`) |

## Repository Structure

```
ai-for-good/
├── CLAUDE.md                   # This file (root brain)
├── .claude/                    # Orchestration (agents, commands, KB, SDD)
│   ├── CLAUDE.md               # Detailed orchestration doc
│   ├── agents/                 # SubAgents (ai-ml/, code-quality/, domain/, exploration/)
│   ├── commands/               # Custom slash commands (core/, train/, submit/, workflow/)
│   ├── kb/                     # Knowledge base with _index.yaml registry
│   ├── sdd/                    # Spec-Driven Development (5-phase workflow)
│   │   ├── architecture/       # WORKFLOW_CONTRACTS.yaml + ARCHITECTURE.md
│   │   ├── templates/          # BRAINSTORM / DEFINE / DESIGN / BUILD_REPORT / SHIPPED
│   │   ├── features/           # Active features (BRAINSTORM_*.md, DEFINE_*.md, DESIGN_*.md)
│   │   ├── reports/            # BUILD_REPORT_*.md
│   │   └── archive/            # SHIPPED features, one folder per feature
│   └── storage/                # Session memory archive (memory-YYYY-MM-DD.md)
├── src/
│   ├── dynamis/                # Physics-informed modules
│   ├── data/                   # Sentinel-2 pipeline (loader, consolidator, extractor, indices, builder)
│   ├── models/                 # DynamisCropClassifier + baselines
│   ├── training/               # Trainer, metrics, visualisations
│   └── submission/             # Predict + package
├── notebooks/
│   ├── 00_colab_setup.ipynb
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_vs_dynamis.ipynb    # ← MAIN Colab T4 deliverable
│   ├── 03_full_training.ipynb
│   └── 04_inference_submission.ipynb
├── tests/
├── docs/
│   ├── SOLUTION_DESIGN.md      # 40% of score
│   └── DYNAMIS_TERRA_PAPER.md  # J-FET draft
├── requirements.txt
└── pyproject.toml
```

## Quickstart

### Local (sample data only, 131 MB)
```bash
python -m pip install -r requirements.txt
pytest tests/                 # smoke tests against test_input_sample/
```

### Colab (full data, 70 GB)
1. Open `notebooks/00_colab_setup.ipynb`, set Runtime → T4 GPU.
2. Mount Drive — the shared data folder is:
   ```
   /content/drive/Shareddrives/SKYVIDYA/AI for Good/datasets_final_round
   ```
3. Run `notebooks/02_baseline_vs_dynamis.ipynb` end-to-end (~30 min on T4).

## Key References (other workspaces, read-only sources)

| Source | Used for |
|---|---|
| `C:\Users\eluzq\workspace\00_benchmarking\skyvidya_core\skyvidya_dynamis` | Dynamis core theory (`dynamis_core.py` copied verbatim) |
| `C:\Users\eluzq\workspace\dynamis-finance-ai` | PIA pattern (adapted to `chaos_attention.py`) |
| `C:\Users\eluzq\workspace\ai-powered-dynamis-nlp` | Transfer Proof protocol |
| `C:\Users\eluzq\workspace\semana-ai-data-engineer-main` | `.claude/` orchestration template |

## Technical Support

- Challenge platform: [spaceaichallenge.zero2x.org](https://spaceaichallenge.zero2x.org)
- Technical support: `spaceapp@zhejianglab.org`
- DingTalk Group ID: `lnn19940969`

## Conventions

- Python 3.10+, type hints, `from __future__ import annotations`.
- No mocks in integration tests — use `background/.../test_input_sample/` for smoke tests.
- Seeds fixed for reproducibility (`torch.manual_seed(42)` in notebooks).
- Same fold splits for baseline vs Dynamis (GroupKFold by `point_id`).
- Feature names live in `src/data/temporal_builder.py::FEATURE_NAMES` (17 = 12 bands + 5 indices).
- Phenophases live in `src/dynamis/phenology_prior.py::PHENOPHASES` (7 in canonical order).

## Action Plan

Approved plan at `C:\Users\eluzq\.claude\plans\nifty-snuggling-hare.md`. Timeline: 24 days with explicit buffer before the 2026-05-10 submission deadline.

## SDD Workflow (5 phases)

Adopted from AgentSpec v3.0.0 (2026-04-17). Every Dynamis iteration (v3, v4, ...) flows through:

```
/brainstorm → /define → /design → /build → /ship      (+ /iterate cross-phase)
```

Outputs live in `.claude/sdd/features/` (in-flight) and `.claude/sdd/archive/{FEATURE}/` (shipped).

| Command | Phase | Output |
|---|---|---|
| `/brainstorm` | 0 | `BRAINSTORM_{FEATURE}.md` — explore approaches, YAGNI |
| `/define` | 1 | `DEFINE_{FEATURE}.md` — requirements + clarity score ≥12/15 |
| `/design` | 2 | `DESIGN_{FEATURE}.md` — architecture + file manifest |
| `/build` | 3 | Code + `BUILD_REPORT_{FEATURE}.md` |
| `/ship` | 4 | `SHIPPED_{DATE}.md` + lessons learned |
| `/iterate` | cross | Update any phase doc with cascade detection |

Rules of the road: see `.claude/sdd/architecture/WORKFLOW_CONTRACTS.yaml`.
