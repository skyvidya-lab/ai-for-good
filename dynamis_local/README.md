# dynamis_local

Local Windows/CUDA training pipeline for Dynamis Terra. Ports the Colab
notebook `02_baseline_vs_dynamis_v8_full_audit.ipynb` to pure Python.

## Prerequisites

- Python 3.10+
- A CUDA-enabled GPU with a matching PyTorch install. See
  <https://pytorch.org> for the correct `pip install torch ...` command
  for your CUDA version.
- ~150 GB free disk (67 GB for the zips, 67 GB extracted, plus cache / checkpoints / reports).
- The `ai-for-good` private repo cloned locally (for `src/`).
- The 5 `track1_download_link_*.zip` files downloaded from the project
  Drive folder to a single directory on your disk.

## Install dependencies

```cmd
pip install numpy pandas rasterio lightgbm tqdm hilbertcurve scikit-learn
```

Plus PyTorch with CUDA per the instructions on pytorch.org.

## Configuration

Two environment variables control all paths:

- `DYNAMIS_ROOT` — folder containing the 5 zips.
- `DYNAMIS_REPO` — root of the `ai-for-good` repo (must contain `src/`).

Defaults (if env vars not set):

- `DYNAMIS_ROOT` → `%USERPROFILE%\workspace\ai-for-good\datasets_final_round`
- `DYNAMIS_REPO` → the current working directory

Everything else (extracted TIFFs, cache, models, reports) is derived from
these two values and lives as siblings under `%DYNAMIS_ROOT%\..\`.

## Running

From inside this directory (or after adding it to your `PYTHONPATH`):

```cmd
set DYNAMIS_ROOT=D:\data\track1_zips
set DYNAMIS_REPO=D:\code\ai-for-good
python run.py extract    :: First time: ~30-60 min for 67 GB
python run.py build      :: ~15-30 min on NVMe (build cache)
python run.py baseline   :: ~1 min (LightGBM CV)
python run.py dynamis    :: ~30-60 min on T4-class GPU (5-fold CV + final)
python run.py report     :: seconds (markdown)
```

Or run everything in sequence:

```cmd
python run.py all
```

Each stage is idempotent — if its output already exists, it skips. Pass
`--force` to redo a stage.

## Outputs

Under `%DYNAMIS_ROOT%\..\`:

- `extracted/region_train_{1..4}/*.tiff` — TIFFs ready for the feature build
- `sample/points_train_label.csv` — labels + supporting CSVs
- `cache/series_v7_*.pkl` — pickled `PointSeries` cache
- `models/dynamis_terra_v7.pt` — final checkpoint with everything the
  inference script needs (weights, config, temperature, OOD threshold,
  normalisation stats, band order, crop classes, phenophase classes)
- `reports/02_baseline_vs_dynamis_v7/` — metrics JSON + markdown report

## --help

```cmd
python run.py --help
```

prints the full CLI including the `--force`, `--verbose`, and
`--skip-validate` flags.

## Troubleshooting

**"Workspace is missing zips"** — `DYNAMIS_ROOT` doesn't point at the
folder with the 5 `track1_download_link_*.zip` files. Check the path
(use `dir %DYNAMIS_ROOT%` to verify) or set the env var.

**"Cannot import `src`"** — `DYNAMIS_REPO` doesn't point at the
`ai-for-good` repo root. The repo must have a `src/` package at its
top level. Either `cd` to the repo root before running, or set the
env var.

**CUDA not detected** — runs on CPU (slow). Double-check your PyTorch
install with `python -c "import torch; print(torch.cuda.is_available())"`.
