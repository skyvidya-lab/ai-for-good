"""Single source of truth for all paths, versioning, and environment config.

All other modules import from this file. Never hardcode a path elsewhere.

Configuration is resolved from environment variables with Windows-friendly
defaults. Two variables matter:

  DYNAMIS_ROOT   folder containing the 5 track1_download_link_*.zip files
                 (default: %USERPROFILE%/workspace/ai-for-good/datasets_final_round)

  DYNAMIS_REPO   folder with the ai-for-good repo containing src/
                 (default: current working directory)

Call `config.validate()` once at process start to catch misconfiguration
before anything else runs.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Version — bump for each training run so outputs don't overwrite each other.
# ---------------------------------------------------------------------------
VERSION: int = 7
RUN_TAG: str = f'02_baseline_vs_dynamis_v{VERSION}'

# ---------------------------------------------------------------------------
# Paths. Resolved lazily so env var changes before `validate()` still take
# effect, but accessed via attributes so consumers see the final values.
# ---------------------------------------------------------------------------

def _default_root() -> Path:
    return Path.home() / 'workspace' / 'ai-for-good' / 'datasets_final_round'

def _default_repo() -> Path:
    return Path.cwd()

WORKSPACE: Path = Path(os.environ.get('DYNAMIS_ROOT', str(_default_root())))
REPO_PATH: Path = Path(os.environ.get('DYNAMIS_REPO', str(_default_repo())))

# Derived paths. Siblings of WORKSPACE so they sit alongside the dataset root.
SAMPLE_DIR: Path = WORKSPACE.parent / 'sample'      # small CSVs + scratch files
MODELS_DIR: Path = WORKSPACE.parent / 'models'      # checkpoints
REPORTS_DIR: Path = WORKSPACE.parent / 'reports' / RUN_TAG
CACHE_DIR: Path = WORKSPACE.parent / 'cache'        # pickled series_list, OOF logits

# TIFF extraction: persistent destination. On Colab we had a split between
# Drive (persistent, slow) and ephemeral local SSD (fast) — locally there's
# just one disk, so EXTRACTED_DIR doubles as LOCAL_TIFF_DIR.
EXTRACTED_DIR: Path = WORKSPACE / 'extracted'
LOCAL_TIFF_DIR: Path = EXTRACTED_DIR  # alias for clarity in downstream code

# ---------------------------------------------------------------------------
# Expected input files.
# ---------------------------------------------------------------------------
ZIPS: dict[str, Path] = {
    'region_train_1': WORKSPACE / 'track1_download_link_5.zip',   # also contains points_train_label.csv
    'region_train_2': WORKSPACE / 'track1_download_link_4.zip',
    'region_train_3': WORKSPACE / 'track1_download_link_3.zip',
    'region_train_4': WORKSPACE / 'track1_download_link_2.zip',
}
GUIDE_ZIP: Path = WORKSPACE / 'track1_download_link_1.zip'

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def pick_device() -> str:
    """Return 'cuda' | 'mps' | 'cpu' in priority order, logging the choice."""
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f'[device] CUDA available: {name} | torch {torch.__version__}')
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f'[device] MPS available (Apple Silicon) | torch {torch.__version__}')
        return 'mps'
    print(f'[device] CPU only — training will be slow | torch {torch.__version__}')
    return 'cpu'


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------
def ensure_dirs() -> None:
    """Create the writable output directories if missing."""
    for d in (SAMPLE_DIR, MODELS_DIR, REPORTS_DIR, CACHE_DIR, EXTRACTED_DIR):
        d.mkdir(parents=True, exist_ok=True)


def validate() -> None:
    """Abort early if the environment is misconfigured.

    Checks, in order:
      1. All 5 input zips exist under WORKSPACE.
      2. REPO_PATH contains an importable `src` package.
      3. Writable output dirs are creatable.
    """
    # 1. Zips
    missing = [z.name for _, z in ZIPS.items() if not z.exists()]
    if not GUIDE_ZIP.exists():
        missing.append(GUIDE_ZIP.name)
    if missing:
        raise FileNotFoundError(
            f'WORKSPACE={WORKSPACE} is missing zips: {missing}\n'
            f'Set DYNAMIS_ROOT to the folder with the 5 track1_download_link_*.zip files, '
            f'or place them at the default path.'
        )

    # 2. src/ importable
    repo_str = str(REPO_PATH)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    try:
        import src  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f'Cannot import `src` from REPO_PATH={REPO_PATH}. '
            f'Set DYNAMIS_REPO to the ai-for-good repo root, or run from inside it. '
            f'Underlying error: {e}'
        )

    # 3. Output dirs
    ensure_dirs()

    print(f'[config] WORKSPACE     = {WORKSPACE}')
    print(f'[config] REPO_PATH     = {REPO_PATH}')
    print(f'[config] EXTRACTED_DIR = {EXTRACTED_DIR}')
    print(f'[config] MODELS_DIR    = {MODELS_DIR}')
    print(f'[config] REPORTS_DIR   = {REPORTS_DIR}')
    print(f'[config] CACHE_DIR     = {CACHE_DIR}')
    print(f'[config] VERSION       = {VERSION}')


# ---------------------------------------------------------------------------
# Output artifact paths. Each stage reads/writes these; keeping them in one
# place means any rename happens once.
# ---------------------------------------------------------------------------
def labels_csv_path() -> Path:
    return SAMPLE_DIR / 'points_train_label.csv'

def series_cache_path(fingerprint: str) -> Path:
    return CACHE_DIR / f'series_v{VERSION}_{fingerprint}.pkl'

def baseline_metrics_path() -> Path:
    return REPORTS_DIR / 'baseline_metrics.json'

def dynamis_metrics_path() -> Path:
    return REPORTS_DIR / 'dynamis_metrics.json'

def dynamis_oof_path() -> Path:
    return REPORTS_DIR / 'dynamis_oof.npz'

def final_checkpoint_path() -> Path:
    return MODELS_DIR / f'dynamis_terra_v{VERSION}.pt'

def report_md_path() -> Path:
    return REPORTS_DIR / 'report.md'
