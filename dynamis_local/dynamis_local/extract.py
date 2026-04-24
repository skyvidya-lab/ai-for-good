"""Stage 1 — extract TIFFs from source zips to local disk.

Idempotent: files already present at the correct size are skipped. Resume
after a partial run is automatic.

This stage produces:
  - EXTRACTED_DIR/<folder_name>/<basename>.tiff for each selected region
  - SAMPLE_DIR/points_train_label.csv  (and the small guide CSVs)

It does NOT decide which regions to select — that's done at build time.
So we extract **every** TIFF from every zip: the full dataset. Size on
disk: ~70 GB. If disk is tight, see build.py's wave processing (TODO).
"""
from __future__ import annotations

import os
import re
import time
import zipfile
from pathlib import Path

from . import config


def _iter_tiff_infos(zf: zipfile.ZipFile):
    """Yield (info, region_id, basename) for every TIFF in a zip."""
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = os.path.basename(info.filename)
        if not name.lower().endswith(('.tif', '.tiff')):
            continue
        m = re.match(r'region_?(\d+)', name)
        if not m:
            continue
        yield info, f'region{int(m.group(1)):02d}', name


def _extract_zip_tiffs(
    zip_path: Path,
    folder_name: str,
    dest_root: Path,
    region_filter: set[str] | None,
    *,
    force: bool = False,
) -> tuple[int, int, int, float]:
    """Extract TIFFs from one zip, idempotently.

    Returns (n_extracted_now, n_already_present, n_bytes_written, elapsed_s).

    If region_filter is None, extracts everything. Otherwise only regions
    whose ID is in the set.
    """
    if not zip_path.exists():
        print(f'  [skip] {zip_path} not found')
        return 0, 0, 0, 0.0

    dest = dest_root / folder_name
    dest.mkdir(parents=True, exist_ok=True)

    n_new = n_skip = n_bytes = 0
    t0 = time.time()
    with zipfile.ZipFile(zip_path) as zf:
        targets = [
            (info, name) for info, rid, name in _iter_tiff_infos(zf)
            if (region_filter is None or rid in region_filter)
        ]
        for info, name in targets:
            target = dest / name
            if not force and target.exists() and target.stat().st_size == info.file_size:
                n_skip += 1
                continue
            # Stream copy with 4 MB buffer for decent sequential throughput
            with zf.open(info) as src, open(target, 'wb') as dst:
                while True:
                    chunk = src.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            n_new += 1
            n_bytes += info.file_size

    return n_new, n_skip, n_bytes, time.time() - t0


def _extract_csvs(zip_path: Path, dest: Path) -> int:
    """Extract all .csv entries from a zip into dest (flat)."""
    if not zip_path.exists():
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.lower().endswith('.csv'):
                continue
            # Flatten — just use the basename
            target = dest / os.path.basename(info.filename)
            if target.exists() and target.stat().st_size == info.file_size:
                continue
            with zf.open(info) as src, open(target, 'wb') as dst:
                dst.write(src.read())
            n += 1
    return n


def run(*, force: bool = False, regions: set[str] | None = None) -> None:
    """Execute the extract stage.

    Args:
        force:   ignore idempotence checks and re-extract everything.
        regions: if set, extract only TIFFs for these region IDs (e.g. {'region00'}).
                 If None, extract everything. Typically used for testing.
    """
    print(f'[extract] source    : {config.WORKSPACE}')
    print(f'[extract] dest      : {config.EXTRACTED_DIR}')
    if regions:
        print(f'[extract] regions   : {sorted(regions)}  ({len(regions)} selected)')
    else:
        print(f'[extract] regions   : all')
    print(f'[extract] force     : {force}')

    # --- Pass 1: small CSVs from every zip (labels + samples + guide) ---
    print('\n[extract] pass 1: CSVs')
    csv_total = 0
    for folder, zip_path in config.ZIPS.items():
        n = _extract_csvs(zip_path, config.SAMPLE_DIR)
        csv_total += n
        print(f'  [{folder}] {n} CSV(s) extracted')
    n = _extract_csvs(config.GUIDE_ZIP, config.SAMPLE_DIR)
    csv_total += n
    print(f'  [guide]         {n} CSV(s) extracted')

    # Sanity: labels file must now exist
    labels = config.labels_csv_path()
    if not labels.exists():
        # It might be under a nested dir — search
        for cand in config.SAMPLE_DIR.rglob('points_train_label.csv'):
            target = config.labels_csv_path()
            if cand != target:
                target.write_bytes(cand.read_bytes())
            break
    if not labels.exists():
        raise FileNotFoundError(
            f'[extract] points_train_label.csv not found anywhere under '
            f'{config.SAMPLE_DIR}. The zip containing it is '
            f'track1_download_link_5.zip — check DYNAMIS_ROOT.'
        )
    print(f'  [labels] {labels}')

    # --- Pass 2: heavy TIFF extraction ---
    print('\n[extract] pass 2: TIFFs')
    t0 = time.time()
    total_new = total_skip = total_bytes = 0
    for folder_name, zip_path in config.ZIPS.items():
        n_new, n_skip, n_bytes, dt = _extract_zip_tiffs(
            zip_path, folder_name, config.EXTRACTED_DIR, regions, force=force,
        )
        total_new += n_new
        total_skip += n_skip
        total_bytes += n_bytes
        mbps = (n_bytes / 1024**2) / max(dt, 0.001) if n_bytes else 0.0
        print(
            f'  [{folder_name}] +{n_new} new, {n_skip} already present '
            f'({n_bytes / 1024**3:.1f} GB in {dt:.0f}s = {mbps:.1f} MB/s)'
        )
    dt_total = time.time() - t0
    mbps_avg = (total_bytes / 1024**2) / max(dt_total, 0.001)
    print(
        f'[extract] total: {total_new} extracted, {total_skip} already present, '
        f'{total_bytes / 1024**3:.1f} GB in {dt_total:.0f}s '
        f'(avg {mbps_avg:.1f} MB/s)'
    )


def is_complete() -> bool:
    """Cheap check: labels CSV and at least one TIFF per folder exist."""
    if not config.labels_csv_path().exists():
        return False
    for folder in config.ZIPS:
        d = config.EXTRACTED_DIR / folder
        if not d.exists():
            return False
        if not any(d.glob('*.tif*')):
            return False
    return True
