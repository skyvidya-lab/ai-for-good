"""Stage 2 — build PointSeries cache from extracted TIFFs.

Produces: CACHE_DIR/series_v{VERSION}_{fingerprint}.pkl

The cache fingerprint is md5(sorted(SAMPLE_REGIONS) + n_points), so if
SAMPLE_REGIONS changes (different sampler parameters), the cache is
invalidated automatically.

This stage depends on:
  - extract.run() having populated EXTRACTED_DIR + points_train_label.csv
  - src.data.build_point_series being importable

Running this stage with --force ignores the cache and rebuilds.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from . import config


def _select_sample_regions(labels_df: pd.DataFrame):
    """Pick regions to include in training.

    Follows the v7 strategy: run the stratified sampler with permissive
    constraints (per_class_min=3, max_regions=everything). If it still
    returns fewer than 30 regions, fall back to "every region with labels",
    which is always correct for spatial CV because class balance only
    needs to hold in aggregate across the sample.
    """
    # Lazy import so config.validate() hasn't run doesn't crash module-load
    from src.data import stratified_region_sample, sample_summary

    all_regions = set(labels_df['region'].dropna().unique())
    n_available = len(all_regions)
    print(f'[build] available regions with labels: {n_available}')

    selected = stratified_region_sample(
        labels_df, per_class_min=3, max_regions=n_available,
    )
    print(f'[build] stratified sampler selected: {len(selected)} regions')

    if len(selected) < 30:
        print(
            f'[build] [fallback] {len(selected)} < 30 — using every region with labels '
            f'({n_available}) instead'
        )
        selected = all_regions

    selected = set(selected)
    print(f'[build] final: {len(selected)} regions')

    summary = sample_summary(labels_df, selected)
    totals = summary.sum(axis=0)
    print(f'[build] class totals: {totals.to_dict()}')

    if (totals < 10).any():
        raise ValueError(
            f'Some crop class has < 10 points in the sample: {totals.to_dict()}. '
            f'Check stratified_region_sample and per_class_min.'
        )

    return selected


def _fingerprint(sample_regions: set[str], n_points: int) -> str:
    seed = repr(sorted(sample_regions)) + f'_n{n_points}'
    return hashlib.md5(seed.encode()).hexdigest()[:10]


def _load_and_tag_labels() -> pd.DataFrame:
    """Load points_train_label.csv and attach a `region` column via bbox lookup."""
    from src.data import index_region_bboxes, assign_region_to_points

    labels_path = config.labels_csv_path()
    if not labels_path.exists():
        raise FileNotFoundError(
            f'[build] labels not found at {labels_path}. Did you run `extract`?'
        )
    labels_df = pd.read_csv(labels_path)
    print(
        f'[build] labels loaded: {len(labels_df)} rows, '
        f'{labels_df["point_id"].nunique()} unique points'
    )

    # Index region bboxes from one TIFF per region across all extracted folders.
    folders = [str(config.EXTRACTED_DIR / f) for f in config.ZIPS]
    bbox_index = index_region_bboxes(
        folders=folders,
        workdir=str(config.SAMPLE_DIR / 'region_index'),
    )
    print(f'[build] indexed {len(bbox_index)} regions from bbox')

    labels_df = assign_region_to_points(labels_df, bbox_index)
    missing = labels_df['region'].isna().sum()
    print(f'[build] points without a region match: {missing}')
    return labels_df


def _build_series_list(sample_labels: pd.DataFrame, view: dict) -> list:
    """Build PointSeries for each unique point_id in sample_labels."""
    from src.data import build_point_series

    series = []
    for pid, group in tqdm(
        sample_labels.groupby('point_id'),
        total=sample_labels['point_id'].nunique(),
        desc='[build] point series',
    ):
        row0 = group.iloc[0]
        region = row0['region']
        pheno_map = dict(zip(group['phenophase_date'], group['phenophase_name']))
        ps = build_point_series(
            point_id=int(pid),
            lon=float(row0['Longitude']),
            lat=float(row0['Latitude']),
            region=region,
            consolidated_view=view,
            phenophase_by_date=pheno_map,
            crop_type=str(row0['crop_type']),
        )
        if ps.features.shape[0] == 0:
            continue
        series.append(ps)
    return series


def run(*, force: bool = False) -> None:
    """Execute the build stage.

    Args:
        force: ignore any existing cache and rebuild from TIFFs.
    """
    print(f'[build] version    : v{config.VERSION}')
    print(f'[build] cache dir  : {config.CACHE_DIR}')

    labels_df = _load_and_tag_labels()
    sample_regions = _select_sample_regions(labels_df)
    sample_labels = labels_df[labels_df['region'].isin(sample_regions)].copy()
    n_points = sample_labels['point_id'].nunique()

    fp = _fingerprint(sample_regions, n_points)
    cache = config.series_cache_path(fp)
    print(f'[build] cache path : {cache}')

    if not force and cache.exists():
        print(f'[build] cache hit — skipping build')
        with open(cache, 'rb') as f:
            series = pickle.load(f)
        print(f'[build] loaded {len(series)} point series from cache')
        return

    # --- Consolidate TIFF folders ---
    from src.data import consolidate_regions
    folders = [str(config.EXTRACTED_DIR / f) for f in config.ZIPS]
    view = consolidate_regions(folders, regions_filter=sample_regions)
    print(f'[build] consolidated {len(view)} regions')
    for r, dates in view.items():
        print(f'  {r}: {len(dates)} unique dates')

    # Filter labels to regions that actually have TIFFs
    sample_labels = sample_labels[sample_labels['region'].isin(view.keys())].copy()
    print(
        f'[build] after region intersection: {sample_labels["point_id"].nunique()} points '
        f'across {sample_labels.groupby("crop_type")["point_id"].nunique().to_dict()}'
    )

    # --- Build series ---
    series = _build_series_list(sample_labels, view)
    print(f'[build] built {len(series)} point series')
    if series:
        first = series[0]
        print(f'[build] first point: T={first.features.shape[0]}, F={first.features.shape[1]}')

    # --- Cache ---
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, 'wb') as f:
            pickle.dump(series, f)
        print(f'[build] cached to {cache}')
    except Exception as e:
        print(f'[build] [warn] cache save failed ({type(e).__name__}: {e}); continuing')


def is_complete() -> bool:
    """True if any series cache exists for the current version."""
    if not config.CACHE_DIR.exists():
        return False
    return any(config.CACHE_DIR.glob(f'series_v{config.VERSION}_*.pkl'))


def load_cache() -> list:
    """Load the most recent series cache for this version. Used by later stages."""
    if not config.CACHE_DIR.exists():
        raise FileNotFoundError(f'[build] no cache dir at {config.CACHE_DIR}')
    candidates = sorted(
        config.CACHE_DIR.glob(f'series_v{config.VERSION}_*.pkl'),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f'[build] no series cache at {config.CACHE_DIR} for v{config.VERSION}. '
            f'Run `build` first.'
        )
    with open(candidates[0], 'rb') as f:
        return pickle.load(f)
