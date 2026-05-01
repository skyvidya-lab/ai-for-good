"""Load aggregated parquet cache → list[PointSeries].

Drop-in replacement for the legacy `_cache_series_v*.pkl` files. Notebooks
v12/v13/+ swap one line:

    series_list = load_aggregated_cache(CACHE_DIR, enriched=True)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cache_writer import AGRO_FEATURES, EXTENDED_AGRO_FEATURES
from .sentinel2_loader import MODEL_BANDS
from .temporal_builder import FEATURE_NAMES, PointSeries
from .vegetation_indices import EXTENDED_INDEX_NAMES

EXTENDED_FEATURE_NAMES = tuple(MODEL_BANDS) + EXTENDED_INDEX_NAMES  # 25
FULL_FEATURE_NAMES = EXTENDED_FEATURE_NAMES + AGRO_FEATURES  # 29
FULL_PLUS_FEATURE_NAMES = EXTENDED_FEATURE_NAMES + EXTENDED_AGRO_FEATURES  # 36


def load_manifest(cache_dir: Path) -> dict:
    cache_dir = Path(cache_dir)
    return json.loads((cache_dir / "MANIFEST.json").read_text())


_VARIANT_TO_FILE: dict[str, tuple[str, tuple[str, ...]]] = {
    "base":      ("observations_base.parquet",      tuple(FEATURE_NAMES)),                              # 17
    "enriched":  ("observations_enriched.parquet",  tuple(FEATURE_NAMES) + AGRO_FEATURES),              # 21
    "extended":  ("observations_extended.parquet",  EXTENDED_FEATURE_NAMES),                            # 25
    "full":      ("observations_full.parquet",      FULL_FEATURE_NAMES),                                # 29
    "full_plus": ("observations_full_plus.parquet", FULL_PLUS_FEATURE_NAMES),                           # 36
}


def load_aggregated_cache(
    cache_dir: Path,
    enriched: bool = False,
    variant: str | None = None,
) -> list[PointSeries]:
    """Reconstruct list[PointSeries] from parquet artifacts.

    Args:
        cache_dir: directory containing the cache artifacts.
        enriched: legacy switch (False -> 'base', True -> 'enriched').
        variant: one of 'base' | 'enriched' | 'extended' | 'full'. Overrides
            `enriched` when provided. 'extended' adds 8 Tier-1 vegetation
            indices (25 feat); 'full' adds the 4 agro features on top (29).

    Returns:
        list[PointSeries] sorted by point_id, with `features` shape (T, F)
        where F = 17 (base) | 21 (enriched) | 25 (extended) | 29 (full).
    """
    cache_dir = Path(cache_dir)
    if variant is None:
        variant = "enriched" if enriched else "base"
    if variant not in _VARIANT_TO_FILE:
        raise ValueError(f"unknown variant {variant!r}; expected one of {list(_VARIANT_TO_FILE)}")

    obs_filename, feature_tuple = _VARIANT_TO_FILE[variant]
    obs_path = cache_dir / obs_filename
    meta_path = cache_dir / "points_meta.geoparquet"
    pheno_path = cache_dir / "phenophases.parquet"

    if not obs_path.exists():
        raise FileNotFoundError(obs_path)

    feature_cols: list[str] = list(feature_tuple)

    obs = pd.read_parquet(obs_path)
    meta = pd.read_parquet(meta_path)
    pheno = pd.read_parquet(pheno_path) if pheno_path.exists() else pd.DataFrame()

    pheno_map: dict[int, dict[str, str]] = {}
    if not pheno.empty:
        for pid, grp in pheno.groupby("point_id"):
            pheno_map[int(pid)] = dict(zip(grp["pheno_date"], grp["phenophase"], strict=False))

    meta_by_pid = {int(r["point_id"]): r for _, r in meta.iterrows()}

    series_list: list[PointSeries] = []
    for pid, grp in obs.sort_values(["point_id", "date"]).groupby("point_id", sort=True):
        m = meta_by_pid[int(pid)]
        dates = grp["date"].astype(str).tolist()
        feats = grp[feature_cols].to_numpy(dtype=np.float64, na_value=np.nan)
        mask = grp["mask"].to_numpy(dtype=bool)
        ps = PointSeries(
            point_id=int(pid),
            region=str(m["region"]),
            lon=float(m["lon"]),
            lat=float(m["lat"]),
            dates=dates,
            features=feats,
            mask=mask,
            crop_type=(str(m["crop_type"]) if m["crop_type"] is not None else None),
            phenophase_by_date=pheno_map.get(int(pid)),
        )
        series_list.append(ps)
    return series_list


__all__ = ["load_aggregated_cache", "load_manifest"]
