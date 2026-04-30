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

from .cache_writer import AGRO_FEATURES
from .temporal_builder import FEATURE_NAMES, PointSeries


def load_manifest(cache_dir: Path) -> dict:
    cache_dir = Path(cache_dir)
    return json.loads((cache_dir / "MANIFEST.json").read_text())


def load_aggregated_cache(
    cache_dir: Path,
    enriched: bool = False,
) -> list[PointSeries]:
    """Reconstruct list[PointSeries] from parquet artifacts.

    Args:
        cache_dir: directory containing the cache artifacts.
        enriched: when True, load observations_enriched.parquet (21 features);
            otherwise load observations_base.parquet (17 features).

    Returns:
        list[PointSeries] sorted by point_id, with `features` shape (T, F)
        where F = 17 (base) or 21 (enriched).
    """
    cache_dir = Path(cache_dir)
    obs_path = cache_dir / ("observations_enriched.parquet" if enriched else "observations_base.parquet")
    meta_path = cache_dir / "points_meta.geoparquet"
    pheno_path = cache_dir / "phenophases.parquet"

    if not obs_path.exists():
        raise FileNotFoundError(obs_path)

    feature_cols: list[str] = list(FEATURE_NAMES)
    if enriched:
        feature_cols = feature_cols + list(AGRO_FEATURES)

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
