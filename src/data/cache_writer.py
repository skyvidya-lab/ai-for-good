"""Serialize a list[PointSeries] to parquet/geoparquet artifacts.

Output layout (under `cache_dir`):
    points_meta.geoparquet         1 row/point + Point(lon,lat) geometry
    observations_base.parquet      1 row/(point_id, date), 17 features + mask
    observations_enriched.parquet  same + 4 agro features (optional)
    phenophases.parquet            labels: (point_id, pheno_date, phenophase, idx)
    MANIFEST.json                  version, fingerprint, counts, feature names, md5s
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .temporal_builder import FEATURE_NAMES, PointSeries
# ── Phenophase index mapping ──────────────────────────────────────────────────
# SINGLE SOURCE OF TRUTH: src/dynamis/phenology_prior.py → PHENOPHASES
# This tuple MUST stay in sync with phenology_prior.PHENOPHASES.
# Canonical chronological order verified 2026-05-03 across 778 training points:
#   Greenup(0) → MidGreenup(1) → Maturity(2) → Peak(3)
#   → Senescence(4) → MidSenescence(5) → Dormancy(6)
# NOTE: "Maturity" precedes "Peak" by dataset convention (grain-fill onset ≠ harvest maturity).
PHENOPHASES_CANON: tuple[str, ...] = (
    "Greenup", "MidGreenup", "Maturity", "Peak",
    "Senescence", "MidSenescence", "Dormancy",
)
PHENO_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(PHENOPHASES_CANON)}


AGRO_FEATURES: tuple[str, ...] = ("precip_acc", "soil_moisture", "temp", "smap_wetness")


# Tier-2 GEE expansion: ET (MODIS MOD16), LST day/night (MODIS MOD11),
# surface solar radiation (ERA5), Vapor Pressure Deficit derived from
# 2m temperature and 2m dewpoint (ERA5), and 10m wind speed (ERA5).
EXTRA_AGRO_FEATURES: tuple[str, ...] = (
    "et", "pet", "lst_day", "lst_night", "solar_rad", "vpd", "wind_10m",
)
EXTENDED_AGRO_FEATURES: tuple[str, ...] = AGRO_FEATURES + EXTRA_AGRO_FEATURES  # 11


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _observations_dataframe(
    series_list: list[PointSeries],
    agro_by_point: dict[int, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Explode list[PointSeries] into a long-format obs DataFrame.

    If `agro_by_point` provided, append AGRO_FEATURES columns aligned by date
    using AgroclimateExtractor.align_to_ps() semantics (5-day backward window).
    Caller passes the already-aligned per-point frame to keep this module pure.
    """
    n_base = len(FEATURE_NAMES)
    enriched = agro_by_point is not None

    rows: list[dict] = []
    for ps in series_list:
        pid = int(ps.point_id)
        feats = ps.features  # (T, 17)
        mask = ps.mask        # (T,)
        agro_df = agro_by_point.get(pid) if enriched else None
        # agro_df expected shape: (T, 4) in AGRO_FEATURES order, indexed 0..T-1
        for t, date in enumerate(ps.dates):
            row: dict[str, object] = {
                "point_id": pid,
                "date": date,
                "mask": bool(mask[t]),
            }
            for i, name in enumerate(FEATURE_NAMES):
                v = feats[t, i]
                row[name] = float(v) if np.isfinite(v) else None
            if enriched and agro_df is not None and t < len(agro_df):
                arow = agro_df.iloc[t]
                for name in AGRO_FEATURES:
                    v = arow.get(name)
                    row[name] = float(v) if v is not None and np.isfinite(v) else None
            elif enriched:
                for name in AGRO_FEATURES:
                    row[name] = None
            rows.append(row)
    df = pd.DataFrame(rows)
    df = df.sort_values(["point_id", "date"]).reset_index(drop=True)
    return df


def _points_meta_geodataframe(series_list: list[PointSeries]):
    """Build a GeoDataFrame with one row per point and Point(lon, lat) geometry."""
    import geopandas as gpd
    from shapely.geometry import Point

    records = []
    for ps in series_list:
        records.append({
            "point_id": int(ps.point_id),
            "region": ps.region,
            "lon": float(ps.lon),
            "lat": float(ps.lat),
            "crop_type": ps.crop_type,
            "n_obs": int(len(ps.dates)),
            "n_valid": int(np.sum(ps.mask)),
            "geometry": Point(float(ps.lon), float(ps.lat)),
        })
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    return gdf.sort_values("point_id").reset_index(drop=True)


def _phenophases_dataframe(series_list: list[PointSeries]) -> pd.DataFrame:
    rows = []
    for ps in series_list:
        if not ps.phenophase_by_date:
            continue
        for d, name in ps.phenophase_by_date.items():
            rows.append({
                "point_id": int(ps.point_id),
                "pheno_date": str(d),
                "phenophase": str(name),
                "phenophase_idx": PHENO_TO_IDX.get(str(name), -1),
            })
    return pd.DataFrame(rows).sort_values(["point_id", "pheno_date"]).reset_index(drop=True)


def write_aggregated_cache(
    series_list: list[PointSeries],
    cache_dir: Path,
    version: str = "v1",
    agro_by_point: dict[int, pd.DataFrame] | None = None,
    overwrite: bool = True,
) -> dict[str, Path]:
    """Write all parquet/geoparquet artifacts to `cache_dir`.

    Args:
        series_list: list of PointSeries (base 17 features).
        cache_dir: target directory (created if missing).
        version: tag stored in MANIFEST + filenames.
        agro_by_point: optional {point_id -> DataFrame(T, AGRO_FEATURES)}; when
            provided, also writes observations_enriched.parquet.
        overwrite: if False, raises when a target file already exists.

    Returns:
        dict mapping artifact name -> path.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {
        "points_meta": cache_dir / "points_meta.geoparquet",
        "observations_base": cache_dir / "observations_base.parquet",
        "phenophases": cache_dir / "phenophases.parquet",
        "manifest": cache_dir / "MANIFEST.json",
    }
    if agro_by_point is not None:
        paths["observations_enriched"] = cache_dir / "observations_enriched.parquet"

    if not overwrite:
        for k, p in paths.items():
            if k != "manifest" and p.exists():
                raise FileExistsError(p)

    gdf_meta = _points_meta_geodataframe(series_list)
    gdf_meta.to_parquet(paths["points_meta"], compression="zstd")

    df_base = _observations_dataframe(series_list, agro_by_point=None)
    df_base.to_parquet(paths["observations_base"], compression="zstd", index=False)

    if agro_by_point is not None:
        df_enr = _observations_dataframe(series_list, agro_by_point=agro_by_point)
        df_enr.to_parquet(paths["observations_enriched"], compression="zstd", index=False)

    df_pheno = _phenophases_dataframe(series_list)
    df_pheno.to_parquet(paths["phenophases"], compression="zstd", index=False)

    manifest = {
        "version": version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_points": len(series_list),
        "n_regions": int(gdf_meta["region"].nunique()),
        "n_obs_base": int(len(df_base)),
        "feature_names_base": list(FEATURE_NAMES),
        "feature_names_agro": list(AGRO_FEATURES) if agro_by_point is not None else [],
        "phenophases_canon": list(PHENOPHASES_CANON),
        "files": {
            k: {"path": p.name, "md5": _md5(p), "bytes": p.stat().st_size}
            for k, p in paths.items() if k != "manifest"
        },
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return paths


__all__ = [
    "AGRO_FEATURES",
    "EXTRA_AGRO_FEATURES",
    "EXTENDED_AGRO_FEATURES",
    "PHENOPHASES_CANON",   # kept for backward compat — now equals phenology_prior.PHENOPHASES
    "PHENO_TO_IDX",        # kept for backward compat — now equals phenology_prior.PHENO_TO_IDX
    "write_aggregated_cache",
]
