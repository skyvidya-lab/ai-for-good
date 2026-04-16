"""
Stratified sampling strategy for the Track 1 dataset.

Problem
-------
The first run picked 5 regions by geographic convenience and produced a sample
with 28 rice + 28 corn + **0 soybean**. The `points_train_label.csv` covers all
778 points, but each point's region must be resolved via bbox lookup — and
arbitrary region picks do not guarantee class coverage.

Strategy
--------
1. Index bboxes of ALL 50 regions by reading only 1 TIFF per region (lightweight).
2. Assign a `region` to every one of the 778 training points via bbox lookup.
3. Greedily pick regions until each of the 3 crop classes has at least
   `per_class_min` points represented.
4. Cache the bbox index on Drive so subsequent notebooks skip the re-read.

This gives a class-balanced sample of ~5-10 regions while keeping the heavy
TIFF extraction focused only on regions that actually matter.
"""
from __future__ import annotations

import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

try:
    import rasterio
    from rasterio.warp import transform_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


def _region_from_name(name: str) -> str | None:
    """Parse `regionNN` from a TIFF filename, tolerating the `region54...` typo."""
    m = re.match(r"region_?(\d+)", name)
    if not m:
        return None
    return f"region{int(m.group(1)):02d}"


def index_region_bboxes(
    zip_paths: list[str | Path],
    cache_path: str | Path | None = None,
    workdir: str | Path = "/tmp/region_index",
    force_refresh: bool = False,
) -> dict[str, tuple[float, float, float, float]]:
    """
    Build {region: bbox_wgs84} for every region present in the zips.

    Only extracts ONE tiff per region to read its georeference — not the whole zip.

    Args:
        zip_paths: paths to the 4 `track1_download_link_{2,3,4,5}.zip` files.
        cache_path: optional JSON path to cache the index (recommended).
        workdir: scratch dir for the lightweight extractions.
        force_refresh: ignore the cache and rebuild.

    Returns:
        {"region00": (min_lon, min_lat, max_lon, max_lat), ...} in WGS84.
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError("rasterio required to read TIFF bboxes")

    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists() and not force_refresh:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return {k: tuple(v) for k, v in data.items()}

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    picked_path: dict[str, Path] = {}
    for zp in zip_paths:
        zp = Path(zp)
        if not zp.exists():
            continue
        with zipfile.ZipFile(zp) as zf:
            # Walk members; for each region we haven't seen, grab the first TIFF
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = Path(info.filename).name
                if not name.lower().endswith((".tif", ".tiff")):
                    continue
                region = _region_from_name(name)
                if region is None or region in picked_path:
                    continue
                # Extract just this one
                target = workdir / region / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    with zf.open(info) as src, open(target, "wb") as dst:
                        dst.write(src.read())
                picked_path[region] = target

    bbox_index: dict[str, tuple[float, float, float, float]] = {}
    for region, path in sorted(picked_path.items()):
        with rasterio.open(path) as r:
            bounds = transform_bounds(r.crs, "EPSG:4326", *r.bounds)
        bbox_index[region] = tuple(float(x) for x in bounds)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(bbox_index, indent=2), encoding="utf-8")

    return bbox_index


def assign_region_to_points(
    labels_df: pd.DataFrame,
    bbox_index: dict[str, tuple[float, float, float, float]],
    lon_col: str = "Longitude",
    lat_col: str = "Latitude",
) -> pd.DataFrame:
    """
    Add a `region` column by looking up each (lon, lat) in the bbox index.

    Points outside all bboxes get `region = NaN` and are logged at summary time
    (caller should drop / investigate).
    """
    df = labels_df.copy()

    def _lookup(row) -> str | None:
        lon, lat = row[lon_col], row[lat_col]
        for region, (minx, miny, maxx, maxy) in bbox_index.items():
            if minx <= lon <= maxx and miny <= lat <= maxy:
                return region
        return None

    df["region"] = df.apply(_lookup, axis=1)
    return df


def stratified_region_sample(
    labels_with_region: pd.DataFrame,
    crop_col: str = "crop_type",
    region_col: str = "region",
    point_col: str = "point_id",
    per_class_min: int = 20,
    max_regions: int = 12,
) -> set[str]:
    """
    Greedily pick regions until each crop class has >= `per_class_min` unique points.

    Rule: iterate regions by descending total unique point count; add a region if
    it contributes to any class still below the target.

    Args:
        labels_with_region: df with point_id, crop_type, region columns.
        per_class_min: minimum unique points per crop class.
        max_regions: cap on the selection (avoid sampling the whole dataset).

    Returns:
        Set of selected region names.

    Raises:
        RuntimeError if `max_regions` is reached before all classes meet the target.
    """
    df = labels_with_region.dropna(subset=[region_col]).drop_duplicates(
        subset=[point_col]
    )
    # Unique points per (region, crop)
    per_region = (
        df.groupby([region_col, crop_col])[point_col]
        .nunique()
        .unstack(fill_value=0)
    )
    region_totals = per_region.sum(axis=1).sort_values(ascending=False)

    classes = per_region.columns.tolist()
    selected: set[str] = set()
    counts: Counter[str] = Counter({c: 0 for c in classes})

    for region in region_totals.index:
        if len(selected) >= max_regions:
            break
        needed = [c for c in classes if counts[c] < per_class_min]
        if not needed:
            break
        # Does this region add to any class still in need?
        contrib = per_region.loc[region, needed].sum()
        if contrib > 0:
            selected.add(region)
            for c in classes:
                counts[c] += int(per_region.loc[region, c])

    missing = [c for c in classes if counts[c] < per_class_min]
    if missing:
        raise RuntimeError(
            f"Could not reach {per_class_min} points for classes {missing} "
            f"within {max_regions} regions. Counts: {dict(counts)}. "
            "Increase max_regions or lower per_class_min."
        )
    return selected


def sample_summary(
    labels_with_region: pd.DataFrame,
    selected: set[str],
    crop_col: str = "crop_type",
    region_col: str = "region",
    point_col: str = "point_id",
) -> pd.DataFrame:
    """Pretty summary: crop x region counts for the chosen sample."""
    df = labels_with_region[labels_with_region[region_col].isin(selected)]
    return (
        df.drop_duplicates(subset=[point_col])
        .groupby([region_col, crop_col])[point_col]
        .nunique()
        .unstack(fill_value=0)
    )


__all__ = [
    "index_region_bboxes",
    "assign_region_to_points",
    "stratified_region_sample",
    "sample_summary",
]
