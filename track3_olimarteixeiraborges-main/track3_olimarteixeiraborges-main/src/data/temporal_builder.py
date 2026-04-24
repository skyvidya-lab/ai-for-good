"""
Build per-point temporal tensors from consolidated folders + labels.

Output shape per point:
    X: (T, F) where T = number of dates for the point's region, F = 12 bands + 5 indices = 17
    dates: list[str] (ISO dates)
    mask: (T,) bool (True where all bands are available)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .folder_consolidator import dates_in_order
from .point_extractor import extract_bands_at_point
from .sentinel2_loader import MODEL_BANDS
from .vegetation_indices import INDEX_NAMES, compute_all_indices

FEATURE_NAMES: tuple[str, ...] = tuple(MODEL_BANDS) + INDEX_NAMES
N_FEATURES = len(FEATURE_NAMES)  # 12 + 5 = 17


@dataclass
class PointSeries:
    """Multi-temporal feature sequence for a single training/test point."""

    point_id: int
    region: str
    lon: float
    lat: float
    dates: list[str]        # length T
    features: np.ndarray    # (T, 17) — bands + indices
    mask: np.ndarray        # (T,) bool — True where the full band set was available
    crop_type: str | None = None
    phenophase_by_date: dict[str, str] | None = None  # {date: phenophase_name}


def point_region_from_coords(
    lon: float,
    lat: float,
    region_bboxes: dict[str, tuple[float, float, float, float]],
) -> str | None:
    """
    Find the region whose bbox contains (lon, lat). Returns None if no match.

    bbox format: (min_lon, min_lat, max_lon, max_lat)
    """
    for region, (minx, miny, maxx, maxy) in region_bboxes.items():
        if minx <= lon <= maxx and miny <= lat <= maxy:
            return region
    return None


def build_point_series(
    point_id: int,
    lon: float,
    lat: float,
    region: str,
    consolidated_view: dict[str, dict[str, dict[str, Path]]],
    phenophase_by_date: dict[str, str] | None = None,
    crop_type: str | None = None,
) -> PointSeries:
    """
    Build the (T, F) tensor for a single point by extracting all its region's dates.

    Args:
        point_id, lon, lat, region: point identity.
        consolidated_view: output of consolidate_regions().
        phenophase_by_date: optional labels (training only).
        crop_type: optional label (training only).
    """
    region_view = consolidated_view.get(region, {})
    dates = dates_in_order(region_view)

    T = len(dates)
    X = np.full((T, N_FEATURES), np.nan, dtype=np.float64)
    mask = np.zeros(T, dtype=bool)

    for t, date in enumerate(dates):
        band_paths = region_view[date]
        bands_vec = extract_bands_at_point(band_paths, lon, lat, list(MODEL_BANDS))
        if np.all(np.isnan(bands_vec[:8])):  # visible+NIR must be present
            continue
        X[t, : len(MODEL_BANDS)] = bands_vec
        indices = compute_all_indices(bands_vec, scale=True)
        for i, name in enumerate(INDEX_NAMES):
            X[t, len(MODEL_BANDS) + i] = indices[name]
        mask[t] = not np.any(np.isnan(bands_vec))

    return PointSeries(
        point_id=point_id,
        region=region,
        lon=lon,
        lat=lat,
        dates=dates,
        features=X,
        mask=mask,
        crop_type=crop_type,
        phenophase_by_date=phenophase_by_date,
    )


def build_training_set(
    labels_df: pd.DataFrame,
    consolidated_view: dict[str, dict[str, dict[str, Path]]],
    point_to_region: dict[int, str],
    verbose: bool = False,
) -> list[PointSeries]:
    """
    Build PointSeries for each unique point in the training labels.

    Assumes `labels_df` has columns: point_id, Longitude, Latitude, phenophase_date,
    crop_type, phenophase_name (per dataset analysis).
    """
    series_list: list[PointSeries] = []
    grouped = labels_df.groupby("point_id")
    for point_id, group in grouped:
        region = point_to_region.get(int(point_id))
        if region is None or region not in consolidated_view:
            if verbose:
                print(f"[skip] point_id={point_id}: no region match or no data")
            continue
        row0 = group.iloc[0]
        pheno_map = dict(zip(group["phenophase_date"], group["phenophase_name"], strict=False))
        crop = str(row0.get("crop_type", "")) or None
        ps = build_point_series(
            point_id=int(point_id),
            lon=float(row0["Longitude"]),
            lat=float(row0["Latitude"]),
            region=region,
            consolidated_view=consolidated_view,
            phenophase_by_date=pheno_map,
            crop_type=crop,
        )
        series_list.append(ps)
    return series_list


__all__ = [
    "PointSeries",
    "FEATURE_NAMES",
    "N_FEATURES",
    "build_point_series",
    "build_training_set",
    "point_region_from_coords",
]
