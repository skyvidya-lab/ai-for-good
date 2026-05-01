"""
Consolidate the 4 overlapping `region_train_{1..4}` folders.

Strategy:
    The folders are temporal/reprocessing batches of the same spatial set.
    For each (region, date) present in ANY folder, keep one copy. Dates that
    differ across folders increase temporal depth per region.

Precedence when the same (region, date, band) appears in multiple folders:
    Later folder index wins (4 > 3 > 2 > 1) — tie-break arbitrary but stable.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .sentinel2_loader import TiffMeta, inventory_tiffs


def consolidate_regions(
    folder_roots: list[str | Path],
    regions_filter: set[str] | None = None,
) -> dict[str, dict[str, dict[str, Path]]]:
    """
    Build a consolidated view: region -> date -> band -> path.

    Args:
        folder_roots: ordered list of folder paths; later roots override earlier on ties.
        regions_filter: optional whitelist of regions.

    Returns:
        Nested dict {region: {date: {band: Path}}}.
    """
    view: dict[str, dict[str, dict[str, Path]]] = defaultdict(lambda: defaultdict(dict))
    for root in folder_roots:
        metas = inventory_tiffs([root], regions_filter=regions_filter)
        for m in metas:
            view[m.region][m.date][m.band] = m.path
    # Freeze into plain dicts
    return {r: {d: dict(bands) for d, bands in dates.items()} for r, dates in view.items()}


def summarise_consolidation(
    view: dict[str, dict[str, dict[str, Path]]],
) -> dict[str, dict[str, int]]:
    """Return {region: {n_dates, n_files}} summary for logging."""
    out: dict[str, dict[str, int]] = {}
    for region, dates in view.items():
        n_files = sum(len(bands) for bands in dates.values())
        out[region] = {"n_dates": len(dates), "n_files": n_files}
    return out


def dates_in_order(view_region: dict[str, dict[str, Path]]) -> list[str]:
    """Return ISO dates sorted chronologically for a single region."""
    return sorted(view_region.keys())


__all__ = ["consolidate_regions", "summarise_consolidation", "dates_in_order"]
