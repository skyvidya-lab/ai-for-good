"""
Sentinel-2 L2A TIFF filename parser and region/date inventory.

Canonical pattern:
    regionNN_YYYY-MM-DD-HH-MM_YYYY-MM-DD-HH-MM_Sentinel-2_L2A_BXX_(Raw).tiff

Known edge cases (from dataset analysis):
    - Some files use `_` instead of `-` in HH_MM (e.g. 2018-10-11-00_00_...)
    - At least 1 file is missing the underscore after region (`region542018-07-23-...`)
    - B10 is present (unusual for L2A — kept but flagged)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ALL_BANDS: tuple[str, ...] = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
)

# Bands used for modelling (skip B10 — cirrus, unusual in L2A)
MODEL_BANDS: tuple[str, ...] = tuple(b for b in ALL_BANDS if b != "B10")

# Tolerant regex:
#  - region(NN) optionally followed by underscore
#  - two date/time blocks separated by underscore; inside each block, `_` or `-` between H and M
#  - band token B01..B12 or B8A
_FILENAME_RE = re.compile(
    r"""
    ^region_?(?P<region>\d+)_?
    (?P<date1>\d{4}-\d{2}-\d{2})[-_](?P<time1>\d{2})[-_](?P<min1>\d{2})
    _
    (?P<date2>\d{4}-\d{2}-\d{2})[-_](?P<time2>\d{2})[-_](?P<min2>\d{2})
    _Sentinel-2_L2A_
    (?P<band>B0[1-9]|B10|B11|B12|B8A)
    _\(Raw\)\.tiff?$
    """,
    re.VERBOSE | re.IGNORECASE,
)


@dataclass(frozen=True)
class TiffMeta:
    """Parsed metadata from a single Sentinel-2 TIFF filename."""

    region: str  # e.g. "region00"
    date: str    # ISO date, e.g. "2018-07-23"
    band: str    # e.g. "B04"
    path: Path

    @property
    def region_id(self) -> int:
        return int(self.region.replace("region", ""))

    @property
    def key(self) -> tuple[str, str]:
        """(region, date) tuple used as grouping key."""
        return (self.region, self.date)


def parse_tiff_filename(path: str | Path) -> TiffMeta | None:
    """Parse a Sentinel-2 TIFF path. Returns None if it does not match the pattern."""
    p = Path(path)
    m = _FILENAME_RE.match(p.name)
    if not m:
        return None
    region = f"region{int(m.group('region')):02d}"
    return TiffMeta(
        region=region,
        date=m.group("date1"),
        band=m.group("band").upper(),
        path=p,
    )


def inventory_tiffs(
    roots: list[str | Path],
    regions_filter: set[str] | None = None,
) -> list[TiffMeta]:
    """
    Recursively inventory all Sentinel-2 TIFFs under the given roots.

    Args:
        roots: directories to scan (e.g. the 4 consolidated region_train folders).
        regions_filter: optional set of region names to keep (e.g. {"region00", "region02"}).

    Returns:
        List of TiffMeta, skipping files that don't match the pattern.
    """
    metas: list[TiffMeta] = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for p in root_path.rglob("*.tif*"):
            meta = parse_tiff_filename(p)
            if meta is None:
                continue
            if regions_filter is not None and meta.region not in regions_filter:
                continue
            metas.append(meta)
    return metas


def group_by_region_date(metas: list[TiffMeta]) -> dict[tuple[str, str], dict[str, Path]]:
    """
    Group metas by (region, date) → {band: path}.

    Silently overwrites duplicates (e.g. same band across folders) — consolidator
    is responsible for deciding precedence.
    """
    grouped: dict[tuple[str, str], dict[str, Path]] = {}
    for m in metas:
        grouped.setdefault(m.key, {})[m.band] = m.path
    return grouped


__all__ = [
    "ALL_BANDS",
    "MODEL_BANDS",
    "TiffMeta",
    "parse_tiff_filename",
    "inventory_tiffs",
    "group_by_region_date",
]
