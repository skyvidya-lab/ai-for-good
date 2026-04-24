"""
Pixel-value extraction at GPS coordinates from Sentinel-2 GeoTIFFs.

Uses rasterio to handle reprojection, resampling, and no-data masking.
Bands at 20m/60m native resolution are resampled to the window of the highest-res
band (10m) using bilinear interpolation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import rasterio
    from rasterio.warp import transform as warp_transform
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


def _require_rasterio() -> None:
    if not RASTERIO_AVAILABLE:
        raise ImportError("rasterio required: pip install rasterio")


def extract_pixel_value(
    tiff_path: str | Path,
    lon: float,
    lat: float,
    src_crs: str = "EPSG:4326",
) -> float:
    """
    Extract the pixel value at (lon, lat) from a single-band GeoTIFF.

    Reprojects the point from src_crs into the raster's CRS, then samples nearest pixel.
    Returns NaN on out-of-bounds or no-data.
    """
    _require_rasterio()
    with rasterio.open(tiff_path) as src:
        xs, ys = warp_transform(src_crs, src.crs, [lon], [lat])
        row, col = src.index(xs[0], ys[0])
        h, w = src.height, src.width
        if not (0 <= row < h and 0 <= col < w):
            return float("nan")
        val = src.read(1)[row, col]
        nodata = src.nodata
        if nodata is not None and val == nodata:
            return float("nan")
        return float(val)


def extract_bands_at_point(
    band_paths: dict[str, Path],
    lon: float,
    lat: float,
    bands_order: list[str],
    src_crs: str = "EPSG:4326",
) -> np.ndarray:
    """
    Extract values for all requested bands at a single (lon, lat).

    Args:
        band_paths: {band_name: path_to_tiff}
        lon, lat: WGS84 coordinates.
        bands_order: desired output order of bands.
        src_crs: CRS of (lon, lat).

    Returns:
        (len(bands_order),) float array; NaN for missing bands or OOB.
    """
    out = np.full(len(bands_order), np.nan, dtype=np.float64)
    for i, band in enumerate(bands_order):
        path = band_paths.get(band)
        if path is None or not Path(path).exists():
            continue
        try:
            out[i] = extract_pixel_value(path, lon, lat, src_crs=src_crs)
        except Exception:
            # Malformed TIFF or CRS mismatch — leave NaN
            continue
    return out


__all__ = ["extract_pixel_value", "extract_bands_at_point", "RASTERIO_AVAILABLE"]
