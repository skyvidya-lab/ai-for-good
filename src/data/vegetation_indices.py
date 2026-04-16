"""
Vegetation indices derived from Sentinel-2 L2A bands.

All indices operate on reflectance values (divide by 10000 if using raw L2A DNs).
The loader returns raw integer values; indices here assume scaled reflectance.

Conventions:
    B02 = Blue (490 nm)
    B03 = Green (560 nm)
    B04 = Red (665 nm)
    B08 = NIR (842 nm)
    B8A = Narrow NIR (865 nm)
    B11 = SWIR-1 (1610 nm)
    B12 = SWIR-2 (2190 nm)
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6
L2A_SCALE = 10000.0  # Sentinel-2 L2A raw values are reflectance * 10000


def scale_l2a(raw: np.ndarray) -> np.ndarray:
    """Convert raw L2A DN to reflectance [0, 1]."""
    return np.asarray(raw, dtype=np.float64) / L2A_SCALE


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index."""
    nir = np.asarray(nir, dtype=np.float64)
    red = np.asarray(red, dtype=np.float64)
    return (nir - red) / (nir + red + EPS)


def evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """Enhanced Vegetation Index (coefficients for Sentinel-2)."""
    nir = np.asarray(nir, dtype=np.float64)
    red = np.asarray(red, dtype=np.float64)
    blue = np.asarray(blue, dtype=np.float64)
    return 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + EPS)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalized Difference Water Index (McFeeters 1996)."""
    green = np.asarray(green, dtype=np.float64)
    nir = np.asarray(nir, dtype=np.float64)
    return (green - nir) / (green + nir + EPS)


def savi(nir: np.ndarray, red: np.ndarray, L: float = 0.5) -> np.ndarray:
    """Soil-Adjusted Vegetation Index."""
    nir = np.asarray(nir, dtype=np.float64)
    red = np.asarray(red, dtype=np.float64)
    return (1 + L) * (nir - red) / (nir + red + L + EPS)


def lswi(nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """Land Surface Water Index (rice-sensitive)."""
    nir = np.asarray(nir, dtype=np.float64)
    swir1 = np.asarray(swir1, dtype=np.float64)
    return (nir - swir1) / (nir + swir1 + EPS)


# Band name -> index in MODEL_BANDS (B01..B09, B8A, B11, B12)
# MODEL_BANDS order: ("B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12")
_BAND_IDX = {
    "B01": 0, "B02": 1, "B03": 2, "B04": 3, "B05": 4, "B06": 5,
    "B07": 6, "B08": 7, "B8A": 8, "B09": 9, "B11": 10, "B12": 11,
}


def compute_all_indices(bands_vec: np.ndarray, scale: bool = True) -> dict[str, float]:
    """
    Compute the 5 vegetation indices from a (12,) band vector in MODEL_BANDS order.

    Returns dict {ndvi, evi, ndwi, savi, lswi}. NaN-propagating.
    """
    v = scale_l2a(bands_vec) if scale else np.asarray(bands_vec, dtype=np.float64)
    blue = v[_BAND_IDX["B02"]]
    green = v[_BAND_IDX["B03"]]
    red = v[_BAND_IDX["B04"]]
    nir = v[_BAND_IDX["B08"]]
    swir1 = v[_BAND_IDX["B11"]]
    return {
        "ndvi": float(ndvi(nir, red)),
        "evi": float(evi(nir, red, blue)),
        "ndwi": float(ndwi(green, nir)),
        "savi": float(savi(nir, red)),
        "lswi": float(lswi(nir, swir1)),
    }


INDEX_NAMES: tuple[str, ...] = ("ndvi", "evi", "ndwi", "savi", "lswi")


__all__ = [
    "scale_l2a",
    "ndvi",
    "evi",
    "ndwi",
    "savi",
    "lswi",
    "compute_all_indices",
    "INDEX_NAMES",
    "L2A_SCALE",
]
