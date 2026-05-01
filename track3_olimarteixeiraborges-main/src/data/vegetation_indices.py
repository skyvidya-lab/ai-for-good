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


# ─────────────────────────────────────────────────────────────────────
# Extended indices (Tier-1 expansion) — derived from MODEL_BANDS only.
# Designed for: water discrimination (rice flooded), chlorophyll
# (Maturity / Senescence), biophysical proxies (LAI / FAPAR / FCOVER),
# and non-photosynthetic vegetation (residue / harvest).
# ─────────────────────────────────────────────────────────────────────
def mndwi(green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """Modified NDWI (Xu 2006). Strong for surface water."""
    green = np.asarray(green, dtype=np.float64)
    swir1 = np.asarray(swir1, dtype=np.float64)
    return (green - swir1) / (green + swir1 + EPS)


def awei(green: np.ndarray, nir: np.ndarray, swir1: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """Automated Water Extraction Index (Feyisa 2014, no-shadow variant)."""
    green = np.asarray(green, dtype=np.float64)
    nir = np.asarray(nir, dtype=np.float64)
    swir1 = np.asarray(swir1, dtype=np.float64)
    swir2 = np.asarray(swir2, dtype=np.float64)
    return 4.0 * (green - swir1) - (0.25 * nir + 2.75 * swir2)


def ndre(b8a: np.ndarray, b5: np.ndarray) -> np.ndarray:
    """Normalised Difference Red-Edge — chlorophyll proxy (Barnes 2000)."""
    b8a = np.asarray(b8a, dtype=np.float64)
    b5 = np.asarray(b5, dtype=np.float64)
    return (b8a - b5) / (b8a + b5 + EPS)


def mtci(b6: np.ndarray, b5: np.ndarray, b4: np.ndarray) -> np.ndarray:
    """MERIS Terrestrial Chlorophyll Index (Dash & Curran 2004)."""
    b6 = np.asarray(b6, dtype=np.float64)
    b5 = np.asarray(b5, dtype=np.float64)
    b4 = np.asarray(b4, dtype=np.float64)
    return (b6 - b5) / (b5 - b4 + EPS)


def fcover_proxy(ndvi_v: np.ndarray) -> np.ndarray:
    """Fraction Vegetation Cover via NDVI rescaling (Carlson & Ripley 1997).
    fcover ≈ ((NDVI - NDVI_soil) / (NDVI_veg - NDVI_soil))**2, clipped to [0,1]."""
    n = np.clip(np.asarray(ndvi_v, dtype=np.float64), 0.0, 1.0)
    fc = (n - 0.05) / 0.85
    return np.clip(fc, 0.0, 1.0)


def lai_proxy(ndvi_v: np.ndarray) -> np.ndarray:
    """LAI proxy via Beer's law on FCOVER. Empirical, capped at 7."""
    fc = fcover_proxy(ndvi_v)
    fc = np.clip(fc, 0.0, 0.99)
    lai = -2.0 * np.log(1.0 - fc)
    return np.clip(lai, 0.0, 7.0)


def fapar_proxy(ndvi_v: np.ndarray) -> np.ndarray:
    """FAPAR proxy from Myneni & Williams 1994 linear approximation."""
    n = np.asarray(ndvi_v, dtype=np.float64)
    return np.clip(1.24 * n - 0.168, 0.0, 1.0)


def fnpv_proxy(swir1: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """Non-photosynthetic vegetation proxy (Guerschman 2009 simplified):
    increases with cellulose / lignin → senesced canopy and crop residue."""
    swir1 = np.asarray(swir1, dtype=np.float64)
    swir2 = np.asarray(swir2, dtype=np.float64)
    return swir2 / (swir1 + EPS)


EXTRA_INDEX_NAMES: tuple[str, ...] = (
    "mndwi", "awei", "ndre", "mtci", "fcover", "lai", "fapar", "fnpv",
)
EXTENDED_INDEX_NAMES: tuple[str, ...] = INDEX_NAMES + EXTRA_INDEX_NAMES  # 13 total


def compute_extended_indices(bands_vec: np.ndarray, scale: bool = True) -> dict[str, float]:
    """Compute all 13 extended indices (5 base + 8 new) from MODEL_BANDS vector."""
    v = scale_l2a(bands_vec) if scale else np.asarray(bands_vec, dtype=np.float64)
    blue = v[_BAND_IDX["B02"]]
    green = v[_BAND_IDX["B03"]]
    red = v[_BAND_IDX["B04"]]
    b5 = v[_BAND_IDX["B05"]]
    b6 = v[_BAND_IDX["B06"]]
    nir = v[_BAND_IDX["B08"]]
    b8a = v[_BAND_IDX["B8A"]]
    swir1 = v[_BAND_IDX["B11"]]
    swir2 = v[_BAND_IDX["B12"]]
    n = float(ndvi(nir, red))
    return {
        "ndvi": n,
        "evi": float(evi(nir, red, blue)),
        "ndwi": float(ndwi(green, nir)),
        "savi": float(savi(nir, red)),
        "lswi": float(lswi(nir, swir1)),
        "mndwi": float(mndwi(green, swir1)),
        "awei": float(awei(green, nir, swir1, swir2)),
        "ndre": float(ndre(b8a, b5)),
        "mtci": float(mtci(b6, b5, red)),
        "fcover": float(fcover_proxy(n)),
        "lai": float(lai_proxy(n)),
        "fapar": float(fapar_proxy(n)),
        "fnpv": float(fnpv_proxy(swir1, swir2)),
    }


__all__ = [
    "scale_l2a",
    "ndvi",
    "evi",
    "ndwi",
    "savi",
    "lswi",
    "mndwi",
    "awei",
    "ndre",
    "mtci",
    "fcover_proxy",
    "lai_proxy",
    "fapar_proxy",
    "fnpv_proxy",
    "compute_all_indices",
    "compute_extended_indices",
    "INDEX_NAMES",
    "EXTRA_INDEX_NAMES",
    "EXTENDED_INDEX_NAMES",
    "L2A_SCALE",
]
