# Sentinel-2 Bands

| Band | Centre λ (nm) | Native res (m) | Purpose | In `MODEL_BANDS`? |
|---|---:|---:|---|:---:|
| B01 | 443 | 60 | Coastal aerosol | ✓ |
| B02 | 490 | 10 | Blue | ✓ |
| B03 | 560 | 10 | Green | ✓ |
| B04 | 665 | 10 | Red | ✓ |
| B05 | 705 | 20 | Red Edge 1 | ✓ |
| B06 | 740 | 20 | Red Edge 2 | ✓ |
| B07 | 783 | 20 | Red Edge 3 | ✓ |
| B08 | 842 | 10 | NIR | ✓ |
| B8A | 865 | 20 | Narrow NIR | ✓ |
| B09 | 945 | 60 | Water vapour | ✓ |
| B10 | 1375 | 60 | Cirrus (L1C only) | ✗ excluded |
| B11 | 1610 | 20 | SWIR 1 | ✓ |
| B12 | 2190 | 20 | SWIR 2 | ✓ |

**12 bands in `MODEL_BANDS`** (see [src/data/sentinel2_loader.py](../../../../src/data/sentinel2_loader.py)).

## Why exclude B10?

B10 is a cirrus detection band, normally stripped from L2A products. Its presence in this dataset is unusual. Without a validated use case, we exclude it from features to avoid introducing noise. If future analysis shows discriminative signal, it can be re-added.

## Scaling

L2A raw values are reflectance × 10 000. Use `src.data.vegetation_indices.scale_l2a()` to get `[0, 1]`.
