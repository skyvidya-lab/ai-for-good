# Dataset Anatomy — Track 1 Final Round

Derived from `background/.../Guide to the Second Round_track1/analise-dataset.md` (2026-04-16).

## Download Bundle (Google Drive)

`/Shareddrives/SKYVIDYA/AI for Good/datasets_final_round/`

| File | Size | Content |
|---|---:|---|
| `track1_download_link_1.zip` | 129 MB | Guide PDFs + `points_train_label.csv` + `test_point.csv` + test sample |
| `track1_download_link_2.zip` | 17.48 GB | `region_train_4/` GeoTIFFs |
| `track1_download_link_3.zip` | 17.49 GB | `region_train_3/` GeoTIFFs |
| `track1_download_link_4.zip` | 17.48 GB | `region_train_2/` GeoTIFFs |
| `track1_download_link_5.zip` | 17.48 GB | `region_train_1/` + `points_train_label.csv` |
| **Total** | ~70.2 GB | 10 245 files |

## Folder Layout (after extraction)

```
region_train_1/     # plain, no subdirs
region_train_2/
region_train_3/
region_train_4/
points_train_label.csv
```

## Filename Pattern (GeoTIFFs)

`regionNN_YYYY-MM-DD-HH-MM_YYYY-MM-DD-HH-MM_Sentinel-2_L2A_BXX_(Raw).tiff`

Known edge cases:
- Some files use `_` instead of `-` between `HH` and `MM` (e.g. `2018-10-11-00_00`).
- At least 1 file has a typo (`region542018-07-23-...` — missing underscore).
- Parser in [src/data/sentinel2_loader.py](../../../src/data/sentinel2_loader.py) is tolerant.

## Region Inventory

- **50 unique region IDs** among `region00..region57`.
- Gaps: `03, 05, 14, 19, 22, 23, 30, 41, 55` do not exist.
- Each folder covers 47–49 of the 50 regions (not a partition).

## Temporal Depth

| Folder | Mean dates/region | Min | Max |
|---|---:|---:|---:|
| region_train_1 | 4.7 | 1 | 12 |
| region_train_2 | 4.4 | 1 | 15 |
| region_train_3 | 4.4 | 1 | 12 |
| region_train_4 | 4.5 | 1 | 11 |

**Consolidated across folders** a region can have up to ~18 unique dates. This unlocks Hurst temporal viability for most points.

## Bands

13 bands per (region, date): `B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12`.

- **B10** is unusual in L2A products (it's a L1C cirrus band). **Excluded** from `MODEL_BANDS`.
- Native resolutions: B02/B03/B04/B08 = 10 m; B05/B06/B07/B8A/B11/B12 = 20 m; B01/B09 = 60 m.
- `point_extractor.py` samples at the point coordinate; rasterio handles per-band CRS/resolution internally.

## Labels

`points_train_label.csv` schema:

```
point_id, Longitude, Latitude, phenophase_date, crop_type, phenophase_name
```

- 5 447 rows
- 778 unique `point_id`
- Each `point_id` has **exactly 7 rows**, one per phenophase
- Phenophases: `Dormancy, Greenup, MidGreenup, Peak, Maturity, MidSenescence, Senescence`
- Class distribution: perfectly balanced (778 samples × 7 stages = 5 446; actual 5 447, likely 1 duplicate)

## Crop Classes

| Class | Points | Labeled rows |
|---|---:|---:|
| rice | 367 | 2 569 |
| corn | 229 | 1 603 |
| soybean | 182 | 1 274 |
| **Total** | **778** | **5 447** |

**Note:** README mentions a 4th class `background`, but it does **not** appear in training labels. It may appear in the private test set → use Kalman `trace(P)` as OOD detector, threshold routes uncertain points to background.

## Split Strategy

- **No official split** is provided.
- **Mandatory:** `GroupKFold(n_splits=5, groups=point_id)` to avoid leakage across the 7 phenophases of the same point.
- Recommended: stratify by `crop_type` when selecting fold assignments.

## Points of Attention

1. **B10 usability** — keep out of model bands unless future analysis shows signal.
2. **Typo region54** — parser tolerant.
3. **Irregular date counts (1–15)** — filter points in regions with < 3 dates, or rely on masking + NaN-aware Hurst.
4. **4 folders overlap spatially** — consolidate, do NOT treat as splits.
5. **No background class in training** — handle via uncertainty threshold.
