"""Local, idempotent build of the aggregated parquet cache from zipped TIFFs.

Replaces notebook 05_build_full_aggregated_cache_2.ipynb on machines that
cannot afford to extract the ~70 GB Sentinel-2 archives. Reads TIFFs in place
through GDAL's `/vsizip/` virtual filesystem.

Inputs (in `data/` relative to repo root):
    track1_download_link_1.zip            CSVs (labels + guide). Required.
    track1_download_link_{2,3,4,5}.zip    TIFFs. Any subset present is used.
    saci_enrichment_v9.pkl                Optional; not used (enrichment is skipped).

Outputs (in `data/cache/`):
    points_meta.geoparquet
    observations_base.parquet
    phenophases.parquet
    MANIFEST.json
    _checkpoint_pts/{point_id}.pkl    Per-point pickles (resume support).

Re-run after dropping additional zips into `data/`: only points whose region
became reachable will be processed (existing checkpoints are reused).
"""
from __future__ import annotations

import gc
import pickle
import statistics as st
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

# Make src/ importable when running from the repo root or anywhere else.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from rasterio.warp import transform_bounds  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from src.data.cache_loader import load_aggregated_cache, load_manifest  # noqa: E402
from src.data.cache_writer import write_aggregated_cache  # noqa: E402
from src.data.point_extractor import extract_bands_at_point  # noqa: E402
from src.data.sample_strategy import assign_region_to_points  # noqa: E402
from src.data.sentinel2_loader import MODEL_BANDS, parse_tiff_filename  # noqa: E402
from src.data.temporal_builder import (  # noqa: E402
    N_FEATURES,
    PointSeries,
    build_point_series,
)
from src.data.vegetation_indices import INDEX_NAMES, compute_all_indices  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
CSV_CACHE = DATA_DIR / "_csv_cache"
CKPT_DIR = CACHE_DIR / "_checkpoint_pts"
VERSION = "v1_full778"

TIFF_ZIP_NUMBERS = ("2", "3", "4", "5")  # link_1 holds CSVs only


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def vsizip_uri(zip_path: Path, inner_name: str) -> str:
    """GDAL VSI URI for a member inside a zip. Forward slashes only."""
    zip_posix = zip_path.resolve().as_posix()
    inner_posix = inner_name.replace("\\", "/")
    return f"/vsizip/{zip_posix}/{inner_posix}"


def find_tiff_zips() -> list[Path]:
    """Return present TIFF zips in deterministic, ascending link-number order."""
    found: list[Path] = []
    for n in TIFF_ZIP_NUMBERS:
        p = DATA_DIR / f"track1_download_link_{n}.zip"
        if p.exists():
            found.append(p)
    return found


def extract_csvs_once(dest: Path) -> Path:
    """Find/extract `points_train_label.csv` to `dest` (idempotent).

    Search order:
        1. Already in dest (any subfolder).
        2. Already loose in DATA_DIR (any subfolder).
        3. Inside any track1_download_link_*.zip.

    Returns the path to points_train_label.csv. Raises FileNotFoundError with
    actionable message if not found anywhere.
    """
    dest.mkdir(parents=True, exist_ok=True)

    found = next(dest.rglob("points_train_label.csv"), None)
    if found is not None:
        return found
    found = next(DATA_DIR.rglob("points_train_label.csv"), None)
    if found is not None:
        return found

    for zp in sorted(DATA_DIR.glob("track1_download_link_*.zip")):
        try:
            with zipfile.ZipFile(zp) as zf:
                for info in zf.infolist():
                    if info.filename.lower().endswith(".csv"):
                        zf.extract(info, dest)
        except zipfile.BadZipFile:
            print(f"  warning: {zp.name} is not a valid zip — skipped")
            continue

    found = next(dest.rglob("points_train_label.csv"), None)
    if found is None:
        zips_seen = sorted(p.name for p in DATA_DIR.glob("track1_download_link_*.zip"))
        raise FileNotFoundError(
            "points_train_label.csv was not found in data/ or in any of the zips "
            f"present ({zips_seen}). It typically ships with track1_download_link_5.zip "
            "(= region_train_1). Drop that zip (or the loose CSV) into data/ and re-run."
        )
    return found


def build_view_from_zips(
    tiff_zips: list[Path],
) -> tuple[dict[str, dict[str, dict[str, str]]], dict[str, str]]:
    """Build the consolidated view directly from zip namelists.

    Returns:
        view: {region: {date: {band: vsizip_uri}}}
        zip_compression: {zip_filename: 'STORED' | 'DEFLATE' | 'OTHER:N'}
    """
    view: dict[str, dict[str, dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
    compression_summary: dict[str, str] = {}

    for zp in tiff_zips:
        with zipfile.ZipFile(zp) as zf:
            n_tiffs = 0
            n_parsed = 0
            comp_codes: set[int] = set()
            for info in zf.infolist():
                if info.is_dir():
                    continue
                lower = info.filename.lower()
                if not lower.endswith((".tif", ".tiff")):
                    continue
                n_tiffs += 1
                comp_codes.add(info.compress_type)
                meta = parse_tiff_filename(info.filename)
                if meta is None:
                    continue
                n_parsed += 1
                uri = vsizip_uri(zp, info.filename)
                # Later zip wins on (region, date, band) collision (matches notebook semantics).
                view[meta.region][meta.date][meta.band] = uri

        if not comp_codes:
            comp = "EMPTY"
        elif comp_codes == {zipfile.ZIP_STORED}:
            comp = "STORED"
        elif comp_codes == {zipfile.ZIP_DEFLATED}:
            comp = "DEFLATE"
        elif comp_codes == {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
            comp = "MIXED(STORED+DEFLATE)"
        else:
            comp = "OTHER:" + ",".join(str(c) for c in sorted(comp_codes))
        compression_summary[zp.name] = comp
        print(f"  [{zp.name}] {n_tiffs} TIFFs, {n_parsed} parsed, compression={comp}")

    # Freeze nested defaultdicts → plain dicts.
    frozen = {r: {d: dict(b) for d, b in dates.items()} for r, dates in view.items()}
    return frozen, compression_summary


def merge_or_build_point(
    point_id: int,
    lon: float,
    lat: float,
    region: str,
    view: dict[str, dict[str, dict[str, str]]],
    pheno_map: dict[str, str],
    crop: str | None,
    ckpt_dir: Path,
) -> tuple[str, PointSeries | None]:
    """Build or extend a PointSeries.

    Returns (status, ps) where status ∈ {"built", "merged", "up_to_date"}.
    - "built": no checkpoint existed; full extraction via build_point_series.
    - "merged": checkpoint existed and the new view introduces dates the
      pickle didn't have. Old (date,features,mask) rows are preserved;
      missing dates are extracted on the fly. Pheno_map/crop refreshed.
    - "up_to_date": checkpoint already covers every date in the new view —
      no work needed. The on-disk pickle is left untouched.
    """
    pkl = ckpt_dir / f"{point_id}.pkl"
    region_view = view.get(region, {})
    new_dates = set(region_view.keys())

    if not pkl.exists():
        ps = build_point_series(
            point_id=point_id, lon=lon, lat=lat, region=region,
            consolidated_view=view,
            phenophase_by_date=pheno_map, crop_type=crop,
        )
        return ("built", ps)

    with open(pkl, "rb") as f:
        old: PointSeries = pickle.load(f)

    old_dates_set = set(old.dates)
    missing = new_dates - old_dates_set
    if not missing:
        return ("up_to_date", old)

    old_idx = {d: i for i, d in enumerate(old.dates)}
    final_dates = sorted(old_dates_set | new_dates)
    T = len(final_dates)
    X = np.full((T, N_FEATURES), np.nan, dtype=np.float64)
    mask = np.zeros(T, dtype=bool)

    for t, d in enumerate(final_dates):
        if d in old_idx:
            i = old_idx[d]
            X[t] = old.features[i]
            mask[t] = old.mask[i]
            continue
        # New date — extract from current zip view
        band_paths = region_view[d]
        bands_vec = extract_bands_at_point(band_paths, lon, lat, list(MODEL_BANDS))
        if np.all(np.isnan(bands_vec[:8])):
            continue
        X[t, : len(MODEL_BANDS)] = bands_vec
        indices = compute_all_indices(bands_vec, scale=True)
        for i, name in enumerate(INDEX_NAMES):
            X[t, len(MODEL_BANDS) + i] = indices[name]
        mask[t] = not np.any(np.isnan(bands_vec))

    merged_ps = PointSeries(
        point_id=point_id,
        region=region,
        lon=lon,
        lat=lat,
        dates=final_dates,
        features=X,
        mask=mask,
        crop_type=crop,
        phenophase_by_date=pheno_map,
    )
    return ("merged", merged_ps)


def index_region_bboxes_via_vsizip(
    view: dict[str, dict[str, dict[str, str]]],
) -> dict[str, tuple[float, float, float, float]]:
    """Open one vsizip URI per region and reproject its bounds to WGS84."""
    bboxes: dict[str, tuple[float, float, float, float]] = {}
    for region, dates in view.items():
        # Pick first available date and any band (B02 preferred for size).
        first_date = next(iter(dates))
        bands = dates[first_date]
        path = bands.get("B02") or next(iter(bands.values()))
        with rasterio.open(path) as src:
            bounds_wgs84 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        bboxes[region] = tuple(float(x) for x in bounds_wgs84)
    return bboxes


# ─────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────
def main() -> int:
    print(f"Repo root : {REPO_ROOT}")
    print(f"Data dir  : {DATA_DIR}")
    print(f"Cache dir : {CACHE_DIR}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Discover zips
    tiff_zips = find_tiff_zips()
    if not tiff_zips:
        print("ERROR: no track1_download_link_{2,3,4,5}.zip found in data/.")
        return 2
    print("\n[1/6] TIFF zips found:")
    for zp in tiff_zips:
        size_gb = zp.stat().st_size / 1024**3
        print(f"  {zp.name}  ({size_gb:.2f} GB)")

    # ── 2. Extract CSVs from link_1
    print("\n[2/6] CSV extraction (idempotent)")
    label_csv = extract_csvs_once(CSV_CACHE)
    print(f"  labels: {label_csv}")
    labels_df = pd.read_csv(label_csv)
    print(f"  labels rows: {len(labels_df)}, unique points: {labels_df['point_id'].nunique()}")

    # ── 3. Consolidate (no extraction)
    print("\n[3/6] Consolidating regions from zips (vsizip)")
    view, comp = build_view_from_zips(tiff_zips)
    if not view:
        print("ERROR: no parseable TIFFs found across the provided zips.")
        return 3
    n_dates = [len(dates) for dates in view.values()]
    n_files = sum(sum(len(b) for b in dates.values()) for dates in view.values())
    print(
        f"  regions: {len(view)} | "
        f"dates/region min={min(n_dates)} median={int(st.median(n_dates))} max={max(n_dates)} | "
        f"total band-files: {n_files}"
    )
    deflate_zips = [name for name, c in comp.items() if "DEFLATE" in c]
    if deflate_zips:
        print(
            "  NOTE: DEFLATE compression detected in "
            f"{deflate_zips} — random-access reads will be slower than STORED."
        )

    # ── 4. Bboxes + region assignment
    print("\n[4/6] Bboxes + region assignment")
    bboxes = index_region_bboxes_via_vsizip(view)
    print(f"  indexed {len(bboxes)} region bboxes")

    points_df = (
        labels_df.drop_duplicates("point_id")[["point_id", "Longitude", "Latitude"]]
        .reset_index(drop=True)
    )
    p2r_df = assign_region_to_points(points_df, bboxes)
    matched = p2r_df["region"].notnull().sum()
    print(f"  points matched to a region: {matched}/{len(points_df)}")
    point_to_region = dict(zip(p2r_df["point_id"], p2r_df["region"]))

    # ── 5. Per-point build / merge (sequential, checkpointed)
    print("\n[5/6] Per-point build / merge")
    on_disk = {int(p.stem) for p in CKPT_DIR.glob("*.pkl")}
    print(f"  checkpoint: {len(on_disk)} points already on disk (will be merged with new view)")

    tasks: list[tuple[int, float, float, str | None, dict[str, str], str | None]] = []
    skipped_no_region = 0
    for pid, group in labels_df.groupby("point_id"):
        pid_i = int(pid)
        region = point_to_region.get(pid_i)
        # Region not present in current view: skip if no checkpoint either;
        # if a checkpoint exists, leave it untouched (it carries data from
        # an earlier run with a different zip set).
        if region is None or region not in view:
            if pid_i not in on_disk:
                skipped_no_region += 1
            continue
        row0 = group.iloc[0]
        pheno_map = dict(zip(group["phenophase_date"], group["phenophase_name"]))
        crop = str(row0.get("crop_type", "")) or None
        tasks.append(
            (pid_i, float(row0["Longitude"]), float(row0["Latitude"]), region, pheno_map, crop)
        )
    print(
        f"  to process: {len(tasks)} points "
        f"(skipped {skipped_no_region} with no region in current zips and no prior checkpoint)"
    )

    counters = {"built": 0, "merged": 0, "up_to_date": 0}
    errors: list[int] = []
    pbar = tqdm(tasks, desc="extract")
    for pid_i, lon, lat, region, pheno_map, crop in pbar:
        try:
            status, ps = merge_or_build_point(
                pid_i, lon, lat, region, view, pheno_map, crop, CKPT_DIR
            )
            if ps is not None and status != "up_to_date":
                with open(CKPT_DIR / f"{ps.point_id}.pkl", "wb") as pf:
                    pickle.dump(ps, pf)
            counters[status] += 1
            pbar.set_postfix(counters, refresh=False)
        except Exception as e:
            print(f"  error point {pid_i}: {e}")
            errors.append(pid_i)
        if (counters["built"] + counters["merged"]) % 50 == 0:
            gc.collect()
    print(f"  counters: {counters}")
    if errors:
        print(f"  errors on {len(errors)} points: {errors}")

    # Reload from disk → series_list (sorted)
    print("  reloading checkpoints…")
    series_list = []
    for pf in tqdm(sorted(CKPT_DIR.glob("*.pkl"), key=lambda p: int(p.stem)), desc="load"):
        with open(pf, "rb") as f:
            series_list.append(pickle.load(f))
    if not series_list:
        print("ERROR: no PointSeries built — aborting before cache write.")
        return 4

    T_lens = [len(ps.dates) for ps in series_list]
    valid = [int(ps.mask.sum()) for ps in series_list]
    print(
        f"  series_list: {len(series_list)} pts | "
        f"T per pt min={min(T_lens)} median={int(st.median(T_lens))} max={max(T_lens)} | "
        f"valid obs min={min(valid)} median={int(st.median(valid))} max={max(valid)}"
    )

    # ── 6. Write cache + smoke-test reload
    print("\n[6/6] Writing parquet cache (base, 17 features, no enrichment)")
    paths = write_aggregated_cache(
        series_list=series_list,
        cache_dir=CACHE_DIR,
        version=VERSION,
        agro_by_point=None,
    )
    for k, p in paths.items():
        size_mb = p.stat().st_size / 1024**2 if p.exists() else 0.0
        print(f"  {k}: {p}  ({size_mb:.2f} MB)")

    print("\n[validation] Reloading via cache_loader.load_aggregated_cache")
    manifest = load_manifest(CACHE_DIR)
    print(f"  manifest: version={manifest['version']} n_points={manifest['n_points']} "
          f"n_regions={manifest['n_regions']} n_obs_base={manifest['n_obs_base']}")
    rs_base = load_aggregated_cache(CACHE_DIR, enriched=False)
    import numpy as np
    by_pid = {ps.point_id: ps for ps in rs_base}
    parity_errs = 0
    for ps in series_list:
        p2 = by_pid.get(ps.point_id)
        if p2 is None or ps.dates != p2.dates:
            parity_errs += 1
            continue
        if not np.allclose(ps.features, p2.features, equal_nan=True, rtol=0, atol=1e-6):
            parity_errs += 1
        if not np.array_equal(ps.mask, p2.mask):
            parity_errs += 1
    print(f"  parity errors: {parity_errs} (must be 0)")

    meta = pd.read_parquet(CACHE_DIR / "points_meta.geoparquet")
    print("  crop_type counts:")
    print(meta["crop_type"].value_counts().to_string())
    print(f"  regions covered: {meta['region'].nunique()}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
