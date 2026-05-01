"""Extend the existing parquet cache with 8 new vegetation indices.

Reads the band columns (B01..B12, B8A) already present in
`observations_base.parquet` and `observations_enriched.parquet` and computes
the Tier-1 expansion in-place — no TIFF I/O, no zip access. Output:

    data/cache/observations_extended.parquet  (25 features = 12 bands + 13 indices)
    data/cache/observations_full.parquet      (29 features = extended + 4 agro)
    data/cache/MANIFEST.json                  (re-emitted with new file blocks)

Idempotent. Safe to re-run after every cache rebuild.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.cache_writer import AGRO_FEATURES, PHENOPHASES_CANON  # noqa: E402
from src.data.sentinel2_loader import MODEL_BANDS  # noqa: E402
from src.data.vegetation_indices import (  # noqa: E402
    EXTENDED_INDEX_NAMES,
    EXTRA_INDEX_NAMES,
    INDEX_NAMES,
    awei,
    fapar_proxy,
    fcover_proxy,
    fnpv_proxy,
    lai_proxy,
    mndwi,
    mtci,
    ndre,
    scale_l2a,
)

CACHE_DIR = REPO_ROOT / "data" / "cache"
EXTENDED_FEATURE_NAMES = tuple(MODEL_BANDS) + EXTENDED_INDEX_NAMES  # 25
FULL_FEATURE_NAMES = EXTENDED_FEATURE_NAMES + AGRO_FEATURES  # 29


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _compute_extra_columns(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return dict of new index columns aligned to df rows.

    Bands stored as raw L2A DN (reflectance × 10000); scale before deriving.
    """
    n = len(df)
    out = {name: np.full(n, np.nan, dtype=np.float64) for name in EXTRA_INDEX_NAMES}

    bands = {b: df[b].to_numpy(dtype=np.float64, na_value=np.nan) for b in MODEL_BANDS}
    valid = np.ones(n, dtype=bool)
    for b in ("B03", "B05", "B06", "B08", "B8A", "B11", "B12"):
        valid &= np.isfinite(bands[b])

    if not valid.any():
        return out

    sel = valid
    g = bands["B03"][sel] / 10000.0
    b5 = bands["B05"][sel] / 10000.0
    b6 = bands["B06"][sel] / 10000.0
    red = bands["B04"][sel] / 10000.0  # noqa: F841 (kept for clarity)
    nir = bands["B08"][sel] / 10000.0
    b8a = bands["B8A"][sel] / 10000.0
    s1 = bands["B11"][sel] / 10000.0
    s2 = bands["B12"][sel] / 10000.0

    out["mndwi"][sel] = mndwi(g, s1)
    out["awei"][sel] = awei(g, nir, s1, s2)
    out["ndre"][sel] = ndre(b8a, b5)
    out["mtci"][sel] = mtci(b6, b5, bands["B04"][sel] / 10000.0)
    # NDVI is already a column in the parquet — reuse it for FCOVER/LAI/FAPAR
    n_ndvi = df["ndvi"].to_numpy(dtype=np.float64, na_value=np.nan)[sel]
    out["fcover"][sel] = fcover_proxy(n_ndvi)
    out["lai"][sel] = lai_proxy(n_ndvi)
    out["fapar"][sel] = fapar_proxy(n_ndvi)
    out["fnpv"][sel] = fnpv_proxy(s1, s2)
    return out


def extend_parquet(src_path: Path, dst_path: Path, *, has_agro: bool) -> Path:
    print(f"\n[extend] {src_path.name} -> {dst_path.name}")
    df = pd.read_parquet(src_path)
    expected_cols = ["point_id", "date", "mask", *MODEL_BANDS, *INDEX_NAMES]
    if has_agro:
        expected_cols += list(AGRO_FEATURES)
    missing = [c for c in expected_cols if c not in df.columns]
    assert not missing, f"missing columns in {src_path}: {missing}"

    new_cols = _compute_extra_columns(df)
    for name, arr in new_cols.items():
        df[name] = arr

    # Reorder columns: meta, bands, all indices, optional agro, mask
    ordered = ["point_id", "date"]
    ordered += list(MODEL_BANDS)
    ordered += list(EXTENDED_INDEX_NAMES)
    if has_agro:
        ordered += list(AGRO_FEATURES)
    ordered += ["mask"]
    df = df[ordered]
    df.to_parquet(dst_path, compression="zstd", index=False)
    size_mb = dst_path.stat().st_size / 1024 ** 2
    n_valid = int(df[list(EXTRA_INDEX_NAMES)].notna().any(axis=1).sum())
    print(f"  rows: {len(df)}  cols: {len(df.columns)}  valid extra-index rows: {n_valid}  ({size_mb:.2f} MB)")
    return dst_path


def update_manifest(paths: dict[str, Path]) -> Path:
    mp = CACHE_DIR / "MANIFEST.json"
    manifest = json.loads(mp.read_text())
    manifest["created_utc_extended"] = datetime.now(timezone.utc).isoformat()
    manifest["feature_names_extended"] = list(EXTENDED_FEATURE_NAMES)
    manifest["feature_names_full"] = list(FULL_FEATURE_NAMES)
    manifest.setdefault("files", {})
    for k, p in paths.items():
        if not p.exists():
            continue
        manifest["files"][k] = {
            "path": p.name,
            "md5": _md5(p),
            "bytes": p.stat().st_size,
        }
    mp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return mp


def main() -> int:
    base = CACHE_DIR / "observations_base.parquet"
    enriched = CACHE_DIR / "observations_enriched.parquet"
    if not base.exists():
        print(f"ERROR: {base} not found. Run scripts/build_full_aggregated_cache.py first.")
        return 2

    out: dict[str, Path] = {}
    out["observations_extended"] = extend_parquet(
        base, CACHE_DIR / "observations_extended.parquet", has_agro=False,
    )
    if enriched.exists():
        out["observations_full"] = extend_parquet(
            enriched, CACHE_DIR / "observations_full.parquet", has_agro=True,
        )
    else:
        print("note: observations_enriched.parquet missing — skipping observations_full.parquet")

    update_manifest(out)
    print(f"\nManifest updated. Wrote {len(out)} extended parquet(s).")
    print(f"Extended feature count : {len(EXTENDED_FEATURE_NAMES)} (12 bands + 13 indices)")
    print(f"Full feature count     : {len(FULL_FEATURE_NAMES)} (extended + {len(AGRO_FEATURES)} agro)")

    # Smoke check: round-trip via pandas
    for k, p in out.items():
        df = pd.read_parquet(p)
        n_finite = int(df[list(EXTRA_INDEX_NAMES)].notna().any(axis=1).sum())
        print(f"[verify] {p.name}: {len(df)} rows, {n_finite} with at least one new index")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
