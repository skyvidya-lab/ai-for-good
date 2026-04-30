"""Add ERA5/SMAP agroclimate features to the existing base cache.

Reads:
    data/cache/observations_base.parquet  (must already exist)
    data/saci_enrichment_v9.pkl           (precomputed via GEE in Colab)

Writes:
    data/cache/observations_enriched.parquet  (17 base + 4 agro = 21 features)
    data/cache/MANIFEST.json                  (re-emitted with both file blocks)

No TIFF I/O. CPU-only, ~minutes for 778 points.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.data.cache_loader import load_aggregated_cache  # noqa: E402
from src.data.cache_writer import AGRO_FEATURES, write_aggregated_cache  # noqa: E402


# Replicated from src.data.agroclimate_enrichment.AgroclimateExtractor.align_to_ps
# to avoid importing the `ee` (earthengine-api) dependency, which is only needed
# when extracting fresh values from GEE — not when aligning a cached pickle.
def align_to_ps(ps_dates: list[str], agro_df) -> pd.DataFrame:
    default = {"precip_acc": 0.0, "soil_moisture": 0.0, "temp": 20.0, "smap_wetness": 0.5}
    if agro_df is None or (isinstance(agro_df, pd.DataFrame) and agro_df.empty):
        return pd.DataFrame([default] * len(ps_dates))

    agro_df = agro_df.copy()
    agro_df["date"] = pd.to_datetime(agro_df["date"])
    rows = []
    for d in ps_dates:
        target = pd.to_datetime(d)
        mask = (agro_df["date"] <= target) & (agro_df["date"] > target - pd.Timedelta(days=5))
        win = agro_df[mask]
        if len(win) > 0:
            rows.append({
                "precip_acc": float(win["total_precipitation_sum"].sum() * 1000.0
                                    if win["total_precipitation_sum"].notna().any() else 0.0),
                "soil_moisture": float(win["volumetric_soil_water_layer_1"].mean()
                                       if win["volumetric_soil_water_layer_1"].notna().any() else 0.0),
                "temp": float(win["temperature_2m"].mean() - 273.15
                              if win["temperature_2m"].notna().any() else 20.0),
                "smap_wetness": float(win["sm_surface_wetness"].mean()
                                      if win["sm_surface_wetness"].notna().any() else 0.5),
            })
        else:
            rows.append(default)
    return pd.DataFrame(rows)

CACHE_DIR = REPO_ROOT / "data" / "cache"
SACI_PATH = REPO_ROOT / "data" / "saci_enrichment_v9.pkl"
VERSION = "v1_full778"  # keep manifest version stable


def main() -> int:
    if not (CACHE_DIR / "observations_base.parquet").exists():
        print("ERROR: data/cache/observations_base.parquet not found. "
              "Run scripts/build_full_aggregated_cache.py first.")
        return 2
    if not SACI_PATH.exists():
        print(f"ERROR: {SACI_PATH} missing. Cannot enrich without GEE pickle.")
        return 3

    print(f"[1/3] Loading base cache from {CACHE_DIR}")
    series_list = load_aggregated_cache(CACHE_DIR, enriched=False)
    print(f"  {len(series_list)} PointSeries loaded "
          f"(F={series_list[0].features.shape[1]})")

    print(f"\n[2/3] Loading SACI agroclimate pickle ({SACI_PATH.name})")
    with open(SACI_PATH, "rb") as f:
        saci = pickle.load(f)
    print(f"  saci entries: {len(saci)}")

    agro_by_point = {}
    missing = 0
    for ps in series_list:
        src = saci.get(ps.point_id)
        agro_by_point[ps.point_id] = align_to_ps(ps.dates, src)
        if src is None:
            missing += 1
    print(f"  aligned {len(agro_by_point)} points (missing source: {missing})")
    print(f"  features added: {AGRO_FEATURES}")

    print("\n[3/3] Writing enriched cache")
    paths = write_aggregated_cache(
        series_list=series_list,
        cache_dir=CACHE_DIR,
        version=VERSION,
        agro_by_point=agro_by_point,
    )
    for k, p in paths.items():
        size_mb = p.stat().st_size / 1024**2 if p.exists() else 0.0
        print(f"  {k}: {p.name}  ({size_mb:.2f} MB)")

    # Smoke check
    print("\n[validation]")
    rs_enr = load_aggregated_cache(CACHE_DIR, enriched=True)
    F = rs_enr[0].features.shape[1]
    print(f"  reloaded enriched: {len(rs_enr)} pts, F={F} (expect 21)")
    assert F == 21, f"expected 21 features, got {F}"
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
