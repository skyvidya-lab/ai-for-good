import pandas as pd
import numpy as np

CACHE_DIR = r'C:\Users\eluzq\workspace\ai-for-good\data\cache'

phenophases = pd.read_parquet(f'{CACHE_DIR}/phenophases.parquet')
meta        = pd.read_parquet(f'{CACHE_DIR}/points_meta.geoparquet')

# ─── 1. Check phenophase_idx mapping in cache vs our phenology_prior.py
print("=== CACHE phenophase_idx MAPPING ===")
mapping = phenophases.groupby('phenophase')['phenophase_idx'].first().sort_values()
print(mapping)

# ─── 2. Merge with meta to get crop_type and region
df = phenophases.merge(meta[['point_id', 'crop_type', 'region', 'lon', 'lat']], on='point_id')
df['pheno_date'] = pd.to_datetime(df['pheno_date'], format='mixed')

# Sort by point and date, compute intervals
df = df.sort_values(['point_id', 'pheno_date'])
df['prev_date'] = df.groupby('point_id')['pheno_date'].shift(1)
df['prev_phase'] = df.groupby('point_id')['phenophase'].shift(1)
df['interval_days'] = (df['pheno_date'] - df['prev_date']).dt.days
df['transition'] = df['prev_phase'].fillna('') + ' -> ' + df['phenophase']
transitions = df[df['prev_phase'].notna()].copy()

# Canonical order for display
CANONICAL_TRANSITIONS = [
    "Greenup -> MidGreenup",
    "MidGreenup -> Maturity",
    "Maturity -> Peak",
    "Peak -> Senescence",
    "Senescence -> MidSenescence",
    "MidSenescence -> Dormancy",
]

# ─── 3. Per-crop × per-region analysis
print("\n=== INTERVALS BY CROP × REGION (mean days) ===")
tbl = transitions[transitions['transition'].isin(CANONICAL_TRANSITIONS)].copy()
pivot_mean = tbl.pivot_table(
    index=['crop_type', 'region'],
    columns='transition',
    values='interval_days',
    aggfunc='mean'
)[CANONICAL_TRANSITIONS].round(1)
print(pivot_mean.to_string())

# ─── 4. Summary: between-region variance per crop per transition
print("\n=== BETWEEN-REGION STD (per crop per transition) ===")
region_means = tbl.groupby(['crop_type', 'transition', 'region'])['interval_days'].mean().reset_index()
between_std = region_means.groupby(['crop_type', 'transition'])['interval_days'].agg(['mean', 'std', 'min', 'max']).round(2)
# Reorder
between_std = between_std.reset_index()
between_std['trans_order'] = between_std['transition'].map({t: i for i, t in enumerate(CANONICAL_TRANSITIONS)})
between_std = between_std.sort_values(['crop_type', 'trans_order'])
print(between_std[['crop_type', 'transition', 'mean', 'std', 'min', 'max']].to_string(index=False))

# ─── 5. Conclusion: is between-region variance large enough to justify per-region priors?
print("\n=== CONCLUSION: Within-point STD vs Between-region STD ===")
within = tbl.groupby(['crop_type', 'transition'])['interval_days'].std().round(2)
between = region_means.groupby(['crop_type', 'transition'])['interval_days'].std().round(2)
comp = pd.DataFrame({'within_point_std': within, 'between_region_std': between})
comp['signal_to_noise'] = (comp['between_region_std'] / comp['within_point_std']).round(2)
print(comp.to_string())
print("\nSignal-to-noise > 1.0 means between-region variation is LARGER than within-point noise.")
