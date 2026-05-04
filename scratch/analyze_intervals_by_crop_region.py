import pandas as pd
import numpy as np

file_path = r'C:\Users\eluzq\workspace\ai-for-good\data\points_train_label.csv'
df = pd.read_csv(file_path)
df['phenophase_date'] = pd.to_datetime(df['phenophase_date'], format='mixed')
df_sorted = df.sort_values(['point_id', 'phenophase_date'])

# Canonical order
PHENOPHASES = ["Greenup", "MidGreenup", "Maturity", "Peak", "Senescence", "MidSenescence", "Dormancy"]

# Add interval and previous phenophase
df_sorted['phenophase_interval'] = df_sorted.groupby('point_id')['phenophase_date'].diff().dt.days.fillna(0)
df_sorted['prev_phase'] = df_sorted.groupby('point_id')['phenophase_name'].shift(1)
df_sorted['transition'] = df_sorted['prev_phase'].fillna('') + ' -> ' + df_sorted['phenophase_name']

# Only keep real transitions (not the first row with interval=0)
transitions = df_sorted[df_sorted['prev_phase'].notna()].copy()

# We need the region per point — get it from the metadata or approximate from lon/lat
# For now use crop_type from CSV directly
print("="*60)
print("INTERVALS BY CROP TYPE")
print("="*60)
for crop in ['rice', 'corn', 'soybean']:
    sub = transitions[transitions['crop_type'] == crop]
    print(f"\n--- {crop.upper()} ---")
    stats = sub.groupby('transition')['phenophase_interval'].agg(['mean', 'std', 'count'])
    # Reorder by canonical sequence
    ordered_transitions = [f"{PHENOPHASES[i]} -> {PHENOPHASES[i+1]}" for i in range(len(PHENOPHASES)-1)]
    for t in ordered_transitions:
        if t in stats.index:
            row = stats.loc[t]
            print(f"  {t:<35} mean={row['mean']:.1f}d  std={row['std']:.1f}d  n={int(row['count'])}")

# ─── Check variance by region (approximated from Longitude clusters)
print("\n" + "="*60)
print("LATITUDE RANGE PER CROP (proxy for region variation)")
print("="*60)
meta = df[['point_id', 'Longitude', 'Latitude', 'crop_type']].drop_duplicates('point_id')
for crop in ['rice', 'corn', 'soybean']:
    sub = meta[meta['crop_type'] == crop]
    print(f"  {crop:10}: lat [{sub['Latitude'].min():.2f}, {sub['Latitude'].max():.2f}]  "
          f"lon [{sub['Longitude'].min():.2f}, {sub['Longitude'].max():.2f}]  n={len(sub)}")

# ─── Cluster points into approximate regions by longitude quartiles
# and check if intervals differ significantly across those clusters
meta['lon_bin'] = pd.qcut(meta['Longitude'], q=4, labels=['W', 'CW', 'CE', 'E'])
transitions2 = transitions.merge(meta[['point_id', 'crop_type', 'lon_bin']], on=['point_id', 'crop_type'], how='left')

print("\n" + "="*60)
print("INTERVAL VARIATION BY CROP × LONGITUDE BAND")
print("="*60)
focus_transitions = ["Maturity -> Peak", "Peak -> Senescence", "Greenup -> MidGreenup"]
for t in focus_transitions:
    print(f"\n  Transition: {t}")
    sub = transitions2[transitions2['transition'] == t]
    tbl = sub.groupby(['crop_type', 'lon_bin'])['phenophase_interval'].agg(['mean', 'std', 'count'])
    print(tbl.to_string())

# ─── ANOVA-style: std between crops vs within crops
print("\n" + "="*60)
print("BETWEEN-CROP vs WITHIN-CROP STD (key transitions)")
print("="*60)
for t in focus_transitions:
    sub = transitions[transitions['transition'] == t]
    overall_std = sub['phenophase_interval'].std()
    means_by_crop = sub.groupby('crop_type')['phenophase_interval'].mean()
    within_stds   = sub.groupby('crop_type')['phenophase_interval'].std()
    print(f"\n{t}:")
    print(f"  Overall std: {overall_std:.2f}d")
    print(f"  Means by crop: {means_by_crop.to_dict()}")
    print(f"  Within-crop std: {within_stds.to_dict()}")
    print(f"  Between-crop range: {means_by_crop.max() - means_by_crop.min():.2f}d")
