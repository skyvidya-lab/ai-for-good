import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

# Fix paths
REPO_PATH = Path('C:/Users/eluzq/workspace/ai-for-good')
if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))

from src.data.cache_loader import load_aggregated_cache
CACHE_DIR = REPO_PATH / 'data' / 'cache'

print("Loading cache...")
try:
    series_agro = load_aggregated_cache(CACHE_DIR, variant="full_plus")
except Exception:
    try:
        series_agro = load_aggregated_cache(CACHE_DIR, variant="full")
    except Exception:
        series_agro = load_aggregated_cache(CACHE_DIR, enriched=True)

CROPS = ['rice', 'corn', 'soybean']

# Test 1: Background bug (E8)
unknown_crops = []
for ps in series_agro:
    if ps.crop_type not in CROPS:
        unknown_crops.append(ps.crop_type)

print(f"\n--- TEST 1: Background bug ---")
print(f"Unknown crops count: {len(unknown_crops)}")
if len(unknown_crops) > 0:
    print(f"Unique unknown crops: {set(unknown_crops)}")

# Test 2: Region purity (E3)
regions = []
crop_y = []
for ps in series_agro:
    regions.append(ps.region)
    crop_y.append(ps.crop_type)

df = pd.DataFrame({'region': regions, 'crop': crop_y})
purity = df.groupby('region')['crop'].nunique()
pure_regions = (purity == 1).sum()
total_regions = len(purity)
print(f"\n--- TEST 2: Region purity ---")
print(f"Total regions: {total_regions}")
print(f"Mono-crop regions: {pure_regions} ({(pure_regions/total_regions)*100:.1f}%)")
