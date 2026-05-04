import pandas as pd
import numpy as np

CACHE_DIR = r'C:\Users\eluzq\workspace\ai-for-good\data\cache'

# Load the full cache files
phenophases = pd.read_parquet(f'{CACHE_DIR}/phenophases.parquet')
meta        = pd.read_parquet(f'{CACHE_DIR}/points_meta.geoparquet')

print("=== PHENOPHASES PARQUET ===")
print(f"Shape: {phenophases.shape}")
print(f"Columns: {phenophases.columns.tolist()}")
print(phenophases.head(10))
print(f"\nUnique phenophases: {sorted(phenophases.iloc[:, -1].unique()) if phenophases.shape[1] > 0 else 'N/A'}")

print("\n=== POINTS META ===")
print(f"Shape: {meta.shape}")
print(f"Columns: {meta.columns.tolist()}")
print(meta.head(5))
print(f"\nCrop types: {meta['crop_type'].value_counts().to_dict() if 'crop_type' in meta.columns else 'N/A'}")
print(f"Regions:    {meta['region'].value_counts().to_dict() if 'region' in meta.columns else 'N/A'}")
