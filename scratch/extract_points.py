import zipfile
import pandas as pd
import os
from pathlib import Path

# Try common locations for the ZIP
workspace = Path(r"C:\Users\eluzq\workspace\ai-for-good")
# Check in track3_olimarteixeiraborges-main or root
zips = list(workspace.rglob("track1_download_link_5.zip"))

if zips:
    zip_path = zips[0]
    dest = workspace / "sample_points.csv"
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if 'points_train_label.csv' in name:
                with zf.open(name) as src, open(dest, 'wb') as dst:
                    dst.write(src.read())
                print(f"Extracted points_train_label.csv to {dest}")
                break
else:
    print("ZIP track1_download_link_5.zip not found.")
