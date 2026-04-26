import ee
import pandas as pd
import numpy as np
import time
import os
import pickle
from pathlib import Path
from tqdm import tqdm

class AgroclimateExtractor:
    """
    Ultra-fast GEE Extractor using Temporal Stacking.
    Stacks time series into bands to perform a single spatial reduction.
    """
    
    def __init__(self, project='agente-bdr-sdr'):
        try:
            ee.Initialize(project=project)
            print(f"GEE initialized with project: {project}")
        except Exception as e:
            print(f"Standard initialization failed: {e}")
            try:
                import google.colab
                print("Google Colab detected. Triggering ee.Authenticate()...")
                ee.Authenticate()
                ee.Initialize(project=project)
            except (ImportError, Exception):
                try:
                    ee.Initialize()
                    print("GEE initialized with default project.")
                except Exception:
                    print("GEE not initialized. Authentication required.")

    def get_time_series_fast(self, lat, lon, start_date='2018-04-01', end_date='2018-11-01'):
        """
        Extracts all dates in one single request using toBands().
        """
        point = ee.Geometry.Point([lon, lat])
        start = ee.Date(start_date)
        end = ee.Date(end_date)
        n_days = end.difference(start, 'days')
        dates = ee.List.sequence(0, n_days.subtract(1)).map(lambda d: start.advance(d, 'day'))

        # Collections
        era5_col = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
            .filterDate(start_date, end_date)
        smap_col = ee.ImageCollection("NASA/SMAP/SPL4SMGP/008") \
            .filterDate(start_date, end_date)

        def daily_aggregate(d):
            d = ee.Date(d)
            date_str = d.format('yyyyMMdd')
            e5 = era5_col.filterDate(d, d.advance(1, 'day')).mean() \
                .select(['volumetric_soil_water_layer_1', 'total_precipitation_sum', 'temperature_2m'])
            sm = smap_col.filterDate(d, d.advance(1, 'day')).mean() \
                .select(['sm_surface_wetness'])
            
            # Prefix bands with date to identify them later
            return e5.addBands(sm).rename([
                date_str.cat('_soil'), date_str.cat('_precip'), 
                date_str.cat('_temp'), date_str.cat('_smap')
            ])

        # Stack all days into one image with ~800 bands
        stacked_image = ee.ImageCollection.fromImages(dates.map(daily_aggregate)).toBands()

        try:
            # Single spatial reduction for the whole stack
            data = stacked_image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=500
            ).getInfo()

            if not data:
                return pd.DataFrame()

            # Parse results back into a clean DataFrame
            rows = []
            # The band names in toBands() are usually 'index_datestr_var'
            # But with our rename, they should be 'index_YYYYMMDD_var'
            # Let's parse all keys
            temp_dict = {}
            for k, v in data.items():
                # Format: "0_20180401_soil" or just "20180401_soil" depending on GEE version
                parts = k.split('_')
                date_str = next((p for p in parts if len(p) == 8 and p.isdigit()), None)
                var_name = parts[-1]
                
                if date_str:
                    if date_str not in temp_dict: temp_dict[date_str] = {}
                    temp_dict[date_str][var_name] = v

            for d_str, vals in temp_dict.items():
                rows.append({
                    'date': pd.to_datetime(d_str, format='%Y%m%d'),
                    'volumetric_soil_water_layer_1': vals.get('soil'),
                    'total_precipitation_sum': vals.get('precip'),
                    'temperature_2m': vals.get('temp'),
                    'sm_surface_wetness': vals.get('smap')
                })
            
            return pd.DataFrame(rows).sort_values('date')

        except Exception as e:
            print(f"Fast extraction error for ({lat}, {lon}): {e}")
            return pd.DataFrame()

    def align_to_ps(self, ps_dates, agro_df):
        if agro_df is None or (isinstance(agro_df, pd.DataFrame) and agro_df.empty):
            return pd.DataFrame([{'precip_acc': 0.0, 'soil_moisture': 0.0, 'temp': 20.0, 'smap_wetness': 0.5}] * len(ps_dates))

        agro_df = agro_df.copy()
        agro_df['date'] = pd.to_datetime(agro_df['date'])
        
        results = []
        for d in ps_dates:
            target_dt = pd.to_datetime(d)
            mask = (agro_df['date'] <= target_dt) & (agro_df['date'] > target_dt - pd.Timedelta(days=5))
            window = agro_df[mask]
            
            if len(window) > 0:
                results.append({
                    'precip_acc': float(window['total_precipitation_sum'].sum() * 1000.0 if window['total_precipitation_sum'].notna().any() else 0.0),
                    'soil_moisture': float(window['volumetric_soil_water_layer_1'].mean() if window['volumetric_soil_water_layer_1'].notna().any() else 0.0),
                    'temp': float(window['temperature_2m'].mean() - 273.15 if window['temperature_2m'].notna().any() else 20.0),
                    'smap_wetness': float(window['sm_surface_wetness'].mean() if window['sm_surface_wetness'].notna().any() else 0.5)
                })
            else:
                results.append({'precip_acc': 0.0, 'soil_moisture': 0.0, 'temp': 20.0, 'smap_wetness': 0.5})
        return pd.DataFrame(results)

def batch_enrich(points_df, output_path, project='agente-bdr-sdr'):
    """
    Enriches points one by one but using the super-fast Temporal Stacking method.
    """
    extractor = AgroclimateExtractor(project=project)
    unique_points = points_df.drop_duplicates('point_id')
    
    all_results = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, 'rb') as f:
                all_results = pickle.load(f)
        except: pass

    to_process = unique_points[~unique_points['point_id'].isin(all_results.keys())]
    print(f"Total: {len(unique_points)} | Processed: {len(all_results)} | Remaining: {len(to_process)}")
    
    if len(to_process) == 0:
        print("All points done.")
        return

    # One by one is now safe because each point takes < 1 second
    for _, row in tqdm(to_process.iterrows(), total=len(to_process)):
        pid = row['point_id']
        df = extractor.get_time_series_fast(row['Latitude'], row['Longitude'])
        if not df.empty:
            all_results[pid] = df
            # Save every 10 points to be safe
            if len(all_results) % 10 == 0:
                with open(output_path, 'wb') as f:
                    pickle.dump(all_results, f)
        
    with open(output_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Complete! Saved to {output_path}")
