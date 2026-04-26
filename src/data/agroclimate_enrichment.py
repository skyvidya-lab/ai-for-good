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
    Optimized Extractor for SACI agroclimate variables.
    Handles temporal aggregation (Daily) to avoid GEE element limits.
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

    def get_batch_time_series(self, points_list, start_date='2018-04-01', end_date='2018-11-01'):
        """
        Extracts daily data for a list of points.
        """
        features = [
            ee.Feature(ee.Geometry.Point([lon, lat]), {'point_id': str(pid)})
            for pid, lat, lon in points_list
        ]
        fc = ee.FeatureCollection(features)
        
        start = ee.Date(start_date)
        end = ee.Date(end_date)
        n_days = end.difference(start, 'days')
        dates = ee.List.sequence(0, n_days.subtract(1)).map(lambda d: start.advance(d, 'day'))

        # 1. ERA5-Land (Daily Aggregate)
        era5_col = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
            .filterDate(start_date, end_date) \
            .select(['volumetric_soil_water_layer_1', 'total_precipitation_sum', 'temperature_2m'])

        # 2. SMAP L4 (3-hourly -> Daily Mean)
        smap_col = ee.ImageCollection("NASA/SMAP/SPL4SMGP/008") \
            .filterDate(start_date, end_date) \
            .select(['sm_surface_wetness'])

        def daily_aggregate(d):
            d = ee.Date(d)
            # Aggregate ERA5 (already daily, but ensures alignment)
            e5 = era5_col.filterDate(d, d.advance(1, 'day')).mean()
            # Aggregate SMAP (3-hourly to daily mean)
            sm = smap_col.filterDate(d, d.advance(1, 'day')).mean()
            
            return e5.addBands(sm).set('millis', d.millis())

        # Create a single daily collection with all bands
        combined_daily = ee.ImageCollection.fromImages(dates.map(daily_aggregate))

        def reduce_img(img):
            return img.reduceRegions(
                collection=fc,
                reducer=ee.Reducer.mean(),
                scale=500
            ).map(lambda f: f.set('millis', img.get('millis')))

        try:
            # Single request for all bands!
            res = combined_daily.map(reduce_img).flatten().getInfo()
            df = pd.DataFrame([f['properties'] for f in res['features']])

            if df.empty:
                return {}

            df['date'] = pd.to_datetime(df['millis'], unit='ms').dt.normalize()
            
            results = {}
            for pid, _, _ in points_list:
                p_df = df[df['point_id'] == str(pid)].sort_values('date')
                if not p_df.empty:
                    # Rename back if needed or clean up
                    results[pid] = p_df[['date', 'volumetric_soil_water_layer_1', 'total_precipitation_sum', 'temperature_2m', 'sm_surface_wetness']]
            
            return results

        except Exception as e:
            print(f"Batch extraction error: {e}")
            return {}

    def align_to_ps(self, ps_dates, agro_df):
        """Aligns daily agro data to PointSeries dates."""
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
                    'precip_acc': float(window['total_precipitation_sum'].sum() * 1000.0),
                    'soil_moisture': float(window['volumetric_soil_water_layer_1'].mean()),
                    'temp': float(window['temperature_2m'].mean() - 273.15),
                    'smap_wetness': float(window['sm_surface_wetness'].mean())
                })
            else:
                results.append({'precip_acc': 0.0, 'soil_moisture': 0.0, 'temp': 20.0, 'smap_wetness': 0.5})
        return pd.DataFrame(results)

def batch_enrich(points_df, output_path, project='agente-bdr-sdr', batch_size=20):
    """Efficiently enriches points using batching and daily aggregation."""
    extractor = AgroclimateExtractor(project=project)
    unique_points = points_df.drop_duplicates('point_id')
    pts = [(row['point_id'], row['Latitude'], row['Longitude']) for _, row in unique_points.iterrows()]
    
    all_results = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, 'rb') as f:
                all_results = pickle.load(f)
        except: pass

    to_process = [p for p in pts if p[0] not in all_results]
    print(f"Total points: {len(pts)} | Already processed: {len(all_results)} | To process: {len(to_process)}")
    
    if not to_process:
        print("All points already processed.")
        return

    for i in range(0, len(to_process), batch_size):
        batch = to_process[i : i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(to_process)-1)//batch_size + 1} ({len(batch)} points)...")
        
        batch_results = extractor.get_batch_time_series(batch)
        if batch_results:
            all_results.update(batch_results)
            with open(output_path, 'wb') as f:
                pickle.dump(all_results, f)
        
        time.sleep(1)
        
    print(f"Enrichment complete. Final size: {len(all_results)}. Saved to {output_path}")
