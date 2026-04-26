import ee
import pandas as pd
import numpy as np
import time
import os
from pathlib import Path
from tqdm import tqdm

class AgroclimateExtractor:
    """
    Enriches crop data with agroclimate variables from GEE (SACI inspired).
    Focus: ERA5-Land (Soil Moisture, Precip, Temp) and SMAP (Surface Wetness).
    """
    
    def __init__(self, project='agente-bdr-sdr'):
        try:
            # Try initializing with the project from SACI
            ee.Initialize(project=project)
            print(f"GEE initialized with project: {project}")
        except Exception as e:
            print(f"Standard initialization failed: {e}")
            try:
                # Fallback for generic environment
                ee.Initialize()
                print("GEE initialized with default project.")
            except Exception:
                print("GEE not initialized. Authentication required.")

    def get_time_series(self, lat, lon, start_date='2018-04-01', end_date='2018-11-01'):
        """
        Extracts multi-source time series for a specific point.
        """
        point = ee.Geometry.Point([lon, lat])
        
        # 1. ERA5-Land Daily (9km)
        era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
            .filterDate(start_date, end_date) \
            .filterBounds(point) \
            .select([
                'volumetric_soil_water_layer_1',
                'volumetric_soil_water_layer_2',
                'total_precipitation_sum',
                'temperature_2m'
            ])
            
        # 2. SMAP L4 (Surface Wetness - 9km)
        smap = ee.ImageCollection("NASA/SMAP/SPL4SMGP/008") \
            .filterDate(start_date, end_date) \
            .filterBounds(point) \
            .select(['sm_surface_wetness'])

        def _extract_collection(img_col, bands):
            def _get_val(img):
                res = img.reduceRegion(ee.Reducer.mean(), point, 500)
                return ee.Feature(None, {
                    'millis': img.date().millis(),
                    **{b: res.get(b) for b in bands}
                })
            
            features = img_col.map(_get_val).getInfo()
            rows = [f['properties'] for f in features['features']]
            df = pd.DataFrame(rows)
            if not df.empty:
                df['date'] = pd.to_datetime(df['millis'], unit='ms')
            return df

        try:
            df_era5 = _extract_collection(era5, ['volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2', 'total_precipitation_sum', 'temperature_2m'])
            df_smap = _extract_collection(smap, ['sm_surface_wetness'])
            
            if df_era5.empty or df_smap.empty:
                return pd.DataFrame()

            # Merge on date (ignoring hours if any)
            df_era5['date'] = df_era5['date'].dt.normalize()
            df_smap['date'] = df_smap['date'].dt.normalize()
            
            merged = pd.merge(df_era5, df_smap, on='date', how='outer').sort_values('date')
            return merged
        except Exception as e:
            print(f"Error extracting GEE data for point ({lat}, {lon}): {e}")
            return pd.DataFrame()

    def align_to_ps(self, ps_dates, agro_df):
        """
        Aligns daily agro data to PointSeries dates.
        ps_dates: list of strings 'YYYY-MM-DD' or datetime objects
        """
        if agro_df.empty:
            # Return zeros if no data found
            return pd.DataFrame([{
                'precip_acc': 0.0, 'soil_moisture': 0.0, 
                'temp': 20.0, 'smap_wetness': 0.5
            }] * len(ps_dates))

        agro_df = agro_df.copy()
        agro_df['date'] = pd.to_datetime(agro_df['date'])
        
        results = []
        for d in ps_dates:
            target_dt = pd.to_datetime(d)
            # Window: 5 days prior to observation
            mask = (agro_df['date'] <= target_dt) & (agro_df['date'] > target_dt - pd.Timedelta(days=5))
            window = agro_df[mask]
            
            if len(window) > 0:
                results.append({
                    'precip_acc': float(window['total_precipitation_sum'].sum() * 1000.0), # m to mm
                    'soil_moisture': float(window['volumetric_soil_water_layer_1'].mean()),
                    'temp': float(window['temperature_2m'].mean() - 273.15), # K to C
                    'smap_wetness': float(window['sm_surface_wetness'].mean())
                })
            else:
                # Fallback to nearest or average
                results.append({
                    'precip_acc': 0.0,
                    'soil_moisture': 0.0,
                    'temp': 20.0,
                    'smap_wetness': 0.5
                })
        return pd.DataFrame(results)

def batch_enrich(points_df, output_path, project='agente-bdr-sdr'):
    """
    Runs extraction for a dataframe of points.
    Expected columns: point_id, Latitude, Longitude
    """
    extractor = AgroclimateExtractor(project=project)
    unique_points = points_df.drop_duplicates('point_id')
    
    all_results = {}
    print(f"Starting SACI enrichment for {len(unique_points)} unique points...")
    
    for _, row in tqdm(unique_points.iterrows(), total=len(unique_points)):
        pid = row['point_id']
        lat, lon = row['Latitude'], row['Longitude']
        
        df = extractor.get_time_series(lat, lon)
        if not df.empty:
            all_results[pid] = df
            
        # Small sleep to avoid GEE rate limits
        time.sleep(0.1)
        
    # Save to a pickle for easy loading in notebooks
    import pickle
    with open(output_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Enrichment complete. Saved to {output_path}")

if __name__ == "__main__":
    # Test script with a known point if running locally
    print("SACI Enrichment Module Ready.")
