# Processing the images

import geopandas as gpd
import contextily as ctx
from shapely.ops import polygonize
from rasterio.mask import mask
import rasterio
import glob
import os
import random
import cv2
import numpy as np
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def robust_pipeline(root_dir, image_dir, dataset_dir, samples_per_city=50):
    print("Starting Robust Multi-City Pipeline...")
    
    # 1. Identify Cities and Land Use Files
    cities = {} # city_name -> [list of shapefiles with Land_use]
    
    # Search recursively
    # We use a generator or fast walk
    for subdir, dirs, files in os.walk(root_dir):
        # Check if this is a leaf node with shapefiles
        # Skip T0 baseline historical folders because they don't match modern satellite imagery
        if "T0" in subdir:
            continue
            
        shps = [f for f in files if f.endswith(".shp")]
        if not shps:
            continue
            
        # Infer city name from path (e.g. .../Accra/Paris/Paris_T0 -> Paris)
        # Path parts
        parts = Path(subdir).parts
        # Assuming struct: root/Accra/CityName/...
        # We'll use the folder name just below 'Accra' as the key if possible, or just the immediate parent
        # Let's use the immediate parent for grouping, it's safer
        
        # ACTUALLY, strict city grouping is hard. 
        # Let's just create a list of ALL candidates and then group key by the 4th component?
        # data/raw/atlas_10ha/Accra/Paris/...
        
        valid_shps = []
        for s in shps:
            shp_path = os.path.join(subdir, s)
            try:
                # Fast check for column
                # We need to read metadata. GeoPandas read_file with rows=1 is best attempt
                gdf = gpd.read_file(shp_path, rows=1)
                cols = [c.lower() for c in gdf.columns]
                if 'land_use' in cols or 'landuse' in cols:
                    valid_shps.append(shp_path)
            except:
                pass
                
        if valid_shps:
            # Add to city group
            # Identify city name: The folder directly under 'atlas_10ha' is 'Accra', 
            # the one under 'Accra' is likely the real city name?
            # Let's try to extract the city name dynamically
            try:
                # Find index of 'atlas_10ha'
                idx = parts.index('atlas_10ha')
                # Next is 'Accra' (the container), Next is City
                if len(parts) > idx + 2:
                    city_name = parts[idx+2]
                else:
                    city_name = "Unknown"
            except:
                city_name = os.path.basename(subdir)
            
            if city_name not in cities:
                cities[city_name] = []
            cities[city_name].extend(valid_shps)
            print(f"  {city_name}: Found {len(valid_shps)} valid files so far...")

    print("\n--- City Summary ---")
    for city, files in cities.items():
        print(f"{city}: {len(files)} files")
        
    # 2. Process Samples
    os.makedirs(image_dir, exist_ok=True)
    
    total_crops = 0
    
    for city, files in cities.items():
        print(f"\nProcessing {city}...")
        
        # Sample
        if len(files) > samples_per_city:
            selection = random.sample(files, samples_per_city)
        else:
            selection = files
            
        for shp_path in selection:
            try:
                basename = Path(shp_path).stem
                img_path = os.path.join(image_dir, f"{basename}.tif")
                
                # Load GDF (Full this time)
                gdf = gpd.read_file(shp_path)
                if gdf.crs is None:
                    gdf.set_crs(epsg=4326, inplace=True)
                
                # Standardize column name
                if 'Land_use' not in gdf.columns and 'landuse' in gdf.columns:
                    gdf['Land_use'] = gdf['landuse']
                
                # A. Fetch Image if needed
                if not os.path.exists(img_path):
                    # Bounds
                    w, s, e, n = gdf.total_bounds
                    # Buffer
                    b = 0.001
                    
                    import time
                    fetch_success = False
                    max_retries = 4
                    for attempt in range(max_retries):
                        try:
                            # Ctx fetch
                            ctx.bounds2raster(w-b, s-b, e+b, n+b, img_path, zoom='auto', source=ctx.providers.Esri.WorldImagery, ll=True)
                            print(f"  Fetched {basename}.tif")
                            fetch_success = True
                            break
                        except Exception as err:
                            if attempt < max_retries - 1:
                                wait_time = (2 ** attempt) * 5 + random.uniform(1, 5)
                                print(f"  API Error on {basename}: {err}. Retrying in {wait_time:.1f}s...")
                                time.sleep(wait_time)
                            else:
                                print(f"  Fetch failed permanently for {basename}.")
                    
                    if not fetch_success:
                        continue
                
                # B. Polygonize & Label
                # The shapefile itself MIGHT contain polygons (if it's Land Use Polygons)
                # Or Lines (if it's Land Use Lines).
                # Check geometry type
                geom_type = gdf.geometry.type.iloc[0]
                
                labeled_polys = []
                
                if 'Polygon' in geom_type:
                    # It's already polygons! Direct Use.
                    labeled_polys = gdf
                else:
                    # It's lines. Polygonize.
                    lines = list(gdf.geometry)
                    polys = list(polygonize(lines))
                    if not polys:
                        continue
                    poly_df = gpd.GeoDataFrame(geometry=polys, crs=gdf.crs)
                    # Spatial Join
                    labeled_polys = gpd.sjoin(poly_df, gdf, how="inner", predicate="intersects")
                    # Dedup
                    labeled_polys = labeled_polys[~labeled_polys.index.duplicated(keep='first')]
                
                # C. Crop & Save
                with rasterio.open(img_path) as src:
                    # Align CRS
                    if labeled_polys.crs != src.crs:
                        labeled_polys = labeled_polys.to_crs(src.crs)
                        
                    for idx, row in labeled_polys.iterrows():
                        label = row['Land_use']
                        geom = row.geometry
                        try:
                            # Add 15m context padding (assumes epsg 3857/meters)
                            geom = geom.buffer(15.0)
                        except:
                            pass
                        
                        try:
                            # 80/20 Split
                            split = "train" if random.random() < 0.8 else "val"
                            
                            out_dir = os.path.join(dataset_dir, split, str(label))
                            os.makedirs(out_dir, exist_ok=True)
                            
                            out_filename = f"{basename}_{idx}.jpg"
                            out_path = os.path.join(out_dir, out_filename)
                            
                            if os.path.exists(out_path): dest_exists=True # Skip if already done? No, maybe overwrite.
                            
                            out_image, out_transform = mask(src, [geom], crop=True)
                            
                             # Check validity
                            if out_image.shape[0] == 0: continue
                            
                            out_img_cv = np.moveaxis(out_image, 0, -1)
                            out_img_cv = cv2.cvtColor(out_img_cv, cv2.COLOR_RGB2BGR)
                            
                            if out_img_cv.shape[0] < 5 or out_img_cv.shape[1] < 5: continue
                            
                            cv2.imwrite(out_path, out_img_cv)
                            total_crops += 1
                            
                        except:
                            pass
                            
            except Exception as e:
                print(f"  Error processing {shp_path}: {e}")

    print(f"\nPipeline Complete. Generated {total_crops} training chips.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw/atlas_10ha")
    parser.add_argument("--images", default="data/raw/fetched_images_robust")
    parser.add_argument("--output", default="data/classification_dataset_robust")
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()
    
    robust_pipeline(args.root, args.images, args.output, args.samples)
