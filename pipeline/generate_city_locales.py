import os
import argparse
import random
import math
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx
from pathlib import Path

import glob

def generate_random_locales(city_bounds, num_locales=10, radius_meters=178.4, existing_centers=None):
    """
    Generates random non-overlapping circular locales within a city boundary.
    """
    min_lon, min_lat, max_lon, max_lat = city_bounds
    
    # Approximation: 1 degree of latitude is roughly 111,320 meters.
    # Enforce a minimum distance between centers so circles don't overlap.
    min_dist_deg = (radius_meters * 2) / 111320.0 
    
    centers = existing_centers.copy() if existing_centers else []
    attempts = 0
    max_attempts = 10000 
    target_len = len(centers) + num_locales
    
    center_lon = (min_lon + max_lon) / 2.0
    center_lat = (min_lat + max_lat) / 2.0
    lon_std_dev = (max_lon - min_lon) / 4.0
    lat_std_dev = (max_lat - min_lat) / 4.0
    
    while len(centers) < target_len and attempts < max_attempts:
        attempts += 1
        rand_lon = random.gauss(center_lon, lon_std_dev)
        rand_lat = random.gauss(center_lat, lat_std_dev)
        
        # Ensure the point is still within the bounding box
        if not (min_lon <= rand_lon <= max_lon) or not (min_lat <= rand_lat <= max_lat):
            continue
        
        too_close = False
        for (c_lon, c_lat) in centers:
            dist = math.sqrt((rand_lon - c_lon)**2 + (rand_lat - c_lat)**2)
            if dist < min_dist_deg:
                too_close = True
                break
                
        if not too_close:
            centers.append((rand_lon, rand_lat))
            
    if len(centers) < target_len:
        print(f"Warning: Only found {len(centers)} non-overlapping locales after {max_attempts} attempts.")
        
    geometry = [Point(lon, lat) for lon, lat in centers]
    gdf_centers = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326")
    
    # Project to Web Mercator (EPSG:3857) to buffer in accurate meters
    gdf_centers_metric = gdf_centers.to_crs("EPSG:3857")
    
    # Buffer centers by radius
    gdf_circles_metric = gdf_centers_metric.copy()
    gdf_circles_metric['geometry'] = gdf_circles_metric.geometry.buffer(radius_meters)
    
    # Project back to EPSG:4326
    gdf_circles = gdf_circles_metric.to_crs("EPSG:4326")
    
    return gdf_circles

def main():
    parser = argparse.ArgumentParser(description="Generate random locales and fetch satellite images.")
    parser.add_argument("--city", type=str, required=True, help="Name of the city (used for output folders)")
    parser.add_argument("--bounds", type=float, nargs=4, required=True, help="Bounding box: min_lon min_lat max_lon max_lat")
    parser.add_argument("--num_locales", type=int, default=10, help="Number of locales to generate")
    parser.add_argument("--radius", type=float, default=178.4, help="Radius of locales in meters")
    parser.add_argument("--output_dir", type=str, default="data/processed/locales", help="Base output directory")
    parser.add_argument("--append", action="store_true", help="Append new locales to existing ones")
    
    args = parser.parse_args()
    
    city_dir = os.path.join(args.output_dir, args.city.replace(" ", "_"))
    gis_dir = os.path.join(city_dir, "gis")
    img_dir = os.path.join(city_dir, "images")
    
    os.makedirs(gis_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    
    existing_centers = []
    if args.append:
        existing_gis = sorted(glob.glob(os.path.join(gis_dir, "locale_*.geojson")))
        for f in existing_gis:
            if "all_locales" in f: continue
            try:
                gdf = gpd.read_file(f)
                centroid = gdf.to_crs("EPSG:3857").centroid.to_crs("EPSG:4326")
                existing_centers.append((centroid.x.iloc[0], centroid.y.iloc[0]))
            except:
                pass
                
    print(f"Generating {args.num_locales} additional locales for {args.city}...")
    locales_gdf = generate_random_locales(args.bounds, num_locales=args.num_locales, radius_meters=args.radius, existing_centers=existing_centers)
    
    for idx, row in locales_gdf.iterrows():
        # Save individual GIS file for the circle (GeoJSON is clean and easy)
        locale_id = f"locale_{idx+1:02d}"
        single_gdf = gpd.GeoDataFrame([row], crs="EPSG:4326")
        
        geojson_path = os.path.join(gis_dir, f"{locale_id}.geojson")
        single_gdf.to_file(geojson_path, driver="GeoJSON")
        
        # Calculate bounds for image fetching
        w, s, e, n = single_gdf.total_bounds
        
        # Add a tiny buffer (e.g. 0.001 deg) for the image context if desired. Let's do 0.0005.
        buffer_val = 0.0005
        w -= buffer_val
        s -= buffer_val
        e += buffer_val
        n += buffer_val
        
        img_path = os.path.join(img_dir, f"{locale_id}.tif")
        
        if args.append and os.path.exists(img_path):
            print(f"[{locale_id}] Image already exists, skipping fetch.")
        else:
            print(f"[{locale_id}] Fetching satellite tile...")
            try:
                ctx.bounds2raster(
                    w, s, e, n,
                    img_path,
                    zoom=18, # using zoom 18 for max detail without breaking Esri
                    source=ctx.providers.Esri.WorldImagery,
                    ll=True
                )
                print(f"[{locale_id}] Saved GIS: {geojson_path}, Image: {img_path}")
            except Exception as e_msg:
                print(f"[{locale_id}] Failed to fetch image: {e_msg}")
            
    # Also save the combined locales into one file for convenience
    combined_path = os.path.join(gis_dir, f"{args.city.replace(' ', '_')}_all_locales.geojson")
    locales_gdf.to_file(combined_path, driver="GeoJSON")
    print(f"Saved combined locales to {combined_path}")

if __name__ == "__main__":
    main()
