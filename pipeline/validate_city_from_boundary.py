"""
validate_city_from_boundary.py
Uses a saved city boundary GeoJSON to randomly generate N non-overlapping
10-hectare circles strictly inside the city boundary, extracts OSM blocks,
runs the model, and produces city-wide statistics.

Usage:
    .venv/bin/python3 validate_city_from_boundary.py \
        --boundary data/processed/buenos_aires_2010_boundary.geojson \
        --city "Buenos_Aires_2010" \
        --num_locales 200
"""

import os
import sys
import math
import random
import argparse
import subprocess
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx
from pathlib import Path

# Add project root to sys.path so absolute imports resolve correctly at runtime and in Pylance
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from archive.extract_blocks import extract_blocks_for_locale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary",    required=True,  help="Path to city boundary GeoJSON")
    parser.add_argument("--city",        required=True,  help="City name for output folders")
    parser.add_argument("--num_locales", type=int, required=True)
    parser.add_argument("--model",       default="models/landuse_resnet18.pth")
    parser.add_argument("--output_dir",  default="data/processed/locales")
    parser.add_argument("--append",      action="store_true", help="Append to existing locales")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Guaranteed Boundary Validation: {args.city}")
    print(f"  Target: EXACTLY {args.num_locales} valid 10-ha circles")
    print(f"{'='*60}\n")

    # 1. Load boundary
    boundary_gdf = gpd.read_file(args.boundary)
    if boundary_gdf.crs is None:
        boundary_gdf = boundary_gdf.set_crs("EPSG:4326")
    boundary_metric = boundary_gdf.to_crs("EPSG:3857")
    city_poly = boundary_metric.geometry.unary_union
    minx, miny, maxx, maxy = city_poly.bounds
    # Calculate centroid for Urban Core biasing
    centroid = city_poly.centroid
    center_x = centroid.x
    center_y = centroid.y
    # Standard deviation: roughly 1/6th of total span means 99.7% of throws fall within the box naturally
    sigma_x = (maxx - minx) / 6.0
    sigma_y = (maxy - miny) / 6.0

    # Directories
    city_dir = os.path.join(args.output_dir, args.city)
    gis_dir   = os.path.join(city_dir, "gis")
    img_dir   = os.path.join(city_dir, "images")
    blocks_dir= os.path.join(city_dir, "blocks")
    
    os.makedirs(gis_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(blocks_dir, exist_ok=True)

    # State variables
    radius_meters = 178.4
    min_dist_m = radius_meters * 2
    accepted_centers = []
    accepted_gdfs = []

    valid_locales_count = 0
    attempts = 0
    max_attempts = max(5000, args.num_locales * 50)  # Safety break

    if args.append:
        import glob
        existing_gis = sorted(glob.glob(os.path.join(gis_dir, "locale_*.geojson")))
        for f in existing_gis:
            if "all_locales" in f: continue
            try:
                # Check if BOTH blocks and images exist for this locale to count it as fully valid
                locale_id = os.path.basename(f).replace(".geojson", "")
                blocks_path = os.path.join(blocks_dir, f"{locale_id}_blocks.geojson")
                img_path = os.path.join(img_dir, f"{locale_id}.tif")
                if not os.path.exists(blocks_path) or not os.path.exists(img_path):
                    # Incomplete, cleanup to avoid corrupting the count
                    if os.path.exists(f): os.remove(f)
                    if os.path.exists(blocks_path): os.remove(blocks_path)
                    if os.path.exists(img_path): os.remove(img_path)
                    continue
                
                gdf = gpd.read_file(f)
                accepted_gdfs.append(gdf)
                # Ensure it's in a projected CRS to get centroid accurately in meters, 
                # but our centers are stored in EPSG:4326 from the gaussian throw? 
                # Wait, rx, ry are in EPSG:3857? No, center_x, center_y are from city_poly which is EPSG:3857!
                # Let's check where the centroid comes from.
                # In line 86, pt = Point(rx, ry). rx, ry are in EPSG:3857 because city_poly is in EPSG:3857.
                # So accepted_centers expects EPSG:3857 coordinates.
                centroid_metric = gdf.to_crs("EPSG:3857").centroid
                accepted_centers.append((centroid_metric.x.iloc[0], centroid_metric.y.iloc[0]))
                valid_locales_count += 1
            except Exception as e:
                print(f"Failed to load existing locale {f}: {e}")
                pass
        print(f"Resuming with {valid_locales_count} existing locales.")

    print(f"\n[1/3] Generating, Validating OSM, and Fetching imagery (need {args.num_locales - valid_locales_count} more)...")
    while valid_locales_count < args.num_locales and attempts < max_attempts:
        attempts += 1
        
        # Throw Dart using Gaussian Distribution to perfectly bias towards the Urban Center / Downtown!
        rx = random.gauss(center_x, sigma_x)
        ry = random.gauss(center_y, sigma_y)
        pt = Point(rx, ry)

        # Basic Check: Is it inside the legal boundary?
        if not city_poly.contains(pt):
            continue

        # Basic Check: Does it overlap with an already accepted circle?
        too_close = any(
            math.sqrt((rx - cx) ** 2 + (ry - cy) ** 2) < min_dist_m
            for cx, cy in accepted_centers
        )
        if too_close:
            continue

        # It's a mathematically valid candidate! Create geometry.
        candidate_geom = pt.buffer(radius_meters)
        candidate_gdf = gpd.GeoDataFrame(geometry=[candidate_geom], crs="EPSG:3857").to_crs("EPSG:4326")
        
        # Save temporary GeoJSON
        target_idx = valid_locales_count + 1
        locale_id = f"locale_{target_idx:02d}"
        geojson_path = os.path.join(gis_dir, f"{locale_id}.geojson")
        candidate_gdf.to_file(geojson_path, driver="GeoJSON")

        # Step 1: Prove it has city blocks (OSM extraction)
        # We do this BEFORE fetching Esri imagery because API fetching is slow and expensive!
        print(f"\n---> Testing candidate {locale_id} (Attempt {attempts})...")
        success = extract_blocks_for_locale(geojson_path, blocks_dir)
        
        if not success:
            print(f"     [!] Invalid spot. No city blocks found. Discarding and retrying...")
            # Cleanup the dummy geojson so we don't clutter the logic
            if os.path.exists(geojson_path):
                os.remove(geojson_path)
            continue
            
        # Step 2: Fetch Esri Imagery
        w, s, e, n = candidate_gdf.total_bounds
        buf = 0.0005
        img_path = os.path.join(img_dir, f"{locale_id}.tif")
        try:
            ctx.bounds2raster(w - buf, s - buf, e + buf, n + buf,
                              img_path, zoom=18,
                              source=ctx.providers.Esri.WorldImagery, ll=True)
            print(f"     [+] Success! Locale locked in. ({valid_locales_count + 1}/{args.num_locales})")
        except Exception as e_msg:
            print(f"     [!] Failed to fetch imagery: {e_msg}. Discarding and retrying...")
            if os.path.exists(geojson_path): os.remove(geojson_path)
            blocks_path = os.path.join(blocks_dir, f"{locale_id}_blocks.geojson")
            if os.path.exists(blocks_path): os.remove(blocks_path)
            continue

        # If everything passes seamlessly:
        accepted_centers.append((rx, ry))
        accepted_gdfs.append(candidate_gdf)
        valid_locales_count += 1

    if valid_locales_count < args.num_locales:
        print(f"\nWARNING: Safety limit reached ({max_attempts} attempts). City boundary might be physically too small to fit {args.num_locales} non-overlapping valid circles.")
        print(f"Stopping at {valid_locales_count} successful locales.")
    else:
        print(f"\n[2/3] Successfully guaranteed exactly {valid_locales_count} matching locales!")

    # Combine all valid ones
    if accepted_gdfs:
        import pandas as pd
        combined_gdf = gpd.GeoDataFrame(pd.concat(accepted_gdfs, ignore_index=True), crs="EPSG:4326")
        combined_gdf.to_file(os.path.join(gis_dir, f"{args.city}_all_locales.geojson"), driver="GeoJSON")

    # 4. Classify blocks
    # Note: Because the while loop natively extracted blocks from `extract_blocks_for_locale`, we completely skip the subprocess extraction layer! We just run classification natively!
    print("\n[3/3] Running classification...")
    subprocess.run([
        ".venv/bin/python3", "pipeline/run_locales_classification.py",
        "--city_dir", city_dir,
        "--model", args.model
    ], check=True)

    print(f"\n{'='*60}")
    print(f"  Done! Results successfully locked in:")
    print(f"  {os.path.join(city_dir, 'classification_results')}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
