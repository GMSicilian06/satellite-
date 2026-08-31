"""
aggregate_city_stats.py
Aggregates predicted land use statistics from the classification results directory.
Calculates area-weighted percentages of each class.

Usage:
    .venv/bin/python3 aggregate_city_stats.py --results_dir data/processed/locales/Buenos_Aires_2010/classification_results
"""

import os
import glob
import argparse
import geopandas as gpd
import pandas as pd

CLASS_NAMES = {
    0: "Open Space",
    1: "Non-residential",
    2: "Atomistic settlements",
    3: "Informal subdivisions",
    4: "Formal subdivisions",
    5: "Housing projects",
    6: "Class 6",
    7: "Class 7",
    8: "None / Other",
    -1: "Error",
}

def aggregate_stats(results_dir):
    print(f"Aggregating statistics from: {results_dir}")
    
    all_results = []
    # Find all predicted_classified.geojson files
    pattern = os.path.join(results_dir, "**", "predicted_classified.geojson")
    geojsons = glob.glob(pattern, recursive=True)
    
    if not geojsons:
        print("No classification results found.")
        return

    for g_path in geojsons:
        try:
            gdf = gpd.read_file(g_path)
            # Ensure CRS is metric for accurate area calculation
            if gdf.crs is None:
                gdf.set_crs("EPSG:4326", inplace=True)
            gdf_m = gdf.to_crs("EPSG:3857")
            
            # Calculate areas
            gdf_m['area_m2'] = gdf_m.geometry.area
            
            # Group by pred_class_code
            if 'pred_class_code' in gdf_m.columns:
                stats = gdf_m.groupby('pred_class_code')['area_m2'].sum().reset_index()
                all_results.append(stats)
        except Exception as e:
            print(f"Error processing {g_path}: {e}")

    if not all_results:
        print("No valid data found in results.")
        return

    # Combine all results
    combined = pd.concat(all_results)
    final_stats = combined.groupby('pred_class_code')['area_m2'].sum().reset_index()
    
    total_area = final_stats['area_m2'].sum()
    final_stats['percentage'] = (final_stats['area_m2'] / total_area) * 100
    
    # Map class names
    final_stats['class_name'] = final_stats['pred_class_code'].map(CLASS_NAMES).fillna("Unknown")
    
    print("\n" + "="*50)
    print("  PREDICTED CITY-WIDE LAND USE DISTRIBUTION")
    print("="*50)
    print(f"{'Class Name':<25} {'Area (km²)':>12} {'Percentage':>12}")
    print("-" * 50)
    
    for _, row in final_stats.sort_values('percentage', ascending=False).iterrows():
        area_km2 = row['area_m2'] / 1_000_000
        print(f"{row['class_name']:<25} {area_km2:>12.2f} {row['percentage']:>11.1f}%")
        
    print("-" * 50)
    print(f"{'TOTAL':<25} {total_area/1_000_000:>12.2f} {100.0:>11.1f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, help="Directory containing classification results")
    args = parser.parse_args()
    aggregate_stats(args.results_dir)
