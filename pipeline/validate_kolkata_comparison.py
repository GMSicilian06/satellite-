import os
import sys
import argparse
import geopandas as gpd
import pandas as pd
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.validate_city_from_boundary import main as generate_locales
from pipeline.run_locales_classification import classify_locales

def run_kolkata_validation():
    city = "Kolkata_Comparison"
    boundary_path = "data/processed/kolkata_boundary.geojson"
    num_locales = 35
    
    # Model paths
    global_model = "models/landuse_efficientnet_v2_s.pth"
    regional_model = "models/landuse_southeast_asia_improved.pth"
    
    print(f"\n{'='*60}")
    print(f"  KOLKATA VALIDATION: Global vs Regional")
    print(f"  Locales: {num_locales}")
    print(f"{'='*60}\n")

    # 1. Generate 35 Locales and Fetch Imagery
    # We use the existing validate_city_from_boundary logic
    sys.argv = [
        "validate_city_from_boundary.py",
        "--boundary", boundary_path,
        "--city", city,
        "--num_locales", str(num_locales)
    ]
    generate_locales()

    city_dir = f"data/processed/locales/{city}"

    # 2. Run Classification with Global Model
    print("\n--- Running Global Model Classification ---")
    classify_locales(city_dir, global_model)
    # Move results to a global-specific folder
    os.rename(os.path.join(city_dir, "classification_results"), os.path.join(city_dir, "results_global"))

    # 3. Run Classification with Regional Model
    print("\n--- Running Regional Model Classification ---")
    classify_locales(city_dir, regional_model)
    # Move results to a regional-specific folder
    os.rename(os.path.join(city_dir, "classification_results"), os.path.join(city_dir, "results_regional"))

    # 4. Aggregate Statistics
    print("\n--- Aggregating Statistics for Kolkata ---")
    stats = compare_stats(city_dir)
    
    print(f"\nFinal Statistics for Kolkata (35 random locales):")
    print(stats.to_string())
    
    stats.to_csv(os.path.join(city_dir, "kolkata_comparison_stats.csv"))

def compare_stats(city_dir):
    def get_summary(results_path):
        all_blocks = []
        import glob
        for geojson in glob.glob(os.path.join(results_path, "*", "predicted_classified.geojson")):
            gdf = gpd.read_file(geojson)
            all_blocks.append(gdf)
        
        if not all_blocks: return pd.Series()
        
        full_gdf = pd.concat(all_blocks)
        # Class mapping
        class_map = {
            0: "Open Space", 1: "Non-Residential", 2: "Atomistic",
            3: "Informal", 4: "Formal", 5: "Housing Projects"
        }
        full_gdf['class_name'] = full_gdf['pred_class_code'].map(class_map)
        return full_gdf['class_name'].value_counts(normalize=True) * 100

    global_stats = get_summary(os.path.join(city_dir, "results_global"))
    regional_stats = get_summary(os.path.join(city_dir, "results_regional"))
    
    comparison = pd.DataFrame({
        "Global Model (%)": global_stats,
        "Regional Model (%)": regional_stats
    }).fillna(0)
    
    return comparison

if __name__ == "__main__":
    run_kolkata_validation()
