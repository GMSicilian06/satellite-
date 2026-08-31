import os
import argparse
import subprocess
import osmnx as ox

def run_validation(city_name, num_locales=40, model_path="models/landuse_efficientnet_v2_s.pth", append=False):
    print(f"======================================================")
    print(f" Starting automated validation pipeline for: {city_name}")
    print(f"======================================================")
    
    # 1. Automatically fetch bounding box from OpenStreetMap
    print(f"\n[1/5] Fetching geographic bounding box for '{city_name}' via OSM...")
    try:
        city_gdf = ox.geocode_to_gdf(city_name)
        min_lon, min_lat, max_lon, max_lat = city_gdf.total_bounds
        print(f"      Resolved Bounding Box: [{min_lon:.4f}, {min_lat:.4f}, {max_lon:.4f}, {max_lat:.4f}]")
    except Exception as e:
        print(f"      Error: Could not resolve '{city_name}' on OpenStreetMap. Please check spelling.")
        print(f"      Details: {e}")
        return

    safe_city_name = city_name.replace(" ", "_").replace(",", "")
    
    # 2. Generate Random Locales
    print(f"\n[2/5] Generating {num_locales} random 10-hectare locales from map servers...")
    cmd_gen = [
        ".venv/bin/python3", "pipeline/generate_city_locales.py", 
        "--city", safe_city_name,
        "--bounds", str(min_lon), str(min_lat), str(max_lon), str(max_lat),
        "--num_locales", str(num_locales)
    ]
    if append:
        cmd_gen.append("--append")
    subprocess.run(cmd_gen, check=True)
    
    # 3. Extract OSM Blocks
    city_dir = os.path.join("data/processed/locales", safe_city_name)
    gis_dir = os.path.join(city_dir, "gis")
    blocks_dir = os.path.join(city_dir, "blocks")
    
    print(f"\n[3/5] Querying exact OSM road networks to cut geometric blocks...")
    cmd_ext = [
        ".venv/bin/python3", "archive/extract_blocks.py",
        "--gis_dir", gis_dir,
        "--out_dir", blocks_dir
    ]
    if append:
        cmd_ext.append("--append")
    subprocess.run(cmd_ext, check=True)
    
    # 4. Classify Blocks using the model
    print(f"\n[4/5] Activating EfficientNet-V2-S model to classify cut locales...")
    cmd_class = [
        ".venv/bin/python3", "pipeline/run_locales_classification.py",
        "--city_dir", city_dir,
        "--model", model_path
    ]
    if append:
        cmd_class.append("--append")
    subprocess.run(cmd_class, check=True)

    # 5. Measure Convergence
    print(f"\n[5/5] Analyzing classification convergence and generating statistics...")
    cmd_conv = [
        ".venv/bin/python3", "src/validation/measure_classification_convergence.py",
        "--city_dir", city_dir
    ]
    subprocess.run(cmd_conv, check=True)

    print(f"\n======================================================")
    print(f" Pipeline Complete! Visual validation results are saved at:")
    print(f" {os.path.join(city_dir, 'classification_results')}")
    print(f" Convergence stats and plot are saved in: {city_dir}")
    print(f"======================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fully automated script to test new model against any city on Earth.")
    parser.add_argument("--city", type=str, required=True, help="Name of the city (e.g., 'Bogota, Colombia')")
    parser.add_argument("--num_locales", type=int, default=40, help="Number of 10-hectare circles to randomly generate")
    parser.add_argument("--model", type=str, default="models/landuse_efficientnet_v2_s.pth", help="Path to your trained .pth model file")
    parser.add_argument("--append", action="store_true", help="Append to existing locales instead of overwriting")
    args = parser.parse_args()
    
    run_validation(args.city, args.num_locales, args.model, args.append)
