import os
import glob
import argparse
import sys
import random

# Add project root to sys.path to resolve 'src' modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.validation.offline_analysis import run_offline_analysis

def evaluate_aue_city(city_name, num_locales=10, model_path="models/landuse_efficientnet_v2_s_hd.pth"):
    print(f"======================================================")
    print(f" Starting AUE Ground Truth Evaluation for: {city_name}")
    print(f" Model: {model_path}")
    print(f"======================================================")
    
    # Path to the specific city in the AUE database
    base_dir = os.path.join("data/AUE_Everything", city_name)
    
    if not os.path.exists(base_dir):
        print(f"Error: Could not find AUE directory for {city_name} at {base_dir}")
        return
        
    # Search for all shapefiles recursively
    shp_files = glob.glob(os.path.join(base_dir, "**", "*.shp"), recursive=True)
    
    # Filter for valid ground truth blocks (files ending in 11.shp usually represent blocks)
    # Exclude historical T0 baselines and arterial roads
    valid_shps = [s for s in shp_files if s.endswith("11.shp") and "T0" not in s and "Arterial" not in s]
    
    if len(valid_shps) == 0:
        print(f"Warning: No files ending in 11.shp found. Falling back to all T1/T3 shapefiles.")
        valid_shps = [s for s in shp_files if "T0" not in s and "Arterial" not in s]
        
    print(f"Found {len(valid_shps)} valid AUE ground truth locales in {city_name}.")
    
    # Shuffle for randomness, then take the requested number
    random.seed(42) # Deterministic shuffle for reproducibility
    valid_shps.sort() # Ensure consistent order before shuffling
    random.shuffle(valid_shps)
    
    selected_shps = valid_shps[:num_locales]
    print(f"Executing HD classification pipeline on {len(selected_shps)} selected locales...\n")
    
    for shp_path in selected_shps:
        # Extract the sample ID (filename without extension, usually 15 digits)
        sample_id = os.path.splitext(os.path.basename(shp_path))[0]
        # Run the full offline analysis pipeline!
        run_offline_analysis(city_name, sample_id, shp_path, model_path)
        
    print(f"\n======================================================")
    print(f" Evaluation Complete! Check the offline_comparison folders.")
    print(f"======================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a specific AUE City using HD Model.")
    parser.add_argument("--city", type=str, required=True, help="Name of the city (e.g., 'Mexico City')")
    parser.add_argument("--num", type=int, default=10, help="Number of locales to evaluate")
    parser.add_argument("--model", type=str, default="models/landuse_efficientnet_v2_s_hd.pth", help="Model to evaluate")
    args = parser.parse_args()
    
    evaluate_aue_city(args.city, args.num, args.model)
