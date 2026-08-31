import os
import glob
import shutil
import argparse
from tqdm import tqdm

def create_regional_dataset(cities, global_dataset_dir, output_dir, aue_dir="data/AUE_Everything"):
    cities_str = ", ".join(cities)
    print(f"Creating regional dataset for {cities_str}...")
    
    locale_ids = set()
    for city_name in cities:
        # 1. Find the locale IDs for the city
        city_path = os.path.join(aue_dir, city_name)
        if not os.path.exists(city_path):
            print(f"Warning: Could not find city directory at {city_path}. Skipping {city_name}.")
            continue
            
        shp_files = glob.glob(os.path.join(city_path, "**", "*.shp"), recursive=True)
        if not shp_files:
            print(f"Warning: Could not find any shapefiles in {city_path}. Skipping {city_name}.")
            continue
            
        found_for_city = 0
        for shp_file in shp_files:
            basename = os.path.basename(shp_file)
            if len(basename) >= 15 and basename[:15].isdigit():
                locale_ids.add(basename[:15])
                found_for_city += 1
                
        print(f"Found {found_for_city} unique locale IDs for {city_name}.")
        
    print(f"Total: Found {len(locale_ids)} unique locale IDs across all cities.")
    
    if len(locale_ids) == 0:
        print("No valid 15-digit locale IDs found. Exiting.")
        return

    # 2. Setup output directory structure
    for split in ['train', 'val']:
        split_dir = os.path.join(global_dataset_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        classes = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        for cls in classes:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)
            
    # 3. Copy matched chips
    total_copied = 0
    for split in ['train', 'val']:
        split_dir = os.path.join(global_dataset_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        print(f"\nProcessing {split} split...")
        classes = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        
        for cls in classes:
            src_cls_dir = os.path.join(split_dir, cls)
            dst_cls_dir = os.path.join(output_dir, split, cls)
            
            files = os.listdir(src_cls_dir)
            matched_files = [f for f in files if any(f.startswith(loc_id) for loc_id in locale_ids)]
            
            if matched_files:
                print(f"  Class {cls}: Found {len(matched_files)} regional chips.")
                for f in tqdm(matched_files, desc=f"Copying {cls}", leave=False):
                    src_file = os.path.join(src_cls_dir, f)
                    dst_file = os.path.join(dst_cls_dir, f)
                    if not os.path.exists(dst_file):
                        shutil.copy2(src_file, dst_file)
                total_copied += len(matched_files)
                
    print(f"\nFinished! Copied a total of {total_copied} chips to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a regional dataset from the global dataset")
    parser.add_argument("--cities", nargs='+', required=True, help="List of city names (e.g. Dhaka Lahore)")
    parser.add_argument("--global_data", default="data/classification_dataset_robust", help="Path to global dataset")
    parser.add_argument("--output_data", default=None, help="Output path (default: data/classification_{cities})")
    
    args = parser.parse_args()
    
    if args.output_data is None:
        cities_str = "_".join([c.lower().replace(" ", "_") for c in args.cities])
        args.output_data = f"data/classification_dataset_{cities_str}"
        
    create_regional_dataset(args.cities, args.global_data, args.output_data)
