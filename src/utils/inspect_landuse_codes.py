import geopandas as gpd
import glob
import sys

def inspect_land_use(root_dir):
    # Find all ...111.shp files (Land Use layer)
    # Recursively
    files = glob.glob(f"{root_dir}/**/*111.shp", recursive=True)
    all_values = set()
    
    print(f"Found {len(files)} Land Use files.")
    
    for f in files:
        try:
            gdf = gpd.read_file(f)
            if 'Land_use' in gdf.columns:
                unique_vals = gdf['Land_use'].unique()
                for v in unique_vals:
                    all_values.add(v)
            else:
                print(f"No 'Land_use' column in {f}")
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    print("\nUnique Land Use Values Found:")
    for v in sorted(list(all_values)):
        print(v)

if __name__ == "__main__":
    inspect_land_use(sys.argv[1])
