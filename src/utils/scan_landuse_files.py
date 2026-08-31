import geopandas as gpd
import glob
import os
import pandas as pd
from pathlib import Path

def scan_for_landuse(root_dir):
    print(f"Scanning {root_dir}...")
    
    # Get all shapefiles
    all_shps = glob.glob(os.path.join(root_dir, "**/*.shp"), recursive=True)
    print(f"Found {len(all_shps)} shapefiles total.")
    
    matches = []
    
    # We'll sample 1 file from each leaf directory to be faster? 
    # No, we need to find WHICH file in the directory is the Land Use one.
    # Usually directories have ~5-10 files. It's okay to scan headers.
    
    count = 0
    for shp in all_shps:
        count += 1
        if count % 100 == 0:
            print(f"Scanned {count}...")
            
        try:
            # efficient read? read_file reads geometry too. 
            # We can use read_file(rows=1)?
             # GeoPandas 0.11+ supports rows
            gdf = gpd.read_file(shp, rows=1)
            
            cols = [c.lower() for c in gdf.columns]
            if 'land_use' in cols or 'landuse' in cols:
                matches.append(shp)
                print(f"[FOUND] {shp}")
                
        except Exception as e:
            pass
            
    print(f"\nFound {len(matches)} files with Land_use data.")
    
    # Group by filename suffix to find patterns
    suffixes = {}
    for m in matches:
        suff = m[-7:] # last 7 chars e.g. "111.shp"
        suffixes[suff] = suffixes.get(suff, 0) + 1
        
    print("\nCommon Suffixes:")
    for s, c in suffixes.items():
        print(f"{s}: {c}")

if __name__ == "__main__":
    scan_for_landuse("data/raw/atlas_10ha")
