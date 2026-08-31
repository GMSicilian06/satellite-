
import geopandas as gpd
import rasterio
import rasterio.plot
import matplotlib.pyplot as plt
import os

def inspect_paris():
    shp_path = "data/raw/atlas_10ha/Accra/Paris/Paris_T0/100314880002001.shp"
    tif_path = "data/raw/fetched_images_robust/100314880002001.tif"
    
    print(f"--- Inspecting Shapefile: {shp_path} ---")
    if os.path.exists(shp_path):
        gdf = gpd.read_file(shp_path)
        print("Columns:", gdf.columns.tolist())
        if 'Land_use' in gdf.columns:
            print("Unique Land_use values:", gdf['Land_use'].unique())
        if 'Land_use_C' in gdf.columns: # Sometimes names get truncated
             print("Unique Land_use_C values:", gdf['Land_use_C'].unique())
             
        # Check coordinates
        print("Bounds:", gdf.total_bounds)
        print("CRS:", gdf.crs)
    else:
        print("SHP NOT FOUND")
        
    print(f"\n--- Inspecting GeoTIFF: {tif_path} ---")
    if os.path.exists(tif_path):
        with rasterio.open(tif_path) as src:
            print("CRS:", src.crs)
            print("Bounds:", src.bounds)
            print("Profile:", src.profile)
            
            # Try a simple plot test
            fig, ax = plt.subplots()
            rasterio.plot.show(src, ax=ax)
            plt.savefig("data/paris/debug_image_only.png")
            print("Saved debug_image_only.png")
    else:
        print("TIF NOT FOUND")

if __name__ == "__main__":
    inspect_paris()
