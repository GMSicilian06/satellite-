import geopandas as gpd
import sys

def inspect_shapefile(path):
    try:
        gdf = gpd.read_file(path)
        print(f"File: {path}")
        print(f"CRS: {gdf.crs}")
        print(f"Bounds (Total): {gdf.total_bounds}")
        print(f"Shape Count: {len(gdf)}")
        print(f"Columns: {gdf.columns.tolist()}")
    except Exception as e:
        print(f"Error reading shapefile: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_shapefile.py <path_to_shp>")
    else:
        inspect_shapefile(sys.argv[1])
