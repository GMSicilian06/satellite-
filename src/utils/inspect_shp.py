import geopandas as gpd
import sys

try:
    gdf = gpd.read_file("data/ahmedabad_test/100562310072501.shp")
    print(f"Rows: {len(gdf)}")
    print(f"Columns: {gdf.columns}")
    if not gdf.empty:
        print(f"Geometry Type: {gdf.geom_type.unique()}")
        print(f"Bounds: {gdf.total_bounds}")
        print(gdf.head())
    else:
        print("Empty GeoDataFrame")
except Exception as e:
    print(e)
