import geopandas as gpd
import rasterio
from shapely.geometry import shape, mapping
from shapely.affinity import affine_transform
import json
import os

def georeference_geojson(pixel_geojson_path, geotiff_path, output_path):
    print(f"Georeferencing {pixel_geojson_path} using {geotiff_path}...")
    
    # 1. Load Pixel Polygons
    gdf_pixels = gpd.read_file(pixel_geojson_path)
    
    # 2. Load GeoTIFF Transform
    with rasterio.open(geotiff_path) as src:
        transform = src.transform
        crs = src.crs
        print(f"Loaded Transform: {transform}")
        print(f"Target CRS: {crs}")
        
    # 3. Apply Transform
    # Rasterio transform is [a, b, c, d, e, f] mapping (col, row) -> (x, y)
    # Shapely affine_transform expects [a, b, d, e, xoff, yoff]
    # Rasterio: x = a*col + b*row + c
    #           y = d*col + e*row + f
    # Shapely:  x' = a*x + b*y + xoff
    #           y' = d*x + e*y + yoff
    
    # So the mapping is direct since col=x, row=y in shapely geometry if we treat them as such.
    # But wait, image encoding:
    # If the geojson has (x, y) as (col, row - from top left).
    # Then we just apply the affine matrix.
    
    # Rasterio transform matrix:
    # | a b c |
    # | d e f |
    # | 0 0 1 |
    
    # Shapely expects: [a, b, d, e, c, f]
    matrix = [transform.a, transform.b, transform.d, transform.e, transform.c, transform.f]
    
    def apply_affine(geom):
        return affine_transform(geom, matrix)
    
    gdf_pixels['geometry'] = gdf_pixels['geometry'].apply(apply_affine)
    gdf_pixels.crs = crs
    
    # 4. Save
    gdf_pixels.to_file(output_path, driver="GeoJSON")
    print(f"Saved Georeferenced Polygons: {output_path}")

if __name__ == "__main__":
    georeference_geojson(
        "data/ahmedabad_output/plots.geojson",
        "data/ahmedabad_test/image.tif",
        "data/ahmedabad_output/plots_georeferenced.geojson"
    )
