import geopandas as gpd
import argparse
from shapely.affinity import scale
from shapely.geometry import Polygon

def rescale_geojson(input_path, output_path, scale_x, scale_y):
    print(f"Rescaling {input_path} by X={scale_x}, Y={scale_y}...")
    
    gdf = gpd.read_file(input_path)
    if gdf.empty:
        print("Input GeoJSON is empty.")
        return

    # Shapely scale function scales from the center by default.
    # We want to scale from the origin (0,0) because these are pixel coordinates relative to top-left.
    # So we used scaled = scale(geom, xfact, yfact, origin=(0,0))
    
    def scale_geom(geom):
        return scale(geom, xfact=scale_x, yfact=scale_y, origin=(0,0))
        
    gdf['geometry'] = gdf['geometry'].apply(scale_geom)
    
    gdf.to_file(output_path, driver='GeoJSON')
    print(f"Saved rescaled vectors to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sx", type=float, required=True)
    parser.add_argument("--sy", type=float, required=True)
    args = parser.parse_args()
    
    rescale_geojson(args.input, args.output, args.sx, args.sy)
