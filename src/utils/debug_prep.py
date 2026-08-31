import geopandas as gpd
import rasterio
from rasterio.mask import mask
import os
import glob
from pathlib import Path
import cv2
import numpy as np

def debug_single_id(img_path, shp_root):
    basename = Path(img_path).stem # 100943160008110
    print(f"DEBUG: Processing {basename}")
    
    # 1. Find Boundary
    search_pattern = f"{shp_root}/**/{basename}.shp"
    candidates = glob.glob(search_pattern, recursive=True)
    if not candidates:
        print("DEBUG: Boundary shapefile NOT FOUND via glob.")
        return
    
    print(f"DEBUG: Found boundary at {candidates[0]}")
    folder = Path(candidates[0]).parent
    base_id = basename[:-3] # 100943160008
    
    blocks_path = folder / f"{base_id}112.shp"
    landuse_path = folder / f"{base_id}111.shp"
    
    if not blocks_path.exists():
        print(f"DEBUG: Block file MISSING at {blocks_path}")
    else:
        print(f"DEBUG: Block file FOUND")
        
    if not landuse_path.exists():
        print(f"DEBUG: LandUse file MISSING at {landuse_path}")
    else:
        print(f"DEBUG: LandUse file FOUND")
        
    # Load and Join
    try:
        blocks_df = gpd.read_file(blocks_path)
        landuse_df = gpd.read_file(landuse_path)
        print(f"DEBUG: Loaded Blocks ({len(blocks_df)}) and LandUse ({len(landuse_df)})")
        
        if blocks_df.crs != landuse_df.crs:
            print(f"DEBUG: CRS Mismatch! Blocks: {blocks_df.crs}, LandUse: {landuse_df.crs}")
            blocks_df = blocks_df.to_crs(landuse_df.crs)
            
        print(f"DEBUG: Blocks Bounds: {blocks_df.total_bounds}")
        print(f"DEBUG: LandUse Bounds: {landuse_df.total_bounds}")
        print(f"DEBUG: Blocks Geom Type: {blocks_df.geometry.type.unique()}")
        print(f"DEBUG: LandUse Geom Type: {landuse_df.geometry.type.unique()}")
            
        from shapely.ops import polygonize
        from shapely.geometry import MultiLineString
        
        # Try to polygonize LandUse lines
        print("DEBUG: Attempting to polygonize LandUse Lines...")
        lines = list(landuse_df.geometry)
        polygons = list(polygonize(lines))
        print(f"DEBUG: Generated {len(polygons)} polygons from {len(lines)} lines.")
        
        if len(polygons) == 0:
            print("DEBUG: Polygonization failed (no closed loops?).")
            return

        # Create new GeoDataFrame for Polygons
        # We need to assign labels to these polygons.
        # This is tricky. The lines have labels. The polygon is formed by lines.
        # We need to find which label is associated with the polygon.
        
        # Heuristic: Spatial join the new polygons with the original lines to get the label.
        
        blocks_polys = gpd.GeoDataFrame(geometry=polygons, crs=landuse_df.crs)
        
        # Check intersection with LandUse Lines
        # We join Polygons (blocks) to Lines (landuse)
        joined = gpd.sjoin(blocks_polys, landuse_df, how="inner", predicate="intersects")
        print(f"DEBUG: Spatial Join Result (Polys vs Lines): {len(joined)} labeled blocks.")
        
        if len(joined) > 0:
            print("DEBUG: Sample Label:", joined.iloc[0]['Land_use'])
            
            # Check Image cropping with Polygon
            with rasterio.open(img_path) as src:
                if joined.crs != src.crs:
                    joined = joined.to_crs(src.crs)
                
                # Try cropping first block
                geom = joined.iloc[0].geometry
                # mask requires iterable of GeoJSON-like dicts or objects with __geo_interface__
                out_image, out_transform = mask(src, [geom], crop=True)
                print(f"DEBUG: Crop Shape: {out_image.shape}")
                
    except Exception as e:
        print(f"DEBUG: Error: {e}")

if __name__ == "__main__":
    debug_single_id(
        "data/raw/fetched_images/100943160008110.tif",
        "data/raw/atlas_10ha"
    )
