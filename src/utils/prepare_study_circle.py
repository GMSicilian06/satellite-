import geopandas as gpd
import rasterio
from rasterio.mask import mask
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from shapely.geometry import box
import os

def prepare_study_circle(shapefile_path, geotiff_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Study Circle (Ground Truth)
    print(f"Loading Ground Truth: {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)
    
    # Get bounds
    minx, miny, maxx, maxy = gdf.total_bounds
    print(f"Bounds: {minx}, {miny}, {maxx}, {maxy}")
    
    # Create a bounding box geometry
    bbox = box(minx, miny, maxx, maxy)
    
    # 2. Load Satellite Image
    print(f"Loading Satellite Tiff: {geotiff_path}")
    with rasterio.open(geotiff_path) as src:
        # Check CRS alignment
        print(f"Image CRS: {src.crs}, Shapes CRS: {gdf.crs}")
        if src.crs != gdf.crs:
            print("Reprojecting shapes to match image CRS...")
            gdf = gdf.to_crs(src.crs)
            bbox = box(*gdf.total_bounds)
            
        # Crop
        print("Cropping image to study circle bounds...")
        # We assume the shapefile covers the study area. 
        # Using the geometry of the blocks might define the mask, but a bbox crop is safer to get the whole context.
        # Let's crop to bbox.
        
        shapes = [bbox]
        out_image, out_transform = mask(src, shapes, crop=True)
        out_meta = src.meta.copy()

        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        
        # Save Cropped Tiff (optional, for geo-reference)
        out_tiff = os.path.join(output_dir, "study_circle.tif")
        with rasterio.open(out_tiff, "w", **out_meta) as dest:
            dest.write(out_image)
            
        # Save as PNG for Nanobanana
        # Rasterio reads as (Channels, Height, Width). PIP needs (H, W, C)
        # Also need to handle normalization if it's not uint8
        img_array = out_image.transpose(1, 2, 0)
        
        # If > 3 channels, take first 3 (RGB)
        if img_array.shape[2] > 3:
            img_array = img_array[:, :, :3]
            
        # Normalize if needed (e.g. if 16-bit)
        if out_meta['dtype'] == 'uint16':
             img_array = (img_array.astype('float32') / 256).astype('uint8')
        
        img_pil = Image.fromarray(img_array.astype('uint8'))
        out_png = os.path.join(output_dir, "study_circle.png")
        img_pil.save(out_png)
        print(f"Saved cropped image for AI: {out_png}")
        
    # Save the aligned/reprojected shapefile for comparison script
    out_shp = os.path.join(output_dir, "study_circle_truth.shp")
    gdf.to_file(out_shp)
    print(f"Saved aligned truth: {out_shp}")

if __name__ == "__main__":
    prepare_study_circle(
        "data/new_delhi/new_delhi_blocks.shp",
        "data/new_delhi/new_delhi_satellite.tif",
        "data/new_delhi_study_circle"
    )
