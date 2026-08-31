import geopandas as gpd
import rasterio
from rasterio.mask import mask
import os
import glob
from pathlib import Path
import cv2
import numpy as np
from shapely.geometry import box

def prepare_dataset(img_dir, shp_root, output_dir):
    """
    1. Matches Satellite Image -> Blocks (112) -> Land Use (111).
    2. Spatially joins Blocks with Land Use to get Class Label.
    3. Crops Satellite Image for each Block.
    4. Saves to output_dir/class_id/filename.jpg.
    """
    
    # Directories created on the fly
        
    # Find all fetched images
    tif_files = glob.glob(os.path.join(img_dir, "*.tif"))
    print(f"Found {len(tif_files)} satellite images.")
    
    total_blocks = 0
    saved_blocks = 0
    
    for tif_path in tif_files:
        try:
            basename = Path(tif_path).stem # e.g., 100010560000110
            
            # Find corresponding shapefiles
            # Logic: recursively find folder containing basename.shp (the boundary)
            # then look for 112 (blocks) and 111 (land use) in that same folder
            search_pattern = f"{shp_root}/**/{basename}.shp"
            candidates = glob.glob(search_pattern, recursive=True)
            if not candidates:
                print(f"No boundary shapefile found for {basename}, skipping.")
                continue
                
            folder = Path(candidates[0]).parent
            blocks_path = folder / f"{basename[:-1]}2.shp"
            landuse_path = folder / f"{basename[:-1]}1.shp"
            
            if not landuse_path.exists():
                print(f"No Land Use file for {basename}")
                continue
                
            # Load Data
            landuse_df = gpd.read_file(landuse_path)
            
            if landuse_df.empty:
                continue

            # Polygonize Land Use Lines
            # We ignore blocks_df as it is unreliable
            from shapely.ops import polygonize
            lines = list(landuse_df.geometry)
            polygons = list(polygonize(lines))
            
            if not polygons:
                continue
                
            # Create Dataframe from Polygons
            poly_df = gpd.GeoDataFrame(geometry=polygons, crs=landuse_df.crs)
            
            # Spatial Join to get Labels back from lines
            # A polygon is formed by lines. So likely the lines touch the polygon boundary.
            # 'touches' or 'intersects'
            joined = gpd.sjoin(poly_df, landuse_df, how="inner", predicate="intersects")
            
            # Now we have labeled polygons
            # Note: A polygon might intersect multiple lines (its borders).
            # Usually they have the same label if it's a homogeneous area.
            # We drop duplicates to avoid saving the same crop multiple times if it matched multiple lines.
            joined = joined[~joined.index.duplicated(keep='first')]
             
            with rasterio.open(tif_path) as src:
                # Ensure blocks are in the same CRS as the image for masking
                if joined.crs != src.crs:
                    joined = joined.to_crs(src.crs)
                    
                for idx, row in joined.iterrows():
                    geom = row.geometry
                    
                    # Robust Column Check
                    label = None
                    for col in ['Land_use', 'Land_Use', 'landuse', 'Land_use_Code', 'Land_Use_C']:
                         if col in row:
                             label = row[col]
                             break
                    
                    if label is None:
                        # Fallback: try by index? No, too risky.
                        continue
 
                    
                    try:
                        # Mask/Crop
                        out_image, out_transform = mask(src, [geom], crop=True)
                        
                        if out_image.shape[0] == 0 or out_image.shape[1] == 0 or out_image.shape[2] == 0:
                            continue
                            
                        # Reshape for CV2: (H, W, C)
                        out_img_cv = np.moveaxis(out_image, 0, -1)
                        out_img_cv = cv2.cvtColor(out_img_cv, cv2.COLOR_RGB2BGR)
                        
                        if out_img_cv.shape[0] < 5 or out_img_cv.shape[1] < 5:
                            continue
                            
                        # Unique filename: basename + polygon index
                        out_filename = f"{basename}_{idx}.jpg"
                        
                        # 80/20 Split based on random chance
                        import random
                        split = "train" if random.random() < 0.8 else "val"
                        
                        out_dir_final = os.path.join(output_dir, split, str(label))
                        os.makedirs(out_dir_final, exist_ok=True)
                        
                        out_path = os.path.join(out_dir_final, out_filename)
                        
                        cv2.imwrite(out_path, out_img_cv)
                        saved_blocks += 1
                        
                    except Exception as e:
                        pass
                        
            total_blocks += len(polygons)
            
        except Exception as e:
            print(f"Error processing {tif_path}: {e}")
            
    print(f"Processing Complete.")
    print(f"Total Blocks Scanned: {total_blocks}")
    print(f"Labeled & Saved Blocks: {saved_blocks}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--shp_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    
    prepare_dataset(args.img_dir, args.shp_dir, args.output_dir)
