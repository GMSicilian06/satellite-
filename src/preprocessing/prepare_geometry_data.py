import os
import glob
import rasterio
from rasterio import features
import geopandas as gpd
import numpy as np
import cv2
from pathlib import Path

# Size of training tiles
TILE_SIZE = 512

def find_file_recursive(root, filename):
    for path in Path(root).rglob(filename):
        return str(path)
    return None

def build_id_map(root_dir):
    """
    Scans the directory for all *110.shp files and maps IDs to their directory.
    """
    id_map = {}
    for path in Path(root_dir).rglob("*110.shp"):
        # ID is the filename stem
        file_id = path.stem
        # Store the parent directory to find siblings
        id_map[file_id] = path.parent
    return id_map

def rasterize_layer(gdf, transform, shape):
    """
    Rasterizes a GeoDataFrame into a boolean mask.
    """
    if gdf.empty:
        return np.zeros(shape, dtype=np.uint8)
    
    # Rasterize shapes
    # shapes must be (geometry, value) iterable
    shapes = ((geom, 1) for geom in gdf.geometry)
    
    mask = features.rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8
    )
    return mask

def tile_image(image, mask_road, mask_block, output_prefix, output_dirs):
    """
    Cuts the image and masks into TILE_SIZE chunks.
    """
    h, w, c = image.shape
    
    # Simple sliding window with no overlap for now
    for y in range(0, h, TILE_SIZE):
        for x in range(0, w, TILE_SIZE):
            if y + TILE_SIZE > h or x + TILE_SIZE > w:
                continue
                
            img_tile = image[y:y+TILE_SIZE, x:x+TILE_SIZE]
            road_tile = mask_road[y:y+TILE_SIZE, x:x+TILE_SIZE]
            block_tile = mask_block[y:y+TILE_SIZE, x:x+TILE_SIZE]
            
            # Skip empty tiles (optional, but good for data cleaning)
            # If the image is all black (fetched padding) or invalid
            if np.mean(img_tile) < 5: 
                continue

            tile_id = f"{output_prefix}_{x}_{y}"
            
            # Save
            cv2.imwrite(os.path.join(output_dirs['images'], f"{tile_id}.jpg"), img_tile)
            cv2.imwrite(os.path.join(output_dirs['road'], f"{tile_id}.png"), road_tile * 255) # Save as visible 0-255
            cv2.imwrite(os.path.join(output_dirs['block'], f"{tile_id}.png"), block_tile * 255)

def process_data(img_dir, shp_dir, output_root):
    output_dirs = {
        'images': os.path.join(output_root, 'images'),
        'road': os.path.join(output_root, 'masks/roads'),
        'block': os.path.join(output_root, 'masks/blocks')
    }
    
    print("Building ID Map...")
    id_map = build_id_map(shp_dir)
    print(f"Mapped {len(id_map)} locations.")
    
    # Process each fetched image
    fetched_images = glob.glob(os.path.join(img_dir, "*.tif"))
    print(f"Found {len(fetched_images)} images to process.")
    
    for img_path in fetched_images:
        try:
            basename = Path(img_path).stem # e.g. 100010560000110
            
            if basename not in id_map:
                print(f"Skipping {basename}: Source Shapefiles not found in map.")
                continue
                
            shp_folder = id_map[basename]
            
            # Construct sibling filenames
            # Assuming standard naming convention: ...110 -> ...113 (Roads), ...112 (Blocks)
            # We need to construct the exact filename.
            # Usually the prefix matches.
            # Let's try to replace the suffix.
            
            # Handle variable length IDs if needed, but here simple replacement of last 3 chars seems checking.
            # Check if ends with 110
            if not basename.endswith('110'):
                print(f"Skipping {basename}: Doesn't end with 110 convention.")
                continue
                
            base_prefix = basename[:-3] # 100010560000
            road_id = base_prefix + "113"
            block_id = base_prefix + "112"
            
            road_path = os.path.join(shp_folder, f"{road_id}.shp")
            block_path = os.path.join(shp_folder, f"{block_id}.shp")
            
            # Open Image to get transform/shape
            with rasterio.open(img_path) as src:
                # Read image (channels last for cv2: H,W,C)
                # distinct from src.read() which is (C, H, W)
                img_data = src.read() 
                img_data = np.moveaxis(img_data, 0, -1) # Convert to H,W,C
                
                # Convert RGB to BGR for cv2
                img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

                transform = src.transform
                shape = src.shape # H, W
                crs = src.crs

            # Load vectors
            # Handle missing files gracefully (some might not have roads/blocks?)
            if os.path.exists(road_path):
                road_gdf = gpd.read_file(road_path)
                if road_gdf.crs != crs:
                    road_gdf = road_gdf.to_crs(crs)
                road_mask = rasterize_layer(road_gdf, transform, shape)
            else:
                print(f"Missing Roads for {basename}")
                road_mask = np.zeros(shape, dtype=np.uint8)

            if os.path.exists(block_path):
                block_gdf = gpd.read_file(block_path)
                if block_gdf.crs != crs:
                    block_gdf = block_gdf.to_crs(crs)
                block_mask = rasterize_layer(block_gdf, transform, shape)
            else:
                print(f"Missing Blocks for {basename}")
                block_mask = np.zeros(shape, dtype=np.uint8)
                
            # Tile and Save
            tile_image(img_data, road_mask, block_mask, basename, output_dirs)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--shp_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    
    process_data(args.img_dir, args.shp_dir, args.output_dir)
