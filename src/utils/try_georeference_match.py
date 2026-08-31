
import os
import geopandas as gpd
import rasterio
from shapely.affinity import affine_transform

def try_georeference_matches():
    candidates = [
        "100751890072801",
        "101051890072801",
        "100031900072901",
        "100731890072801",
        "100071900072801"
    ]
    
    pixel_geojson = "data/mumbai/nano_extracted/blocks.geojson"
    mumbai_shp_base = "data/raw/atlas_10ha/Accra/Mumbai/Mumbai_T0"
    image_base = "data/raw/fetched_images_robust"
    
    print(f"Testing {len(candidates)} candidates for {pixel_geojson}...")
    
    # Load Pixel Polygons once
    gdf_pixels = gpd.read_file(pixel_geojson)
    
    found_match = False
    
    for cid in candidates:
        tif_path = os.path.join(image_base, cid + ".tif")
        shp_path = os.path.join(mumbai_shp_base, cid + ".shp")
        
        if not os.path.exists(tif_path) or not os.path.exists(shp_path):
            print(f"Skipping {cid}: Missing file.")
            continue
            
        print(f"Testing Candidate: {cid}")
        
        # 1. Load Transform
        with rasterio.open(tif_path) as src:
            transform = src.transform
            crs = src.crs
            
        # 2. Apply Transform (Temp)
        matrix = [transform.a, transform.b, transform.d, transform.e, transform.c, transform.f]
        
        gdf_temp = gdf_pixels.copy()
        gdf_temp['geometry'] = gdf_temp['geometry'].apply(lambda g: affine_transform(g, matrix))
        gdf_temp.crs = crs
        
        # 3. Check Overlap with Truth
        gdf_truth = gpd.read_file(shp_path)
        if gdf_truth.crs != crs:
            gdf_truth = gdf_truth.to_crs(crs)
            
        xmin, ymin, xmax, ymax = gdf_truth.total_bounds
        overlap = gdf_temp.cx[xmin:xmax, ymin:ymax]
        
        count = len(overlap)
        print(f"  -> Overlap Count: {count}")
        
        if count > 20: # Should be substantial overlap
            print(f"SUCCESS! Found matching ID: {cid}")
            found_match = True
            
            # Save Georeferenced File
            out_path = f"data/mumbai/nano_extracted/blocks_georeferenced_{cid}.geojson"
            gdf_temp.to_file(out_path, driver="GeoJSON")
            print(f"  Saved: {out_path}")
            break
            
    if not found_match:
        print("No matching candidate found.")

if __name__ == "__main__":
    try_georeference_matches()
