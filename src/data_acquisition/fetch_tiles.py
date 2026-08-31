import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path

def fetch_image_for_shapefile(shp_path, output_dir, buffer_val=0.001):
    """
    Fetches satellite imagery for the extent of the given shapefile.
    
    Args:
        shp_path (str): Path to the shapefile defining the area of interest (AOI).
        output_dir (str): Directory to save the resulting GeoTIFF.
        buffer_val (float): Buffer to add to bounds (in degrees usually if CRS is 4326). 
                            Contextily handles reprojecting for the tile fetch, 
                            but we want to ensure we cover the edges.
    """
    try:
        # Load the shapefile
        gdf = gpd.read_file(shp_path)
        
        # ensure 3857 for contextily mapping (optional but good practice to know what we are dealing with)
        # However, bounds2raster handles reprojection internally usually, 
        # but the input bounds must be in the target CRS or we specify it.
        # contextily.bounds2raster expects (west, south, east, north)
        
        # Let's verify CRS.
        if gdf.crs is None:
            print(f"Warning: {shp_path} has no CRS. Assuming EPSG:4326")
            gdf.set_crs(epsg=4326, inplace=True)
            
        # Get bounds
        w, s, e, n = gdf.total_bounds
        
        # Add small buffer to ensure we don't clip edges
        w -= buffer_val
        s -= buffer_val
        e += buffer_val
        n += buffer_val
        
        # Prepare output filename
        # Use the same basename as the shapefile (e.g. 100010560000110.shp -> 100010560000110.tif)
        basename = Path(shp_path).stem
        output_path = os.path.join(output_dir, f"{basename}.tif")
        
        print(f"Fetching tile for {basename}...")
        
        # Fetch and save
        # Source: Esri World Imagery (Satellite)
        search = ctx.bounds2raster(
            w, s, e, n,
            output_path,
            zoom='auto', # or specify level like 18
            source=ctx.providers.Esri.WorldImagery,
            ll=True # Input bounds are in Lat/Lon (Long, Lat) if True. 
                    # If gdf is 4326, then yes.
        )
        
        print(f"Saved to {output_path}")
        return output_path

    except Exception as e:
        print(f"Failed to fetch for {shp_path}: {e}")
        return None

def process_directory(root_dir, output_dir):
    """
    Recursively find all *110.shp files (assuming these are the locale boundaries)
    and fetch images for them.
    """
    # Pattern to match the "Locale Boundary" shapefiles often ending in 110 or just generally 
    # looking for the main ID.
    # Based on user data: 100010560000110.shp
    
    search_pattern = str(Path(root_dir) / "**" / "*110.shp")
    shapefiles = glob.glob(search_pattern, recursive=True)
    
    if not shapefiles:
        print("No *110.shp files found. Trying broader search *.shp...")
        shapefiles = glob.glob(str(Path(root_dir) / "**" / "*.shp"), recursive=True)
        # Filter to avoid fetching for every single layer if we only want one per locale
        # But for now, let's limit to the "Boundary" ones if we can deduce them.
        # User example: *110 is boundary, *111 is land use, *112 is plot.
        # We only need the image ONCE per locale (the image is the same for all).
        # So filtering for *110.shp is a good heuristic.
        shapefiles = [f for f in shapefiles if f.endswith("110.shp")]

    print(f"Found {len(shapefiles)} target shapefiles.")
    
    for shp in shapefiles:
        fetch_image_for_shapefile(shp, output_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input directory containing Shapefiles")
    parser.add_argument("--output", required=True, help="Output directory for Satellite Images")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    process_directory(args.input, args.output)
