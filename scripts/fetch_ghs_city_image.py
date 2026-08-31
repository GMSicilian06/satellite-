import os
import argparse
import geopandas as gpd
import contextily as ctx
import rasterio
from rasterio.mask import mask
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Fetches an entire city's satellite image cropped to its GHS-UCDB boundary.")
    parser.add_argument("--city", required=True, help="City name (e.g. 'Paris', 'Lima')")
    parser.add_argument("--db_path", default="/Users/carlossequeira/Downloads/GHS_UCDB_MTUC_GLOBE_R2024A_V1_0/GHS_UCDB_MTUC_GLOBE_R2024A.gpkg", help="Path to GHS_UCDB geopackage")
    parser.add_argument("--zoom", type=int, default=13, help="Satellite zoom level. 13-14 is safe for full cities. 18 will likely crash for an entire city!")
    parser.add_argument("--year", type=int, default=2025, choices=[1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025, 2030], help="Year of the GHS boundary layer.")
    args = parser.parse_args()

    city_query = args.city.strip()
    safe_city_name = city_query.replace(" ", "_").replace(",", "")

    print(f"\n======================================================")
    print(f" Full City Satellite Fetcher (GHS-UCDB)")
    print(f" Target City: {city_query}")
    print(f" Target Zoom: {args.zoom} (Warning: Higher than 14 can be massive)")
    print(f"======================================================")
    
    if not os.path.exists(args.db_path):
        print(f"Error: Database not found at {args.db_path}")
        return

    # 1. Search for the city
    print(f"\n[1/4] Searching for '{city_query}' in database...")
    try:
        # Exact match first
        sql_where = f"GC_UCN_MAI_2025 = '{city_query}'"
        attr_gdf = gpd.read_file(args.db_path, layer="GHSL_UCDB_MTUC_GLOBE_R2024", where=sql_where)
        
        # If no exact match, fallback to LIKE
        if len(attr_gdf) == 0:
            sql_where = f"GC_UCN_MAI_2025 LIKE '%{city_query}%'"
            attr_gdf = gpd.read_file(args.db_path, layer="GHSL_UCDB_MTUC_GLOBE_R2024", where=sql_where)
            if len(attr_gdf) == 0:
                sql_where_alt = f"GC_UCN_LIS_2025 LIKE '%{city_query}%'"
                attr_gdf = gpd.read_file(args.db_path, layer="GHSL_UCDB_MTUC_GLOBE_R2024", where=sql_where_alt)
            
        if len(attr_gdf) == 0:
            print(f"Error: City '{city_query}' not found.")
            return
            
        if len(attr_gdf) > 1:
            print(f"Warning: Found {len(attr_gdf)} matches. Selecting the first:")
            for idx, row in attr_gdf.iterrows():
                print(f"  - {row['GC_UCN_MAI_2025']} ({row['GC_CNT_GAD_2025']})")
                
        target_row = attr_gdf.iloc[0]
        city_id = target_row['ID_MTUC_G0']
        actual_name = target_row['GC_UCN_MAI_2025']
        country = target_row['GC_CNT_GAD_2025']
        print(f"      Matched: {actual_name}, {country}")
    except Exception as e:
         print(f"Error reading DB: {e}")
         return

    # 2. Extract Polygon
    print(f"\n[2/4] Extracting geographical footprint for {args.year}...")
    poly_layer = f"GHSL_UCDB_MTUC_{args.year}_GLOBE_R2024"
    poly_gdf = gpd.read_file(args.db_path, layer=poly_layer, where=f"ID_UC_G0 = {city_id}")
    
    if len(poly_gdf) == 0:
        print(f"Error: Polygon for year {args.year} not found.")
        return

    # Reproject to Web Mercator for proper Contextily fetching
    poly_gdf = poly_gdf.to_crs(epsg=3857)
    
    # Calculate bounds
    w, s, e, n = poly_gdf.total_bounds
    print(f"      Bounds (Mercator): W:{w:.1f}, S:{s:.1f}, E:{e:.1f}, N:{n:.1f}")

    # Set up directories
    out_dir = os.path.join("data", "processed", "full_city_images")
    os.makedirs(out_dir, exist_ok=True)
    temp_tif = os.path.join(out_dir, f"{safe_city_name}_{args.year}_temp.tif")
    final_tif = os.path.join(out_dir, f"{safe_city_name}_{args.year}_city.tif")
    final_img = os.path.join(out_dir, f"{safe_city_name}_{args.year}_city_preview.png")
    final_jpg = os.path.join(out_dir, f"{safe_city_name}_{args.year}_city_preview.jpg")

    # 3. Fetch Imagery
    print(f"\n[3/4] Downloading Esri satellite tiles (Zoom {args.zoom}). This may take a moment...")
    try:
        ctx.bounds2raster(
            w, s, e, n,
            temp_tif,
            zoom=args.zoom,
            source=ctx.providers.Esri.WorldImagery
        )
        print("      Raw tiles downloaded and stitched successfully.")
    except Exception as err:
        print(f"      Download Error: {err}")
        return

    # 4. Mask and Crop strictly to the boundary
    print(f"\n[4/4] Masking image to strictly fit the boundary footprint...")
    try:
        with rasterio.open(temp_tif) as src:
            out_image, out_transform = mask(
                src,
                poly_gdf.geometry,
                crop=True,         # Remove black box padding around boundary
                filled=True,       # Fill pixels outside boundary with fill_value
                nodata=0           # Treat outside as pure black no-data
            )
            out_meta = src.meta.copy()

        # Update metadata with new cropped dimensions
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": 0
        })

        # Save masked GeoTIFF
        with rasterio.open(final_tif, "w", **out_meta) as dest:
            dest.write(out_image)
            
        # Clean up the raw square box
        if os.path.exists(temp_tif):
            os.remove(temp_tif)
            
        print(f"      Saved precisely masked GeoTIFF: {final_tif}")
        
        # BONUS: Save a preview PNG/JPG for easy viewing without GIS software
        print(f"      Generating standard preview image...")
        from rasterio.plot import reshape_as_image
        img_data = reshape_as_image(out_image)
        # Handle alpha/nodata rendering for PNG (making pure black transparent or keeping it)
        plt.figure(figsize=(12, 12))
        plt.imshow(img_data)
        plt.axis('off')
        plt.savefig(final_img, bbox_inches='tight', dpi=200, transparent=True)
        # Also save JPG
        from PIL import Image
        img_pil = Image.open(final_img)
        if img_pil.mode in ("RGBA", "LA") or (img_pil.mode == "P" and "transparency" in img_pil.info):
            background = Image.new("RGB", img_pil.size, (255, 255, 255))
            background.paste(img_pil, mask=img_pil.split()[3])
            img_pil = background
        else:
            img_pil = img_pil.convert("RGB")
        img_pil.save(final_jpg, "JPEG", quality=95)
        plt.close()
        print(f"      Saved transparent preview PNG: {final_img}")
        print(f"      Saved standard JPEG: {final_jpg}")

    except Exception as err:
        print(f"      Masking Error: {err}")
        return

    print(f"\n======================================================")
    print(f" Done! The complete city imagery is ready in:")
    print(f" {out_dir}")
    print(f"======================================================")

if __name__ == "__main__":
    main()
