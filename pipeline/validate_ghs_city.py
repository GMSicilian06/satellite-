import os
import argparse
import subprocess
import geopandas as gpd

def main():
    parser = argparse.ArgumentParser(description="Validates a city directly from the GHS-UCDB global database.")
    parser.add_argument("--city", required=True, help="Name of the city to extract from GHS-UCDB (e.g. 'Buenos Aires', 'Lima')")
    parser.add_argument("--country", default=None, help="Optional country filter to restrict matches")
    parser.add_argument("--db_path", default="/Users/carlossequeira/Downloads/GHS_UCDB_MTUC_GLOBE_R2024A_V1_0/GHS_UCDB_MTUC_GLOBE_R2024A.gpkg", help="Path to GHS_UCDB geopackage")
    parser.add_argument("--num_locales", type=int, required=True, help="Number of locales to generate")
    parser.add_argument("--model", default="models/landuse_efficientnet_v2_s.pth", help="Path to your trained .pth model file")
    parser.add_argument("--append", action="store_true", help="Append to existing locales")
    args = parser.parse_args()
    
    city_query = args.city.strip()
    safe_city_name = city_query.replace(" ", "_").replace(",", "")

    print(f"\n======================================================")
    print(f" GHS-UCDB Boundary Extractor")
    print(f" Target City: {city_query}")
    if args.country:
        print(f" Target Country: {args.country}")
    print(f"======================================================")
    
    if not os.path.exists(args.db_path):
        print(f"Error: Database not found at {args.db_path}")
        return

    # 1. Search for city
    print(f"\n[1/3] Searching for '{city_query}' in GHS-UCDB Database...")
    try:
        # First attempt exact match
        sql_where = f"GC_UCN_MAI_2025 = '{city_query}'"
        attr_gdf = gpd.read_file(args.db_path, layer="GHSL_UCDB_MTUC_GLOBE_R2024", where=sql_where)
        
        # Fallback to LIKE if exact match fails
        if len(attr_gdf) == 0:
            sql_where = f"GC_UCN_MAI_2025 LIKE '%{city_query}%'"
            attr_gdf = gpd.read_file(args.db_path, layer="GHSL_UCDB_MTUC_GLOBE_R2024", where=sql_where)
            if len(attr_gdf) == 0:
                sql_where_alt = f"GC_UCN_LIS_2025 LIKE '%{city_query}%'"
                attr_gdf = gpd.read_file(args.db_path, layer="GHSL_UCDB_MTUC_GLOBE_R2024", where=sql_where_alt)
            
        if len(attr_gdf) == 0:
            print(f"Error: City '{city_query}' not found in the GHS-UCDB database.")
            return
            
        if len(attr_gdf) > 1:
            if args.country:
                # Filter by country name
                filtered = attr_gdf[attr_gdf['GC_CNT_GAD_2025'].str.contains(args.country, case=False, na=False)]
                if len(filtered) > 0:
                    attr_gdf = filtered
            
            # Print matches if still multiple
            if len(attr_gdf) > 1:
                print(f"Warning: Found {len(attr_gdf)} matches for '{city_query}'. Using the first one:")
                for idx, row in attr_gdf.iterrows():
                    print(f"  - {row['GC_UCN_MAI_2025']} (Country: {row['GC_CNT_GAD_2025']})")
                
        # Take the first matched city
        target_row = attr_gdf.iloc[0]
        city_id = target_row['ID_MTUC_G0']
        actual_name = target_row['GC_UCN_MAI_2025']
        country = target_row['GC_CNT_GAD_2025']
        
        print(f"      Matched City: {actual_name}, {country} (ID: {city_id})")
        
    except Exception as e:
        print(f"Error reading database attributes: {e}")
        return

    # 2. Extract the actual Polygon Boundary using the ID
    print(f"\n[2/3] Extracting polygon geometry for ID {city_id}...")
    try:
        poly_layer = "GHSL_UCDB_MTUC_2025_GLOBE_R2024"
        poly_where = f"ID_UC_G0 = {city_id}"
        poly_gdf = gpd.read_file(args.db_path, layer=poly_layer, where=poly_where)
        
        if len(poly_gdf) == 0:
            print(f"Error: Could not retrieve polygon geometry for ID {city_id}.")
            return
            
        # Ensure it's EPSG:4326 for consistency with the rest of the pipeline
        poly_gdf = poly_gdf.to_crs("EPSG:4326")
        
        # Save as temporary GeoJSON
        extents_dir = "data/processed/urban_extents"
        os.makedirs(extents_dir, exist_ok=True)
        boundary_path = os.path.join(extents_dir, f"{safe_city_name}_ghs_boundary.geojson")
        
        poly_gdf.to_file(boundary_path, driver="GeoJSON")
        print(f"      Saved exact boundary to: {boundary_path}")
        
    except Exception as e:
        print(f"Error reading geometry layer: {e}")
        return

    # 3. Hook into the existing boundary validation pipeline
    print(f"\n[3/3] Triggering validation pipeline for {actual_name}...")
    cmd = [
        ".venv/bin/python3", "pipeline/validate_city_from_boundary.py",
        "--boundary", boundary_path,
        "--city", f"{safe_city_name}_GHS",
        "--num_locales", str(args.num_locales),
        "--model", args.model
    ]
    if args.append:
        cmd.append("--append")
    
    print(f"      Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
