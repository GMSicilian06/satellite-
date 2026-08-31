import os
import glob
import geopandas as gpd
import pandas as pd

CLASS_NAMES = {
    0: "Open Space",
    1: "Non-residential",
    2: "Atomistic settlements",
    3: "Informal subdivisions",
    4: "Formal subdivisions",
    5: "Housing projects",
    6: "None / Other",
    -1: "Error",
}

CITIES = [
    "Bogota",
    "Medellín",
    "Cali",
    "Barranquilla",
    "Cartagena",
    "Bucaramanga",
    "Cúcuta",
    "Pereira",
    "Ibagué",
    "Santa Marta",
    "Manizales",
    "Pasto",
    "Montería",
    "Valledupar",
    "Villavicencio",
]

def load_city_results(city_name):
    safe_city_name = city_name.replace(" ", "_").replace(",", "")
    results_dir = os.path.join("data/processed/locales", f"{safe_city_name}_GHS", "classification_results")
    pattern = os.path.join(results_dir, "**", "predicted_classified.geojson")
    geojsons = glob.glob(pattern, recursive=True)
    
    city_gdfs = []
    for g_path in geojsons:
        try:
            gdf = gpd.read_file(g_path)
            if gdf.empty:
                continue
            if gdf.crs is None:
                gdf.set_crs("EPSG:4326", inplace=True)
            gdf_m = gdf.to_crs("EPSG:3857")
            gdf_m['area_m2'] = gdf_m.geometry.area
            city_gdfs.append(gdf_m[['pred_class_code', 'area_m2']])
        except Exception as e:
            pass
           
    if not city_gdfs:
        return None
    return pd.concat(city_gdfs, ignore_index=True)

def main():
    print("Collecting Colombia statistics from disk...")
    city_data = {}
    total_locales_count = 0
    for city in CITIES:
        df = load_city_results(city)
        if df is not None:
            city_data[city] = df
            # Count locales by checking folders
            safe_city_name = city.replace(" ", "_").replace(",", "")
            results_dir = os.path.join("data/processed/locales", f"{safe_city_name}_GHS", "classification_results")
            locales_found = len(glob.glob(os.path.join(results_dir, "locale_*")))
            total_locales_count += locales_found
            print(f"Loaded results for {city} ({locales_found} locales, {len(df)} blocks).")
        else:
            print(f"No results found for {city}.")
            
    if not city_data:
        print("Error: No classification results found for any Colombian city.")
        return
        
    print(f"Total locales compiled across all cities: {total_locales_count}")
    
    # Build combined list for exporting
    rows = []
    
    # 1. City reports
    for city, df in city_data.items():
        total_area = df['area_m2'].sum()
        stats = df.groupby('pred_class_code')['area_m2'].sum().reset_index()
        stats['percentage'] = (stats['area_m2'] / total_area) * 100
        stats['class_name'] = stats['pred_class_code'].map(CLASS_NAMES).fillna("Unknown")
        
        for _, r in stats.iterrows():
            rows.append({
                "City": city,
                "Class_Code": r['pred_class_code'],
                "Class_Name": r['class_name'],
                "Area_m2": r['area_m2'],
                "Area_km2": r['area_m2'] / 1_000_000,
                "Percentage": r['percentage']
            })
            
    # 2. Nationwide report
    combined_df = pd.concat(city_data.values(), ignore_index=True)
    total_nation_area = combined_df['area_m2'].sum()
    nation_stats = combined_df.groupby('pred_class_code')['area_m2'].sum().reset_index()
    nation_stats['percentage'] = (nation_stats['area_m2'] / total_nation_area) * 100
    nation_stats['class_name'] = nation_stats['pred_class_code'].map(CLASS_NAMES).fillna("Unknown")
    
    for _, r in nation_stats.iterrows():
        rows.append({
            "City": "Nationwide (Aggregated)",
            "Class_Code": r['pred_class_code'],
            "Class_Name": r['class_name'],
            "Area_m2": r['area_m2'],
            "Area_km2": r['area_m2'] / 1_000_000,
            "Percentage": r['percentage']
        })
        
    export_df = pd.DataFrame(rows)
    
    # Save CSV
    csv_path = "data/processed/colombia_landuse_statistics.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    export_df.to_csv(csv_path, index=False)
    print(f"Saved CSV file to: {csv_path}")
    
    # Generate Markdown Report
    md_path = "data/processed/colombia_landuse_statistics.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Land Use Statistics Report: Colombia\n\n")
        f.write(f"Generated from random circular locales (compiled {total_locales_count} total locales across {len(city_data)} cities, 10 hectares each) using the fine-tuned EfficientNet model.\n\n")
        
        # Nationwide table
        f.write("## Nationwide Aggregated Statistics\n\n")
        f.write("| Class Name | Area (km²) | Percentage |\n")
        f.write("| :--- | :---: | :---: |\n")
        nation_rows = export_df[export_df['City'] == "Nationwide (Aggregated)"].sort_values('Percentage', ascending=False)
        for _, r in nation_rows.iterrows():
            f.write(f"| {r['Class_Name']} | {r['Area_km2']:.3f} | {r['Percentage']:.1f}% |\n")
        f.write(f"| **TOTAL** | **{total_nation_area/1_000_000:.3f}** | **100.0%** |\n\n")
        
        # Individual cities
        f.write("## Statistics by City\n\n")
        for city in city_data.keys():
            f.write(f"### {city}\n\n")
            f.write("| Class Name | Area (km²) | Percentage |\n")
            f.write("| :--- | :---: | :---: |\n")
            city_rows = export_df[export_df['City'] == city].sort_values('Percentage', ascending=False)
            city_total = city_rows['Area_m2'].sum() / 1_000_000
            for _, r in city_rows.iterrows():
                f.write(f"| {r['Class_Name']} | {r['Area_km2']:.3f} | {r['Percentage']:.1f}% |\n")
            f.write(f"| **TOTAL** | **{city_total:.3f}** | **100.0%** |\n\n")
            
    print(f"Saved Markdown report to: {md_path}")

if __name__ == "__main__":
    main()
