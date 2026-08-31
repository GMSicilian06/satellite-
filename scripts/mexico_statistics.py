import os
import sys
import glob
import argparse
import subprocess
import geopandas as gpd
import pandas as pd

# Standard Class Names matching the fine-tuned model's output indices
# (The model has 7 output classes: 0 -> Open Space, 1 -> Non-residential, 
# 2 -> Atomistic, 3 -> Informal, 4 -> Formal, 5 -> Housing projects, 6 -> None / Other)
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
    "Mexico City",
    "Guadalajara",
    "León",
    "Culiacán",
    "Reynosa",
    "Tijuana",
    "Monterrey",
    "Puebla",
    "Toluca",
    "Ciudad Juárez",
    "Torreón",
    "Querétaro",
    "San Luis Potosí",
    "Mérida",
    "Aguascalientes",
]

def run_pipeline_for_city(city_name, num_locales, model_path, append=False):
    print("\n" + "="*70)
    print(f" PROCESSING CITY: {city_name.upper()}")
    print("="*70)
    
    cmd = [
        sys.executable, "pipeline/validate_ghs_city.py",
        "--city", city_name,
        "--country", "México",
        "--num_locales", str(num_locales),
        "--model", model_path
    ]
    if append:
        cmd.append("--append")
        
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def load_city_results(city_name):
    # Standardize name for folder matching validate_ghs_city.py
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

def print_stats_table(df, title):
    total_area = df['area_m2'].sum()
    if total_area == 0:
        print(f"No valid classified area found for {title}.")
        return
        
    stats = df.groupby('pred_class_code')['area_m2'].sum().reset_index()
    stats['percentage'] = (stats['area_m2'] / total_area) * 100
    stats['class_name'] = stats['pred_class_code'].map(CLASS_NAMES).fillna("Unknown")
    
    print("\n" + "="*60)
    print(f"  LAND USE DISTRIBUTION: {title.upper()}")
    print("="*60)
    print(f"{'Class Name':<28} {'Area (km²)':>12} {'Percentage':>12}")
    print("-" * 60)
    
    for _, row in stats.sort_values('percentage', ascending=False).iterrows():
        area_km2 = row['area_m2'] / 1_000_000
        print(f"{row['class_name']:<28} {area_km2:>12.3f} {row['percentage']:>11.1f}%")
        
    print("-" * 60)
    print(f"{'TOTAL':<28} {total_area/1_000_000:>12.3f} {100.0:>11.1f}%")
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Orchestrate land use statistics for Mexican cities.")
    parser.add_argument("--num_locales", type=int, default=20, help="Number of locales to sample per city")
    parser.add_argument("--model", type=str, default="models/landuse_leon_san_salvador_culiacan_mexico_city_reynosa_guadalajara_tijuana_bogota_valledupar_caracas_cabimas_holguin_quito_efficientnet_v2_s.pth", help="Path to trained .pth model file")
    parser.add_argument("--append", action="store_true", help="Skip existing processed files and resume classification")
    
    args = parser.parse_args()
    
    # 1. Run validation pipeline for each city
    all_city_results = {}
    
    for city in CITIES:
        try:
            run_pipeline_for_city(city, args.num_locales, args.model, args.append)
            # Immediately load results to check success
            city_df = load_city_results(city)
            if city_df is not None:
                all_city_results[city] = city_df
            else:
                print(f"Warning: No results compiled for {city}")
        except Exception as e:
            print(f"Error running pipeline for {city}: {e}")
            continue

    # 2. Export and Display Results
    if not all_city_results:
        print("Error: No classification data was successfully compiled for any city.")
        return
        
    # Build combined list for exporting
    rows = []
    
    # City statistics
    for city, df in all_city_results.items():
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
            
    # Nationwide aggregated statistics
    combined_df = pd.concat(all_city_results.values(), ignore_index=True)
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
    csv_path = "data/processed/mexico_landuse_statistics.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    export_df.to_csv(csv_path, index=False)
    print(f"Saved CSV file to: {csv_path}")
    
    # Generate Markdown Report
    md_path = "data/processed/mexico_landuse_statistics.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Land Use Statistics Report: Mexico\n\n")
        f.write("Generated from random circular locales (10-hectare zones) using the fine-tuned EfficientNet model.\n\n")
        
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
        for city in all_city_results.keys():
            f.write(f"### {city}\n\n")
            f.write("| Class Name | Area (km²) | Percentage |\n")
            f.write("| :--- | :---: | :---: |\n")
            city_rows = export_df[export_df['City'] == city].sort_values('Percentage', ascending=False)
            city_total = city_rows['Area_m2'].sum() / 1_000_000
            for _, r in city_rows.iterrows():
                f.write(f"| {r['Class_Name']} | {r['Area_km2']:.3f} | {r['Percentage']:.1f}% |\n")
            f.write(f"| **TOTAL** | **{city_total:.3f}** | **100.0%** |\n\n")
            
    print(f"Saved Markdown report to: {md_path}")

    # Display results in console
    print("\n\n" + "X"*70)
    print(" SUMMARY OF RESULTS BY CITY")
    print("X"*70)
    for city, df in all_city_results.items():
        print_stats_table(df, city)
        
    print("\n\n" + "X"*70)
    print(" AGGREGATED NATIONWIDE MEXICO STATISTICS")
    print("X"*70)
    print_stats_table(combined_df, "All Processed Mexican Cities")

if __name__ == "__main__":
    main()