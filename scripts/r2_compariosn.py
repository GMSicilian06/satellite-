import os
import sys
import glob
import random
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.metrics import r2_score

# Add project root to sys.path to resolve 'src' modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.validation.offline_analysis import run_offline_analysis

# Target cities (all 10 cities in Central America, Mexico, and Colombia)
CITIES = [
    "San Salvador",
    "Guatemala City",
    "Leon",
    "Culiacan",
    "Mexico City",
    "Guadalajara",
    "Bogota",
    "Valledupar",
    "Reynosa",
    "Tijuana"
]

CLASS_NAMES = {
    0: "Open Space",
    1: "Non-residential",
    2: "Atomistic settlements",
    3: "Informal subdivisions",
    4: "Formal subdivisions",
    5: "Housing projects",
    6: "None / Other"
}

def get_gt_column(df):
    for col in ['Land_use', 'Land_Use', 'Land_use_Code']:
        if col in df.columns:
            return col
    return None

def compute_r2_and_errors(y_true, y_pred):
    """Compute R2, MAE, and RMSE. Handles zero variance cases gracefully."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # R2 is undefined if the true values have zero variance
    if np.var(y_true) == 0:
        r2 = np.nan
    else:
        try:
            r2 = r2_score(y_true, y_pred)
        except Exception:
            r2 = np.nan
    return r2, mae, rmse

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model against AUE ground truth for 10 cities.")
    parser.add_argument("--num_circles", type=int, default=40, help="Number of circles per city to process")
    parser.add_argument("--model", type=str, default="models/landuse_leon_san_salvador_culiacan_mexico_city_reynosa_guadalajara_tijuana_bogota_valledupar_caracas_cabimas_holguin_quito_efficientnet_v2_s.pth", help="Path to trained model checkpoint")
    parser.add_argument("--force_classify", action="store_true", help="Redo classification even if predicted_classified.geojson exists")
    args = parser.parse_args()

    random.seed(42) # Reproducible sampling
    
    print(f"\n{'='*70}")
    print(f" LAND USE R2 EVALUATION PIPELINE")
    print(f" Model: {args.model}")
    print(f" Target Cities: {', '.join(CITIES)}")
    print(f" Samples Per City: {args.num_circles}")
    print(f"{'='*70}\n")

    circle_records = []
    
    for city in CITIES:
        print(f"\n>>> PROCESSING CITY: {city.upper()} <<<")
        base_dir = os.path.join("data/AUE_Everything", city)
        if not os.path.exists(base_dir):
            print(f"Error: AUE directory for {city} does not exist at {base_dir}. Skipping.")
            continue
            
        # Find all shapefiles recursively
        shp_files = glob.glob(os.path.join(base_dir, "**", "*.shp"), recursive=True)
        # Filter for block shapefiles ending in 11.shp
        valid_shps = [s for s in shp_files if s.endswith("11.shp") and "T0" not in s and "Arterial" not in s]
        
        if not valid_shps:
            print(f"Warning: No valid *11.shp block files found for {city}. Skipping.")
            continue
            
        print(f"Found {len(valid_shps)} total study circles in {city}.")
        
        # Sort and shuffle deterministically
        valid_shps.sort()
        random.shuffle(valid_shps)
        
        selected_shps = valid_shps[:args.num_circles]
        print(f"Selected {len(selected_shps)} circles to evaluate.")
        
        processed_count = 0
        for shp_path in selected_shps:
            sample_id = os.path.splitext(os.path.basename(shp_path))[0]
            output_dir = f"data/{city.lower()}/offline_comparison/{sample_id}"
            pred_path = os.path.join(output_dir, "predicted_classified.geojson")
            
            # Check if prediction file exists and we're not forcing re-classification
            if not os.path.exists(pred_path) or args.force_classify:
                print(f"\n--- Running classification for {city} Circle {sample_id} ---")
                try:
                    run_offline_analysis(city, sample_id, shp_path, args.model)
                except Exception as e:
                    print(f"Error classifying circle {sample_id}: {e}. Skipping this circle.")
                    continue
            else:
                print(f"Circle {sample_id} already classified. Loading cached results.")
                
            if not os.path.exists(pred_path):
                print(f"Warning: Classification failed to produce {pred_path}. Skipping.")
                continue
                
            # Process results
            try:
                gdf = gpd.read_file(pred_path)
                if gdf.empty:
                    continue
                
                # Check ground truth column name
                gt_col = get_gt_column(gdf)
                if not gt_col:
                    print(f"Warning: No ground truth column (Land_use) found in {pred_path}. Skipping.")
                    continue
                    
                # Convert to metric projection for accurate area calculations
                gdf_m = gdf.to_crs("EPSG:3857")
                gdf_m['area_m2'] = gdf_m.geometry.area
                
                # Compute total area and areas per class
                gt_areas = {c: 0.0 for c in range(7)}
                pred_areas = {c: 0.0 for c in range(7)}
                
                for _, row in gdf_m.iterrows():
                    area = row['area_m2']
                    
                    # GT
                    gt_val = row[gt_col]
                    if gt_val is not None and not pd.isna(gt_val):
                        try:
                            gt_cls = int(float(gt_val))
                            if gt_cls in gt_areas:
                                gt_areas[gt_cls] += area
                        except ValueError:
                            pass
                            
                    # Pred
                    pred_val = row.get('pred_class_code')
                    if pred_val is not None and not pd.isna(pred_val):
                        try:
                            pred_cls = int(float(pred_val))
                            if pred_cls in pred_areas:
                                pred_areas[pred_cls] += area
                        except ValueError:
                            pass
                            
                # Calculate percentages
                total_gt_area = sum(gt_areas.values())
                total_pred_area = sum(pred_areas.values())
                
                gt_pcts = {c: (gt_areas[c] / total_gt_area) * 100 if total_gt_area > 0 else 0.0 for c in range(7)}
                pred_pcts = {c: (pred_areas[c] / total_pred_area) * 100 if total_pred_area > 0 else 0.0 for c in range(7)}
                
                # Record circle statistics
                record = {
                    "City": city,
                    "Circle_ID": sample_id,
                    "Total_Area_m2": total_gt_area
                }
                for c in range(7):
                    record[f"GT_Area_m2_{c}"] = gt_areas[c]
                    record[f"Pred_Area_m2_{c}"] = pred_areas[c]
                    record[f"GT_Pct_{c}"] = gt_pcts[c]
                    record[f"Pred_Pct_{c}"] = pred_pcts[c]
                    
                circle_records.append(record)
                processed_count += 1
                
            except Exception as e:
                print(f"Error parsing prediction file for circle {sample_id}: {e}")
                continue
                
        print(f"Successfully processed {processed_count}/25 circles for {city}.")

    if not circle_records:
        print("\nError: No valid circles were processed successfully. Exiting.")
        return

    # Convert to DataFrame
    df_results = pd.DataFrame(circle_records)
    csv_path = "data/processed/r2_evaluation_results.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved circle-level results to CSV: {csv_path}")

    # Compute Circle-Level Metrics for each Category
    circle_metrics = []
    print("\n" + "="*70)
    print(f" CIRCLE-LEVEL EVALUATION METRICS (N = {len(df_results)})")
    print("="*70)
    print(f"{'Class Name':<28} {'R2 Score':>10} {'MAE (%)':>10} {'RMSE (%)':>10}")
    print("-" * 70)
    
    for c in range(6): # AUE Ground Truth classes 0-5
        y_true = df_results[f"GT_Pct_{c}"].values
        y_pred = df_results[f"Pred_Pct_{c}"].values
        
        r2, mae, rmse = compute_r2_and_errors(y_true, y_pred)
        r2_str = f"{r2:10.3f}" if not np.isnan(r2) else f"{'N/A':>10}"
        
        print(f"{CLASS_NAMES[c]:<28} {r2_str} {mae:10.2f}% {rmse:10.2f}%")
        circle_metrics.append({
            "Class_Code": c,
            "Class_Name": CLASS_NAMES[c],
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse
        })
    print("="*70)

    # Compute City-Level Aggregated Metrics
    city_grouped = df_results.groupby("City").sum()
    city_records = []
    
    for city_name, row in city_grouped.iterrows():
        city_gt_areas = {c: row[f"GT_Area_m2_{c}"] for c in range(7)}
        city_pred_areas = {c: row[f"Pred_Area_m2_{c}"] for c in range(7)}
        
        total_gt = sum(city_gt_areas.values())
        total_pred = sum(city_pred_areas.values())
        
        city_record = {
            "City": city_name,
        }
        for c in range(7):
            city_record[f"GT_Pct_{c}"] = (city_gt_areas[c] / total_gt) * 100 if total_gt > 0 else 0.0
            city_record[f"Pred_Pct_{c}"] = (city_pred_areas[c] / total_pred) * 100 if total_pred > 0 else 0.0
        city_records.append(city_record)
        
    df_cities = pd.DataFrame(city_records)
    
    city_metrics = []
    print("\n" + "="*70)
    print(f" CITY-LEVEL AGGREGATED METRICS (N = {len(df_cities)})")
    print("="*70)
    print(f"{'Class Name':<28} {'R2 Score':>10} {'MAE (%)':>10} {'RMSE (%)':>10}")
    print("-" * 70)
    
    for c in range(6):
        y_true = df_cities[f"GT_Pct_{c}"].values
        y_pred = df_cities[f"Pred_Pct_{c}"].values
        
        r2, mae, rmse = compute_r2_and_errors(y_true, y_pred)
        r2_str = f"{r2:10.3f}" if not np.isnan(r2) else f"{'N/A':>10}"
        
        print(f"{CLASS_NAMES[c]:<28} {r2_str} {mae:10.2f}% {rmse:10.2f}%")
        city_metrics.append({
            "Class_Code": c,
            "Class_Name": CLASS_NAMES[c],
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse
        })
    print("="*70)

    # Save Markdown Report
    md_path = "data/processed/r2_evaluation_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Land Use R2 and Prediction Error Evaluation Report\n\n")
        f.write(f"This report compares predictions from the fine-tuned model against the official Atlas of Urban Expansion (AUE) ground truth across **8 cities** (sampling up to 25 study circles per city).\n\n")
        
        f.write("## Circle-Level Evaluation Metrics\n")
        f.write("Evaluation computed across all individual study circles ($N = " + str(len(df_results)) + "$). This represents the model's high-resolution performance on localized 10-hectare zones.\n\n")
        f.write("| Class Name | R² Score | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for m in circle_metrics:
            r2_str = f"{m['R2']:.3f}" if not np.isnan(m['R2']) else "N/A"
            f.write(f"| {m['Class_Name']} | {r2_str} | {m['MAE']:.2f}% | {m['RMSE']:.2f}% |\n")
            
        f.write("\n## City-Level Aggregated Metrics\n")
        f.write("Evaluation computed by aggregating all circles for each city to obtain city-wide compositions ($N = " + str(len(df_cities)) + "$). This measures the model's capacity to predict macro-level urban patterns.\n\n")
        f.write("| Class Name | R² Score | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for m in city_metrics:
            r2_str = f"{m['R2']:.3f}" if not np.isnan(m['R2']) else "N/A"
            f.write(f"| {m['Class_Name']} | {r2_str} | {m['MAE']:.2f}% | {m['RMSE']:.2f}% |\n")
            
    print(f"\nSaved Markdown summary report to: {md_path}")

if __name__ == "__main__":
    main()