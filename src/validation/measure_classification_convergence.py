import os
import glob
import argparse
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

CLASS_NAMES = {
    0: "Open Space",
    1: "Non-residential",
    2: "Atomistic settlements",
    3: "Informal subdivisions",
    4: "Formal subdivisions",
    5: "Housing projects",
    6: "Class 6",
    7: "Class 7",
    8: "None / Other",
    -1: "Error",
}

def analyze_convergence(city_dir):
    print(f"Analyzing classification convergence for: {city_dir}")
    
    results_dir = os.path.join(city_dir, "classification_results")
    gis_dir = os.path.join(city_dir, "gis")
    
    # Sort the locales to simulate random sampling order
    locale_dirs = sorted(glob.glob(os.path.join(results_dir, "locale_*")))
    
    if not locale_dirs:
        print("No classification results found.")
        return
        
    locale_stats = []
    
    for ldir in locale_dirs:
        locale_id = os.path.basename(ldir)
        pred_file = os.path.join(ldir, "predicted_classified.geojson")
        gis_file = os.path.join(gis_dir, f"{locale_id}.geojson")
        
        if not os.path.exists(pred_file) or not os.path.exists(gis_file):
            continue
            
        try:
            # Load circle to get total area
            circle_gdf = gpd.read_file(gis_file)
            if circle_gdf.crs is None:
                circle_gdf.set_crs("EPSG:4326", inplace=True)
            circle_gdf = circle_gdf.to_crs("EPSG:3857")
            circle_area = circle_gdf.geometry.area.sum()
            
            # Load predictions
            preds_gdf = gpd.read_file(pred_file)
            class_areas = {}
            if len(preds_gdf) > 0:
                if preds_gdf.crs is None:
                    preds_gdf.set_crs("EPSG:4326", inplace=True)
                preds_gdf = preds_gdf.to_crs("EPSG:3857")
                preds_gdf['area_m2'] = preds_gdf.geometry.area
                
                # Group by class
                if 'pred_class_code' in preds_gdf.columns:
                    class_areas = preds_gdf.groupby('pred_class_code')['area_m2'].sum().to_dict()
                
            stats = {'locale': locale_id, 'circle_area': circle_area}
            
            # Initialize all known classes to 0 area to ensure consistent columns
            for class_code in CLASS_NAMES.keys():
                if class_code != -1:
                    stats[f'class_{class_code}'] = 0.0
                    
            # Populate with actual areas
            for class_code, area in class_areas.items():
                stats[f'class_{class_code}'] = area
                
            locale_stats.append(stats)
            
        except Exception as e:
            print(f"Error processing {locale_id}: {e}")
        
    if not locale_stats:
        print("No valid data found.")
        return
        
    df = pd.DataFrame(locale_stats)
    
    # Identify class columns
    class_cols = [c for c in df.columns if c.startswith('class_')]
    
    # Calculate cumulative running average
    df['cumulative_circle_area'] = df['circle_area'].cumsum()
    
    for c in class_cols:
        df[f'cum_{c}'] = df[c].cumsum()
        df[f'pct_{c}'] = (df[f'cum_{c}'] / df['cumulative_circle_area']) * 100
        
    # Save to CSV
    out_csv = os.path.join(city_dir, "classification_convergence.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved stats to {out_csv}")
    
    # Plot convergence
    plt.figure(figsize=(12, 8))
    
    pct_cols = [c for c in df.columns if c.startswith('pct_')]
    for c in pct_cols:
        # Get class name
        class_code = int(c.split('_')[2])
        class_name = CLASS_NAMES.get(class_code, f"Class {class_code}")
        
        # Only plot lines that have some area > 0 at the end
        if df[c].iloc[-1] > 0.1:
            plt.plot(df.index + 1, df[c], marker='o', markersize=4, label=class_name)
        
    plt.title(f"Land Use Convergence - {os.path.basename(os.path.normpath(city_dir))}")
    plt.xlabel("Number of Locales Combined")
    plt.ylabel("Cumulative Percentage of Circle Area (%)")
    plt.legend()
    plt.grid(True)
    
    out_plot = os.path.join(city_dir, "classification_convergence_plot.png")
    plt.savefig(out_plot, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {out_plot}")
    print(f"When lines flatten out, the percentages have converged, indicating you have enough circles.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure cumulative area of classified regions to analyze convergence.")
    parser.add_argument("--city_dir", type=str, required=True, help="Base directory of the processed city data (e.g., data/processed/locales/Bogota_Colombia)")
    args = parser.parse_args()
    
    analyze_convergence(args.city_dir)
