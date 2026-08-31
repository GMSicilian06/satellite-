import os
import pandas as pd
import numpy as np

def main():
    csv_path = 'data/processed/all_regions_r2_results.csv'
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    print(f"Loading existing results from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Clean text values (strip extra whitespace or quotes)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip('\" ')

    # Filter out Non-residential
    print("Filtering out 'Non-residential' metric rows...")
    df_filtered = df[df['Classification Metric'] != 'Non-residential'].copy()

    # Re-normalize percentages using absolute areas to preserve precision
    print("Re-normalizing percentages of the remaining classes...")
    # Group by city/region/country and sum remaining areas
    totals = df_filtered.groupby(['City', 'Region', 'Country'])[['Model Area (km2)', 'AUE Area (km2)']].sum().reset_index()
    totals.rename(columns={'Model Area (km2)': 'Total Model Area', 'AUE Area (km2)': 'Total AUE Area'}, inplace=True)

    # Merge totals back to calculate new percentages
    df_merged = pd.merge(df_filtered, totals, on=['City', 'Region', 'Country'])

    df_merged['Model Pct'] = np.where(
        df_merged['Total Model Area'] > 0,
        (df_merged['Model Area (km2)'] / df_merged['Total Model Area']) * 100,
        0.0
    )
    df_merged['AUE Pct'] = np.where(
        df_merged['Total AUE Area'] > 0,
        (df_merged['AUE Area (km2)'] / df_merged['Total AUE Area']) * 100,
        0.0
    )

    # Keep only the requested 6 columns (City, Region, Country, Metric, Model Pct, AUE Pct)
    output_cols = ['City', 'Region', 'Country', 'Classification Metric', 'Model Pct', 'AUE Pct']
    df_final = df_merged[output_cols].copy()

    # Round to 6 decimal places for the CSV file
    df_final['Model Pct'] = df_final['Model Pct'].round(6)
    df_final['AUE Pct'] = df_final['AUE Pct'].round(6)

    # Save cleaned CSV back to workspace
    df_final.to_csv(csv_path, index=False)
    print(f"Saved cleaned 6-column CSV to: {csv_path}")

    # Save Markdown Reports
    artifact_paths = [
        "/Users/carlossequeira/.gemini/antigravity-ide/brain/1886d4a0-c314-40da-9625-5170fe657153/all_regions_classification_comparison.md",
        "/Users/carlossequeira/Satellite/data/processed/all_regions_classification_comparison.md"
    ]
    
    for path in artifact_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Master Land Use Classification Metrics Comparison (All Regions)\n\n")
            f.write("This master report compares the land-use classification metrics predicted by our regional fine-tuned models against the official **Atlas of Urban Expansion (AUE)** ground truth across all study circles, excluding the Non-residential class.\n\n")
            f.write("## 1. Percentage Composition Table\n\n")
            
            f.write("| Region | Country | City | Classification Metric | Model Classification Number (%) | AUE Classification Number (%) |\n")
            f.write("| :--- | :--- | :--- | :--- | :---: | :---: |\n")
            for _, row in df_final.iterrows():
                f.write(f"| {row['Region']} | {row['Country']} | {row['City']} | {row['Classification Metric']} | {row['Model Pct']:.2f}% | {row['AUE Pct']:.2f}% |\n")
            f.write("\n")
                
        print(f"Saved Markdown report to: {path}")

if __name__ == "__main__":
    main()
