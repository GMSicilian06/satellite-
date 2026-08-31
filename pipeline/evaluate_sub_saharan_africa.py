import os
import sys
import glob
import shutil
import torch
import torch.nn as nn
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.mask import mask
from torchvision import models, transforms
from PIL import Image
from shapely.ops import polygonize
import warnings
warnings.filterwarnings("ignore")

# Sub-Saharan African configuration
SSA_CITIES = {
    "Accra": {"country": "Ghana", "ending": "11.shp"},
    "Beira": {"country": "Mozambique", "ending": "01.shp"},
    "Johannesburg": {"country": "South Africa", "ending": "01.shp"},
    "Kinshasa": {"country": "DR Congo", "ending": "01.shp"},
    "Lagos": {"country": "Nigeria", "ending": "01.shp"},
    "Luanda": {"country": "Angola", "ending": "01.shp"},
    "Nakuru": {"country": "Kenya", "ending": "01.shp"},
    "Port Elizabeth": {"country": "South Africa", "ending": "01.shp"}
}

CLASS_NAMES = {
    0: "Open Space",
    1: "Non-residential",
    2: "Atomistic settlements",
    3: "Informal subdivisions",
    4: "Formal subdivisions",
    5: "Housing projects",
    6: "None / Other"
}

def load_model(model_path, device):
    loaded_obj = torch.load(model_path, map_location=device)
    if 'classifier.1.weight' in loaded_obj:
        num_classes = loaded_obj['classifier.1.weight'].shape[0]
    else:
        num_classes = 7
    print(f"Loaded weights from {os.path.basename(model_path)} ({num_classes} classes).")
    model = models.efficientnet_v2_s(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(loaded_obj)
    model.eval()
    model.to(device)
    return model, num_classes

def classify_polygon(geom, src, model, transform, device):
    try:
        geom_buffered = geom.buffer(5.0)
        out_image, _ = mask(src, [geom_buffered], crop=True)
        img_array = out_image.transpose(1, 2, 0)
        if img_array.shape[2] > 3:
            img_array = img_array[:, :, :3]
        if img_array.shape[0] < 5 or img_array.shape[1] < 5:
            return -1
        pil_img = Image.fromarray(img_array)
        tensor = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tensor)
            pred = torch.max(outputs, 1)[1].item()
        return pred
    except Exception:
        return -1

def main():
    model_path = "/Users/carlossequeira/Satellite/models/landuse_sub_saharan_africa_efficientnet_v2_s.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        sys.exit(1)
        
    img_all_dir = "/Users/carlossequeira/Satellite/data/raw/fetched_images_all"
    img_robust_dir = "/Users/carlossequeira/Satellite/data/raw/fetched_images_robust"
    
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}", flush=True)
    
    model, num_classes = load_model(model_path, device)
    
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    tifs = [os.path.splitext(f)[0] for f in os.listdir(img_all_dir) if f.endswith(".tif")]
    records = []
    
    for city, config in SSA_CITIES.items():
        print(f"\n>>> Processing {city} ({config['country']}) ...", flush=True)
        base_dir = os.path.join("/Users/carlossequeira/Satellite/data/AUE_Everything", city)
        if not os.path.exists(base_dir):
            print(f"  Directory not found for {city}. Skipping.", flush=True)
            continue
            
        shp_files = glob.glob(os.path.join(base_dir, "**", "*.shp"), recursive=True)
        ending = config["ending"]
        valid_shps = [s for s in shp_files if s.endswith(ending) and "T0" not in s and "Arterial" not in s]
        
        valid_shps.sort()
        selected_shps = []
        for shp in valid_shps:
            name = os.path.splitext(os.path.basename(shp))[0]
            if name in tifs:
                selected_shps.append((shp, name))
                    
        print(f"  Selected {len(selected_shps)} study circles for evaluation.", flush=True)
        if len(selected_shps) == 0:
            continue
        
        city_gt_areas = {c: 0.0 for c in range(num_classes)}
        city_pred_areas = {c: 0.0 for c in range(num_classes)}
        
        for shp_path, shp_name in selected_shps:
            try:
                src_tif = os.path.join(img_all_dir, f"{shp_name}.tif")
                dst_tif = os.path.join(img_robust_dir, f"{shp_name}.tif")
                if not os.path.exists(dst_tif):
                    shutil.copy2(src_tif, dst_tif)
                
                # Load shapefile
                gdf = gpd.read_file(shp_path)
                if gdf.empty:
                    continue
                if gdf.crs is None:
                    gdf = gdf.set_crs("EPSG:4326")
                
                # Case-insensitive ground truth column name detection
                gt_col = None
                for col in gdf.columns:
                    if col.lower() in ['land_use', 'landuse', 'land_use_code']:
                        gt_col = col
                        break
                        
                if not gt_col:
                    continue
                
                gdf_m = gdf.to_crs("EPSG:3857")
                gdf_m['area_m2'] = gdf_m.geometry.area
                
                geom_types = gdf_m['geometry'].geom_type.unique()
                has_lines = any('LineString' in t for t in geom_types)
                has_polys = any('Polygon' in t for t in geom_types)
                
                if has_lines and not has_polys:
                    lines = list(gdf_m.geometry)
                    polys = list(polygonize(lines))
                    if len(polys) == 0:
                        gdf_m['geometry'] = gdf_m.geometry.buffer(5)
                        gdf_m['area_m2'] = gdf_m.geometry.area
                    else:
                        gdf_blocks = gpd.GeoDataFrame(geometry=polys, crs=gdf_m.crs)
                        gdf_m = gpd.sjoin(gdf_blocks, gdf_m, how='left', predicate='intersects')
                        gdf_m = gdf_m[~gdf_m.index.duplicated(keep='first')]
                        gdf_m['area_m2'] = gdf_m.geometry.area
                elif has_lines and has_polys:
                    gdf_m = gdf_m.copy()
                    gdf_m['geometry'] = gdf_m.geometry.apply(lambda g: g.buffer(5) if 'Line' in g.geom_type else g)
                    gdf_m['area_m2'] = gdf_m.geometry.area
                
                # Open Geotiff
                blocks_classified = 0
                with rasterio.open(dst_tif) as src:
                    gdf_rast = gdf_m.to_crs(src.crs)
                    
                    for idx, row in gdf_m.iterrows():
                        try:
                            gt_val = row[gt_col]
                            if gt_val is None or pd.isna(gt_val):
                                continue
                            gt_cls = int(float(gt_val))
                            if gt_cls not in city_gt_areas:
                                continue
                            
                            area_m2 = row['area_m2']
                            if area_m2 < 1:
                                continue
                                
                            geom_for_mask = gdf_rast.loc[idx, 'geometry']
                            pred_cls = classify_polygon(geom_for_mask, src, model, transform, device)
                            
                            if pred_cls != -1 and pred_cls in city_pred_areas:
                                city_gt_areas[gt_cls] += area_m2
                                city_pred_areas[pred_cls] += area_m2
                                blocks_classified += 1
                        except Exception:
                            pass
                print(f"    Circle {shp_name}: Classified {blocks_classified} blocks.", flush=True)
            except Exception as e:
                print(f"    Error processing circle {shp_name}: {e}", flush=True)
                
        # Calculate totals excluding class 1 ("Non-residential")
        total_gt_excl = sum(area for c, area in city_gt_areas.items() if c != 1)
        total_pred_excl = sum(area for c, area in city_pred_areas.items() if c != 1)
        print(f"  Aggregated Area (Excluding Non-residential) for {city}: GT={total_gt_excl/1_000_000:.3f} km², Pred={total_pred_excl/1_000_000:.3f} km²", flush=True)
        
        if total_gt_excl > 0:
            for c in range(num_classes):
                if c == 1:
                    continue  # Skip Non-residential
                gt_pct = (city_gt_areas[c] / total_gt_excl) * 100
                pred_pct = (city_pred_areas[c] / total_pred_excl) * 100 if total_pred_excl > 0 else 0.0
                records.append({
                    "City": city,
                    "Region": "Sub-Saharan Africa",
                    "Country": config["country"],
                    "Classification Metric": CLASS_NAMES[c],
                    "Model Pct": round(pred_pct, 6),
                    "AUE Pct": round(gt_pct, 6)
                })
            
    if len(records) == 0:
        print("\nNo matching cities were evaluated. Exiting.", flush=True)
        return
        
    df_new = pd.DataFrame(records)
    
    # Save Sub-Saharan African CSV results
    ssa_csv_path = "/Users/carlossequeira/Satellite/data/processed/sub_saharan_africa_r2_results.csv"
    os.makedirs(os.path.dirname(ssa_csv_path), exist_ok=True)
    df_new.to_csv(ssa_csv_path, index=False)
    print(f"\nSaved Sub-Saharan African results to CSV: {ssa_csv_path}", flush=True)
    
    # Merge into the master CSV
    master_csv_path = "/Users/carlossequeira/Satellite/data/processed/all_regions_r2_results.csv"
    if os.path.exists(master_csv_path):
        print(f"Loading master CSV: {master_csv_path}...")
        df_master = pd.read_csv(master_csv_path)
        
        # Clean master values
        for col in df_master.columns:
            if df_master[col].dtype == object:
                df_master[col] = df_master[col].astype(str).str.strip('\" ')
                
        # Remove any existing Sub-Saharan Africa rows to prevent duplicates
        df_master = df_master[df_master['Region'] != 'Sub-Saharan Africa']
        
        # Append new results
        df_combined = pd.concat([df_master, df_new], ignore_index=True)
    else:
        df_combined = df_new
        
    df_combined.to_csv(master_csv_path, index=False)
    print(f"Merged and updated master CSV at: {master_csv_path}", flush=True)
    
    # Generate Separate Markdown Reports for Sub-Saharan Africa
    artifact_paths = [
        "/Users/carlossequeira/.gemini/antigravity-ide/brain/1886d4a0-c314-40da-9625-5170fe657153/sub_saharan_africa_classification_comparison.md",
        "/Users/carlossequeira/Satellite/data/processed/sub_saharan_africa_classification_comparison.md"
    ]
    for path in artifact_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Sub-Saharan African Land Use Classification Comparison\n\n")
            f.write("This report presents the land-use classification metrics predicted by our regional fine-tuned Sub-Saharan African model against the official **Atlas of Urban Expansion (AUE)** ground truth across all study circles, excluding the Non-residential class.\n\n")
            f.write("## Percentage Composition Table\n\n")
            
            f.write("| Country | City | Classification Metric | Model Classification Number (%) | AUE Classification Number (%) |\n")
            f.write("| :--- | :--- | :--- | :---: | :---: |\n")
            for _, row in df_new.iterrows():
                f.write(f"| {row['Country']} | {row['City']} | {row['Classification Metric']} | {row['Model Pct']:.2f}% | {row['AUE Pct']:.2f}% |\n")
            f.write("\n")
            
        print(f"Saved Markdown report to: {path}", flush=True)

if __name__ == "__main__":
    main()
