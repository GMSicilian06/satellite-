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

# Define Region Configs
REGIONS_CONFIG = {
    "South Asia": {
        "model_path": "/Users/carlossequeira/Satellite/models/landuse_dhaka_lahore_ahmedabad_belgaum_coimbatore_hindupur_hyderabad_jaipur_jalna_kanpur_kolkata_kozhikode_malegaon_mumbai_parbhani_pune_singrauli_sitapur_vijayawada_efficientnet_v2_s.pth",
        "ending": "01.shp",
        "cities": {
            "Dhaka": "Bangladesh",
            "Lahore": "Pakistan",
            "Ahmedabad": "India",
            "Belgaum": "India",
            "Coimbatore": "India",
            "Hindupur": "India",
            "Hyderabad": "India",
            "Jaipur": "India",
            "Jalna": "India",
            "Kanpur": "India",
            "Kolkata": "India",
            "Kozhikode": "India",
            "Malegaon": "India",
            "Mumbai": "India",
            "Parbhani": "India",
            "Pune": "India",
            "Singrauli": "India",
            "Sitapur": "India",
            "Vijayawada": "India"
        }
    },
    "East Asia": {
        "model_path": "/Users/carlossequeira/Satellite/models/landuse_east_asia_efficientnet_v2_s.pth",
        "ending": "01.shp",
        "cities": {
            "Anqing, Anhui": "China",
            "Beijing, Beijing": "China",
            "Bicheng, Chongqing": "China",
            "Bicheng, Chongqing 2": "China",
            "Busan": "South Korea",
            "Changzhou, Jingsu": "China",
            "Chengdu, Sichuan": "China",
            "Cheonan": "South Korea",
            "Cheonan 2": "South Korea",
            "Fukuoka": "Japan",
            "Gaoyou, Jiangsu": "China",
            "Guangzhou, Guangdong": "China",
            "Gwangju": "South Korea",
            "Haikou, Hainan": "China",
            "Hangzhou, Zhejiang": "China",
            "Hangzhou, Zhejiang 2": "China",
            "Hong Kong, Hong Kong": "China",
            "Jinan, Shandong": "China",
            "Jinju": "South Korea",
            "Kaiping, Guangdong": "China",
            "Leshan, Sichuan": "China",
            "Okayama": "Japan",
            "Osaka": "Japan",
            "Pingxiang, Jiangxi": "China",
            "Pingxiang, Jiangxi 2": "China",
            "Pyongyang": "North Korea",
            "Qingdao, Shandong": "China",
            "Seoul": "South Korea",
            "Shanghai, Shanghai": "China",
            "Shanghai, Shanghai 2": "China",
            "Shenzhen, Guangdong": "China",
            "Suining, Sichuan": "China",
            "Taipei, Taiwan": "Taiwan",
            "Tangshan, Hebei": "China",
            "Tianjin, Tianjin": "China",
            "Tokyo": "Japan",
            "Wuhan, Hubei": "China",
            "Xingping, Shaanxi": "China",
            "Xucheng, Jiangsu": "China",
            "Yamaguchi": "Japan",
            "Yiyang, Hunan": "China",
            "Yucheng, Zhejiang": "China",
            "Yulin, Guangxi": "China",
            "Zhengzhou, Henan": "China",
            "Zunyi, Guizhou": "China"
        }
    },
    "Central America & Mexico": {
        "model_path": "/Users/carlossequeira/Satellite/models/landuse_leon_san_salvador_culiacan_mexico_city_reynosa_guadalajara_tijuana_bogota_valledupar_caracas_cabimas_holguin_quito_efficientnet_v2_s.pth",
        "ending": "11.shp",
        "cities": {
            "Leon": "Nicaragua",
            "San Salvador": "El Salvador",
            "Culiacan": "Mexico",
            "Mexico City": "Mexico",
            "Reynosa": "Mexico",
            "Guadalajara": "Mexico",
            "Tijuana": "Mexico",
            "Guatemala City": "Guatemala",
            "Holguin": "Cuba"
        }
    },
    "South America": {
        "model_path": "/Users/carlossequeira/Satellite/models/landuse_south_america_efficientnet_v2_s.pth",
        "ending": "11.shp",
        "cities": {
            "Belo Horizonte": "Brazil",
            "Bogota": "Colombia",
            "Cabimas": "Venezuela",
            "Caracas": "Venezuela",
            "Cochabamba": "Bolivia",
            "Cordoba": "Argentina",
            "Florianopolis": "Brazil",
            "Ilheus": "Brazil",
            "Palmas": "Brazil",
            "Quito": "Ecuador",
            "Valledupar": "Colombia"
        }
    }
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
    img_all_dir = "/Users/carlossequeira/Satellite/data/raw/fetched_images_all"
    img_robust_dir = "/Users/carlossequeira/Satellite/data/raw/fetched_images_robust"
    
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}", flush=True)
    
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    tifs = [os.path.splitext(f)[0] for f in os.listdir(img_all_dir) if f.endswith(".tif")]
    
    csv_path = "/Users/carlossequeira/Satellite/data/processed/all_regions_r2_results.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    # Write header and empty the file
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("City,Region,Country,Classification Metric,Model Pct,AUE Pct\n")
    
    for reg_name, config in REGIONS_CONFIG.items():
        print(f"\n==================================================")
        print(f"🌐 EVALUATING REGION: {reg_name.upper()}")
        print(f"==================================================")
        
        model_path = config["model_path"]
        if not os.path.exists(model_path):
            print(f"Warning: Model weights not found at {model_path}. Skipping region.")
            continue
            
        model, num_classes = load_model(model_path, device)
        ending = config["ending"]
        
        for city, country in config["cities"].items():
            print(f"\n>>> Processing {city} ({country}) ...", flush=True)
            base_dir = os.path.join("/Users/carlossequeira/Satellite/data/AUE_Everything", city)
            if not os.path.exists(base_dir):
                print(f"  Directory not found for {city}. Skipping.", flush=True)
                continue
                
            shp_files = glob.glob(os.path.join(base_dir, "**", "*.shp"), recursive=True)
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
                with open(csv_path, "a", encoding="utf-8") as f:
                    for c in range(num_classes):
                        if c == 1:
                            continue  # Skip Non-residential
                        gt_pct = (city_gt_areas[c] / total_gt_excl) * 100
                        pred_pct = (city_pred_areas[c] / total_pred_excl) * 100 if total_pred_excl > 0 else 0.0
                        f.write(f'"{city}","{reg_name}","{country}","{CLASS_NAMES[c]}",{pred_pct:.6f},{gt_pct:.6f}\n')
            
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) < 50:
        print("\nNo matching cities were evaluated. Exiting.", flush=True)
        return
        
    print(f"\nFinished evaluation! Results written to CSV: {csv_path}", flush=True)
    df_results = pd.read_csv(csv_path)
    
    # Save Markdown Reports
    artifact_paths = [
        "/Users/carlossequeira/.gemini/antigravity-ide/brain/1886d4a0-c314-40da-9625-5170fe657153/all_regions_classification_comparison.md",
        "/Users/carlossequeira/Satellite/data/processed/all_regions_classification_comparison.md"
    ]
    for path in artifact_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Master Land Use Classification Metrics Comparison (All Regions)\n\n")
            f.write("This master report compares the land-use classification metrics predicted by our regional fine-tuned models against the official **Atlas of Urban Expansion (AUE)** ground truth across all study circles.\n\n")
            f.write("## 1. Percentage Composition Table\n\n")
            
            f.write("| Region | Country | City | Classification Metric | Model Classification Number (%) | AUE Classification Number (%) |\n")
            f.write("| :--- | :--- | :--- | :--- | :---: | :---: |\n")
            for _, row in df_results.iterrows():
                f.write(f"| {row['Region']} | {row['Country']} | {row['City']} | {row['Classification Metric']} | {row['Model Pct']:.2f}% | {row['AUE Pct']:.2f}% |\n")
            f.write("\n")
                
        print(f"Saved Markdown report to: {path}", flush=True)

if __name__ == "__main__":
    main()
