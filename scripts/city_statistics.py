"""
city_statistics.py
Computes area-weighted land use percentages for an entire Atlas city,
comparing ground truth labels vs model predictions across all shapefiles.

Usage (run from repo root):
    python3 scripts/city_statistics.py --city "Dhaka"
    python3 scripts/city_statistics.py --city "Lagos"
"""

import os
import sys
import glob
import argparse
import torch
import torch.nn as nn
import geopandas as gpd
import contextily as ctx
import numpy as np
from torchvision import models, transforms
from PIL import Image
import rasterio
from rasterio.mask import mask as rasterio_mask
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

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

def load_model(model_path, device):
    state = torch.load(model_path, map_location=device)
    num_classes = state['fc.weight'].shape[0]
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model, num_classes

def classify_polygon(geom, src, model, transform_fn, device):
    try:
        out_image, _ = rasterio_mask(src, [geom], crop=True)
        img_array = out_image.transpose(1, 2, 0)
        if img_array.shape[2] > 3:
            img_array = img_array[:, :, :3]
        if img_array.shape[0] < 5 or img_array.shape[1] < 5:
            return -1
        pil_img = Image.fromarray(img_array)
        tensor = transform_fn(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tensor)
            pred = torch.max(outputs, 1)[1].item()
        return pred
    except Exception:
        return -1

def compute_city_stats(city_name, atlas_root, model_path, fetched_dir):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model, num_classes = load_model(model_path, device)
    print(f"Loaded model with {num_classes} output classes.\n")

    transform_fn = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Discover labeled shapefiles
    all_shps = []
    for root_dir, _, files in os.walk(atlas_root):
        if city_name.lower() in root_dir.lower():
            for f in files:
                if f.endswith(".shp"):
                    all_shps.append(os.path.join(root_dir, f))

    valid_shps = []
    for shp in all_shps:
        try:
            gdf = gpd.read_file(shp, rows=1)
            cols = [c.lower() for c in gdf.columns]
            if 'land_use' in cols:
                valid_shps.append((shp, 'Land_use'))
            elif 'landuse' in cols:
                valid_shps.append((shp, 'landuse'))
        except Exception:
            pass

    print(f"Found {len(valid_shps)} labeled shapefiles for '{city_name}'.\n")

    # Area accumulators (in sq meters)
    gt_area   = {}   # class_code -> total area (ground truth)
    pred_area = {}   # class_code -> total area (predictions)

    for shp_path, lu_col in valid_shps:
        sample_id = os.path.splitext(os.path.basename(shp_path))[0]
        img_path  = os.path.join(fetched_dir, f"{sample_id}.tif")

        # Fetch satellite tile if missing
        if not os.path.exists(img_path):
            try:
                gdf = gpd.read_file(shp_path)
                if gdf.crs is None:
                    gdf = gdf.set_crs("EPSG:4326")
                gdf_m = gdf.to_crs("EPSG:3857")
                w, s, e, n = gdf_m.total_bounds
                b = 100
                os.makedirs(fetched_dir, exist_ok=True)
                ctx.bounds2raster(w-b, s-b, e+b, n+b, img_path,
                                  source=ctx.providers.Esri.WorldImagery)
            except Exception:
                continue  # Can't fetch, skip

        try:
            gdf = gpd.read_file(shp_path)
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")

            # Standardize column
            gdf = gdf.rename(columns={lu_col: 'Land_use'})
            gdf_m = gdf.to_crs("EPSG:3857")

            # Handle LineString geometries by buffering to small polygons (3m = half road width)
            geom_types = gdf_m.geometry.geom_type.unique()
            has_lines = any('LineString' in t for t in geom_types)
            if has_lines:
                gdf_m = gdf_m.copy()
                gdf_m['geometry'] = gdf_m.geometry.buffer(3)

            with rasterio.open(img_path) as src:
                gdf_rast = gdf_m.to_crs(src.crs)

                for idx, row in gdf_m.iterrows():
                    try:
                        gt_code = int(row['Land_use'])
                    except Exception:
                        continue
                    area_m2 = row.geometry.area
                    if area_m2 < 1:
                        continue

                    # Accumulate ground truth area
                    gt_area[gt_code] = gt_area.get(gt_code, 0) + area_m2

                    # Run model prediction
                    geom_for_mask = gdf_rast.loc[idx, 'geometry']
                    pred_code = classify_polygon(geom_for_mask, src, model, transform_fn, device)

                    # Accumulate predicted area
                    pred_area[pred_code] = pred_area.get(pred_code, 0) + area_m2

        except Exception as e:
            print(f"  [{sample_id}] Skipped: {e}")
            continue

    # --- Print Results ---
    total_gt   = sum(gt_area.values())
    total_pred = sum(v for k, v in pred_area.items() if k != -1)

    print("\n" + "="*62)
    print(f"  CITY-WIDE LAND USE STATISTICS — {city_name.upper()}")
    print("="*62)
    print(f"\n{'Class':<28} {'GT %':>8}  {'Pred %':>8}")
    print("-"*48)

    all_codes = sorted(set(list(gt_area.keys()) + [k for k in pred_area if k != -1]))
    for code in all_codes:
        name = CLASS_NAMES.get(code, f"Class {code}")
        gt_pct   = (gt_area.get(code, 0)   / total_gt   * 100) if total_gt   > 0 else 0
        pred_pct = (pred_area.get(code, 0) / total_pred * 100) if total_pred > 0 else 0
        print(f"  {name:<26} {gt_pct:>7.1f}%  {pred_pct:>7.1f}%")

    print("-"*48)
    print(f"  {'TOTAL':<26} {100.0:>7.1f}%  {100.0:>7.1f}%")
    print(f"\n  Total area analysed: {total_gt/1_000_000:.2f} km²")
    print("="*62 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city",       type=str, required=True)
    parser.add_argument("--atlas_root", type=str, default="data/raw/atlas_10ha/Accra")
    parser.add_argument("--model",      type=str, default="models/landuse_resnet18.pth")
    parser.add_argument("--fetched",    type=str, default="data/raw/fetched_images_robust")
    args = parser.parse_args()
    compute_city_stats(args.city, args.atlas_root, args.model, args.fetched)
