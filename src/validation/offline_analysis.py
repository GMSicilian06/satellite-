import geopandas as gpd
import rasterio
from rasterio.mask import mask
import rasterio.plot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import torch
from torchvision import transforms
from PIL import Image
import shapely.geometry
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from pipeline.inference import EfficientNetV2_Improved

def offline_analysis(
    existing_blocks_path, 
    study_circle_truth_path, 
    geotiff_path,
    png_path, 
    model_path,
    output_dir
):
    os.makedirs(output_dir, exist_ok=True)
    
    print("--- Phase 1: Spatial Cropping (No API) ---")
    # 1. Load Ground Truth (The Circle)
    gdf_truth = gpd.read_file(study_circle_truth_path)
    circle_bounds = gdf_truth.total_bounds
    bbox_poly = shapely.geometry.box(*circle_bounds)
    
    # 2. Load Existing Full-City Polygons
    print(f"Loading existing data from: {existing_blocks_path}")
    gdf_all = gpd.read_file(existing_blocks_path)
    
    # Ensure CRS match
    if gdf_all.crs != gdf_truth.crs:
        gdf_all = gdf_all.to_crs(gdf_truth.crs)
        
    # 3. Clip/Crop to the Circle
    # Clip/Crop to the Circle
    print("Masking invalid geometries (Robust)...")
    gdf_all = gdf_all[~gdf_all.geometry.is_empty & gdf_all.geometry.notna()]
    
    def safe_fix(geom):
        if geom.is_valid:
            return geom
        try:
            fixed = geom.buffer(0)
            if fixed.is_empty: return None
            return fixed
        except Exception:
            return None

    gdf_all['geometry'] = gdf_all['geometry'].apply(safe_fix)
    gdf_all = gdf_all[gdf_all.geometry.notna()]
    
    print("Clipping polygons to Study Circle...")
    # Using spatial index for speed
    sindex = gdf_all.sindex
    possible_matches_index = list(sindex.intersection(bbox_poly.bounds))
    possible_matches = gdf_all.iloc[possible_matches_index]
    gdf_circle_preds = possible_matches[possible_matches.intersects(bbox_poly)].copy()
    
    # Clip strictly
    gdf_circle_preds = gpd.clip(gdf_circle_preds, bbox_poly)
    print(f"Found {len(gdf_circle_preds)} polygons inside the circle.")
    
    model_name = "EfficientNet" if "efficientnet" in model_path else "ResNet"
    print(f"\n--- Phase 2: Offline Classification ({model_name}) ---")
    # 4. Classify these polygons using the Satellite Image
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Object
    loaded_obj = torch.load(model_path, map_location=device) if os.path.exists(model_path) else None
    
    import torch.nn as nn
    from torchvision import models
    
    if isinstance(loaded_obj, nn.Module):
        model = loaded_obj
    elif isinstance(loaded_obj, dict):
        # Determine number of classes dynamically
        num_classes = list(loaded_obj.values())[-1].shape[0]
        print(f"Detected {num_classes} classes from model checkpoint.")
        
        # Robust Architecture Detection
        sample_key = list(loaded_obj.keys())[0]
        is_improved = any("backbone" in k for k in loaded_obj.keys())
        is_efficientnet = "efficientnet" in model_path.lower() or is_improved
        
        if is_improved:
            print("Detected Improved Architecture (Multi-Sample Dropout).")
            feature_extractor = EfficientNetV2_Improved(num_classes=num_classes)
            feature_extractor.load_state_dict(loaded_obj)
        elif is_efficientnet:
            print("Detected Standard EfficientNet Architecture.")
            feature_extractor = models.efficientnet_v2_s(pretrained=False)
            num_ftrs = feature_extractor.classifier[1].in_features
            feature_extractor.classifier[1] = nn.Linear(num_ftrs, num_classes)
            feature_extractor.load_state_dict(loaded_obj)
        elif "resnet50" in model_path.lower():
            print("Detected ResNet50 Architecture.")
            feature_extractor = models.resnet50(pretrained=False)
            num_ftrs = feature_extractor.fc.in_features
            feature_extractor.fc = nn.Linear(num_ftrs, num_classes)
            feature_extractor.load_state_dict(loaded_obj)
        else:
            print("Detected ResNet18 Architecture (Fallback).")
            feature_extractor = models.resnet18(pretrained=False)
            num_ftrs = feature_extractor.fc.in_features
            feature_extractor.fc = nn.Linear(num_ftrs, num_classes)
            feature_extractor.load_state_dict(loaded_obj)
            
        model = feature_extractor
    else:
        raise ValueError("Could not load model state dict or object.")
        
    model.eval()
    model.to(device)
    
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Open Satellite Image
    classes = ['Residential', 'Commercial', 'Industrial', 'Vegetation'] # Default/Placeholder
    
    predictions = []
    
    predictions = []
    
    with rasterio.open(geotiff_path) as src:
        if gdf_circle_preds.crs != src.crs:
            print(f"Reprojecting Polygons from {gdf_circle_preds.crs} to {src.crs}...")
            gdf_circle_preds = gdf_circle_preds.to_crs(src.crs)
            
        # FORCE POLYGON TYPE for Visualization
        # The user wants "Filled Shapes", but input might be LineStrings (Roads/Blocks boundaries)
        # If LineString, we can Try to Polygonize or Buffer?
        # Buffering a LineString creates a Polygon.
        
        # Check type
        geom_types = gdf_circle_preds.geom_type.unique()
        # --- Polygonize Logic (Fix for colored roads) ---
        from shapely.ops import polygonize
        
        # If we have LineStrings, we must polygonize them to get blocks
        # instead of buffering them (which makes fat roads).
        # If we already have Polygons, we keep them.
        
        geom_types = gdf_circle_preds['geometry'].geom_type.unique()
        print(f"Geometry Types: {geom_types}")
        
        # Check if we need to polygonize (mostly LineStrings)
        has_lines = any('LineString' in t for t in geom_types)
        has_polys = any('Polygon' in t for t in geom_types)
        
        if has_lines and not has_polys:
            print("Converting LineStrings to Polygons (Polygonize + SJoin)...")
            lines = list(gdf_circle_preds.geometry)
            polys = list(polygonize(lines))
            
            if len(polys) == 0:
                print("WARNING: Polygonize produced 0 polygons! Fallback to buffer?")
                # Fallback to buffer if polygonize fails (e.g. valid single lines that don't form loops)
                gdf_circle_preds = gdf_circle_preds.copy()
                gdf_circle_preds['geometry'] = gdf_circle_preds.geometry.buffer(5) 
            else:
                # Create GeoDataFrame from new polygons
                gdf_blocks = gpd.GeoDataFrame(geometry=polys, crs=gdf_circle_preds.crs)
                
                # Spatial Join to recover attributes (Land_Use from the lines that form the block)
                # 'intersects' is robust for lines forming the boundary
                gdf_circle_preds = gpd.sjoin(gdf_blocks, gdf_circle_preds, how='left', predicate='intersects')
                
                # Drop duplicates (a block touches multiple lines, so it gets multiple rows)
                # We keep the first match (assuming homogenous block boundary classes)
                gdf_circle_preds = gdf_circle_preds[~gdf_circle_preds.index.duplicated(keep='first')]
                
                print(f"Polygonized {len(lines)} lines into {len(gdf_circle_preds)} blocks.")

        elif has_lines and has_polys:
             # Mixed? Likely buffering lines is safer or just keeping polys
             print("Mixed geometry. Keeping as is (buffering lines if any).")
             gdf_circle_preds = gdf_circle_preds.copy()
             gdf_circle_preds['geometry'] = gdf_circle_preds.geometry.apply(lambda g: g.buffer(5) if 'Line' in g.geom_type else g)
        else:
            # Already Polygons
            print("Already Polygons. Skipping conversion.")
            gdf_circle_preds = gdf_circle_preds.copy()

        for idx, row in gdf_circle_preds.iterrows():
            try:
                geom = row.geometry
                # Sync Inference with Training: Add 5m context padding (assumes epsg 3857/meters)
                geom = geom.buffer(5.0)
                out_image, out_transform = mask(src, [geom], crop=True)
                
                # Convert to PIL
                img_array = out_image.transpose(1, 2, 0)
                if img_array.shape[2] > 3: img_array = img_array[:,:,:3]
                
                # Skip tiny/empty chips
                if img_array.shape[0] < 5 or img_array.shape[1] < 5:
                    predictions.append("Unclassified")
                    continue
                    
                pil_img = Image.fromarray(img_array)
                
                # DEBUG: Save the chip to see what the model sees
                debug_dir = os.path.join(output_dir, "debug_chips")
                os.makedirs(debug_dir, exist_ok=True)
                if idx < 10: # Save first 10 for inspection
                    pil_img.save(os.path.join(debug_dir, f"poly_{idx}.png"))

                input_tensor = transform(pil_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    val, preds = torch.max(outputs, 1)
                    
                    # Debug Info
                    top_prob, top_cls = probabilities.max(1)
                    if idx < 10:
                        print(f"Poly {idx}: Pred={preds.item()} (Prob={top_prob.item():.2f}) | Raw: {outputs.tolist()}")
                    
                    predictions.append(preds.item()) # Store Raw Integer
                    
            except Exception as e:
                print(f"Prediction Error: {e}")
                predictions.append(-1)
                
    gdf_circle_preds['pred_class_code'] = predictions
    
    # Save Prediction Result
    pred_path = os.path.join(output_dir, "predicted_classified.geojson")
    gdf_circle_preds.to_file(pred_path, driver="GeoJSON")
    # --- Definitions ---
    # Reference: Green (Open), Purple (Residential), Red (Commercial)
    cmap_dict = {
        0: '#FF1493',   # Unknown/Void -> Deep Pink (Very Solid)
        1: '#800080',   # Residential -> Deep Purple (Hex)
        2: '#FF0000',   # Commercial -> Bright Red
        3: '#00FF00',   # Open Space -> Lime Green
        4: '#0000FF',   # Transport -> Blue
        5: '#FFFF00',   # Public -> Yellow
        6: '#00FF00',   # Open Space -> Lime Green
        -1: 'black'     # Error
    }
    
    # Official Atlas of Urban Expansion Classification (0-based in GIS data):
    # 0 -> Open space (Atlas Class 1) -> Lime Green
    # 1 -> Non-residential areas (Atlas Class 2) -> Gray
    # 2 -> Atomistic settlements (Atlas Class 3) -> Purple
    # 3 -> Informal land subdivisions (Atlas Class 4) -> Red
    # 4 -> Formal land subdivisions (Atlas Class 5) -> Blue
    # 5 -> Housing projects (Atlas Class 6) -> Orange
    
    def get_color(class_code):
        if pd.isna(class_code): return 'gray'
        try:
            code = int(float(class_code))
            if code == 0: return 'limegreen'   # Open Space
            if code == 1: return 'gray'        # Non-residential
            if code == 2: return 'purple'      # Atomistic
            if code == 3: return 'red'         # Informal
            if code == 4: return 'blue'        # Formal
            if code == 5: return 'orange'      # Housing projects
            return 'deeppink' # Unknown
        except:
            return 'gray'

    # Handle different column names
    color_col = None
    if 'Land_Use' in gdf_truth.columns:
        color_col = 'Land_Use'
    elif 'Land_use' in gdf_truth.columns:
        color_col = 'Land_use'
    elif 'Land_use_Code' in gdf_truth.columns:
        color_col = 'Land_use_Code'
        
    if color_col:
        print(f"Using color column: {color_col}")
        gdf_truth['color_hex'] = gdf_truth[color_col].apply(get_color)

    # Calculate Accuracy
    correct = 0
    total = 0
    
    # Use the CLIPPED dataframe for truth, ensuring 1:1 match with predictions
    if color_col and color_col in gdf_circle_preds.columns:
        ground_truth_codes = gdf_circle_preds[color_col].astype(int).tolist()
    else:
        ground_truth_codes = []
    
    if len(ground_truth_codes) == len(predictions):
        print("Comparison (GT vs Pred):")
        for gt, pred in zip(ground_truth_codes, predictions):
             print(f"  GT: {gt} | Pred: {pred}")
             if gt == pred:
                 correct += 1
             total += 1
             
    acc = (correct / total * 100) if total > 0 else 0.0
    print(f"Accuracy: {acc:.2f}% ({correct}/{total})")

    # --- Combined Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(24, 12)) # Larger
    ax1, ax2 = axes

    with rasterio.open(geotiff_path) as src:
        img_crs = src.crs
        
        # Reproject Truth to Match Image
        if gdf_truth.crs != img_crs:
             print(f"Viz: Reprojecting Truth from {gdf_truth.crs} to {img_crs}")
             gdf_truth = gdf_truth.to_crs(img_crs)
        
        # Buffer Truth if LineStrings
        # Convert LineString to Polygon if closed (Better than buffering)
        from shapely.geometry import Polygon
        def line_to_poly(geom):
            if geom.geom_type == 'LineString' and geom.is_closed:
                return Polygon(geom.coords)
            elif geom.geom_type == 'LineString' and not geom.is_closed:
                 # Fallback for open lines
                 return geom.buffer(2)
            return geom

        # Apply to Truth (Buffer 0 to fix validity)
        gt_types = gdf_truth.geom_type.unique()
        print(f"Viz: Truth Geoms before: {gt_types}")
        gdf_truth['geometry'] = gdf_truth.geometry.apply(line_to_poly).buffer(0)
        
        # Apply to Preds (Buffer 0 to fix validity)
        pred_types = gdf_circle_preds.geom_type.unique()
        print(f"Viz: Pred Geoms before: {pred_types}")
        gdf_circle_preds['geometry'] = gdf_circle_preds.geometry.apply(line_to_poly).buffer(0)


        # Reproject Preds to Match Image (Ensure it's done)
        if gdf_circle_preds.crs != img_crs:
             print(f"Viz: Reprojecting Preds from {gdf_circle_preds.crs} to {img_crs}")
             gdf_circle_preds = gdf_circle_preds.to_crs(img_crs)

        # Shared Extent setup (Use Bounds of the SHAPES now that they match)
        minx, miny, maxx, maxy = gdf_truth.total_bounds
        span_x = maxx - minx
        span_y = maxy - miny
        xlim = (minx - span_x * 0.1, maxx + span_x * 0.1) # Zoom out slightly
        ylim = (miny - span_y * 0.1, maxy + span_y * 0.1)
    
        # Background Image (Shared)
        # Plot on both
        rasterio.plot.show(src, ax=ax1)
        rasterio.plot.show(src, ax=ax2)

    # --- Map 1: Ground Truth ---
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon as MplPolygon
    from shapely.geometry import MultiPolygon, Polygon as ShapelyPolygon
    import numpy as np

    def plot_gdf_patches(ax, gdf, color_col, exclude_col=None, exclude_vals=[]):
        patches = []
        colors = []
        
        # Debug: Check columns
        if exclude_col and exclude_col not in gdf.columns:
            print(f"WARNING: Exclude column '{exclude_col}' not found in GDF columns: {gdf.columns}")
        
        for idx, row in gdf.iterrows():
            # Exclusion Logic
            if exclude_col and exclude_col in row:
                val = row[exclude_col]
                try:
                    # Robust Int Comparison
                    if int(float(val)) in exclude_vals:
                        continue
                except:
                    pass
            
            geom = row.geometry
            c = row[color_col] if color_col in row else 'red'
            
            if geom.is_empty: continue
            
            if geom.geom_type == 'Polygon':
                patches.append(MplPolygon(np.array(geom.exterior.coords), closed=True))
                colors.append(c)
            elif geom.geom_type == 'MultiPolygon':
                for sub_poly in geom.geoms:
                    patches.append(MplPolygon(np.array(sub_poly.exterior.coords), closed=True))
                    colors.append(c)
                    
        p = PatchCollection(patches, facecolors=colors, edgecolors='none', alpha=0.7)
        ax.add_collection(p)
        return p

    # Determine Class Column (Land_Use)
    cls_col = None
    if 'Land_Use' in gdf_truth.columns: cls_col = 'Land_Use'
    elif 'Land_use' in gdf_truth.columns: cls_col = 'Land_use'
    
    if cls_col:
        print(f"DEBUG: Found Class Column '{cls_col}'. Unique values:")
        print(gdf_truth[cls_col].value_counts())

    
    # Remove the unsafe blind copy. gdf_circle_preds should inherit it.
    print(f"Pred Columns: {gdf_circle_preds.columns}")
    if cls_col and cls_col not in gdf_circle_preds.columns:
         print("WARNING: Class column missing in Preds despite clip. This suggests an issue.")

    if cls_col:
        print(f"Using Exclusion Column: {cls_col}")

    if 'color_hex' in gdf_truth.columns:
         print(f"Plotting Ground Truth...")
         plot_gdf_patches(ax1, gdf_truth, 'color_hex', exclude_col=cls_col, exclude_vals=[])
         ax1.set_title("Ground Truth (Urban Blocks)", fontsize=20, color='white', weight='bold', y=0.95)
    else:
        # No labels available (e.g. random OSM locales) - draw block outlines as reference
        print("No ground truth labels found. Drawing block outlines as spatial reference.")
        if gdf_circle_preds.crs != gdf_truth.crs:
            gdf_circle_preds_plot = gdf_circle_preds.to_crs(gdf_truth.crs)
        else:
            gdf_circle_preds_plot = gdf_circle_preds
        try:
            gdf_circle_preds_plot.boundary.plot(ax=ax1, color='yellow', linewidth=1.0, alpha=0.8)
        except Exception:
            pass
        ax1.set_title("Block Boundaries (No GT Labels)", fontsize=20, color='white', weight='bold', y=0.95)

    # --- Map 2: Prediction ---
    if 'pred_class_code' in gdf_circle_preds.columns:
        # Update colors for preds
        gdf_circle_preds['pred_color'] = gdf_circle_preds['pred_class_code'].apply(get_color)
        
        print(f"Plotting Predictions...")
        preds_view = gdf_circle_preds.copy()
        
        plot_gdf_patches(ax2, preds_view, 'pred_color', exclude_col=cls_col, exclude_vals=[])

    # Titles
    ax1.set_title("Ground Truth (Urban Blocks)", fontsize=20, color='white', weight='bold', y=0.95)
    ax2.set_title("Prediction (Urban Blocks)", fontsize=20, color='white', weight='bold', y=0.95)

    # Styling
    for ax in [ax1, ax2]:
        ax.set_axis_off()
        # Force set limits to the FULL Circle BBox + Margin
        # This addresses "not giving the entire circle"
        minx, miny, maxx, maxy = gdf_truth.total_bounds
        margin = 100 # meters
        ax.set_xlim(minx - margin, maxx + margin)
        ax.set_ylim(miny - margin, maxy + margin)
        
    combined_path = os.path.join(output_dir, "comparison_result_combined.png")
    plt.tight_layout()
    plt.savefig(combined_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved Combined Comparison: {combined_path}")

    print(f"Saved Prediction Map: {pred_path}")

def run_offline_analysis(city_name, sample_id, shp_path, model_path="models/landuse_efficientnet_v2_s.pth"):
    print(f"Setting up analysis for {city_name} - {sample_id}")
    
    # 1. Define Paths
    output_dir = f"data/{city_name.lower()}/offline_comparison/{sample_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Image Path (Try fetched first, else fetch)
    fetched_dir = "data/raw/fetched_images_robust"
    image_path = os.path.join(fetched_dir, f"{sample_id}.tif")
    
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}. Fetching dynamic...")
        import contextily as ctx
        gdf = gpd.read_file(shp_path)
        if gdf.crs is None: gdf.set_crs(epsg=4326, inplace=True)
        w, s, e, n = gdf.to_crs(epsg=3857).total_bounds # Metric bounds for contextily 
        image_path = os.path.join(output_dir, f"{sample_id}_dynamic.tif")
        try:
             # Add buffer
            b = 100 # meters
            ctx.bounds2raster(w-b, s-b, e+b, n+b, image_path, zoom=18, source=ctx.providers.Esri.WorldImagery)
            print("Fetched dynamic image.")
        except Exception as err:
            print(f"Contextily failed: {err}")
            return
            
    print(f"\n========================================")
    print(f"Running Offline Analysis for {sample_id}")
    print(f"========================================")
    
    # Check if shapefile exists
    if not os.path.exists(shp_path):
        print(f"Error: Shapefile not found at {shp_path}")
        return
        
    offline_analysis(
        existing_blocks_path=shp_path,
        study_circle_truth_path=shp_path, 
        geotiff_path=image_path,
        png_path=None,
        model_path=model_path,
        output_dir=output_dir
    )

if __name__ == "__main__":
    pass
