import geopandas as gpd
import contextily as ctx
from shapely.ops import polygonize
import rasterio
from rasterio.mask import mask
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import os
import argparse
import matplotlib.pyplot as plt

# Colors for 0-5
CLASS_COLORS_RGB = {
    0: (255, 0, 0),    # Red
    1: (0, 255, 0),    # Green
    2: (0, 0, 255),    # Blue
    3: (255, 255, 0),  # Yellow
    4: (255, 0, 255),  # Magenta
    5: (0, 255, 255),  # Cyan
}

def test_locale(shp_path, model_path, output_image="test_locale_result.jpg"):
    print(f"Testing on locale: {shp_path}")
    
    # 1. Fetch Image
    basename = os.path.basename(shp_path).replace(".shp", "")
    img_path = f"temp_{basename}.tif"
    
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    
    # Standardize column
    if 'Land_Use' in gdf.columns:
        gdf['Land_use'] = gdf['Land_Use']
    elif 'landuse' in gdf.columns:
        gdf['Land_use'] = gdf['landuse']
        
    if not os.path.exists(img_path):
        print("Fetching satellite image...")
        w, s, e, n = gdf.total_bounds
        b = 0.001
        try:
            ctx.bounds2raster(w-b, s-b, e+b, n+b, img_path, zoom='auto', source=ctx.providers.Esri.WorldImagery, ll=True)
        except Exception as e:
            print(f"Fetch failed: {e}")
            return

    # 2. Polygonize (if lines)
    if 'Polygon' in gdf.geometry.type.iloc[0]:
        blocks = gdf
    else:
        print("Polygonizing lines...")
        lines = list(gdf.geometry)
        polys = list(polygonize(lines))
        poly_df = gpd.GeoDataFrame(geometry=polys, crs=gdf.crs)
        blocks = gpd.sjoin(poly_df, gdf, how="inner", predicate="intersects")
        blocks = blocks[~blocks.index.duplicated(keep='first')]
    
    print(f"Found {len(blocks)} blocks.")
    
    # 3. Load Model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # Training found 'None' directory, so 7 classes
    class_names = ['0', '1', '2', '3', '4', '5', 'None'] 
    
    if "resnet50" in model_path:
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    elif "efficientnet" in model_path:
        model = models.efficientnet_v2_s(pretrained=False)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_names))
    else:
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 4. Predict & Visualize
    # Load Main Image for Viz
    # We need a plain CV2 image for drawing
    # Rasterio can give us the array
    with rasterio.open(img_path) as src:
        # Reproject blocks to match image
        if blocks.crs != src.crs:
            blocks = blocks.to_crs(src.crs)
            
        # Create 2 canvasses: Truth, Prediction
        # Read full image
        img_arr = src.read() # C, H, W
        img_arr = np.moveaxis(img_arr, 0, -1) # H, W, C
        img_vis = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        
        vis_truth = img_vis.copy()
        vis_pred = img_vis.copy()
        
        correct_count = 0
        total_count = 0
        
        for idx, row in blocks.iterrows():
            geom = row.geometry
            true_label = int(row['Land_use'])
            
            # Crop
            try:
                out_image, out_transform = mask(src, [geom], crop=True)
                if out_image.shape[0] == 0: continue
                
                # Ensure 3 channels (RGB)
                if out_image.shape[0] > 3:
                     out_image = out_image[:3, :, :]
                
                crop_cv = np.moveaxis(out_image, 0, -1) # H W C
                if crop_cv.shape[0] < 5: continue
                
                # Predict
                pil_img = Image.fromarray(crop_cv)
                t_img = transform(pil_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    out = model(t_img)
                    _, preds = torch.max(out, 1)
                    pred_label = int(class_names[preds[0]]) # assuming predictions map to int 0-5
                    
                # Stats
                if pred_label == true_label:
                    correct_count += 1
                total_count += 1
                
                def get_pix_coords(geometry, transform):
                    if geometry.geom_type == 'Polygon':
                        ext = list(geometry.exterior.coords)
                        pix = [src.index(x, y) for x, y in ext]
                        pix = [(c, r) for r, c in pix] 
                        return [np.array(pix, np.int32)]
                    return []

                pts = get_pix_coords(geom, src.transform)
                if pts:
                    # Define t_color
                    t_color = CLASS_COLORS_RGB.get(true_label, (255, 255, 255))
                    
                    # BGR
                    bgr_t = (t_color[2], t_color[1], t_color[0])
                    cv2.fillPoly(vis_truth, pts, bgr_t)

                    # Draw Pred
                    p_color = CLASS_COLORS_RGB.get(pred_label, (255, 255, 255))
                    bgr_p = (p_color[2], p_color[1], p_color[0])
                    
                    # Strong Fill
                    cv2.fillPoly(vis_pred, pts, bgr_p)
                    # White Outline
                    cv2.polylines(vis_pred, pts, True, (255, 255, 255), 2)
                    
                    print(f"Drew Block {idx}: Pred={pred_label}, Color={bgr_p}")

            except Exception as e:
                print(f"Block error: {e}")
                pass
                
        # Blend - Stronger Alpha
        alpha = 0.6
        cv2.addWeighted(vis_truth, alpha, img_vis, 1-alpha, 0, vis_truth)
        cv2.addWeighted(vis_pred, alpha, img_vis, 1-alpha, 0, vis_pred)
        
        # Combine Side by Side
        # Resize to reasonable height
        h = 500
        r = h / vis_truth.shape[0]
        dim = (int(vis_truth.shape[1] * r), h)
        
        v_t_small = cv2.resize(vis_truth, dim)
        v_p_small = cv2.resize(vis_pred, dim)
        
        # Add labels
        acc = correct_count/total_count if total_count > 0 else 0
        cv2.putText(v_t_small, f"Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(v_p_small, f"Prediction (Acc: {acc:.2%})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        combined = np.hstack((v_t_small, v_p_small))
        cv2.imwrite(output_image, combined)
        
        # ALSO save separate images as requested
        base = output_image.replace(".jpg", "")
        cv2.imwrite(f"{base}_raw.jpg", img_vis)
        cv2.imwrite(f"{base}_pred.jpg", vis_pred)
        
        # Save Geometry (Blocks Only) image
        vis_geom = img_vis.copy()
        # Draw just outlines on this one
        for idx, row in blocks.iterrows():
             if row.geometry.geom_type == 'Polygon':
                ext = list(row.geometry.exterior.coords)
                pix = [src.index(x, y) for x, y in ext]
                pix = [(c, r) for r, c in pix] 
                pts = [np.array(pix, np.int32)]
                
                # Yellow Outlines for visibility
                cv2.polylines(vis_geom, pts, True, (0, 255, 255), 2)
        
        cv2.imwrite(f"{base}_geom.jpg", vis_geom)
        
        # Save Ground Truth Image
        vis_truth_map = img_vis.copy()
        for idx, row in blocks.iterrows():
            if row.geometry.geom_type == 'Polygon':
                ext = list(row.geometry.exterior.coords)
                pix = [src.index(x, y) for x, y in ext]
                pix = [(c, r) for r, c in pix] 
                pts = [np.array(pix, np.int32)]
                
                # Truth Color
                true_label = int(row['Land_use'])
                t_color = CLASS_COLORS_RGB.get(true_label, (255, 255, 255))
                bgr_t = (t_color[2], t_color[1], t_color[0])
                
                cv2.fillPoly(vis_truth_map, pts, bgr_t)
                cv2.polylines(vis_truth_map, pts, True, (255, 255, 255), 2)
                
        # Blend
        cv2.addWeighted(vis_truth_map, 0.6, img_vis, 0.4, 0, vis_truth_map)
        cv2.imwrite(f"{base}_truth.jpg", vis_truth_map)
        
        print(f"Saved result to {output_image} + _raw.jpg + _geom.jpg + _truth.jpg + _pred.jpg. Accuracy: {correct_count}/{total_count} ({correct_count/total_count:.2%})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shp", required=True)
    parser.add_argument("--model", default="models/landuse_resnet18.pth")
    parser.add_argument("--output", default="test_locale_result.jpg")
    args = parser.parse_args()
    
    test_locale(args.shp, args.model, args.output)
