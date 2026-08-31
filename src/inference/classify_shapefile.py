import geopandas as gpd
import rasterio
from rasterio.mask import mask
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import os
import argparse

# Colors for 0-5
CLASS_COLORS_RGB = {
    0: (255, 0, 0),    # Red
    1: (0, 255, 0),    # Green
    2: (0, 0, 255),    # Blue
    3: (255, 255, 0),  # Yellow
    4: (255, 0, 255),  # Magenta
    5: (0, 255, 255),  # Cyan
    6: (128, 128, 128) # None/Other
}

def classify_shapefile(shp_path, image_path, model_path, output_image):
    print(f"Classifying {shp_path} using {image_path}...")
    
    # 1. Load Data
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        print("Shapefile is empty.")
        return

    # 2. Load Model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
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
    
    # 3. Process
    with rasterio.open(image_path) as src:
        # Reproject blocks to match image CRS
        if src.crs is not None and gdf.crs is not None and gdf.crs != src.crs:
            print(f"Reprojecting shapefile from {gdf.crs} to {src.crs}...")
            gdf = gdf.to_crs(src.crs)
        elif src.crs is None:
             print("Image has no CRS. Assuming geometry is in Pixel Coordinates.")
            
        # Draw canvas
        img_arr = src.read() # C, H, W
        # Ensure 3 channels
        if img_arr.shape[0] > 3:
            img_arr = img_arr[:3, :, :]
            
        img_arr = np.moveaxis(img_arr, 0, -1) # H, W, C
        img_vis = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR) # OpenCV uses BGR
        
        vis_pred = img_vis.copy()
        
        for idx, row in gdf.iterrows():
            geom = row.geometry
            
            # Crop
            try:
                out_image, out_transform = mask(src, [geom], crop=True)
                if out_image.shape[0] == 0: continue
                
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
                    pred_label = int(class_names[preds[0]]) if preds[0] < 6 else 6
                    
                # Visualize
                # Get pixel coords for drawing
                if geom.geom_type == 'Polygon':
                    ext = list(geom.exterior.coords)
                    pix = [src.index(x, y) for x, y in ext]
                    pix = [(c, r) for r, c in pix] # index -> r,c -> y,x. cv2 -> x,y
                    pts = [np.array(pix, np.int32)]
                    
                    p_color = CLASS_COLORS_RGB.get(pred_label, (255, 255, 255))
                    bgr_p = (p_color[2], p_color[1], p_color[0])
                    
                    cv2.fillPoly(vis_pred, pts, bgr_p)
                    cv2.polylines(vis_pred, pts, True, (255, 255, 255), 2)
                    
            except Exception as e:
                # print(f"Error block {idx}: {e}")
                pass
                
        # Blend Prediction
        alpha = 0.6
        cv2.addWeighted(vis_pred, alpha, img_vis, 1-alpha, 0, vis_pred)
        
        cv2.imwrite(output_image, vis_pred)
        print(f"Saved classification to {output_image}")
        
    # 4. Save Geometry Map (Step 2 Visualization)
    # Re-draw clean image
    vis_geom = img_vis.copy()
    
    # Draw Blocks (Yellow)
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == 'Polygon':
            coords = list(geom.exterior.coords)
            points = [src.index(x, y) for x, y in coords]
            points = np.array([(c, r) for r, c in points], np.int32)
            cv2.polylines(vis_geom, [points], True, (0, 255, 255), 2)
            
    # Draw Roads (Cyan) if provided?
    # We don't have roads passed here, but we can if we want.
    # For now just showing Blocks is good for "Blocks identified".
    geom_out = output_image.replace(".jpg", "_geometry.jpg")
    cv2.imwrite(geom_out, vis_geom)
    print(f"Saved geometry visualization to {geom_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shp", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="models/landuse_resnet18.pth")
    parser.add_argument("--output", default="classified_shapefile.jpg")
    args = parser.parse_args()
    
    classify_shapefile(args.shp, args.image, args.model, args.output)
