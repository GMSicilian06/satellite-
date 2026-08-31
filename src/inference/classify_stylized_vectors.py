import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import geopandas as gpd
from shapely.geometry import shape
import numpy as np
import cv2
import argparse
import os

# Class colors for visualization
CLASS_COLORS = {
    '0': (255, 0, 0),    # Red
    '1': (0, 255, 0),    # Green
    '2': (0, 0, 255),    # Blue
    '3': (255, 255, 0),  # Yellow
    '4': (255, 0, 255),  # Magenta
    '5': (0, 255, 255),  # Cyan
}

def classify_screenshot(image_path, blocks_geojson, model_path, output_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model
    # Determine classes from directory structure or hardcode
    # Training found 'None' directory, so 7 classes
    class_names = ['0', '1', '2', '3', '4', '5', 'None']
    
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Load Image & Vectors
    img_cv = cv2.imread(image_path) # BGR
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    # Check if geojson exists
    if not os.path.exists(blocks_geojson):
        print(f"Error: {blocks_geojson} not found")
        return

    gdf = gpd.read_file(blocks_geojson)
    
    # Prepare Output Visualization
    # We will draw filled polygons with transparency
    vis_img = img_cv.copy()
    overlay = img_cv.copy()
    
    print(f"Classifying {len(gdf)} blocks...")
    
    results = []
    
    for idx, row in gdf.iterrows():
        poly = row.geometry
        
        # Crop Logic (Pixel Coords)
        # 1. Get bbox of polygon
        minx, miny, maxx, maxy = poly.bounds
        
        # 2. Crop from image
        # Ensure integer bounds within image dimensions
        minx, miny = max(0, int(minx)), max(0, int(miny))
        maxx, maxy = min(img_cv.shape[1], int(maxx)), min(img_cv.shape[0], int(maxy))
        
        crop = img_rgb[miny:maxy, minx:maxx]
        
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            continue
            
        # 3. Predict
        pil_crop = Image.fromarray(crop)
        input_tensor = transform(pil_crop).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            _, preds = torch.max(outputs, 1)
            pred_class = class_names[preds[0]]
            
        results.append(pred_class)
        
        # 4. Draw
        # Draw on overlay
        color = CLASS_COLORS.get(pred_class, (255, 255, 255))
        
        # OpenCv draws polygons as list of points
        # Exterior
        if poly.geom_type == 'Polygon':
            pts = np.array(list(poly.exterior.coords), np.int32)
            # Y-flip check? 
            # The extract_vectors.py ALREADY flipped Y to match GIS logic (-y).
            # But here we are overlaying on the ORIGINAL IMAGE (Top-Left 0,0).
            # If extract_vectors produced negative Y, we must flip back.
            # Let's check the geojson bounds. If Y is negative, we flip.
            if miny < 0 or maxy < 0:
                 # It was GIS coords (negative Y)
                 # We need to flip back to image coords? 
                 # Actually extract_vectors said:
                 # "We map (x, y) -> (x, -y) to keep orientation correct" (Wait, usually images have +y down)
                 # If extract_vectors output (x, -y), then y was POSITIVE in image, became NEGATIVE in geojson.
                 # So we need to negate Y to get back to image +Y.
                 pts[:, 1] = -pts[:, 1]
            else:
                # If everything is positive, maybe it wasn't flipped or image coords were used directly.
                # Standard GIS usually implies bottom-left origin, images top-left.
                pass

            pts = pts.reshape((-1, 1, 2))
            
            # Strong Fill
            cv2.fillPoly(overlay, [pts], color)
            # White Outline
            cv2.polylines(overlay, [pts], True, (255,255,255), 2)
            
            # Label
            # Disable label text to avoid clutter on large map? 
            # Or make it small. Keeping it small.
            M = cv2.moments(pts)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # cv2.putText(vis_img, pred_class, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Blend overlay - Stronger Alpha
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0, vis_img)
    
    cv2.imwrite(output_path, vis_img)
    print(f"Saved classification result to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--blocks", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="classified_map.jpg")
    args = parser.parse_args()
    
    classify_screenshot(args.image, args.blocks, args.model, args.output)
