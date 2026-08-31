import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import os
import argparse
from tqdm import tqdm

def classify_city(image_path, blocks_shp_path, model_path, output_shp_path, device_name='mps'):
    """
    Classifies every block in the provided shapefile using the satellite image.
    """
    
    # 1. Setup Model
    device = torch.device(device_name if torch.backends.mps.is_available() and device_name=='mps' else "cpu")
    print(f"Using device: {device}")
    
    # Re-instantiate model architecture
    # Note: We hardcode 6 classes as per our training
    num_classes = 6 
    
    if "resnet50" in model_path:
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif "efficientnet" in model_path:
        model = models.efficientnet_v2_s(pretrained=False)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    # Load weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model = model.to(device)
    model.eval()
    
    # Transforms (same as validation)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 2. Load Data
    print(f"Loading blocks from {blocks_shp_path}...")
    blocks_df = gpd.read_file(blocks_shp_path)
    
    # Prepare output column
    predicted_labels = []
    
    print(f"Processing {len(blocks_df)} blocks...")
    
    # 3. Iterate and Classify
    with rasterio.open(image_path) as src:
        # Ensure CRS Match
        if blocks_df.crs != src.crs:
            print("Reprojecting blocks to match image...")
            blocks_df = blocks_df.to_crs(src.crs)
            
        for idx, row in tqdm(blocks_df.iterrows(), total=len(blocks_df)):
            geom = row.geometry
            
            try:
                # Crop
                # Pad slightly? No, context is good but we trained on tight crops.
                out_image, out_transform = mask(src, [geom], crop=True)
                
                # Check validity
                if out_image.shape[0] == 0 or out_image.shape[1] < 5 or out_image.shape[2] < 5:
                    predicted_labels.append(-1) # Error class
                    continue
                
                # Convert to PIL for Transforms
                # out_image is C,H,W (Rasterio) -> needs C,H,W (Torch) but PIL expects H,W,C
                img_np = np.moveaxis(out_image, 0, -1)
                
                # Handle alpha channel if exists (4th channel)
                if img_np.shape[2] == 4:
                    img_np = img_np[:, :, :3]
                    
                pil_img = Image.fromarray(img_np)
                
                # Transform
                input_tensor = preprocess(pil_img)
                input_batch = input_tensor.unsqueeze(0).to(device) # Create mini-batch
                
                # Inference
                with torch.no_grad():
                    output = model(input_batch)
                    _, pred = torch.max(output, 1)
                    predicted_labels.append(pred.item())
                    
            except Exception as e:
                # print(f"Error on block {idx}: {e}")
                predicted_labels.append(-1)

    # 4. Save Prediction
    blocks_df['Pred_Class'] = predicted_labels
    
    # Map class IDs to Names (Optional, based on your definitions)
    # 0: Open Space, 1: Non-Res, 2: Atomistic, 3: Informal, 4: Formal, 5: Housing Project (Example mapping)
    # Ideally we should persist this mapping from training.
    
    blocks_df.to_file(output_shp_path)
    print(f"Saved predictions to {output_shp_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to Satellite TIF")
    parser.add_argument("--blocks", required=True, help="Path to Block Polygons SHP")
    parser.add_argument("--model", default="models/landuse_resnet18.pth", help="Path to trained model")
    parser.add_argument("--output", default="predicted_landuse.shp", help="Path for output Shapefile")
    args = parser.parse_args()
    
    classify_city(args.image, args.blocks, args.model, args.output)
