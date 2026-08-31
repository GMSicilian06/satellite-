import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import argparse
import os

def classify_grid(image_path, model_path, output_path, device_name='mps', grid_size=224):
    """
    Slices a JPG into grid_size x grid_size chunks and classifies them.
    Draws the class label on the image.
    """
    device = torch.device(device_name if torch.backends.mps.is_available() and device_name=='mps' else "cpu")
    print(f"Using device: {device}")
    
    # Load Model
    num_classes = 6
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model = model.to(device)
    model.eval()
    
    # Transforms
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load Image
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("Arial", 20)
    except:
        font = ImageFont.load_default()
        
    # Class names mapping (Inferred from training indices 0-5)
    # Ideally this should be passed in, but we'll use generic placeholders if unknown or verify from dataset
    # From previous context, user defined definitions. 
    # Let's assume the folder structure 0,1,2,3,4,5 maps to the sorted list of Land Use codes found in data.
    # We saw codes 0,1,2,3,4,5. 
    # Let's just use the Code ID for now.
    
    print(f"Processing image {width}x{height} with grid {grid_size}...")
    
    for y in range(0, height, grid_size):
        for x in range(0, width, grid_size):
            box = (x, y, x+grid_size, y+grid_size)
            crop = img.crop(box)
            
            # Predict
            input_tensor = preprocess(crop)
            input_batch = input_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_batch)
                _, pred = torch.max(output, 1)
                label_id = pred.item()
                
            # Draw
            # Color code based on class
            colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'cyan']
            color = colors[label_id % len(colors)]
            
            draw.rectangle(box, outline=color, width=3)
            draw.text((x+5, y+5), str(label_id), fill=color, font=font)
            
    img.save(output_path)
    print(f"Saved demo to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="models/landuse_resnet18.pth")
    parser.add_argument("--output", default="demo_prediction.jpg")
    args = parser.parse_args()
    
    classify_grid(args.image, args.model, args.output)
