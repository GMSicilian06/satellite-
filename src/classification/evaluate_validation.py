import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import argparse

def evaluate_model(data_dir, model_path, output_image="val_results.jpg"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Setup Data
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_dir = os.path.join(data_dir, 'val')
    if not os.path.exists(val_dir):
        print(f"Validation directory not found: {val_dir}")
        return

    val_dataset = datasets.ImageFolder(val_dir, data_transforms)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True) # Shuffle for random viz
    # Force class names to match the 7 classes used in training to prevent shape mismatches
    class_names = ['0', '1', '2', '3', '4', '5', 'None']
    # 2. Load Model
    print("Loading EfficientNet-V2-S...")
    model = models.efficientnet_v2_s(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 3. Run Inference
    all_preds = []
    all_labels = []
    
    print("Running evaluation on validation set...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Metrics
    print("\nClassification Report:")
    # We specify labels to avoid a ValueError if a class is entirely missing from the validation set
    print(classification_report(all_labels, all_preds, labels=list(range(len(class_names))), target_names=class_names, zero_division=0))
    
    # 5. Visual Visualization (First Batch)
    # Get a fresh batch for viz
    inputs, labels = next(iter(val_loader))
    inputs = inputs.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    
    fig = plt.figure(figsize=(12, 12))
    columns = 4
    rows = 4
    
    # Mean/Std for un-normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i in range(min(len(inputs), columns*rows)):
        ax = fig.add_subplot(rows, columns, i + 1)
        img = inputs[i].cpu().numpy().transpose((1, 2, 0))
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        
        true_label = class_names[labels[i]]
        pred_label = class_names[preds[i]]
        
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f"T:{true_label}\nP:{pred_label}", color=color, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_image)
    print(f"Saved visual sample to {output_image}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Classification Model")
    parser.add_argument("--data_dir", required=True, help="Path to validation dataset")
    parser.add_argument("--model", required=True, help="Path to model .pth file")
    parser.add_argument("--output", default="val_results.jpg", help="Output image file")
    
    args = parser.parse_args()
    evaluate_model(args.data_dir, args.model, args.output)
