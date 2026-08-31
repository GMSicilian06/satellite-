#!/usr/bin/env python3
"""
Land Use Vision Model Fine-Tuning Script for South America (Google Colab Edition)

Instructions:
1. Upload this script and `classification_dataset_south_america.zip` to your Google Drive.
2. Upload the base pre-trained model weights `landuse_efficientnet_v2_s_hd.pth` to your Google Drive.
3. Open a Google Colab notebook with a GPU runtime (T4, L4, or A100).
4. Run the cells to mount Drive, unzip the dataset, and execute this training script.
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

def train_model(data_dir, base_weights_path, output_model_path, num_epochs=10, batch_size=32):
    # Data Augmentation & Normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load Datasets
    image_datasets = {}
    for phase in ['train', 'val']:
        phase_dir = os.path.join(data_dir, phase)
        if os.path.isdir(phase_dir):
            image_datasets[phase] = datasets.ImageFolder(phase_dir, data_transforms[phase])
            
    if 'train' not in image_datasets or len(image_datasets['train']) == 0:
        raise ValueError(f"Error: No training images found in {data_dir}/train")
    
    # Class Balancing for Training Data
    train_dataset = image_datasets['train']
    class_counts = [0] * len(train_dataset.classes)
    for _, label in train_dataset.samples:
        class_counts[label] += 1
        
    class_weights = [1.0 / c if c > 0 else 0.0 for c in class_counts]
    sample_weights = [class_weights[label] for _, label in train_dataset.samples]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, sampler=sampler, num_workers=2)
    }
    if 'val' in image_datasets:
        dataloaders['val'] = DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=2)
    
    phases = [p for p in ['train', 'val'] if p in image_datasets and len(image_datasets[p]) > 0]
    dataset_sizes = {x: len(image_datasets[x]) for x in phases}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    
    print(f"Classes found: {class_names}")
    print(f"Dataset sizes: {dataset_sizes}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # --- Setup Model for Fine-Tuning ---
    print(f"Loading EfficientNet-V2-S...")
    model = models.efficientnet_v2_s(weights=None)
    
    # Recreate the classifier head
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    # Load pretrained weights
    print(f"Loading base global weights from {base_weights_path}...")
    if os.path.exists(base_weights_path):
        state_dict = torch.load(base_weights_path, map_location=device)
        
        # Clean 'module.' prefix from state dict keys
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Avoid shape mismatch on classifier layer
        if 'classifier.1.weight' in state_dict and state_dict['classifier.1.weight'].shape[0] != num_classes:
            print("Classifier size mismatch. Loading feature extraction weights only.")
            state_dict.pop('classifier.1.weight', None)
            state_dict.pop('classifier.1.bias', None)
            
        model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(f"Error: Base weights not found at {base_weights_path}")

    # Freeze base feature extraction layers (except final block)
    print("Freezing base layers...")
    for param in model.features.parameters():
        param.requires_grad = False
        
    for param in model.features[-1].parameters():
        param.requires_grad = True
        
    for param in model.classifier.parameters():
        param.requires_grad = True

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.features[-1].parameters(), 'lr': 0.0001},
        {'params': model.classifier.parameters(), 'lr': 0.0005}
    ])
    
    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Wrap dataloader with tqdm
            batch_loop = tqdm(dataloaders[phase], desc=f'{phase} Epoch {epoch}/{num_epochs-1}', leave=False)
            
            for inputs, labels in batch_loop:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                batch_loop.set_postfix(loss=loss.item())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc.item():.4f}')

            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), output_model_path)
                    print(f"New best model saved to {output_model_path}!")
            elif phase == 'train' and 'val' not in phases:
                torch.save(model.state_dict(), output_model_path)
                print(f"Model saved to {output_model_path}")

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc.item() if torch.is_tensor(best_acc) else best_acc:.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Land Use Vision Model on Colab")
    parser.add_argument("--data_dir", default="/content/classification_dataset_south_america", help="Dataset directory path")
    parser.add_argument("--base_weights", default="/content/drive/MyDrive/models/landuse_efficientnet_v2_s_hd.pth", help="Base global model weights path")
    parser.add_argument("--output_model", default="/content/drive/MyDrive/models/landuse_south_america_efficientnet_v2_s.pth", help="Save path for fine-tuned weights")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    train_model(args.data_dir, args.base_weights, args.output_model, args.epochs, args.batch_size)
