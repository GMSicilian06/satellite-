import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import time
from tqdm import tqdm
import ttach as tta  # Ensure this is installed: pip install ttach

# --- 1. IMPROVED MODEL CLASS ---
class EfficientNetV2_Improved(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetV2_Improved, self).__init__()
        # Using weights='DEFAULT' is the modern PyTorch way for best pre-trained weights
        self.backbone = models.efficientnet_v2_s(weights='DEFAULT')
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity() # Remove old head
        
        # Multi-Sample Dropout layers
        self.dropouts = nn.ModuleList([nn.Dropout(0.3) for _ in range(5)])
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        # Average results of 5 dropout masks
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.classifier(dropout(x))
            else:
                out += self.classifier(dropout(x))
        return out / len(self.dropouts)

def train_model(data_dir, output_model_path="models/landuse_model.pth", num_epochs=10, batch_size=32, model_name="resnet50"):
    # --- 2. TRANSFORMS (Updated to 384) ---
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((384, 384)), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(), # Added for satellite imagery
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1), # Robustness to light
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ]),
        'val': transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # --- 3. DATA LOADING & BALANCING ---
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    train_dataset = image_datasets['train']
    class_counts = [0] * len(train_dataset.classes)
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    
    class_weights = [1.0 / c if c > 0 else 0.0 for c in class_counts]
    sample_weights = [class_weights[label] for _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, sampler=sampler, num_workers=0),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=0)
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    # --- 4. MODEL SETUP ---
    if model_name == "efficientnet_v2_s":
        model = EfficientNetV2_Improved(num_classes)
    else:
        # Fallback for ResNet if still desired
        model = getattr(models, model_name)(weights='DEFAULT')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    model = model.to(device)

    # Label Smoothing helps with ambiguous land-use boundaries
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, steps_per_epoch=len(dataloaders['train']), epochs=num_epochs
    )

    # TTA Wrapper for Validation
    tta_model = tta.ClassificationTTAWrapper(model, tta.aliases.flip_transform())

    # --- 5. TRAINING LOOP ---
    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_corrects = 0.0, 0
            batch_loop = tqdm(dataloaders[phase], desc=f'{phase} Epoch {epoch}', leave=False)
            
            for inputs, labels in batch_loop:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # During validation, use TTA model for better accuracy
                    outputs = model(inputs) if phase == 'train' else tta_model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step() # CRITICAL: Update scheduler every batch

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                batch_loop.set_postfix(loss=loss.item())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), output_model_path)

    print(f'Training complete. Best val Acc: {best_acc:.4f}')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/classification_dataset_all")
    parser.add_argument("--model", default="efficientnet_v2_s")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    
    os.makedirs("models", exist_ok=True)
    train_model(args.data_dir, f"models/landuse_{args.model}.pth", args.epochs, args.batch_size, args.model)