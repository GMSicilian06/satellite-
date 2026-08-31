import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import sys
import os
# Add the project root to sys.path so we can import from pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.inference import EfficientNetV2_Improved

def few_shot_regional_finetune(regional_data_dir, base_model_path, output_path, epochs=10):
    # Select Device (MPS for Mac)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  FEW-SHOT REGIONAL FINE-TUNING: Southeast Asia")
    print(f"  Data Source: {regional_data_dir}")
    print(f"{'='*60}")
    
    # 1. Load the Global Model with 6 classes
    model = EfficientNetV2_Improved(num_classes=6)
    
    # Load the weights we just trained
    print(f"Loading base weights from: {base_model_path}")
    state_dict = torch.load(base_model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # 2. FREEZE THE BRAIN (Backbone)
    # This prevents the model from forgetting global features
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # 3. TRAIN THE OUTPUT (Head)
    # We only optimize the dropout layers and the linear classifier
    params_to_update = []
    for name, param in model.named_parameters():
        if "classifier" in name or "dropouts" in name:
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False
        
    model = model.to(device)

    # 4. Regional Augmentation
    data_transforms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(regional_data_dir, data_transforms)
    # Batch size 8 is better for few-shot stability
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Optimizer & Loss (Low learning rate for fine-tuning)
    optimizer = optim.Adam(params_to_update, lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 5. EXECUTE CALIBRATION
    model.train()
    print("\nStarting regional calibration...")
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

    # Save the calibrated regional weights
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"\nSUCCESS: Regional model saved to {output_path}")

if __name__ == "__main__":
    few_shot_regional_finetune(
        regional_data_dir="data/few_shot_southeast_asia", 
        base_model_path="models/landuse_efficientnet_v2_s.pth", 
        output_path="models/landuse_southeast_asia_improved.pth",
        epochs=10
    )
