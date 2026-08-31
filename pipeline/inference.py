import torch.nn as nn
from torchvision import models
import ttach as tta

class EfficientNetV2_Improved(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetV2_Improved, self).__init__()
        self.backbone = models.efficientnet_v2_s(weights='DEFAULT')
        
        # Identify the input features to the final classifier
        in_features = self.backbone.classifier[1].in_features
        
        # Remove the original classifier
        self.backbone.classifier = nn.Identity()
        
        # Create 5 parallel dropout layers
        self.dropouts = nn.ModuleList([nn.Dropout(0.3) for _ in range(5)])
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        # Average the results of the 5 dropout samples
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.classifier(dropout(x))
            else: 
                out += self.classifier(dropout(x))
        return out / len(self.dropouts)

# tta_model = tta.ClassificationTTAWrapper(model, tta.aliases.flip_transform())