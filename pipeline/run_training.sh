#!/bin/zsh
cd /Users/carlossequeira/Satellite
source .venv/bin/activate
echo "=========================================="
echo "🚀 Starting EfficientNet-V2-S Vision Model Training"
echo "=========================================="
python pipeline/train_classifier.py --data_dir data/classification_dataset_robust --model efficientnet_v2_s --epochs 10
echo ""
echo "✅ Training Complete!"
read -k 1 "?Press any key to close this window..."
