#!/bin/zsh
cd "/Users/carlossequeira/MIT Dropbox/Carlos Sequeira/Cities and Environment/Satellite"
source .venv/bin/activate

echo "=========================================="
echo "🌍 Starting Global Resolution Adjustment"
echo "=========================================="
echo "Using the regional fine-tuning logic to freeze 90% of the model"
echo "and quickly adjust the final layers to the new 384x384 resolution"
echo "using the entire robust global dataset!"
echo ""

.venv/bin/python pipeline/finetune_regional.py \
    --data_dir "data/classification_dataset_robust" \
    --base_weights "models/landuse_efficientnet_v2_s.pth" \
    --output_model "models/landuse_efficientnet_v2_s_hd.pth" \
    --epochs 5

echo ""
echo "✅ Global Resolution Adjustment Complete!"
echo "Your new high-definition model is saved as: models/landuse_efficientnet_v2_s_hd.pth"
