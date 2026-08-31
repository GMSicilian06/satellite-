#!/bin/zsh
cd "/Users/carlossequeira/MIT Dropbox/Carlos Sequeira/Cities and Environment/Satellite"
source .venv/bin/activate

CITIES=($@)
if [ ${#CITIES[@]} -eq 0 ]; then
    CITIES=("Dhaka")
fi

# Create a combined lowercase string like "dhaka_lahore"
CITY_STR=$(echo "${CITIES[@]}" | tr ' ' '_' | tr '[:upper:]' '[:lower:]')

echo "=========================================="
echo "🌍 Starting Region-Specific Fine-Tuning for: ${CITIES[@]}"
echo "=========================================="

echo "Step 1: Extracting Regional Dataset..."
.venv/bin/python pipeline/create_regional_dataset.py --cities "${CITIES[@]}"

echo "\nStep 2: Fine-Tuning Classifier Head..."
.venv/bin/python pipeline/finetune_regional.py \
    --data_dir "data/classification_dataset_${CITY_STR}" \
    --base_weights "models/landuse_efficientnet_v2_s_hd.pth" \
    --output_model "models/landuse_${CITY_STR}_efficientnet_v2_s.pth" \
    --epochs 10

echo ""
echo "✅ Regional Fine-Tuning Complete for ${CITIES[@]}!"
