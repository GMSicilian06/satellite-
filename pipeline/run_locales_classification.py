import os
import argparse
import sys
import glob

# Add project root to sys.path so absolute imports resolve correctly at runtime and in Pylance
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.validation.offline_analysis import offline_analysis

def classify_locales(city_dir, model_path, append=False):
    blocks_dir = os.path.join(city_dir, "blocks")
    img_dir = os.path.join(city_dir, "images")
    gis_dir = os.path.join(city_dir, "gis") # Used as "truth" bounding extent
    out_dir = os.path.join(city_dir, "classification_results")
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Process each block file
    for blocks_file in glob.glob(os.path.join(blocks_dir, "*_blocks.geojson")):
        locale_id = os.path.basename(blocks_file).replace('_blocks.geojson', '')
        
        img_file = os.path.join(img_dir, f"{locale_id}.tif")
        truth_file = os.path.join(gis_dir, f"{locale_id}.geojson") # The 10ha circle
        
        if not os.path.exists(img_file):
            print(f"[{locale_id}] Missing image file. Skipping.")
            continue
            
        print(f"\n{'='*40}")
        print(f"Classifying {locale_id}...")
        print(f"{'='*40}")
        
        locale_out_dir = os.path.join(out_dir, locale_id)
        if append and os.path.exists(os.path.join(locale_out_dir, "predicted_classified.geojson")):
            print(f"[{locale_id}] Already classified. Skipping.")
            continue
            
        os.makedirs(locale_out_dir, exist_ok=True)
        
        try:
            # We use the offline_analysis function which loads the ResNet,
            # crops the chips, predicts, and generates a side-by-side comparison.
            # Here: 
            #   existing_blocks_path = The polygons we generated
            #   study_circle_truth_path = The polygons again (or the circle) -> just for visualization matching
            
            offline_analysis(
                existing_blocks_path=blocks_file,
                study_circle_truth_path=blocks_file, # Use blocks as truth so both sides show the blocks
                geotiff_path=img_file,
                png_path=None,
                model_path=model_path,
                output_dir=locale_out_dir
            )
            print(f"[{locale_id}] Classification complete. Results saved to {locale_out_dir}")
            
        except Exception as e:
            print(f"[{locale_id}] Error during classification: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city_dir", required=True, help="Base directory of the processed city data")
    parser.add_argument("--model", default="models/landuse_efficientnet_v2_s.pth", help="Path to trained model")
    parser.add_argument("--append", action="store_true", help="Skip existing classifications")
    args = parser.parse_args()
    
    classify_locales(args.city_dir, args.model, args.append)
