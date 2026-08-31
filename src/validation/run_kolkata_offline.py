
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.validation.offline_analysis import run_offline_analysis

# Kolkata Sample
# ID: 100012250088301
# Path: data/raw/atlas_10ha/Accra/Kolkata/Kolkata_T0/100012250088301.shp

city_name = "Kolkata"
sample_id = "100012250088301"
# Nesting: Accra -> Kolkata
shp_path = f"data/raw/atlas_10ha/Accra/Kolkata/Kolkata_T0/{sample_id}.shp"

print(f"Running Kolkata Offline Analysis for ID: {sample_id}")

try:
    run_offline_analysis(city_name, sample_id, shp_path)
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")
