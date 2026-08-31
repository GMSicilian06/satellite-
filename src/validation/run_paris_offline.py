
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.validation.offline_analysis import run_offline_analysis

# Paris Sample
# ID: 100314880002001
# Path: data/raw/atlas_10ha/Accra/Paris/Paris_T0/100314880002001.shp

city_name = "Paris"
sample_id = "100314880002001"
# Note: The 'Accra' in the path is weird but verified by find_by_name
shp_path = f"data/raw/atlas_10ha/Accra/Paris/Paris_T0/{sample_id}.shp"

print(f"Running Paris Offline Analysis for ID: {sample_id}")

try:
    run_offline_analysis(city_name, sample_id, shp_path)
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")
