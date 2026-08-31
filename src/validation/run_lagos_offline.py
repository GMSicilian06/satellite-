
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.validation.offline_analysis import run_offline_analysis

# Lagos Sample
# ID: 100020640003401
# Path: data/raw/atlas_10ha/Accra/Lagos/Lagos_T0/100020640003401.shp

city_name = "Lagos"
sample_id = "100020640003401"
shp_path = f"data/raw/atlas_10ha/Accra/Lagos/Lagos_T0/{sample_id}.shp"

# We need the image too. 
# It might not be downloaded because fetch_tiles was robust.
# Let's check or download dynamic
# Actually offline_analysis takes the SHAPEFILE and fetches the image if needed using contextily if we don't pass an image path?
# Let's check offline_analysis signature.
# It expects (city_name, shapefile_id, ...). 
# But wait, offline_analysis.py main() hardcodes paths. 
# I should import the function `run_offline_analysis(city_name, shapefile_id, shapefile_path)`

print(f"Running Lagos Offline Analysis for ID: {sample_id}")

try:
    run_offline_analysis(city_name, sample_id, shp_path)
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")
