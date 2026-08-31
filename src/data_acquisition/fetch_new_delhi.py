import contextily as ctx
import matplotlib.pyplot as plt
import os
import sys

# Connaught Place, New Delhi
# N: 28.635, S: 28.625, E: 77.225, W: 77.210
# Small area to ensure high res and fast processing
N, S, E, W = 28.635, 28.625, 77.225, 77.210

OUTPUT_DIR = "data/new_delhi"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_PATH = os.path.join(OUTPUT_DIR, "new_delhi_satellite.tif")

def fetch_data():
    print(f"Fetching Satellite Image for New Delhi ({N},{W})...")
    try:
        # Contextily downloads web tiles. Zoom 18 is good for blocks.
        # bounds2raster expects (West, South, East, North)
        # Note: coordinates are float
        ctx.bounds2raster(W, S, E, N, IMG_PATH, source=ctx.providers.Esri.WorldImagery, zoom=18, ll=True)
        print(f"Saved Image to {IMG_PATH}")
    except Exception as e:
        print(f"Failed to fetch image: {e}")
        return

if __name__ == "__main__":
    fetch_data()
