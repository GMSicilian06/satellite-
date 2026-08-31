
import os

def find_matches():
    # 1. Get Image IDs
    image_dirs = [
        "data/raw/fetched_images",
        "data/raw/fetched_images_robust"
    ]
    image_ids = set()
    for d in image_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith(".tif"):
                    image_ids.add(os.path.splitext(f)[0])
    
    # 2. Check Mumbai SHPs
    mumbai_dir = "data/raw/atlas_10ha/Accra/Mumbai/Mumbai_T0"
    if os.path.exists(mumbai_dir):
        for f in os.listdir(mumbai_dir):
            if f.endswith(".shp"):
                shp_id = os.path.splitext(f)[0]
                if shp_id in image_ids:
                    print(f"MATCH FOUND: Mumbai ID {shp_id}")
                    # Print paths
                    print(f"  Shapefile: {os.path.join(mumbai_dir, f)}")
                    # Find image path
                    for d in image_dirs:
                        img_path = os.path.join(d, shp_id + ".tif")
                        if os.path.exists(img_path):
                            print(f"  Image: {img_path}")
                            pass
                            
if __name__ == "__main__":
    find_matches()
