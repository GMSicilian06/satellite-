
import os

def find_all_matches():
    # 1. Get Image IDs
    image_dirs = [
        "data/raw/fetched_images",
        "data/raw/fetched_images_robust"
    ]
    image_ids = {} # id -> path
    for d in image_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith(".tif"):
                    image_ids[os.path.splitext(f)[0]] = os.path.join(d, f)
    
    print(f"Index: Found {len(image_ids)} images.")

    # 2. Walk Atlas Directory
    atlas_base = "data/raw/atlas_10ha"
    
    matches = []
    
    for root, dirs, files in os.walk(atlas_base):
        for f in files:
            if f.endswith(".shp"):
                shp_id = os.path.splitext(f)[0]
                if shp_id in image_ids:
                    # Found a match!
                    # Get City Name (Parent Dir)
                    # Structure: .../Accra_T0/ID.shp or .../CityName/City_T0/ID.shp
                    parent = os.path.basename(root)
                    
                    # exclude knowns
                    if "Mumbai" in root or "Ahmedabad" in root or "New Delhi" in root:
                        continue
                        
                    matches.append({
                        "id": shp_id,
                        "city_dir": root,
                        "shp_path": os.path.join(root, f),
                        "img_path": image_ids[shp_id]
                    })
    
    print(f"Found {len(matches)} new candidates.")
    for m in matches[:10]:
        print(f"Match: {m['city_dir']} -> ID: {m['id']}")

if __name__ == "__main__":
    find_all_matches()
