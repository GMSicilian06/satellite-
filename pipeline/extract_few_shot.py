import os
import shutil
import random
from pathlib import Path

# Configuration
SOURCE_DIR = "data/classification_dataset_all/train"
AUE_DIR = "data/AUE_Everything"
TARGET_DIR = "data/few_shot_southeast_asia"
SAMPLES_PER_CLASS_PER_CITY = 5 # 5 samples * 25 cities = 125 samples per class (Good for few-shot)

CITIES = [
    "Ahmedabad", "Bangkok", "Belgaum", "Coimbatore", "Dhaka", 
    "Hindupur", "Ho Chi Minh City", "Hyderabad", "Jaipur", "Jalna", 
    "Kanpur", "Karachi", "Kolkata", "Kozhikode", "Lahore", 
    "Malegaon", "Manila", "Mumbai", "Myeik", "Parbhani", 
    "Pune", "Sialkot", "Singapore", "Singrauli", "Vijayawada"
]

def get_city_locales(city_name):
    locales = set()
    city_path = Path(AUE_DIR) / city_name
    if not city_path.exists():
        return locales
    # Find all .shp files recursively
    for shp in city_path.rglob("*.shp"):
        # The filename is usually the LocaleID
        locale_id = shp.stem
        if locale_id.isdigit():
            locales.add(locale_id)
    return locales

def main():
    os.makedirs(TARGET_DIR, exist_ok=True)
    for class_id in range(6):
        os.makedirs(os.path.join(TARGET_DIR, str(class_id)), exist_ok=True)

    city_stats = {}

    for city in CITIES:
        locales = get_city_locales(city)
        if not locales:
            print(f"No locales found for {city}")
            continue
        
        print(f"Processing {city} ({len(locales)} locales)...")
        city_stats[city] = 0

        for class_id in range(6):
            class_source = os.path.join(SOURCE_DIR, str(class_id))
            if not os.path.exists(class_source): continue
            
            # Find images belonging to these locales
            city_images = []
            all_images = os.listdir(class_source)
            for img in all_images:
                locale_id = img.split('_')[0]
                if locale_id in locales:
                    city_images.append(img)
            
            # Sample N images
            sampled = random.sample(city_images, min(len(city_images), SAMPLES_PER_CLASS_PER_CITY))
            
            for img in sampled:
                src = os.path.join(class_source, img)
                dst = os.path.join(TARGET_DIR, str(class_id), f"{city}_{img}")
                shutil.copy2(src, dst)
                city_stats[city] += 1

    print("\nExtraction Complete!")
    for city, count in city_stats.items():
        print(f"{city}: {count} images copied")

if __name__ == "__main__":
    main()
