import geopandas as gpd
import shapely.geometry

def check_alignment(file1, file2):
    print(f"Checking alignment between:\n 1. {file1}\n 2. {file2}")
    
    gdf1 = gpd.read_file(file1)
    gdf2 = gpd.read_file(file2)
    
    print(f"\n--- File 1: {file1} ---")
    print(f"CRS: {gdf1.crs}")
    print(f"Bounds: {gdf1.total_bounds}")
    print(f"Feature Count: {len(gdf1)}")
    
    print(f"\n--- File 2: {file2} ---")
    print(f"CRS: {gdf2.crs}")
    print(f"Bounds: {gdf2.total_bounds}")
    print(f"Feature Count: {len(gdf2)}")
    
    # Check Intersection in File 2's CRS
    if gdf1.crs != gdf2.crs:
        print(f"\nReprojecting File 1 to matches File 2 ({gdf2.crs})...")
        gdf1 = gdf1.to_crs(gdf2.crs)
        print(f"New Bounds 1: {gdf1.total_bounds}")
        
    # Check strict overlap
    box1 = shapely.geometry.box(*gdf1.total_bounds)
    box2 = shapely.geometry.box(*gdf2.total_bounds)
    
    intersection = box1.intersection(box2)
    print(f"\nIntersection Area of Bounding Boxes: {intersection.area}")
    
    if intersection.is_empty:
        print("CRITICAL: No spatial overlap between bounding boxes!")
    else:
        print("Bounding boxes overlap. Checking individual features...")
        dataset_matches = gdf1[gdf1.intersects(box2)]
        print(f"Features in File 1 that intersect File 2's bbox: {len(dataset_matches)}")

if __name__ == "__main__":
    check_alignment(
        "data/ahmedabad_output/plots.geojson",
        "data/ahmedabad_test/100562310072501.shp"
    )
