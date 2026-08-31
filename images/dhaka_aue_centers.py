import os
import glob
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point, Polygon
from shapely.ops import polygonize

def to_polygon(geom):
    if geom.geom_type == "Polygon":
        return geom
    elif geom.geom_type == "LineString":
        return Polygon(geom.coords)
    elif geom.geom_type == "MultiLineString":
        polys = list(polygonize(geom.geoms))
        if polys:
            return polys[0]
    return geom.convex_hull

# -----------------------------
# Settings
# -----------------------------
city_name = "Dhaka"
ghsl_db_path = "data/raw/GHS_UCDB_GLOBE_R2024A.gpkg"
aue_dir = "Dhaka/Dhaka/Dhaka_T0"
output_map = "dhaka_aue_centers.png"

# -----------------------------
# 1. Load AUE Centers
# -----------------------------
print("Loading AUE centers...")
shp_files = glob.glob(os.path.join(aue_dir, "*00.shp"))

centers = []
for f in shp_files:
    try:
        gdf = gpd.read_file(f)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        gdf = gdf.to_crs(epsg=3857)
        # Apply to_polygon
        gdf["geometry"] = gdf.geometry.apply(to_polygon)
        
        # Calculate centroid
        centroid = gdf.geometry.iloc[0].centroid
        centers.append(centroid)
    except Exception as e:
        print(f"Skipping {f}: {e}")

print(f"Loaded {len(centers)} AUE centers.")
centers_gdf = gpd.GeoDataFrame(geometry=centers, crs="EPSG:3857")

# -----------------------------
# 2. Load Dhaka boundary
# -----------------------------
print("Loading Dhaka boundary...")
ghsl = gpd.read_file(
    ghsl_db_path,
    layer="GHS_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A"
)
ghsl.columns = ghsl.columns.str.replace("\ufeff", "", regex=False)
city_gdf = ghsl[
    ghsl["GC_UCN_MAI_2025"].astype(str).str.strip()
    .str.replace("\ufeff", "", regex=False)
    .str.lower() == city_name.lower()
].copy()
city_gdf = city_gdf.to_crs(epsg=3857)
city_gdf["area"] = city_gdf.geometry.area
city_gdf = city_gdf.loc[[city_gdf["area"].idxmax()]].copy()
boundary = city_gdf.geometry.union_all()

# -----------------------------
# 3. Filter centers within boundary
# -----------------------------
centers_in_boundary = centers_gdf[centers_gdf.geometry.within(boundary)]
print(f"Centers inside Dhaka boundary: {len(centers_in_boundary)}")

# -----------------------------
# 4. Plot Map
# -----------------------------
print("Generating Map...")
fig, ax = plt.subplots(figsize=(14, 14))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Plot boundary
city_gdf.boundary.plot(ax=ax, color="cyan", linewidth=2.5, zorder=5)

# Plot centers
centers_gdf.plot(ax=ax, color="red", markersize=150, marker='o', edgecolor="white", linewidth=1.5, zorder=6)

# Set bounds
w, s, e, n = city_gdf.total_bounds
# Add padding
pad_x = (e - w) * 0.1
pad_y = (n - s) * 0.1
ax.set_xlim(w - pad_x, e + pad_x)
ax.set_ylim(s - pad_y, n + pad_y)

try:
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=12)
except Exception as e:
    print(f"Basemap failed: {e}")
    try:
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=11)
    except Exception as e2:
        print(f"Basemap fallback failed: {e2}")

ax.set_axis_off()
for artist in ax.get_children():
    if hasattr(artist, "get_text") and "Esri" in str(artist.get_text()):
        artist.set_visible(False)

plt.tight_layout(pad=0.5)
plt.savefig(output_map, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {output_map}")
