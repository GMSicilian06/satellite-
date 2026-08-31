import os
import math
import random
import warnings
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as ctx
import osmnx as ox
import numpy as np
import torch
import torch.nn as nn
import rasterio
import rasterio.plot
from rasterio.mask import mask as rasterio_mask
from torchvision import transforms, models
from PIL import Image
from shapely.geometry import Point, box
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────── INPUTS ────────────────────────────────────────────
city_name       = "Dhaka"
boundary_path   = "data/processed/urban_extents/Dhaka_ghs_boundary.geojson"
model_path      = "models/landuse_efficientnet_v2_s.pth"

num_circles     = 5
radius_m        = 178.4          # ≈ 10-ha circle
SEED            = 7

# Known dense urban-core lon/lat → EPSG:3857 anchor for the search window
# (Motijheel / Old Dhaka)
CORE_LON, CORE_LAT = 90.4125, 23.7275
WINDOW_M        = 900            # tighter search window around the core for closer circles

output_circles_only = "dhaka_5circles_map1_circles.png"
output_blocks       = "dhaka_5circles_map2_blocks.png"
output_classified   = "dhaka_5circles_map3_classified.png"
CACHE_CLASSIFIED    = "dhaka_5circles_cache_classified.geojson"
CACHE_RAW           = "dhaka_5circles_cache_raw.geojson"

# ─────────────────────────── COLOUR PALETTE ────────────────────────────────────
CLASS_NAMES = [
    "Open Space",
    "Non-residential Areas",
    "Atomistic Settlements",
    "Informal Land Subdivisions",
    "Formal Land Subdivisions",
    "Housing Projects",
    "Road Space",
]

def get_color(class_code):
    if class_code is None: return "gray"
    try:
        code = int(float(class_code))
        if code == 0: return "#4CAF50"   # green
        if code == 1: return "#9E9E9E"   # gray
        if code == 2: return "#9C27B0"   # purple
        if code == 3: return "#F44336"   # red
        if code == 4: return "#2196F3"   # blue
        if code == 5: return "#FF9800"   # orange
        if code == 6: return "#E91E63"   # pink
        return "gray"
    except:
        return "gray"

legend_patches = [
    mpatches.Patch(facecolor=get_color(i), edgecolor="white", label=CLASS_NAMES[i])
    for i in range(len(CLASS_NAMES))
]

# ─────────────────────────── LOAD GHS BOUNDARY ─────────────────────────────────
print("Loading Dhaka GHS boundary...")
boundary_gdf = gpd.read_file(boundary_path)
if boundary_gdf.crs is None:
    boundary_gdf = boundary_gdf.set_crs("EPSG:4326")
boundary_gdf = boundary_gdf.to_crs("EPSG:3857")
boundary = boundary_gdf.geometry.union_all()

# ─────────────────────────── CONVERT CORE TO EPSG:3857 ─────────────────────────
from pyproj import Transformer
_t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
cx, cy = _t.transform(CORE_LON, CORE_LAT)
print(f"Urban core anchor: {cx:.0f}, {cy:.0f}  (EPSG:3857)")

# ─────────────────────────── GENERATE 5 CIRCLES ────────────────────────────────
print(f"\nGenerating {num_circles} random circles within Dhaka urban core...")
random.seed(SEED)

sx1, sy1 = cx - WINDOW_M, cy - WINDOW_M
sx2, sy2 = cx + WINDOW_M, cy + WINDOW_M
search_box = box(sx1, sy1, sx2, sy2).intersection(boundary)

circles  = []
attempts = 0

while len(circles) < num_circles and attempts < 50000:
    x = random.uniform(sx1, sx2)
    y = random.uniform(sy1, sy2)
    pt = Point(x, y)
    if not search_box.contains(pt):
        attempts += 1
        continue
    circle = pt.buffer(radius_m)
    if not boundary.contains(circle):
        attempts += 1
        continue
    too_close = any(
        pt.distance(c["center"]) < radius_m * 2.3
        for c in circles
    )
    if not too_close:
        circles.append({"geometry": circle, "center": pt, "id": len(circles) + 1})
    attempts += 1

print(f"Generated {len(circles)}/{num_circles} circles after {attempts} attempts.")
circles_gdf = gpd.GeoDataFrame(circles, crs="EPSG:3857")

# ─────────────────────────── SHARED WIDE-MAP EXTENT ────────────────────────────
minx, miny, maxx, maxy = circles_gdf.total_bounds
pad  = radius_m * 0.5   # tight zoom — just a half-radius of margin around the outermost circles
xlim = (minx - pad, maxx + pad)
ylim = (miny - pad, maxy + pad)

# ─────────────────────────── LOAD MODEL ────────────────────────────────────────
print("\nLoading EfficientNet-V2-S model...")
device = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

loaded_obj = torch.load(model_path, map_location=device)
if isinstance(loaded_obj, nn.Module):
    model = loaded_obj
elif isinstance(loaded_obj, dict):
    num_classes = list(loaded_obj.values())[-1].shape[0]
    print(f"Detected {num_classes} classes from checkpoint.")
    model = models.efficientnet_v2_s(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(loaded_obj)
else:
    raise ValueError("Unrecognised model format.")

model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ─────────────────────────── EXTRACT OSM BLOCKS + CLASSIFY  (with cache) ───────
if os.path.exists(CACHE_CLASSIFIED) and os.path.exists(CACHE_RAW):
    print("\nLoading cached blocks and classifications...")
    all_classified_blocks = [gpd.read_file(CACHE_CLASSIFIED).to_crs("EPSG:3857")]
    all_raw_blocks        = [gpd.read_file(CACHE_RAW).to_crs("EPSG:3857")]
else:
    print("\nExtracting OSM blocks and classifying all circles...")
    all_classified_blocks = []
    all_raw_blocks        = []   # for the unclassified map

    for c in circles:
        circle_geom  = c["geometry"]      # EPSG:3857
        circle_id    = c["id"]

        circle_4326      = gpd.GeoDataFrame(geometry=[circle_geom], crs="EPSG:3857").to_crs("EPSG:4326")
        circle_geom_4326 = circle_4326.geometry.iloc[0]

        print(f"\nCircle {circle_id}: downloading OSM roads...")
        try:
            buffer_geom   = circle_geom_4326.buffer(0.001)
            G             = ox.graph_from_polygon(buffer_geom, network_type="all", simplify=True)
            _, edges      = ox.graph_to_gdfs(G, nodes=True, edges=True)
            edges_metric  = edges.to_crs("EPSG:3857")
            road_mask     = edges_metric.geometry.buffer(3).union_all()
            locale_metric = gpd.GeoSeries(circle_geom_4326, crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
            blocks_metric = locale_metric.difference(road_mask)
            block_geoms   = (
                list(blocks_metric.geoms)
                if blocks_metric.geom_type == "MultiPolygon"
                else [blocks_metric]
            )
            gdf_blocks = gpd.GeoDataFrame(geometry=block_geoms, crs="EPSG:3857")
            gdf_blocks = gpd.clip(gdf_blocks, circle_geom)
            gdf_blocks = gdf_blocks[gdf_blocks.geometry.area > 50].copy()

            if len(gdf_blocks) == 0:
                print(f"  Circle {circle_id}: no blocks found, skipping.")
                continue

            # Store raw blocks for Map 2
            raw_copy = gdf_blocks.copy()
            raw_copy["circle_id"] = circle_id
            all_raw_blocks.append(raw_copy)

            # Fetch per-circle imagery at zoom 19
            print(f"  Circle {circle_id}: {len(gdf_blocks)} blocks. Fetching imagery (zoom 19)...")
            w2, s2, e2, n2 = circle_4326.total_bounds
            img_path = f"temp_circle_{circle_id}.tif"
            ctx.bounds2raster(
                w2 - 0.0005, s2 - 0.0005, e2 + 0.0005, n2 + 0.0005,
                img_path, zoom=19,
                source=ctx.providers.Esri.WorldImagery, ll=True
            )

            print(f"  Circle {circle_id}: classifying {len(gdf_blocks)} blocks...")
            pred_colors, pred_codes = [], []

            with rasterio.open(img_path) as src:
                gdf_blocks_reproj = gdf_blocks.to_crs(src.crs)
                for _, row in gdf_blocks_reproj.iterrows():
                    try:
                        geom_buf     = row.geometry.buffer(15.0)
                        out_image, _ = rasterio_mask(src, [geom_buf], crop=True)
                        img_array    = out_image.transpose(1, 2, 0)
                        if img_array.shape[2] > 3:
                            img_array = img_array[:, :, :3]
                        if img_array.shape[0] < 5 or img_array.shape[1] < 5:
                            pred_colors.append(get_color(1))
                            pred_codes.append(-1)
                            continue
                        pil_img = Image.fromarray(img_array.astype(np.uint8))
                        tensor  = transform(pil_img).unsqueeze(0).to(device)
                        with torch.no_grad():
                            output   = model(tensor)
                            pred_idx = output.argmax(dim=1).item()
                        pred_colors.append(get_color(pred_idx))
                        pred_codes.append(pred_idx)
                    except Exception:
                        pred_colors.append(get_color(1))
                        pred_codes.append(-1)

            gdf_blocks["pred_color"] = pred_colors
            gdf_blocks["pred_code"]  = pred_codes
            gdf_blocks["circle_id"]  = circle_id
            all_classified_blocks.append(gdf_blocks)

            if os.path.exists(img_path):
                os.remove(img_path)
            print(f"  Circle {circle_id}: done.")

        except Exception as e:
            print(f"  Circle {circle_id}: failed — {e}")

    # ── Save cache after all circles processed ──────────────────────────────
    if all_classified_blocks and not os.path.exists(CACHE_CLASSIFIED):
        gpd.GeoDataFrame(
            pd.concat(all_classified_blocks, ignore_index=True), crs="EPSG:3857"
        ).to_file(CACHE_CLASSIFIED, driver="GeoJSON")
        print(f"Cached classified blocks → {CACHE_CLASSIFIED}")
    if all_raw_blocks and not os.path.exists(CACHE_RAW):
        gpd.GeoDataFrame(
            pd.concat(all_raw_blocks, ignore_index=True), crs="EPSG:3857"
        ).to_file(CACHE_RAW, driver="GeoJSON")
        print(f"Cached raw blocks → {CACHE_RAW}")

# ─────────────────────────── MAP 1: Circles only ───────────────────────────────
print("\nGenerating Map 1: circles only...")
fig, ax = plt.subplots(figsize=(14, 14))
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=17)

circles_gdf.plot(ax=ax, facecolor="none", edgecolor="white", linewidth=2.5, zorder=8)

ax.set_axis_off()
for artist in ax.get_children():
    if hasattr(artist, "get_text") and "Esri" in str(artist.get_text()):
        artist.set_visible(False)

plt.tight_layout(pad=0.5)
plt.savefig(output_circles_only, dpi=300, bbox_inches="tight", facecolor="black")
plt.close()
print(f"Saved: {output_circles_only}")

# ─────────────────────────── MAP 2: Circles + unclassified blocks ──────────────
print("Generating Map 2: circles + unclassified blocks...")
fig, ax = plt.subplots(figsize=(14, 14))
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=17)

if all_raw_blocks:
    blocks_all_raw = gpd.GeoDataFrame(
        pd.concat(all_raw_blocks, ignore_index=True), crs="EPSG:3857"
    )
    blocks_all_raw.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=1.5, alpha=0.9, zorder=6)

circles_gdf.plot(ax=ax, facecolor="none", edgecolor="white", linewidth=2.5, zorder=8)

ax.set_axis_off()
for artist in ax.get_children():
    if hasattr(artist, "get_text") and "Esri" in str(artist.get_text()):
        artist.set_visible(False)

plt.tight_layout(pad=0.5)
plt.savefig(output_blocks, dpi=300, bbox_inches="tight", facecolor="black")
plt.close()
print(f"Saved: {output_blocks}")

# ─────────────────────────── MAP 3: Circles + classified blocks ─────────────────
print("Generating Map 3: circles + classified blocks...")
fig, ax = plt.subplots(figsize=(14, 14))
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=17)

if all_classified_blocks:
    blocks_all = gpd.GeoDataFrame(
        pd.concat(all_classified_blocks, ignore_index=True), crs="EPSG:3857"
    )
    for _, row in blocks_all.iterrows():
        geom  = row.geometry
        color = row["pred_color"]
        if geom is None or geom.is_empty: continue
        if geom.geom_type == "Polygon":
            x_c, y_c = geom.exterior.xy
            ax.fill(x_c, y_c, alpha=1.0, facecolor=color,
                    edgecolor="white", linewidth=0.4, zorder=6)
        elif geom.geom_type == "MultiPolygon":
            for sub in geom.geoms:
                x_c, y_c = sub.exterior.xy
                ax.fill(x_c, y_c, alpha=1.0, facecolor=color,
                        edgecolor="white", linewidth=0.4, zorder=6)

circles_gdf.plot(ax=ax, facecolor="none", edgecolor="white", linewidth=2.5, zorder=8)

ax.legend(handles=legend_patches, loc="upper right",
          fontsize=10, framealpha=0.85,
          facecolor="#1a1a1a", edgecolor="white",
          labelcolor="white", title="Land Use",
          title_fontsize=11)
ax.set_axis_off()
for artist in ax.get_children():
    if hasattr(artist, "get_text") and "Esri" in str(artist.get_text()):
        artist.set_visible(False)

plt.tight_layout(pad=0.5)
plt.savefig(output_classified, dpi=300, bbox_inches="tight", facecolor="black")
plt.close()
print(f"Saved: {output_classified}")

print("\nAll 3 maps saved:")
print(f"  1. {output_circles_only}")
print(f"  2. {output_blocks}")
print(f"  3. {output_classified}")
