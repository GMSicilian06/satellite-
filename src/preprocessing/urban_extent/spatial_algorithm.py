import numpy as np
import rasterio
from scipy.ndimage import convolve, binary_dilation, binary_closing
from skimage.measure import label, regionprops
import geopandas as gpd
import os
from shapely.geometry import Polygon
import rasterio.features

class AUESpatialAlgorithm:
    def __init__(self, resolution=30):
        """
        Implementation of the Atlas of Urban Expansion geometric rules.
        resolution: pixel size in meters (default 30m for Landsat)
        """
        self.resolution = resolution
        
        # Area of 1 pixel in square meters
        self.pixel_area = resolution * resolution
        
        # 1 square kilometer area = 1,000,000 sq meters. 
        # Calculate how many pixels that is:
        self.pixels_in_1sqkm = 1_000_000 / self.pixel_area
        
        # Create a circular kernel approximating 1 sq km for the density test
        # Radius of a 1sqkm circle = sqrt(1000000 / pi) ≈ 564.19 meters
        self.walking_radius_m = np.sqrt(1_000_000 / np.pi)
        self.walking_radius_px = int(np.round(self.walking_radius_m / self.resolution))
        
        # Create the kernel
        self.density_kernel = self._create_circular_kernel(self.walking_radius_px)
        
        # 100m fringe open space buffer
        self.fringe_radius_px = int(np.round(100 / self.resolution))
        self.fringe_kernel = self._create_circular_kernel(self.fringe_radius_px)
        
        # 200m connection buffer (for clustering/merging parts)
        self.connect_radius_px = int(np.round(200 / self.resolution))
        self.connect_kernel = self._create_circular_kernel(self.connect_radius_px)

    def _create_circular_kernel(self, radius):
        """Creates a boolean circular kernel of a given radius in pixels."""
        y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
        mask = x**2 + y**2 <= radius**2
        return mask.astype(np.float32)

    def calculate_urban_extent(self, classified_image, built_up_class=1, open_space_class=3, water_class=2):
        """
        Runs the full AUE pipeline given a classified raster array.
        Returns a boolean mask of the final Urban Extent.
        """
        print("Starting AUE Density Test...")
        # 1. Pixel Classification Masks
        built_up_mask = (classified_image == built_up_class).astype(np.float32)
        open_space_mask = (classified_image == open_space_class).astype(np.float32)
        water_mask = (classified_image == water_class).astype(np.float32)

        # 2. Density Test (The Walking Distance Circle)
        # Calculate the percentage of built-up area within 1 sq km of every pixel
        # Convolve counts the number of built up pixels in the circular window
        built_up_counts = convolve(built_up_mask, self.density_kernel, mode='constant', cval=0.0)
        
        # Divide by total number of pixels in the 1 sq km circle
        circle_area_px = np.sum(self.density_kernel)
        density_percentage = built_up_counts / circle_area_px
        
        # Apply the AUE rules
        urban_built_up = (built_up_mask == 1) & (density_percentage > 0.50)
        suburban_built_up = (built_up_mask == 1) & (density_percentage >= 0.25) & (density_percentage <= 0.50)
        # Note: Rural pixels (<25%) are essentially ignored from here on out by not being in these arrays
        
        valid_built_up = urban_built_up | suburban_built_up

        print("Capturing Open Space...")
        # 3. Fringe Open Space
        # Pixels within 100m of an Urban/Suburban built-up pixel
        urban_dilation = binary_dilation(valid_built_up, structure=self.fringe_kernel)
        fringe_open_space = urban_dilation & (open_space_mask == 1)
        
        # 4. Captured Open Space
        # Open spaces completely surrounded by Built-up and Fringe Open Space.
        # This is equivalent to "filling holes" in the valid_built_up + fringe_open_space mask
        combined_cluster = valid_built_up | fringe_open_space
        
        from scipy.ndimage import binary_fill_holes
        # Fill holes
        filled_extent = binary_fill_holes(combined_cluster)
        
        # Anything that was filled and is open space is "Captured"
        # Water is NOT open space by AUE rules (a river isn't a park)
        # So we ensure the whole filled area is considered the true footprint
        
        print("Clustering and Merging...")
        # 5. Clustering and Merging (200m Contiguity Rule)
        # Dilation and closing to connect clusters separated by 200m or less.
        # "If two urban clusters are within 200 meters of each other... they are stitched together"
        final_morphological = binary_closing(filled_extent, structure=self.connect_kernel)
        
        # Return final mask
        return final_morphological

    def mask_to_shapefile(self, boolean_mask, transform, crs, output_path):
        """
        Converts the boolean raster mask into a vector shapefile polygon.
        """
        print("Vectorizing Urban Extent...")
        # Create shapes from mask
        # Shapes returns a tuple of (geometry, value)
        shapes = rasterio.features.shapes((boolean_mask).astype('uint8'), transform=transform)
        
        # Convert to shapely polygons
        polygons = []
        for geom, val in shapes:
            if val == 1: # Only want the "True" regions
                polygons.append(Polygon(geom['coordinates'][0]))
                
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
        
        # Clean up the output - keep only the largest cluster if there's noise,
        # or simplify it.
        if not gdf.empty:
            # We want to dissolve intersecting polygons if any didn't merge
            # and sort by area to focus on the MAIN city cluster
            
            # GEE exports usually in EPSG:4326, project to a local CRS to calculate area if needed
            # For simplicity, we just save the bounds.
            
            print(f"Writing shapefile to {output_path}...")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            gdf.to_file(output_path)
            print("Done.")
        else:
            print("No valid urban extent found.")
        return gdf
