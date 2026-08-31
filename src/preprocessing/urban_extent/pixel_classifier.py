import ee
import geemap
import numpy as np
import rasterio
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class PixelClassifier:
    def __init__(self, project_id=None):
        """
        Initializes the Earth Engine API.
        Requires prior authentication via `earthengine authenticate`.
        """
        try:
            if project_id:
                ee.Initialize(project=project_id)
            else:
                ee.Initialize()
            print("Earth Engine initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize Earth Engine: {e}")
            print("Please run 'earthengine authenticate' in your terminal.")
            raise

    def get_landsat_composite(self, roi_geometry, start_date='2023-01-01', end_date='2023-12-31'):
        """
        Fetches a cloud-free composite of Landsat 9 Level-2 Surface Reflectance 
        for a given Region of Interest (ROI).
        """
        # Load Landsat 9 Surface Reflectance collection
        l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        
        # Filter by bounds, date, and cloud cover
        collection = (l9
            .filterBounds(roi_geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUD_COVER', 10)))
            
        print(f"Found {collection.size().getInfo()} images for composite.")

        # Create a median composite (reduces remaining clouds/shadows)
        # Apply scaling factors for Surface Reflectance
        def apply_scale_factors(image):
            opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
            thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
            return image.addBands(opticalBands, None, True).addBands(thermalBands, None, True)

        scaled_collection = collection.map(apply_scale_factors)
        composite = scaled_collection.median().clip(roi_geometry)
        
        # Select the relevant bands for land cover classification
        # B2: Blue, B3: Green, B4: Red, B5: NIR, B6: SWIR1, B7: SWIR2
        bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
        return composite.select(bands)

    def extract_training_data(self, image, polygons_fc, label_col='class'):
        """
        Extracts spectral signatures (pixels) from the image using labeled polygons.
        polygons_fc: ee.FeatureCollection containing training areas.
        """
        # Sample the image at the locations of the polygons
        # 30m is the native resolution of Landsat
        training_samples = image.sampleRegions(
            collection=polygons_fc,
            properties=[label_col],
            scale=30,
            tileScale=16 # Helps prevent memory errors on GEE side
        )
        return training_samples

    def train_random_forest(self, training_samples, label_col='class', num_trees=100):
        """
        Trains a Random Forest classifier directly on Google Earth Engine servers.
        Classes: 1 (Built-up), 2 (Water), 3 (Open Space)
        """
        print(f"Training Random Forest on Earth Engine with {num_trees} trees...")
        
        # The list of band names used for training
        bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
        
        # Instantiate the classifier
        classifier = ee.Classifier.smileRandomForest(num_trees).train(
            features=training_samples,
            classProperty=label_col,
            inputProperties=bands
        )
        
        # Optional: Get training accuracy to verify it learned something
        train_accuracy = classifier.confusionMatrix().accuracy().getInfo()
        print(f"GEE Training Accuracy: {train_accuracy:.4f}")
        
        return classifier

    def classify_image(self, image, classifier):
        """
        Applies the trained classifier to the full Landsat image.
        Returns a single-band image where pixel values represent the class ID.
        """
        print("Applying classifier to image...")
        classified = image.classify(classifier)
        return classified

    def export_to_geotiff(self, ee_image, roi_geometry, output_path, scale=30):
        """
        Downloads the classified Earth Engine image as a local GeoTIFF.
        Because limits apply to direct downloads, we use geemap or direct REST URL 
        depending on size, but for a city ROI, geemap's ee_export_image is robust.
        """
        print(f"Exporting classified raster to {output_path}...")
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        
        try:
            geemap.ee_export_image(
                ee_image, 
                filename=output_path, 
                scale=scale, 
                region=roi_geometry, 
                file_per_band=False
            )
            print("Export complete.")
        except Exception as e:
            print(f"Export failed: {e}")
            raise
