# Dataset Preparation & Patch Generation Pipeline

This directory contains data curation, preprocessing, and patch extraction utilities for compiling paired Landsat 9 datasets from raw USGS EarthExplorer GeoTIFFs.

---

## Workflow Overview

```
Raw Landsat 9 GeoTIFF Products (_B2, _B3, _B4, _B10, _MTL.xml)
       │
       ├──> 1. merge_rgb.py: Composite Bands 4 (R), 3 (G), 2 (B) to 30m RGB
       ├──> 2. downscale.py: Resample 30m RGB and Band 10 TIR to target 100m / 200m
       ├──> 3. create_patches.py: Extract co-registered 256x256 / 512x512 non-overlapping patches
       └──> 4. preprocess_metadata.py: Parse XML headers for solar angles & radiometric calibration
```

---

## Scripts Description

- **`driver.py`**: Automated end-to-end driver that iterates over all raw product folders, downscales the required spectral bands, and tiles them into paired patches.
- **`preprocess_metadata.py`**: Parses Landsat 9 `*_MTL.xml` metadata files to extract `radiance_mult_band_10`, `radiance_add_band_10`, `sun_elevation`, `sun_azimuth`, and center coordinates into `data/metadata_mapping.json`.
- **`create_output_zip.py`**: Bundles the processed patch hierarchy for easy cloud/remote distribution.
- **`scripts/download_mpc.py`**: Helper script to fetch Landsat 9 scenes directly via Microsoft Planetary Computer STAC API.
- **`scripts/downscale.py`**: GDAL-based spatial downscaler with box/average pooling.
- **`scripts/merge_rgb.py`**: Combines red, green, and blue GeoTIFF bands into a single 3-channel composite.
- **`scripts/create_patches.py`**: Slices full satellite swathes into co-registered 16-tile grids per scene.
