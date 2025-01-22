import glob
import os

import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from shapely.geometry import Point

# ------------------------------------------------------------------------------
# USER-DEFINED PATHS
# ------------------------------------------------------------------------------
# 1) The path to your input CSV
csv_input = "/media/data/Datasets/Model_A+_Soil+Sentinel_v4_with_weather.csv"

# 2) Two directories containing T 2010 data for Theme 5 and Theme 6
#    (Adjust these paths to your actual folder structure)
t2010_directories = [
    "/media/data/Datasets/Crop/Theme 5 Actual Yields and Production/T/2010",
    "/media/data/Datasets/Crop/Theme 6 Yield and Production Gaps/T/2010"
]

# 3) Output Excel file
csv_output = "/media/data/Datasets/Model_A+_Soil+Sentinel_v4_with_weather+yield.csv"

# ------------------------------------------------------------------------------
# FUNCTION: SAMPLE RASTER AT GIVEN LAT, LON
# ------------------------------------------------------------------------------
def sample_raster(raster_path, lat, lon):
    """
    Open a GeoTIFF and sample the pixel value at given latitude/longitude.
    Assumes the raster is georeferenced. If the raster CRS is not EPSG:4326,
    we transform the coordinate accordingly.
    """
    with rasterio.open(raster_path) as src:
        raster_crs = CRS.from_wkt(src.crs.to_wkt())
        latlon_crs = CRS.from_epsg(4326)

        # If the raster is NOT in EPSG:4326, transform (lon, lat) to raster CRS
        if raster_crs != latlon_crs:
            transformer = Transformer.from_crs(latlon_crs, raster_crs, always_xy=True)
            x, y = transformer.transform(lon, lat)
        else:
            x, y = (lon, lat)

        # Sample the raster at the transformed coordinate
        # sample() returns an iterator of ndarrays (one per band).
        val = next(src.sample([(x, y)]))[0]

    return val

# ------------------------------------------------------------------------------
# MAIN SCRIPT
# ------------------------------------------------------------------------------
def main():
    # 1. Read original CSV into a DataFrame
    df = pd.read_csv(csv_input)

    # Ensure we have the necessary coordinate columns
    if "TH_LAT" not in df.columns or "TH_LONG" not in df.columns:
        raise ValueError("CSV must have columns named TH_LAT and TH_LONG.")

    # 2. Collect all *yld.tif and *qga.tif file paths from both directories
    tif_files = []
    for directory in t2010_directories:
        # Recursively search for all *yld.tif and *qga.tif
        tif_files.extend(glob.glob(os.path.join(directory, "*yld.tif")))
        tif_files.extend(glob.glob(os.path.join(directory, "*qga.tif")))

    # Sort the list
    tif_files = sorted(tif_files)

    # 3. For each qga.tif, sample the raster for each row in df
    #    We'll create one new column per TIFF.
    for tif_path in tif_files:
        # Generate a short column name from the file
        # e.g., "brl_2010_qga" from "brl_2010_qga.tif"
        base_name = os.path.splitext(os.path.basename(tif_path))[0]
        col_name = f"{base_name}_val"  # e.g. "brl_2010_qga_val"

        print(f"Sampling from: {tif_path} -> {col_name}")

        # Prepare a list for storing sampled values
        pixel_values = []

        # Loop through each coordinate in the DataFrame
        for idx, row in df.iterrows():
            lat = row["TH_LAT"]
            lon = row["TH_LONG"]
            val = sample_raster(tif_path, lat, lon)
            pixel_values.append(val)

        # Add the column to the DataFrame
        df[col_name] = pixel_values

    # 4. Write out the enriched DataFrame to an Excel file
    df.to_csv(csv_output, index=False)
    print(f"Enriched CSV file written to: {csv_output}")

# ------------------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
