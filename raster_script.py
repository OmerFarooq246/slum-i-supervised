# pip install leafmap rasterio matplotlib geopandas

# # Download the installer
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# # Run the installer
# bash Miniconda3-latest-Linux-x86_64.sh

# source ~/miniconda3/etc/profile.d/conda.sh

# # Create the environment
# conda create -n geo_env python=3.10 -c conda-forge -y

# # Activate it
# conda activate geo_env

# # Install the required packages
# conda install -c conda-forge leafmap rasterio geopandas matplotlib fiona -y

import os
import leafmap
import rasterio
import matplotlib.pyplot as plt
from rasterio.features import rasterize
import geopandas as gpd

KML_PATH = "/netscratch/mukhtar/ncode_slum_i_taha-main/karachi_data/karachi.kml"
# geotiff_path = "/netscratch/mukhtar/ncode_slum_i_taha-main/karachi_data/karachi.tif"
geotiff_path = "/netscratch/mukhtar/Archive/tmp/karachi.tif"
BBOX = [66.87592219999999, 24.792996800000008, 67.28321259999998, 25.052278200000003]

def gen_lahore_sat():
    # print("[+] Reading KML to determine BBOX...")
    # gdf = gpd.read_file(KML_PATH, driver='KML')

    # BBOX = list(gdf.total_bounds)
    # print(f"[+] Calculated BBOX: {BBOX}")

    # print("[+] Downloading satellite imagery...")
    # leafmap.map_tiles_to_geotiff(
    #     output=geotiff_path,
    #     bbox=BBOX,
    #     zoom=19,
    #     source="SATELLITE",
    #     overwrite=True,
    #     quiet=False,
    #     options=["BIGTIFF=YES"]
    # )
    print("[+] Creating binary mask...")
    with rasterio.open(geotiff_path) as src:
        transform = src.transform
        raster_crs = src.crs
        height, width = src.height, src.width

    gdf = gpd.read_file(KML_PATH, driver='KML')
    if gdf.crs != raster_crs:
        print(f"converting kml crs to raster crs")
        gdf = gdf.to_crs(raster_crs)

    gdf = gpd.read_file(KML_PATH, driver='KML').to_crs(raster_crs)
    shapes = [(geom, 1) for geom in gdf.geometry]

    mask = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=rasterio.uint8,
        all_touched=False
    )

    mask_path = "/netscratch/mukhtar/ncode_slum_i_taha-main/karachi_data/karachi_ground_truth.tif"
    with rasterio.open(
        mask_path,
        'w',
        driver='GTiff',
        width=width,
        height=height,
        count=1,
        dtype=rasterio.uint8,
        crs=raster_crs,
        transform=transform,
        nodata=0
    ) as dst:
        dst.write(mask, 1)

    print(f"[+] Pipeline complete!\nGeoTIFF: {geotiff_path}\nMask: {mask_path}")
    # plt.imshow(mask, cmap='gray')
    # plt.show()

if __name__ == "__main__":
    gen_lahore_sat()