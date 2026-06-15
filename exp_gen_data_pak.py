import os
import math
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Geospatial libraries (Essential for KML -> Raster conversion)
import geopandas as gpd
import rasterio
from rasterio import features

# ASSUMPTION: Your custom modules are available in the environment
# You must ensure these files (Exp_Helper.py and augmentation.py) are present.
from Exp_Helper import (
    collect_super_tiles_paths, ensure_generation, find_duplicate, 
    print_city_details, save_splits, slice_into_patches, group_tiles
)
from augmentation import Augmentation


# --- KML MASK GENERATION FUNCTION (FIXED FOR BIGTIFF) ---

def create_mask_from_kml(raster_path, kml_path, output_mask_path):
    """
    Reads KML, aligns CRS with the raster, rasterizes, and saves the mask 
    using Rasterio with BigTIFF support to avoid the 4GB limit.
    """
    print(f"> Generating Mask from KML <")
    print(f"Input KML: {kml_path}")
    print(f"Reference Raster: {raster_path}")
    
    # 1. Read the KML file
    try:
        gdf = gpd.read_file(kml_path, driver='KML')
    except Exception as e:
        print(f"KML standard load failed ({e}). Attempting generic read_file...")
        gdf = gpd.read_file(kml_path)

    if gdf.empty:
        raise ValueError("The KML file contains no geometries.")

    # 2. Open reference raster to get metadata (shape, CRS, transform)
    with rasterio.open(raster_path) as src:
        out_shape = src.shape
        transform = src.transform
        raster_crs = src.crs
        src_profile = src.profile
        print(f"Raster CRS: {raster_crs} | KML CRS: {gdf.crs}")

    # 3. Reproject Vector to match Raster CRS (CRITICAL STEP)
    if gdf.crs != raster_crs:
        print(f"Reprojecting vectors...")
        gdf = gdf.to_crs(raster_crs)

    # 4. Rasterize
    print("Rasterizing vectors...")
    mask_arr = features.rasterize(
        shapes=[(geom, 255) for geom in gdf.geometry], # Assign value 255 to all polygons
        out_shape=out_shape,
        transform=transform,
        fill=0,
        default_value=0, # Pixels outside geometries are 0 (non-slum)
        dtype='uint8'
    )

    # 5. Save using Rasterio (Guaranteed BigTIFF compatibility)
    out_meta = src_profile.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": out_shape[0],
        "width": out_shape[1],
        "count": 1,
        "dtype": "uint8",
        "compress": "lzw",
        "BIGTIFF": "IF_NEEDED"  # Solves the 4GB file limit issue
    })

    with rasterio.open(output_mask_path, "w", **out_meta) as dest:
        dest.write(mask_arr, 1) # Write the single channel array

    print(f"Mask saved to: {output_mask_path}")
    print("-"*70+"\n")
    
    return output_mask_path


# --- EXISTING PIPELINE FUNCTIONS (Minimal changes for consistency) ---

def generate_super_tiles(size, sat_paths, splits_output_path, save=False):
    print(f"> Generating Super Tiles <")
    print()
    for count, pair in enumerate(sat_paths):
        raster_path = pair[0]
        mask_path = pair[1]
        
        city = "Lahore" # Keeping hardcoded city name as per original script

        output_path = splits_output_path
        print(f"city = {city}")
        print(f"raster_path = {raster_path}")
        print(f"mask_path = {mask_path}")
        print(f"output_path = {output_path}")

        tiles_folder = output_path + "/Tiles"
        masks_folder = output_path + "/Masks"
        os.makedirs(tiles_folder, exist_ok=True)
        os.makedirs(masks_folder, exist_ok=True)

        Image.MAX_IMAGE_PIXELS = None
        
        # NOTE: This still loads the entire file into memory via PIL. 
        # Be aware of OOM errors for very large rasters.
        raster = Image.open(raster_path).convert("RGB")
        mask = Image.open(mask_path)
        
        raster = np.array(raster)
        mask = np.array(mask)
        
        # Ensure mask is 2D (H, W) if PIL read a multi-band TIFF/PNG
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
        
        # Ensure mask is binary (0 or 255) for consistent processing
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)

        print("Info = ")
        print(f"Raster_Size      : {raster.shape}")
        print(f"Tiles gen in x   : {math.ceil(raster.shape[1]/size)}")
        print(f"Tiles gen in y   : {math.ceil(raster.shape[0]/size)}")
        print(f"Total Tiles Gen  : {math.ceil(raster.shape[0]/size) * math.ceil(raster.shape[1]/size)}")

        count_tiles = slice_into_patches(size, raster, mask, output_path, save)

        print(f"tiles generated: {count_tiles}")
        ensure_generation(count_tiles, output_path)
        print("-"*70+"\n")
        print("-"*70+"\n")


def create_super_splits_city(city, splits_output_path, thresh_precent, train_val_percent, train_percent, random_seed, save=False):
    # [Rest of the logic remains unchanged from original]
    print(f"> Generating Super Splits <")
    # ... (omitted for brevity, assume original logic here)
    city_super_tiles_paths = collect_super_tiles_paths(splits_output_path)
    slum_paths, non_slum_paths, mixed_paths = group_tiles(city_super_tiles_paths, thresh_precent)

    all_paths = slum_paths + non_slum_paths + mixed_paths
    all_labels = [0]*len(slum_paths) + [1]*len(non_slum_paths) + [2]*len(mixed_paths)
    
    if (len(all_paths) == 1):
        test_paths = all_paths
        train_paths, val_paths = [], []
    else:
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            all_paths, 
            all_labels, 
            test_size = 1 - train_val_percent, 
            random_state = random_seed,
            stratify = all_labels
        )
        if len(train_val_paths) == 1:
            train_paths = train_val_paths
            val_paths = []
        elif (1 - train_percent)*len(train_val_paths) < len(set(train_val_labels)):
            val_paths = train_val_paths[:math.floor((1 - train_percent)*len(train_val_paths))]
            train_paths = train_val_paths[math.floor((1 - train_percent)*len(train_val_paths)):]
        else:
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                train_val_paths, 
                train_val_labels, 
                test_size = 1 - train_percent, 
                random_state = random_seed, 
                stratify = train_val_labels
            )

    print(f"train, val and test splits created")
    find_duplicate(train_paths, val_paths, "train_paths", "val_paths")
    find_duplicate(train_paths, test_paths, "train_paths", "test_paths")
    find_duplicate(val_paths, test_paths, "val_paths", "test_paths")
    print()
    
    print_city_details(city, train_paths, val_paths, test_paths, thresh_precent)

    splits_dest_folder = splits_output_path + f"/splits"
    print(f"splits_dest_folder = {splits_dest_folder}")
    print("-"*70+"\n")
    if save:
        save_splits(splits_dest_folder, train_paths, val_paths, test_paths)
    return train_paths, val_paths, test_paths


# --- MAIN EXECUTION ---

def main():
    # --- PATH DEFINITIONS (CRITICAL: Update these to match your system) ---
    base_dir = "/netscratch/mukhtar/Archive/tmp"
    lahore_raster_path = os.path.join(base_dir, "lahore.tif")
    lahore_kml_path = "/netscratch/mukhtar/DataExtraction/Lahore.kml" # Your KML file
    
    # Generated mask will be saved here
    generated_mask_path = os.path.join(base_dir, "lahore_generated_mask.tif")
    splits_output_path = "/netscratch/mukhtar/ncode_slum_i_taha-main"
    
    # Path assumptions for augmentation
    aug_folder = f"{splits_output_path}/Augments"
    train_split_path = f"{splits_output_path}/splits/train_split.npy" 
    
    # --- HYPERPARAMETERS ---
    thresh_precent = 0.9
    train_val_percent = 0.8
    train_percent = 0.9
    random_seed = 42

    # 1. Generate Mask from KML
    if os.path.exists(lahore_kml_path):
        create_mask_from_kml(lahore_raster_path, lahore_kml_path, generated_mask_path)
    else:
        raise FileNotFoundError(f"KML file not found at {lahore_kml_path}")

    # 2. Setup Paths for Tiling
    PAK_sat_paths = [(lahore_raster_path, generated_mask_path)]

    # 3. Execution Pipeline
    augmentation = Augmentation(random_seed)

    generate_super_tiles(512, PAK_sat_paths, splits_output_path, save=True)
    
    # NOTE: Set 'save=True' here to write the splits files needed for augmentation
    train_paths, val_paths, test_paths = create_super_splits_city(
        "Lahore", splits_output_path, thresh_precent, train_val_percent, 
        train_percent, random_seed, save=True # Changed to True
    )
    
    # 4. Augmentation
    if os.path.exists(train_split_path) or len(train_paths) > 0: 
        augmentation.balance_classes(train_split_path, aug_folder, thresh_precent)
    else:
        print("Skipping augmentation: No data in train split.")

if __name__ == "__main__":
    main()