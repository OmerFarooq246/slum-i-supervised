import os
from Exp_Helper import collect_super_tiles_paths, ensure_generation, find_duplicate, group_tiles_binary, print_city_details_binary, save_splits, slice_into_patches
from PIL import Image
import numpy as np
import math
from sklearn.model_selection import train_test_split
import cv2


def generate_super_tiles(size, sat_paths, splits_output_path, city, sat = "PAK", save = False):
    print(f"> Generating Super Tiles <")
    print()
    for count, pair in enumerate(sat_paths):
        raster_path = pair[0]
        mask_path = pair[1]

        output_path = splits_output_path
        print(f"city = {city}")
        print(f"raster_path = {raster_path}")
        print(f"mask_path = {mask_path}")
        print(f"output_path = {output_path}")

        tiles_folder = output_path + "/Tiles"
        masks_folder = output_path + "/Masks"
        os.makedirs(tiles_folder, exist_ok=True)
        os.makedirs(masks_folder, exist_ok=True)

        if sat == "DG":
            raster = cv2.imread(raster_path, cv2.IMREAD_UNCHANGED)
            raster = cv2.cvtColor(raster, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        else:
            Image.MAX_IMAGE_PIXELS = None
            raster = Image.open(raster_path).convert("RGB")
            mask = Image.open(mask_path)
            raster = np.array(raster)
            mask = np.array(mask)
        
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


def create_super_splits_city(city, splits_output_path, train_val_percent, train_percent, random_seed, save = False):
    print(f"> Generating Super Splits <")
    print(f"city = {city}")
    city_super_tiles_paths = collect_super_tiles_paths(splits_output_path)
    slum_paths, non_slum_paths = group_tiles_binary(city_super_tiles_paths)

    all_paths = slum_paths + non_slum_paths
    all_labels = [1]*len(slum_paths) + [0]*len(non_slum_paths)
    
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
    
    print_city_details_binary(city, train_paths, val_paths, test_paths)

    splits_dest_folder = splits_output_path + f"/splits"
    print(f"splits_dest_folder = {splits_dest_folder}")
    print("-"*70+"\n")
    if save:
        save_splits(splits_dest_folder, train_paths, val_paths, test_paths)
    return train_paths, val_paths, test_paths


def distribute_into_splits(train_split_path, val_split_path, test_split_path, splits_folder, save=False):
    super_train_paths = np.load(train_split_path)
    super_val_paths = np.load(val_split_path)
    super_test_paths = np.load(test_split_path)

    generate_tiles_on_split(super_train_paths, splits_folder, "train", save)
    generate_tiles_on_split(super_val_paths, splits_folder, "val", save)
    generate_tiles_on_split(super_test_paths, splits_folder, "test", save)


def generate_tiles_on_split(super_split_paths, splits_folder, split, save=False):
    output_folder = f"{splits_folder}/{split}"

    os.makedirs(f"{output_folder}/Tiles", exist_ok=True)
    os.makedirs(f"{output_folder}/Masks", exist_ok=True)
    print(f">> {split} <<")
    count_tiles = 0

    for super_tile_path in super_split_paths:
        super_mask_path = super_tile_path.replace("Tiles", "Masks")
        super_mask_path = super_mask_path.replace("tile", "mask")
        super_tile = np.load(super_tile_path)
        super_mask = np.load(super_mask_path)
        
        tile_id = super_tile_path.split("/")[-1]
        id = tile_id[tile_id.find("_") + 1:][:-4]
        output_tile_path = f"{output_folder}/Tiles/tile_{id}.npy"
        output_mask_path = f"{output_folder}/Masks/mask_{id}.npy"
        if save:
            np.save(output_tile_path, super_tile)
            np.save(output_mask_path, super_mask)
        
        count_tiles += 1

    ensure_generation(count_tiles, output_folder)


def create_paths_dg():
    paths_dict = {
        "medellin": {
            "raster": "/netscratch/mukhtar/ncode_slum_i_taha-main/patrick_data/Colombia/Medellin/DG/training/Medellin_40cm.tif",
            "mask": "/netscratch/mukhtar/ncode_slum_i_taha-main/patrick_data/Colombia/Medellin/DG/training/Medellin_ground_truth.tif",
        },
        "lower": {
            "raster": "/netscratch/mukhtar/ncode_slum_i_taha-main/patrick_data/Kenya/Nairobi/Lower/DG/training/Lower_30cm.tif",
            "mask": "/netscratch/mukhtar/ncode_slum_i_taha-main/patrick_data/Kenya/Nairobi/Lower/DG/training/Lower_30cm_ground_truth.tif",
        },
        "makoko": {
            "raster": "/netscratch/mukhtar/ncode_slum_i_taha-main/patrick_data/Nigeria/Makoko/DG/training/Makoko_50cm.tif",
            "mask": "/netscratch/mukhtar/ncode_slum_i_taha-main/patrick_data/Nigeria/Makoko/DG/training/Makoko_50cm_large_ground_truth.tif",
        },
        "eldaein": {
            "raster": "/netscratch/mukhtar/ncode_slum_i_taha-main/patrick_data/Sudan/ElDaein/DG/training/ElDaien_40cm.tif",
            "mask": "/netscratch/mukhtar/ncode_slum_i_taha-main/patrick_data/Sudan/ElDaein/DG/training/ElDaien_40cm_ground_truth.tif",
        },
        "elgeneina": {
            "raster": "/netscratch/mukhtar/ncode_slum_i_taha-main/patrick_data/Sudan/ElGeneina/DG/training/ElGeneina_40cm.tif",
            "mask": "/netscratch/mukhtar/ncode_slum_i_taha-main/patrick_data/Sudan/ElGeneina/DG/training/ElGeneina_40cm_ground_truth.tif",
        },
    }
    return paths_dict


def main():
    paths_dict = create_paths_dg()
    cities = list(paths_dict.keys())

    for city in cities:
        raster_path = paths_dict[city]["raster"]
        mask_path = paths_dict[city]["mask"]

        sat_paths = [(raster_path, mask_path)]
        
        train_val_percent = 0.8
        train_percent = 0.9
        random_seed = 42

        sat = "DG"

        sizes = [128, 64, 32]
        for size in sizes:
            splits_output_path = f"/netscratch/mukhtar/ncode_slum_i_taha-main/patrick_data_processed/{city}/{size}"
            train_split_path = f"{splits_output_path}/splits/train_split.npy"
            val_split_path = f"{splits_output_path}/splits/val_split.npy"
            test_split_path = f"{splits_output_path}/splits/test_split.npy"

            splits_folder = f"{splits_output_path}/splits"

            print(f"size: {size}")
            generate_super_tiles(size, sat_paths, splits_output_path, city, sat, save=True)
            create_super_splits_city(city, splits_output_path, train_val_percent, train_percent, random_seed, save=True)
            distribute_into_splits(train_split_path, val_split_path, test_split_path, splits_folder, save=True)

if __name__ == "__main__":
    main()