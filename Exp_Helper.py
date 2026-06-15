import os
import numpy as np

def save_splits(splits_dest_folder, train_paths, val_paths, test_paths):
    os.makedirs(splits_dest_folder, exist_ok=True)
    np.save(splits_dest_folder + "/train_split.npy", np.array(train_paths))
    np.save(splits_dest_folder + "/val_split.npy", np.array(val_paths))
    np.save(splits_dest_folder + "/test_split.npy", np.array(test_paths))


def print_city_details(city, train_paths, val_paths, test_paths, thresh_precent):
    total = len(train_paths) + len(val_paths) + len(test_paths)
    print(f"{city}_train_paths = {len(train_paths)} - {round(len(train_paths)/total, 2)}")
    group_tiles(train_paths, thresh_precent, debug = True)
    print(f"{city}_val_paths   = {len(val_paths)} - {round(len(val_paths)/total, 2)}")
    group_tiles(val_paths, thresh_precent, debug = True)
    print(f"{city}_test_paths  = {len(test_paths)} - {round(len(test_paths)/total, 2)}")
    group_tiles(test_paths, thresh_precent, debug = True)
    print(f"{city}_Total       = {total}")


def print_city_details_binary(city, train_paths, val_paths, test_paths):
    total = len(train_paths) + len(val_paths) + len(test_paths)

    print(f"{city}_train_paths = {len(train_paths)} - {round(len(train_paths)/total, 2)}")
    group_tiles_binary(train_paths, debug=True)

    print(f"{city}_val_paths   = {len(val_paths)} - {round(len(val_paths)/total, 2)}")
    group_tiles_binary(val_paths, debug=True)

    print(f"{city}_test_paths  = {len(test_paths)} - {round(len(test_paths)/total, 2)}")
    group_tiles_binary(test_paths, debug=True)

    print(f"{city}_Total       = {total}")


def find_duplicate(paths_array_1, paths_array_2, array_name_1, array_name_2):
    duplicate_paths = set(paths_array_1) & set(paths_array_2)
    if duplicate_paths:
        print(f"found duplicates between {array_name_1} and {array_name_2} = {duplicate_paths}")
        return duplicate_paths
    else:
        print(f"no duplicates found between {array_name_1} and {array_name_2}")


def group_tiles(tiles_path_array, thresh_precent, debug = False):
    if (len(tiles_path_array) == 0):
        if debug: print("empty array - no elements\n")
        else: print("empty array - no elements\n")
        return [], [], []
    else:
        slum_tiles_paths = []
        non_slum_tiles_paths = []
        mixed_tiles_paths = []

        for tile_path in tiles_path_array:
            mask_path = tile_path.replace("Tiles", "Masks")
            mask_path = mask_path.replace("tile", "mask")

            mask = np.load(mask_path)
            mask_max = np.max(mask)
            mask_min = np.min(mask)

            if mask_max == 0 and mask_min == 0:
                non_slum_tiles_paths.append(tile_path)
            elif mask_max == 1 and mask_min == 1:
                slum_tiles_paths.append(tile_path)
            else:
                if (np.sum(mask == 1) / mask.size) >= thresh_precent:
                    slum_tiles_paths.append(tile_path)
                elif (np.sum(mask == 0) / mask.size) >= thresh_precent:
                    non_slum_tiles_paths.append(tile_path)
                else:
                    mixed_tiles_paths.append(tile_path)
                
        total = len(slum_tiles_paths) + len(non_slum_tiles_paths) + len(mixed_tiles_paths)
        print(f"slum_tiles_paths     = {len(slum_tiles_paths)} - {round(len(slum_tiles_paths)/total, 2)}")
        print(f"non_slum_tiles_paths = {len(non_slum_tiles_paths)} - {round(len(non_slum_tiles_paths)/total, 2)}")
        print(f"mixed_tiles_paths    = {len(mixed_tiles_paths)} - {round(len(mixed_tiles_paths)/total, 2)}")
        print(f"Total                = {total}")
        print()
            
        if not debug: 
            return slum_tiles_paths, non_slum_tiles_paths, mixed_tiles_paths


def group_tiles_binary(tiles_path_array, debug=False):
    if len(tiles_path_array) == 0:
        print("empty array - no elements\n")
        return [], []

    slum_tiles_paths = []
    non_slum_tiles_paths = []

    for tile_path in tiles_path_array:
        mask_path = tile_path.replace("Tiles", "Masks")
        mask_path = mask_path.replace("tile", "mask")

        mask = np.load(mask_path)

        white_ratio = np.sum(mask == 1) / mask.size

        if white_ratio > 0.5:
            slum_tiles_paths.append(tile_path)
        else:
            non_slum_tiles_paths.append(tile_path)

    total = len(slum_tiles_paths) + len(non_slum_tiles_paths)

    print(f"slum_tiles_paths     = {len(slum_tiles_paths)} - {round(len(slum_tiles_paths)/total, 2)}")
    print(f"non_slum_tiles_paths = {len(non_slum_tiles_paths)} - {round(len(non_slum_tiles_paths)/total, 2)}")
    print(f"Total                = {total}")
    print()

    if not debug:
        return slum_tiles_paths, non_slum_tiles_paths


def collect_super_tiles_paths(splits_output_path):
    tiles_paths_array = []
    folder_path = splits_output_path + "/Tiles"
    for i in os.listdir(folder_path):
        temp_path = os.path.join(folder_path, i).replace("\\", "/")
        tiles_paths_array.append(temp_path)
    return tiles_paths_array


def slice_into_patches(patch_size, raster, mask, output_folder, save = False):
    raster_w = raster.shape[1]
    raster_h = raster.shape[0]
    count_tiles = 0
    
    for j in range(0, raster_h, patch_size):
        for i in range(0, raster_w, patch_size):
            i_temp = i
            j_temp = j
            term = ""

            if (raster_w - i) < patch_size :
                i_temp = raster_w - patch_size
                term = "_end"

            if (raster_h - j) < patch_size :
                j_temp = raster_h - patch_size
                term = "_end"

            count_tiles = count_tiles + 1
            
            tile_array = raster[j_temp:j_temp + patch_size, i_temp:i_temp + patch_size, :]
            mask_array = mask[j_temp:j_temp + patch_size, i_temp:i_temp + patch_size]
            
            output_path_tile_npy = output_folder + f"/Tiles/tile_{i_temp}_{j_temp}" + term + ".npy"
            output_path_mask_npy = output_folder + f"/Masks/mask_{i_temp}_{j_temp}" + term + ".npy"

            if save:
                np.save(output_path_tile_npy, tile_array.astype("uint8"))
                np.save(output_path_mask_npy, mask_array.astype("uint8"))
    return count_tiles


def ensure_generation(prev_count, output_path):
    folder_tiles = output_path + "/Tiles"
    folder_masks = output_path + "/Masks"
    count_tile = 0
    count_mask = 0
    for tile_path in os.listdir(folder_tiles):
        count_tile += 1
        path = os.path.join(folder_tiles, tile_path)
        if 0 in np.load(path).shape:
            print(f"Error = tile found with dimension zero")
            print(f"path = {path}")
    for mask_path in os.listdir(folder_masks):
        count_mask += 1
        path = os.path.join(folder_masks, mask_path)
        if 0 in np.load(path).shape:
            print(f"Error = mask found with dimension zero")
            print(f"path = {path}")
    print(f"tile and mask count = {count_tile} & {count_mask}")
    if count_tile == count_mask:
        if prev_count == count_tile:
            print("generation successful")
        else:
            print("runtime generation and folder does not match")
    else:
        print("masks and tiles in folder do not match")


def print_exp_info(
    main_dir,
    tile_size_weights,
    epochs,
    batch_size,
    lr,
    pre_weights,
    verbose,
    sat,
    size,
    experiment,
    train_city,
    mode,
    random_seed,
    cpus,
    channels_first,
    gray,
    normalize
):
    print(f"main_dir: {main_dir}")
    print(f"tile_size_weights: {tile_size_weights}")
    print(f"epochs: {epochs}")
    print(f"batch_size: {batch_size}")
    print(f"lr: {lr}")
    print(f"pre_weights: {pre_weights}")
    print(f"verbose: {verbose}")
    print(f"sat: {sat}")
    print(f"size: {size}")
    print(f"experiment: {experiment}")
    print(f"train_city: {train_city}")
    print(f"mode: {mode}")
    print(f"random_seed: {random_seed}")
    print(f"cpus: {cpus}")
    print(f"channels_first: {channels_first}")
    print(f"gray: {gray}")
    print(f"normalize: {normalize}")
    print()

def collect_split_tiles_paths(split_path):
    tiles_paths_array = []
    for i in os.listdir(split_path):
        temp_path = os.path.join(split_path, i).replace("\\", "/")
        tiles_paths_array.append(temp_path)
    return tiles_paths_array

def collect_augment_paths(aug_folder):
    aug_path_array = []
    for i in os.listdir(aug_folder):
        temp_path = os.path.join(aug_folder, i).replace("\\", "/")
        aug_path_array.append(temp_path)
    return aug_path_array

def load_train_val_paths(cities, augs, split_path_train, split_path_val, aug_folder, do_group_tiles, thresh_precent, return_groups, mode):
    train_data_paths = []
    aug_paths = []
    val_data_paths = []
    print("> loading train and val splits & augments<")
    for city in cities:
        print(f"city: {city}")
        temp_train = collect_split_tiles_paths(split_path_train)
        if augs: augments = collect_augment_paths(aug_folder)
        temp_val = collect_split_tiles_paths(split_path_val)
        print(f"train_split: {len(temp_train)}")
        if augs: print(f"augments: {len(augments)}")
        print(f"val_split: {len(temp_val)}")
        train_data_paths.extend(temp_train)
        if augs: aug_paths.extend(augments)
        val_data_paths.extend(temp_val)
        print()
    all_train_data_paths = train_data_paths + aug_paths
    print(f"Total train data: {len(train_data_paths)}")
    print(f"Total aug data: {len(aug_paths)}")
    print(f"Total val data: {len(val_data_paths)}")
    print()
    print("All Train(train + aug) data details")
    
    if do_group_tiles: 
        if mode == "small_tile_classification":
            slum_tiles_paths, non_slum_tiles_paths = group_tiles_binary(all_train_data_paths, debug = not return_groups)
            if return_groups:
                return slum_tiles_paths, non_slum_tiles_paths, val_data_paths
        
        else:
            slum_tiles_paths, non_slum_tiles_paths, mixed_tiles_paths = group_tiles(all_train_data_paths, thresh_precent, debug = not return_groups)
            if return_groups:
                return slum_tiles_paths, non_slum_tiles_paths, mixed_tiles_paths, val_data_paths
    
    return all_train_data_paths, val_data_paths


def load_train_val_paths_DG(cities, augs, do_group_tiles, thresh_precent, seed):
    train_data_paths = []
    aug_paths = []
    val_data_paths = []
    print("> loading train and val splits & augments<")
    for city in cities:
        print(f"city: {city}")
        split_path_train = f"/netscratch/mukhtar/ncode_slum_i_taha-main/patrick_data_processed/{city}/splits_{seed}/train/Tiles"
        split_path_val = f"/netscratch/mukhtar/ncode_slum_i_taha-main/patrick_data_processed/{city}/splits_{seed}/val/Tiles"
        aug_folder = f"/netscratch/mukhtar/ncode_slum_i_taha-main/patrick_data_processed/{city}/splits_{seed}/Augments/Tiles"

        temp_train = collect_split_tiles_paths(split_path_train)
        if augs: augments = collect_augment_paths(aug_folder)
        temp_val = collect_split_tiles_paths(split_path_val)
        print(f"train_split: {len(temp_train)}")
        if augs: print(f"augments: {len(augments)}")
        print(f"val_split: {len(temp_val)}")
        train_data_paths.extend(temp_train)
        if augs: aug_paths.extend(augments)
        val_data_paths.extend(temp_val)
        print()
    all_train_data_paths = train_data_paths + aug_paths
    print(f"Total train data: {len(train_data_paths)}")
    print(f"Total aug data: {len(aug_paths)}")
    print(f"Total val data: {len(val_data_paths)}")
    print()
    print("All Train(train + aug) data details")
    if do_group_tiles: group_tiles(all_train_data_paths, thresh_precent, debug = True)
    return all_train_data_paths, val_data_paths