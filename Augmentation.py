from Exp_Helper import collect_split_tiles_paths
from exp_gen_data_pak import group_tiles
import numpy as np
import tensorflow as tf
import os

class Augmentation():
    def __init__(self, random_seed):
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

    def augment_class(self, class_type, class_data, max_tiles, aug_folder):
        if len(class_data) > 0:
            multiple = (max_tiles // len(class_data)) - 1
            last_multiple = max_tiles % len(class_data)
            print(f"class = {class_type} - {len(class_data)}")
            print(f"multiple = {multiple}")
            print(f"last_multiple = {last_multiple}")
            count = 0

            for count in range(multiple):
                print(f"round {count + 1}")
                count += self.apply_augmentation(class_data, aug_folder, round=count)
        
            if last_multiple > 0:
                print(f"round last_multiple")
                last_multiple_indices = np.random.choice(len(class_data), last_multiple, replace=False)
                last_multiple_data = np.array(class_data)[last_multiple_indices] #should give error if not np array
                count += self.apply_augmentation(last_multiple_data, aug_folder, round="last")
            print(f"Total Augments = {multiple * len(class_data) + last_multiple}")
            print()
        else:
            print(f"class = {class_type} has zero samples - can't augment")

    def apply_augmentation(self, data, aug_folder, round):
        for count, tile_path in enumerate(data):
            mask_path = tile_path.replace("Tiles", "Masks")
            mask_path = mask_path.replace("tile", "mask")
            tile = np.load(tile_path)
            mask = np.load(mask_path)
            aug_tile, aug_mask = self.augment_instance(tile, mask)
            aug_tile_path = aug_folder + "/Tiles/" + tile_path.split("/")[-1][:-4] + f"_aug_{round}.npy"
            aug_mask_path = aug_folder + "/Masks/" + mask_path.split("/")[-1][:-4] + f"_aug_{round}.npy"
            np.save(aug_tile_path, aug_tile)
            np.save(aug_mask_path, aug_mask)
        return count

    def balance_classes(self, train_split_path, aug_folder, thresh_precent):
        print()
        train_paths = collect_split_tiles_paths(train_split_path)
        slum_paths, non_slum_paths, mixed_paths = group_tiles(train_paths, thresh_precent)
        max_class_tiles = max([len(slum_paths), len(non_slum_paths), len(mixed_paths)])
        
        os.makedirs(aug_folder, exist_ok=True)
        os.makedirs(aug_folder + "/Tiles", exist_ok=True)
        os.makedirs(aug_folder + "/Masks", exist_ok=True)
        print(f"aug_folder = {aug_folder}")
        print(f"max_tiles = {max_class_tiles}")
        print()

        self.augment_class("slum", slum_paths, max_class_tiles, aug_folder)
        self.augment_class("non_slum", non_slum_paths, max_class_tiles, aug_folder)
        self.augment_class("mixed", mixed_paths, max_class_tiles, aug_folder)
        print()
        self.ensure_balance(aug_folder, train_paths, max_class_tiles, thresh_precent)
        print("-"*70+"\n")

    def collect_augment_paths(self, aug_folder):
        aug_path_array = []
        for i in os.listdir(aug_folder):
            temp_path = os.path.join(aug_folder, i).replace("\\", "/")
            aug_path_array.append(temp_path)
        return aug_path_array

    def ensure_balance(self, aug_folder, train_paths, max_class_tiles, thresh_precent):
        aug_paths = self.collect_augment_paths(aug_folder)
        balanced_paths = list(train_paths) + aug_paths
        print(f"After Augmentation")
        slum_paths, non_slum_paths, mixed_paths = group_tiles(balanced_paths, thresh_precent)

        if len(slum_paths) > 0 and len(slum_paths) != max_class_tiles:
            print("slum_paths not balanced")
        else:
            print("slum_paths balanced")
        if len(non_slum_paths) > 0 and len(non_slum_paths) != max_class_tiles:
            print("non_slum_paths not balanced")
        else:
            print("non_slum_paths balanced")
        if len(mixed_paths) > 0 and len(mixed_paths) != max_class_tiles:
            print("mixed_paths not balanced")
        else:
            print("mixed_paths balanced")
    
    def augment_instance(self, image, mask):
        mask = tf.expand_dims(mask, axis=-1)
        image, mask = self.random_flip(image, mask)
        image, mask = self.random_rotate(image, mask)
        image, mask = self.random_brightness(image, mask)
        image, mask = self.random_contrast(image, mask)
        image = image.numpy()
        mask = np.squeeze(mask.numpy())
        return image, mask

    def random_flip(self, image, mask):
        if np.random.choice([0, 1]):
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
        if np.random.choice([0, 1]):
            image = tf.image.flip_up_down(image)
            mask = tf.image.flip_up_down(mask)
        return image, mask

    def random_rotate(self, image, mask):
        times = [1, 2, 3] #1=90, 2=180, 3=270
        selected_time = np.random.choice(times)
        image = tf.image.rot90(image, k=selected_time)
        mask = tf.image.rot90(mask, k=selected_time)
        return image, mask

    def random_brightness(self, image, mask):
        image = tf.image.random_brightness(image, max_delta=0.2)
        return image, mask

    def random_contrast(self, image, mask):
        image = tf.image.random_contrast(image, lower=0.95, upper=1.1)
        return image, mask