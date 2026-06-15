#slum = 0, non_slum = 1, mixed = 2

import tensorflow as tf
import numpy as np
from transformers import SegformerImageProcessor

class Data_Loader():
    def __init__(self, random_seed, mode, train_data_paths, val_data_paths, cpus, batch_size, tile_size, channels_first, pre_weights, gray, normalize = True, thresh_precent = 0.9):
        self.random_seed = random_seed
        tf.random.set_seed(self.random_seed)
        self.mode = mode
        self.train_data_paths = list(train_data_paths)
        self.val_data_paths = list(val_data_paths)
        self.cpus = cpus
        self.batch_size = batch_size
        self.tile_size = tile_size
        self.channels_first = channels_first
        imgpros_weights = pre_weights
        if (pre_weights == None):
            imgpros_weights = "nvidia/mit-b5"
        self.image_processor = SegformerImageProcessor.from_pretrained(imgpros_weights)
        self.thresh_precent = thresh_precent
        self.gray = gray
        self.normalize = normalize

    def load_all_splits(self):
        train_data = self.load_data(self.train_data_paths, "train")
        val_data = self.load_data(self.val_data_paths, "val")
        return train_data, val_data

    def load_data(self, paths, split_type):
        data = tf.data.Dataset.from_tensor_slices(paths)
        data = data.shuffle(len(data))
        
        if split_type == "train":
            data = data.map(self.wrapper_training, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            data = data.map(self.wrapper, num_parallel_calls=tf.data.AUTOTUNE)
        
        data = data.batch(self.batch_size)
        data = data.prefetch(buffer_size=tf.data.AUTOTUNE) 
        return data


    def load_all_splits_classificaiton(self):
        train_data = self.load_data_classification(self.train_data_paths, "train")
        val_data = self.load_data_classification(self.val_data_paths, "val")
        return train_data, val_data

    def load_data_classification(self, paths, split_type):
        data = tf.data.Dataset.from_tensor_slices(paths)
        
        if split_type == "train":
            data = data.map(self.wrapper_training, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            data = data.map(self.wrapper, num_parallel_calls=tf.data.AUTOTUNE)
        
        data = data.cache()
        data = data.shuffle(len(data))
        data = data.batch(self.batch_size)
        data = data.prefetch(buffer_size=tf.data.AUTOTUNE)
        return data

    def wrapper(self, tile_path):
        tile, mask = tf.py_function(self.load_tile_mask, inp=[tile_path, False], Tout=[tf.float32, tf.float32])
        if self.channels_first:
            tile.set_shape([self.tile_size[2], self.tile_size[0], self.tile_size[1]])
            if self.mode == "classification":
                mask.set_shape([3,])
            elif self.mode == "small_tile_classification":
                mask.set_shape([2,])
            else:
                mask.set_shape([self.tile_size[0], self.tile_size[1]])

        else:
            tile.set_shape(self.tile_size)
            if self.mode == "classification":
                mask.set_shape([3,])
            elif self.mode == "small_tile_classification":
                mask.set_shape([2,])
            else:
                mask.set_shape([self.tile_size[0], self.tile_size[1], 2])

        return tile, mask

    def wrapper_training(self, tile_path):
        tile, mask = tf.py_function(self.load_tile_mask, inp=[tile_path, True], Tout=[tf.float32, tf.float32])
        if self.channels_first:
            tile.set_shape([self.tile_size[2], self.tile_size[0], self.tile_size[1]])
            if self.mode == "classification":
                mask.set_shape([3,])
            elif self.mode == "small_tile_classification":
                mask.set_shape([2,])
            else:
                mask.set_shape([self.tile_size[0], self.tile_size[1]])

        else:
            tile.set_shape(self.tile_size)
            if self.mode == "classification":
                mask.set_shape([3,])
            elif self.mode == "small_tile_classification":
                mask.set_shape([2,])
            else:
                mask.set_shape([self.tile_size[0], self.tile_size[1], 2])

        return tile, mask

    def load_tile_mask(self, tile_path, train_load):
        tile_path = tile_path.numpy().decode("utf-8")
        mask_path = tile_path.replace("Tiles", "Masks")
        mask_path = mask_path.replace("tile", "mask")
        tile = np.load(tile_path)
        mask = np.load(mask_path)

        if train_load:
            tile, mask = self.augment_train_data(tile, mask)

        if self.channels_first:
            if self.mode == "classification" or self.mode == "small_tile_classification":
                tile = self.image_processor(tile, mask, return_tensors="np", do_resize=False)["pixel_values"].squeeze()
                mask = self.return_class_hot_vector(mask, self.mode).astype("float32")
            else:
                encodes = self.image_processor(tile, mask, return_tensors="np")
                tile = encodes["pixel_values"].squeeze()
                mask = encodes["labels"].squeeze()
        else:
            if self.normalize: 
                tile = tile / 255.0
            if self.mode == "classification" or self.mode == "small_tile_classification":
                mask = self.return_class_hot_vector(mask, self.mode).astype("float32")
            else:
                mask = tf.one_hot(mask, depth = 2)

        if self.gray:
            tile = np.mean(tile, axis = 0)
            tile = np.stack((tile, tile, tile), axis=0)
        return (tile, mask)
    
    def return_class_hot_vector(self, mask, mode):
        mask_max = np.max(mask)
        mask_min = np.min(mask)
        class_hot_vector = None
        if mask_max == 0 and mask_min == 0:
            if mode == "classification":
                class_hot_vector = [0, 1, 0]
            elif mode == "small_tile_classification":
                class_hot_vector = [0, 1]
        elif mask_max == 1 and mask_min == 1:
            if mode == "classification":
                class_hot_vector = [1, 0, 0]
            elif mode == "small_tile_classification":
                class_hot_vector = [1, 0]
        else:
            if mode == "classification":
                if (np.sum(mask == 1) / mask.size) >= self.thresh_precent:
                    class_hot_vector = [1, 0, 0]
                elif (np.sum(mask == 0) / mask.size) >= self.thresh_precent:
                    class_hot_vector = [0, 1, 0]
                else:
                    class_hot_vector = [0, 0, 1]
            elif mode == "small_tile_classification":
                if (np.sum(mask == 1) / mask.size) >= 0.5:
                    class_hot_vector = [0, 1]
                else:
                    class_hot_vector = [1, 0]
        return np.array(class_hot_vector)
    
    def augment_train_data(self, tile, mask):
        # mask = tf.expand_dims(mask, axis = -1)
        # tile = tf.image.random_flip_left_right(tile)
        # mask = tf.image.random_flip_left_right(mask)
        # tile = tf.image.random_flip_up_down(tile)
        # mask = tf.image.random_flip_up_down(mask)
        # tile = tf.image.random_brightness(tile, max_delta=0.2)
        # tile = tf.image.random_contrast(tile, lower=0.7, upper=1.25)
        # mask = np.squeeze(mask)
        return tile, mask