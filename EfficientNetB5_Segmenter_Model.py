from EfficientNet_Segmenters.EfficientNetB5_Segmenter import EfficientNetB5_Segmenter
from Pipeline import setup_callbacks, CustomMetricsCallback
import os
import numpy as np
import keras

class EfficientNetB5_Segmenter_Model():
    def __init__(self, config, identifier, tile_size, epochs, batch_size, lr, pre_weights, verbose, sat, size, experiment, city, results_dir, mode):
        self.tile_size = tile_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model_size = None
        self.verbose = verbose
        self.sat = sat
        self.size = size
        self.experiment = experiment
        self.weights_folder = results_dir + f"/{self.sat}/Results/{self.sat}_{self.size}/{self.experiment}/{city}/Weights/{identifier}"
        self.prediction_folder = results_dir + f"/{self.sat}/Results/{self.sat}_{self.size}/{self.experiment}/{city}/Preds/{identifier}"
        self.experiment_folder = results_dir + f"/{self.sat}/Results/{self.sat}_{self.size}/{self.experiment}"
        self.history_path = f"{self.weights_folder}/history.npy"
        if results_dir and results_dir != "None":
            os.makedirs(self.weights_folder, exist_ok=True)
        self.save_best_weights, self.reduce_lr = setup_callbacks(self.weights_folder, lr_patience=3, lr_min_delata=0.001)
        self.mode = mode
        self.cities = ["Medellin", "Lower", "Makoko", "ElDaein", "ElGeneina"]
        self.image_processor = None
        self.ENB5_Segmenter = EfficientNetB5_Segmenter(config)

    def build_model(self):
        self.ENB5_Segmenter.compile(
            optimizer = keras.optimizers.AdamW(self.lr),
            loss = keras.losses.CategoricalCrossentropy(from_logits=True),
        )
    
    def train_model(self, train_data, val_data):
        customMetricsCallback = CustomMetricsCallback(train_data, val_data, self.tile_size, channels_first=False)
        callbacks = [self.reduce_lr, self.save_best_weights, customMetricsCallback]
        
        history = self.ENB5_Segmenter.fit(
            train_data,
            validation_data = val_data,
            epochs = self.epochs, 
            batch_size = self.batch_size, 
            verbose = self.verbose,
            callbacks = callbacks
        )

        np.save(self.history_path, np.array([history.history]))
        last_weight_paths = self.weights_folder + "/last_weights.h5"
        self.ENB5_Segmenter.save_weights(last_weight_paths)