from Pipeline import setup_callbacks
import os
import numpy as np
import keras
import keras.layers as Layers

class Efficient_Net_B5():
    def __init__(self, identifier, tile_size, epochs, batch_size, lr, pre_weights, verbose, sat, size, experiment, city, results_dir, mode, normalize):
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
        self.image_processor = None
        self.normalize = normalize

    def construct_model(self):
        efficient_net = keras.applications.EfficientNetB5(
            input_shape = self.tile_size,
            include_top = False,
            weights = "imagenet"
        )

        input = Layers.Input(shape = self.tile_size)
        x = efficient_net(input)
        x = Layers.GlobalAveragePooling2D()(x)
        x = Layers.Dropout(0.2)(x)
        output = Layers.Dense(2)(x)

        model = keras.models.Model(input, output, name = "efficient_net")
        return model

    def build_model(self):
        self.efficient_net = self.construct_model()
        self.efficient_net.compile(
            optimizer = keras.optimizers.AdamW(self.lr),
            loss = keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.CategoricalAccuracy(),
                keras.metrics.F1Score()
            ]
        )
    
    def train_model(self, train_data, val_data):
        callbacks = [self.reduce_lr, self.save_best_weights]
        
        history = self.efficient_net.fit(
            train_data,
            validation_data = val_data,
            epochs = self.epochs, 
            batch_size = self.batch_size, 
            verbose = self.verbose,
            callbacks = callbacks
        )

        np.save(self.history_path, np.array([history.history]))
        last_weight_paths = self.weights_folder + "/last_weights.h5"
        self.efficient_net.save_weights(last_weight_paths)