from transformers import TFSegformerForSemanticSegmentation, TFSegformerForImageClassification, SegformerImageProcessor
from Pipeline import setup_callbacks, CustomMetricsCallback
import tensorflow as tf
import os
import numpy as np
import keras
from transformers import SegformerConfig

class SegFormer_Model():
    def __init__(self, identifier, tile_size, epochs, batch_size, lr, pre_weights, verbose, sat, size, experiment, city, results_dir, mode, lr_patience=4, lr_min_delata=0.0001):
        self.tile_size = tile_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model_size = pre_weights
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
        self.save_best_weights, self.reduce_lr = setup_callbacks(self.weights_folder, lr_patience=lr_patience, lr_min_delata=lr_min_delata)
        self.mode = mode
        self.cities = ["Medellin", "Lower", "Makoko", "ElDaein", "ElGeneina"]
        imgpros_weights = pre_weights
        if (pre_weights == None):
            imgpros_weights = "nvidia/mit-b5"
        self.image_processor = SegformerImageProcessor.from_pretrained(imgpros_weights)

    def build_model_segmentation(self):
        id2label = {0: "non_slum", 1: "slum"}
        label2id = {label: id for id, label in id2label.items()}
        self.segformer = TFSegformerForSemanticSegmentation.from_pretrained(
            self.model_size,
            num_labels = 2,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )

        self.segformer.compile(
            optimizer = keras.optimizers.AdamW(self.lr)
        )

    #uses mit-b5 config
    def build_model_segmentation_scratch(self, config):

        # config = SegformerConfig(
        #     num_labels=2,
        #     id2label={0: "non_slum", 1: "slum"},
        #     label2id={"non_slum": 0, "slum": 1},
        #     depths=[3, 6, 40, 3],
        #     hidden_sizes=[64, 128, 320, 512],
        #     decoder_hidden_size=768,
        # )
        self.segformer = TFSegformerForSemanticSegmentation(config)
        self.segformer.compile(
            optimizer = keras.optimizers.AdamW(self.lr)
        )

    def get_segmenter_encoder_weights(self, old_weights):
        id2label = {0: "non_slum", 1: "slum"}
        label2id = {label: id for id, label in id2label.items()}
        old_segformer = TFSegformerForSemanticSegmentation.from_pretrained(
            self.model_size,
            num_labels = 2,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        old_segformer.load_weights(old_weights)
        encoder_weights = old_segformer.layers[0].get_weights()
        del old_segformer
        return encoder_weights

    def build_model_classification(self, old_weights = None):        
        if self.mode == "classification":
            labels = 3
        elif self.mode == "small_tile_classification":
            labels = 2
        self.segformer = TFSegformerForImageClassification.from_pretrained(
            self.model_size, 
            num_labels = labels, 
            ignore_mismatched_sizes=True
        )

        if old_weights:
            print()
            print(f"setting older encoder weights from: {old_weights}")
            print()
            encoder_weights = self.get_segmenter_encoder_weights(old_weights)
            self.segformer.layers[0].set_weights(encoder_weights)
            self.segformer.layers[0].trainable = False
            for layer in self.segformer.layers:
                print(f"{layer.name}.trainable: {layer.trainable}")
            print()

        self.segformer.compile(
            optimizer = keras.optimizers.AdamW(self.lr),
            loss = keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.CategoricalAccuracy(),
                keras.metrics.F1Score()
            ]
        )
    
    def train_model(self, train_data, val_data):
        callbacks = []
        if self.mode == "segmentation":
            customMetricsCallback = CustomMetricsCallback(train_data, val_data, self.tile_size, channels_first=True)
            callbacks = [self.reduce_lr, self.save_best_weights, customMetricsCallback]
        elif self.mode == "classification" or self.mode == "small_tile_classification":
            callbacks = [self.reduce_lr, self.save_best_weights]
        
        history = self.segformer.fit(
            train_data,
            validation_data = val_data,
            epochs = self.epochs, 
            batch_size = self.batch_size, 
            verbose = self.verbose,
            callbacks = callbacks
        )

        np.save(self.history_path, np.array([history.history]))
        last_weight_paths = self.weights_folder + "/last_weights.h5"
        self.segformer.save_weights(last_weight_paths)