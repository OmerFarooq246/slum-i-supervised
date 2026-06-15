import numpy as np
from Pipeline import make_predictions, evaluate_model
from SegFormer_Model import SegFormer_Model
from Data_Loader import Data_Loader
from Evaluation_Metrics import Evaluation_Metrics
# from Classes.Create_Splits import Create_Splits
from Exp_Helper import print_exp_info, load_train_val_paths_DG
import tensorflow as tf
import random

results_dir = "/netscratch/mukhtar/ncode_slum_i_taha-main/results"
tile_size_weights = [512, 512, 3]
epochs = 15
batch_size = 16
lr = 0.00001
pre_weights = "nvidia/mit-b5"
verbose = 2
sat = "DG"
size = 512
experiment = "Segmentation_on_DG"
train_city = "bari bari"
mode = "segmentation"
random_seed = 42
cpus = 2
channels_first = True
gray = False
normalize = True
print_exp_info(results_dir, tile_size_weights, epochs, batch_size, lr, pre_weights, verbose, sat, size, experiment, train_city, mode, random_seed, cpus, channels_first, gray, normalize)

tf.random.set_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)

thresh_precent = 0.9
train_val_percent = 0.8
train_percent = 0.9

evaluation_metrics = Evaluation_Metrics()
cities = ["medellin", "lower", "makoko", "eldaein", "elgeneina"]
random_seeds = [42, 1337, 2023, 3407, 9999]
print(f"random_seeds: {random_seeds}")

for seed in random_seeds:
    print(f"seed: {seed}")

    all_train_data_paths, val_data_paths = load_train_val_paths_DG(
        cities,
        augs = True,
        do_group_tiles=False,
        thresh_precent=thresh_precent,
        seed=seed
    )

    data_loader = Data_Loader(random_seed, mode, all_train_data_paths, val_data_paths, cpus, batch_size, tile_size_weights, channels_first, pre_weights, gray, thresh_precent=thresh_precent)
    train_data, val_data = data_loader.load_all_splits()
    print()
    print(f"train_data by loader: {train_data.element_spec} - batches: {train_data.cardinality()}")
    print(f"val_data by loader: {val_data.element_spec} - batches: {val_data.cardinality()}")
    print()

    segformer = SegFormer_Model("SegFormer-mit-B5", tile_size_weights, epochs, batch_size, lr, pre_weights, verbose, sat, size, f"{experiment}_{seed}", train_city, results_dir, mode)
    segformer.build_model_segmentation()
    segformer.train_model(train_data, val_data)
    print()
    make_predictions(segformer.segformer, segformer.image_processor, segformer.prediction_folder, cities, None, segformer.weights_folder, "best", "test", channels_first, seed=seed, sat="DG")
    print()
    evaluate_model(thresh_precent, segformer.mode, segformer.prediction_folder, cities, None, evaluation_metrics, data_loader, "test", channels_first, seed=seed, sat="DG")

    del segformer
    del train_data
    del val_data
    del all_train_data_paths
    del val_data_paths
    print("-"*70+"\n")