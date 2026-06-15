import numpy as np
from Pipeline import make_predictions, evaluate_model
from SegFormer_Model import SegFormer_Model
from Data_Loader import Data_Loader
from Evaluation_Metrics import Evaluation_Metrics
# from Classes.Create_Splits import Create_Splits
from Exp_Helper import print_exp_info, load_train_val_paths
import tensorflow as tf

sizes = [128, 64, 32]

for size in sizes:
    results_dir = "/netscratch/mukhtar/ncode_slum_i_taha-main/results"
    tile_size_weights = [size, size, 3]
    epochs = 15

    if size == 32:
        batch_size = 256
    elif size == 64:
        batch_size = 128
    elif size == 128:
        batch_size = 32
    
    lr = 0.00001
    pre_weights = "nvidia/mit-b5"
    verbose = 2
    sat = "PAK"
    # size = 512
    experiment = f"Binary_Classification_{size}"
    train_city = "bari bari"
    mode = "small_tile_classification"
    random_seed = 42
    cpus = 2
    channels_first = True
    gray = False
    normalize = True
    print_exp_info(results_dir, tile_size_weights, epochs, batch_size, lr, pre_weights, verbose, sat, size, experiment, train_city, mode, random_seed, cpus, channels_first, gray, normalize)

    tf.random.set_seed(random_seed)

    train_val_percent = 0.8
    train_percent = 0.9

    evaluation_metrics = Evaluation_Metrics()

    cities = ["Islamabad"]
    train_city = "Islamabad"
    print(f"> Training on city: {train_city} <")
    print(f"size: {size}")

    split_path_train = f"/netscratch/mukhtar/ncode_slum_i_taha-main/islamabad_data/{size}/splits/train/Tiles"
    split_path_val = f"/netscratch/mukhtar/ncode_slum_i_taha-main/islamabad_data/{size}/splits/val/Tiles"
    split_path_test = f"/netscratch/mukhtar/ncode_slum_i_taha-main/islamabad_data/{size}/splits/test/Tiles"


    slum_tiles_paths, non_slum_tiles_paths, val_data_paths = load_train_val_paths(
        cities,
        augs = False,
        split_path_train=split_path_train,
        split_path_val=split_path_val,
        aug_folder=None,
        do_group_tiles=True,
        thresh_precent=None,
        return_groups=True,
        mode=mode
    )

    min_len = len(slum_tiles_paths)
    if len(non_slum_tiles_paths) < len(slum_tiles_paths):
        min_len = len(non_slum_tiles_paths)

    slum_tiles_paths_trunc = slum_tiles_paths[:min_len]
    non_slum_tiles_paths_trunc = non_slum_tiles_paths[:min_len]

    all_train_data_paths = slum_tiles_paths_trunc + non_slum_tiles_paths_trunc
    print(f"all_train_data_paths: {len(all_train_data_paths)}")

    data_loader = Data_Loader(
        random_seed,
        mode,
        all_train_data_paths,
        val_data_paths,
        cpus,
        batch_size,
        tile_size_weights,
        channels_first,
        pre_weights,
        gray,
        thresh_precent=None
    )
    train_data, val_data = data_loader.load_all_splits_classificaiton()
    print()
    print(f"train_data by loader: {train_data.element_spec} - batches: {train_data.cardinality()}")
    print(f"val_data by loader: {val_data.element_spec} - batches: {val_data.cardinality()}")
    print()

    segformer = SegFormer_Model(
        "SegFormer-mit-B5",
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
        results_dir,
        mode
    )
    segformer.build_model_classification()
    segformer.train_model(train_data, val_data)
    print()
    make_predictions(
        segformer.segformer,
        segformer.image_processor,
        segformer.prediction_folder,
        cities,
        split_path_test,
        segformer.weights_folder,
        "best",
        "test",
        channels_first,
        seed=None
    )
    print()
    evaluate_model(
        None,
        segformer.mode,
        segformer.prediction_folder,
        cities,
        split_path_test,
        evaluation_metrics,
        data_loader,
        "test",
        channels_first,
        seed=None
    )
    
    del segformer
    del train_data
    del val_data
    del all_train_data_paths
    del val_data_paths
    print("-"*70+"\n")