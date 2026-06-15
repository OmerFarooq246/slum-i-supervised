from Pipeline import make_prediction_on_city, evaluate_model_on_city, apply_evalutaion_metrics
from classification.Efficient_Net_B5 import Efficient_Net_B5
from Data_Loader import Data_Loader
from Evaluation_Metrics import Evaluation_Metrics
from Exp_Helper import print_exp_info, load_train_val_paths, collect_split_tiles_paths
import tensorflow as tf
import os

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
    pre_weights = None
    verbose = 2
    sat = "DG"
    experiment = f"Binary_Classification_{size}"
    train_city = "all"
    mode = "small_tile_classification"
    random_seed = 42
    cpus = 2
    channels_first = False
    gray = False
    normalize = True
    print_exp_info(results_dir, tile_size_weights, epochs, batch_size, lr, pre_weights, verbose, sat, size, experiment, train_city, mode, random_seed, cpus, channels_first, gray, normalize)

    tf.random.set_seed(random_seed)

    train_val_percent = 0.8
    train_percent = 0.9

    evaluation_metrics = Evaluation_Metrics()

    cities = ["medellin", "lower", "makoko", "eldaein", "elgeneina"]

    all_train_paths = []
    all_val_paths = []
    all_test_dirs = {}

    for city in cities:
        print(f"> Collecting city: {city} <")
        print(f"size: {size}")

        split_path_train = f"/netscratch/mukhtar/ncode_slum_i_taha-main/patrick_data_processed/{city}/{size}/splits/train/Tiles"
        split_path_val = f"/netscratch/mukhtar/ncode_slum_i_taha-main/patrick_data_processed/{city}/{size}/splits/val/Tiles"
        split_path_test = f"/netscratch/mukhtar/ncode_slum_i_taha-main/patrick_data_processed/{city}/{size}/splits/test/Tiles"

        train_data_paths, val_data_paths = load_train_val_paths(
            [city],
            augs = False,
            split_path_train=split_path_train,
            split_path_val=split_path_val,
            aug_folder=None,
            do_group_tiles=False,
            thresh_precent=None,
            return_groups=False,
            mode=mode
        )
        print(f"train_data_paths: {len(train_data_paths)}")
        print(f"val_data_paths: {len(val_data_paths)}")

        all_train_paths.extend(train_data_paths)
        all_val_paths.extend(val_data_paths)
        all_test_dirs[city] = split_path_test
    
    print(f"all_train_paths: {len(all_train_paths)}")
    print(f"all_val_paths: {len(all_val_paths)}")
    print(f"all_test_dirs: {len(all_test_dirs.keys())}")

    data_loader = Data_Loader(
        random_seed,
        mode,
        all_train_paths,
        all_val_paths,
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

    ENB5 = Efficient_Net_B5(
        "Efficient_Net_B5",
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
        mode,
        normalize
    )
    ENB5.build_model()
    ENB5.train_model(train_data, val_data)
    print()

    print("> Making Predictions <")
    selected_weights_path = f"{ENB5.weights_folder}/best_weights.h5"
    print(f"weights_path: {selected_weights_path}")
    ENB5.efficient_net.load_weights(selected_weights_path)
    print()

    for city in cities:
        os.makedirs(f"{ENB5.prediction_folder}/{city}", exist_ok=True)
        split_path = all_test_dirs[city]
        print(f"city test_splits_dir: {split_path}")
        city_test_split_paths = collect_split_tiles_paths(split_path)
        print(f"test_split: {len(city_test_split_paths)}")
        make_prediction_on_city(ENB5.efficient_net, ENB5.image_processor, ENB5.prediction_folder, city_test_split_paths, city, channels_first, normalize)
        print()

    print("> Evalutaing Model <")
    all_test_masks = []
    all_preds = []
    all_test_split_paths = []
    for city in cities:
        print(f"city = {city}")
        print(f"prediction_folder: {ENB5.prediction_folder}")
        split_path = all_test_dirs[city]
        print(f"city test_splits_dir: {split_path}")
        test_masks, preds, test_split_paths = evaluate_model_on_city(None, mode, ENB5.prediction_folder, split_path, city, evaluation_metrics, data_loader, None, channels_first)
        all_test_masks.extend(test_masks)
        all_preds.extend(preds)
        all_test_split_paths.extend(test_split_paths)
    print(f"city = all")
    apply_evalutaion_metrics(mode, evaluation_metrics, data_loader, all_test_masks, all_preds, all_test_split_paths)

    del ENB5
    print("-"*70+"\n")