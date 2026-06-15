from Exp_Helper import collect_split_tiles_paths, group_tiles
import keras
import os
import numpy as np
import tensorflow as tf

def setup_callbacks(weights_folder, lr_patience, lr_min_delata, monitor='val_loss'):
    print(f"lr_patience: {lr_patience}")
    print(f"lr_min_delata: {lr_min_delata}")

    save_best_weights = keras.callbacks.ModelCheckpoint(
        filepath=f"{weights_folder}/best_weights.h5",
        monitor=monitor,
        save_best_only=True,       
        mode='min',
        save_weights_only=True,    
        verbose=1
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience = lr_patience,
        verbose=1,
        min_delta = lr_min_delata,
        cooldown=0
    )
    return save_best_weights, reduce_lr

def make_preds_folders(cities, prediction_folder):
    for city in cities:
        os.makedirs(f"{prediction_folder}/{city}", exist_ok=True)

#slum = 0, non_slum = 1, mixed = 2
def group_city_preds(city_preds_paths):
    slums = []
    non_slums = []
    mixed = []
    for pred_path in city_preds_paths:
        pred = int(tf.cast(tf.argmax(np.load(pred_path), axis=-1), tf.uint8)[0])
        if pred == 0:
            slums.append(pred_path)
        elif pred == 1:
            non_slums.append(pred_path)
        else:
            mixed.append(pred_path)
    print(f"slum_preds: {len(slums)}")
    print(f"non_slum_preds: {len(non_slums)}")
    print(f"mixed_preds: {len(mixed)}")
    return slums, non_slums, mixed

def make_predictions_on_make_predictions(
    model, 
    image_processor, 
    prediction_folder,
    cities,
    sat,
    size,
    paths_control,
    weights_folder,
    weights_type,
    split,
    channels_first,
    experiment,
    train_city,
    identifier,
    main_dir,
    normalize = True
):
    print("> Making Predictions on Predictions <")
    selected_weights_path = f"{weights_folder}/{weights_type}_weights.h5"
    model.load_weights(selected_weights_path)
    print(f"weights_path: {selected_weights_path}")
    print()
    make_preds_folders(cities, prediction_folder)
    for city in cities:
        print(f"city = {city}")
        city_preds_paths = paths_control.collect_preds_paths(sat, size, city, experiment, train_city, identifier)
        print(f"city_preds_paths: {len(city_preds_paths)}")
        slums_pred_paths, non_slums_pred_paths, mixed_pred_paths = group_city_preds(city_preds_paths)
        mixed_tiles_paths = []
        for path in mixed_pred_paths:
            city = path.split("/")[-2]
            tile_id = path.split("/")[-1]
            id = tile_id[tile_id.find("_") + 1:][:-4]
            temp_path = f"{main_dir}/{sat}/Datasets/{sat}_{size}/{city}/test/Tiles/tile_{id}.npy"
            mixed_tiles_paths.append(temp_path)
        print(f"pred_mixed_tiles_paths: {len(mixed_tiles_paths)}")
        make_prediction_on_city(model, image_processor, prediction_folder, mixed_tiles_paths, city, channels_first, normalize)
        print()

def make_predictions(model, image_processor, prediction_folder, cities, split_path, weights_folder, weights_type, split, channels_first, seed, normalize = True, sat = "PAK"):
    print("> Making Predictions <")
    selected_weights_path = f"{weights_folder}/{weights_type}_weights.h5"
    model.load_weights(selected_weights_path)
    print(f"weights_path: {selected_weights_path}")
    print()
    make_preds_folders(cities, prediction_folder)
    for city in cities:
        print(f"city = {city}")
        if sat == "DG" and seed:
            split_path = f"/netscratch/mukhtar/ncode_slum_i_taha-main/patrick_data_processed/{city}/splits_{seed}/test/Tiles"
        city_test_split_paths = collect_split_tiles_paths(split_path)
        print(f"{split}_split: {len(city_test_split_paths)}")
        make_prediction_on_city(model, image_processor, prediction_folder, city_test_split_paths, city, channels_first, normalize)
        print()

def make_prediction_on_city(model, image_processor, prediction_folder, city_test_split_paths, city, channels_first, normalize):
    print(f"prediction_folder: {prediction_folder}")
    print(f"city: {city}")
    for test_path in city_test_split_paths:
        test_tile = np.load(test_path)
        
        if image_processor != None:
            test_tile = image_processor(test_tile, return_tensors="np")["pixel_values"].squeeze()
        else:
            if normalize: test_tile = test_tile / 255.0

        if channels_first:
            pred = model.predict(tf.expand_dims(test_tile, axis = 0), verbose = 0)
            if isinstance(pred, dict) and "logits" in pred:
                pred = pred.logits
        else:
            pred = model.predict(tf.expand_dims(test_tile, axis = 0), verbose = 0)

        pred_id = test_path.split("/")[-1].replace("tile", "pred")
        pred_path = f"{prediction_folder}/{city}/{pred_id}"
        np.save(pred_path, pred)

def evaluate_model(thresh_precent, mode, prediction_folder, cities, split_path, evaluation_metrics, data_loader, split, channels_first, seed, sat = "PAK"):
    print("> Evalutaing Model <")
    print()
    all_test_masks = []
    all_preds = []
    all_test_split_paths = []
    for city in cities:
        print(f"city = {city}")
        print(f"prediction_folder: {prediction_folder}")
        if sat == "DG" and seed:
            split_path = f"/netscratch/mukhtar/ncode_slum_i_taha-main/patrick_data_processed/{city}/splits_{seed}/test/Tiles"
        test_masks, preds, test_split_paths = evaluate_model_on_city(thresh_precent, mode, prediction_folder, split_path, city, evaluation_metrics, data_loader, split, channels_first)
        all_test_masks.extend(test_masks)
        all_preds.extend(preds)
        all_test_split_paths.extend(test_split_paths)
    # print(f"city = all")
    # apply_evalutaion_metrics(mode, evaluation_metrics, data_loader, all_test_masks, all_preds, all_test_split_paths)

def evaluate_model_on_city(thresh_precent, mode, prediction_folder, split_path, city, evaluation_metrics, data_loader, split, channels_first):
    test_masks, preds, test_split_paths = load_city_testMasks_and_preds(mode, prediction_folder, split_path, data_loader, city, split, channels_first)
    # group_tiles(test_split_paths, thresh_precent, debug = True)
    apply_evalutaion_metrics(mode, evaluation_metrics, data_loader, test_masks, preds, test_split_paths)
    print("-"*70+"\n")
    return test_masks, preds, test_split_paths

def apply_evalutaion_metrics(mode, evaluation_metrics, data_loader, test_masks, preds, test_split_paths):
    if mode == "segmentation":
        all_f1 = evaluation_metrics.F1_Score(test_masks, preds, mode)
        all_pwas = evaluation_metrics.pixelwise_accuracy(test_masks, preds, mode)
        all_ious = evaluation_metrics.iou(test_masks, preds, mode)
    else:
        all_f1 = evaluation_metrics.F1_Score(test_masks, preds, mode)
        all_accuracy = evaluation_metrics.classification_accuracy(test_masks, preds, mode)
        class_report = evaluation_metrics.make_classification_report(test_masks, preds, mode)
        confusion_mtrx = evaluation_metrics.make_confusion_matrix(test_masks, preds, mode)
    
def load_city_testMasks_and_preds(mode, prediction_folder, split_path, data_loader, city, split, channels_first):
    print(f"loading test tiles from: {split_path}")
    test_split_paths = collect_split_tiles_paths(split_path)
    test_masks = []
    preds = []
    for test_path in test_split_paths:
        mask_path = test_path.replace("Tiles", "Masks")
        mask_path = mask_path.replace("tile", "mask")
        mask = np.load(mask_path)
        
        pred_id = test_path.split("/")[-1].replace("tile", "pred")
        pred_path = f"{prediction_folder}/{city}/{pred_id}"
        pred = np.load(pred_path)
        if mode == "segmentation":
            test_masks.append(mask)
            pred = np.squeeze(pred)
            if channels_first:
                pred = np.transpose(pred, (1, 2, 0))
            pred = tf.image.resize(pred, mask.shape[:2])
            pred = tf.cast(tf.argmax(pred, axis=-1), tf.uint8)
        else:
            hot_vector = tf.expand_dims(data_loader.return_class_hot_vector(mask, mode), axis = 0)
            hot_vector = tf.cast(tf.argmax(hot_vector, axis=-1), tf.uint8)[0]
            test_masks.append(hot_vector)
            pred = tf.cast(tf.argmax(pred, axis=-1), tf.uint8)[0]
        preds.append(pred)
    return test_masks, preds, test_split_paths


class CustomMetricsCallback(keras.callbacks.Callback):
    def __init__(self, train_data, val_data, tile_size, channels_first):
        super().__init__()
        self.tile_size = tile_size
        self.train_data = train_data
        self.val_data = val_data
        self.channels_first = channels_first
        self.slum_iou_fn = keras.metrics.IoU(num_classes=2, target_class_ids=[1], name="slum_iou")
        self.non_slum_iou_fn = keras.metrics.IoU(num_classes=2, target_class_ids=[0], name="non_slum_iou")
        self.f1_score_fn = keras.metrics.F1Score(name="f1_score", average="weighted")
        self.accuracy_fn = keras.metrics.Accuracy(name="pixel_accuracy")
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.compute_metrics(self.train_data, logs, "train")
        self.compute_metrics(self.val_data, logs, "val")
    
    def compute_metrics(self, data, logs, split_type):
        self.slum_iou_fn.reset_states()
        self.non_slum_iou_fn.reset_states()
        self.f1_score_fn.reset_states()
        for batch, (tiles, masks) in enumerate(data):
            if self.channels_first:
                logits = self.model.predict(tiles, verbose = 0)
                if isinstance(logits, dict) and "logits" in logits:
                    logits = logits.logits
                    logits = tf.transpose(logits, perm=[0, 2, 3, 1])
            else:
                logits = self.model.predict(tiles, verbose = 0)
                masks = tf.cast(tf.argmax(masks, axis=-1), tf.float32)
            logits = tf.image.resize(logits, self.tile_size[:2])
            pred_classes = tf.cast(tf.argmax(logits, axis=-1), tf.float32)
            flat_masks = tf.reshape(masks, (masks.shape[0], -1))
            flat_logits = tf.reshape(pred_classes, (pred_classes.shape[0], -1))

            self.slum_iou_fn.update_state(masks, pred_classes)
            self.non_slum_iou_fn.update_state(masks, pred_classes)
            self.f1_score_fn.update_state(flat_masks, flat_logits)
            self.accuracy_fn.update_state(flat_masks, flat_logits)
        logs[f'{split_type}_slum_iou']  = self.slum_iou_fn.result()
        logs[f'{split_type}_non_slum_iou']  = self.non_slum_iou_fn.result()
        logs[f'{split_type}_f1_score']  = self.f1_score_fn.result()
        logs[f'{split_type}_pixel_accuracy'] = self.accuracy_fn.result()