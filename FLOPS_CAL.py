import numpy as np
import time
import tensorflow as tf
from SegFormer_Model import SegFormer_Model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import keras
from EfficientNet_Segmenters.Config import EfficientNet_Segmenter_Config
from EfficientNetB5_Segmenter_Model import EfficientNetB5_Segmenter_Model
import platform
import subprocess
import os

def measure_latency(model, input_shape=(1, 512, 512, 3), iterations=100, warmup_iterations=20):
    # 1. Prepare dummy data
    dummy_input = tf.random.uniform(input_shape)
    
    # 2. WARM-UP 
    # This triggers JIT compilation and hardware cache initialization
    print("Warming up...")
    for _ in range(warmup_iterations):
        _ = model(dummy_input, training=False)
    
    # 3. MEASUREMENT LOOP
    print("Measuring...")
    latencies = []
    for _ in range(iterations):
        start_time = time.perf_counter()
        
        # The actual inference
        _ = model(dummy_input, training=False)
        
        # GPU Synchronization (Crucial for Accurate Research)
        # Note: In pure TF2, model() calls are usually synchronous on the host side, 
        # but to be safe, we ensure the device has finished.
        if isinstance(_, (list, tuple)):
            _ = _[0].numpy()
        elif isinstance(_, dict):
            _ = next(iter(_.values())).numpy()
        else:
            _ = _.numpy()
        
        end_time = time.perf_counter()
        latencies.append(end_time - start_time)
    
    # 4. STATISTICAL ANALYSIS
    mean_latency = np.mean(latencies) * 1000  # Convert to ms
    std_latency = np.std(latencies) * 1000
    fps = 1000 / mean_latency
    
    print(f"Latency: {mean_latency:.2f} ms ± {std_latency:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")
    print()
    return mean_latency


def get_flops(model, input_res=512, channels_first=True):
    # 1. Determine input shape and signature based on format
    if channels_first:
        # SegFormer/HF format: [Batch, Channels, Height, Width]
        input_shape = [1, 3, input_res, input_res]
        spec = tf.TensorSpec(input_shape, tf.float32, name="pixel_values")
        # HF models often require the specific key 'pixel_values'
        concrete_func = tf.function(lambda x: model(pixel_values=x, training=False)).get_concrete_function(spec)
    else:
        # Standard Keras format: [Batch, Height, Width, Channels]
        input_shape = [1, input_res, input_res, 3]
        spec = tf.TensorSpec(input_shape, tf.float32, name="inputs")
        concrete_func = tf.function(lambda x: model(x, training=False)).get_concrete_function(spec)

    # 2. Freeze the graph
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(concrete_func)
    
    # 3. Use the Profiler to count FLOPs
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    
    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, 
        run_meta=run_meta, 
        cmd='op', 
        options=opts
    )
    
    return flops.total_float_ops


def measure_segformer():
    tile_size_weights = [512, 512, 3]
    epochs = 0
    batch_size = 0
    lr = 0.00001
    pre_weights = "nvidia/mit-b5"
    verbose = 2
    sat = "PAK"
    size = 512
    experiment = "measurement"
    train_city = "None"
    mode = "segmentation"
    random_seed = 42
    results_dir = "None"

    segformer = SegFormer_Model("SegFormer-mit-B5", tile_size_weights, epochs, batch_size, lr, pre_weights, verbose, sat, size, experiment, train_city, results_dir, mode)
    segformer.build_model_segmentation()
    segformer.segformer.summary()
    print()

    measure_latency(segformer.segformer, input_shape=(1, 3, 512, 512))

    flops = get_flops(segformer.segformer, input_res=512, channels_first=True)
    print(f"SegFormer FLOPs: {flops / 10**9:.2f} GFLOPs")


def measure_enb5_seg():
    tile_size_weights = [512, 512, 3]
    epochs = 0
    batch_size = 0
    lr = 0.00001
    pre_weights = None
    verbose = 2
    sat = "PAK"
    size = 512
    experiment = "measurement"
    train_city = "None"
    mode = "segmentation"
    random_seed = 42
    results_dir = "None"

    config = EfficientNet_Segmenter_Config()

    ENB5_Segmenter = EfficientNetB5_Segmenter_Model(
        config, 
        "EfficientNetB5_Segmenter", 
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
    ENB5_Segmenter.build_model()
    ENB5_Segmenter.ENB5_Segmenter.build((None,512,512,3))
    ENB5_Segmenter.ENB5_Segmenter.summary()
    print()

    measure_latency(ENB5_Segmenter.ENB5_Segmenter, input_shape=(1, 512, 512, 3))

    flops = get_flops(ENB5_Segmenter.ENB5_Segmenter, input_res=512, channels_first=False)
    print(f"ENB5_Segmenter FLOPs: {flops / 10**9:.2f} GFLOPs")


def log_hardware_info():
    print("="*30 + " HARDWARE INFO " + "="*30)
    
    # 1. Get CPU Info
    try:
        if platform.system() == "Linux":
            cpu_info = subprocess.check_output("lscpu | grep 'Model name'", shell=True).decode().split(":")[1].strip()
            cores = subprocess.check_output("nproc", shell=True).decode().strip()
            print(f"CPU Model: {cpu_info}")
            print(f"Available CPU Cores: {cores}")
        else:
            print(f"Processor: {platform.processor()}")
    except Exception as e:
        print(f"Could not retrieve CPU info: {e}")

    # 2. Get GPU Info (TensorFlow Way)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            print(f"GPU Found: {gpu.name}")
            # Try to get specific model name via nvidia-smi
            try:
                gpu_name = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"], 
                    encoding='utf-8'
                ).strip()
                print(f"GPU Model: {gpu_name}")
            except:
                pass
    else:
        print("GPU: No GPU detected. Running on CPU.")

    # 3. Get OS and Software Info
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"TensorFlow Version: {tf.__version__}")
    print("="*75 + "\n")


if __name__ == "__main__":
    print("-"*70)
    log_hardware_info()
    print("-"*70)
    # measure_segformer()
    measure_enb5_seg()