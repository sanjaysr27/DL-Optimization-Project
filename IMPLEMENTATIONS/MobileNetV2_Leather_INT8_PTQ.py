# -*- coding: utf-8 -*-
"""
Post-Training Quantization (PTQ) — full INT8
Loads the saved leather-defect MobileNetV2 Keras model (FP32, produced by
the base-model training script — base_applications_MobileNetV2_Full_Train.keras)
and converts it to a full INT8 TFLite model using a calibration dataset,
then evaluates accuracy on the reconstructed held-out test split.

This is the same full-INT8 conversion approach (inference_input_type/
inference_output_type = int8, manual quantize/dequantize) as the Cat/Dog
INT8 script, retargeted at the leather CSIR dataset and the leather base
model path/preprocessing:
  - SAVED_BASE_MODEL_PATH points at your actual saved leather model.
  - DATASET_DIR / folder names / sample counts match the leather CSIR
    dataset (0_DefectFree / 1_Defective, 3992 each), not Cat/Dog.
  - SEED = 42 and decode_bmp + mobilenet_v2.preprocess_input, matching
    exactly what the base-model training script used.
  - Test split is reconstructed using the identical tf.data pipeline
    (Python shuffle -> tf.data.Dataset.shuffle -> take/skip) as the
    training script, so it's actually the same held-out images — a plain
    Python-list slice alone gives a different permutation than
    tf.data.Dataset.shuffle even with the same seed, silently pulling in
    images the model was trained on.

NOTE: TFLiteConverter.from_keras_model() converts whatever graph is in the
loaded model regardless of whether MobileNetV2 is nested as a single layer
(as in your base-model script, `x = MobileNetV2(...)( x)`) or flat — so no
layer-name resolution is needed here, unlike the pruning/dropout scripts.

FIXES APPLIED FOR PIPELINE-DRIVER COMPATIBILITY
--------------------------------------------------
1. The metrics block at the end unconditionally computed
   `correct / total * 100`. If every single test image failed to load
   (all skipped by the try/except in the eval loop), `total` stays 0 and
   this raises ZeroDivisionError, crashing the script after it already did
   all the work. Now guarded.
2. The stored `test_accuracy` metric used a 0-100 percentage scale, while
   every other stage in this pipeline (base training, channel pruning,
   dropout, layer pruning) stores Keras's native `evaluate()` accuracy,
   which is a 0-1 fraction. Mixing scales in the same results CSV column
   is easy to misread. This script now stores the 0-1 fraction in the
   metrics file (console output still prints the friendly percentage).
3. `test_loss` is no longer populated with the accuracy value — this
   script never computes an actual loss, so writing accuracy into a field
   called "test_loss" was misleading. Left blank instead.
"""

import os
import random
import numpy as np
import tensorflow as tf

# ==============================================================================
# 1. PATHS & CONFIGURATION
# ==============================================================================
DATASET_DIR           = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\DATASET\CSIR_Datasets"
SAVED_BASE_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\base_applications_MobileNetV2_Full_Train.keras"
QUANTIZED_MODEL_PATH  = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\MobileNetV2_quant_int8.tflite"

IMG_SIZE                = 224
SEED                    = 42
NUM_CALIBRATION_SAMPLES = 200

# ==============================================================================
# 2. LOAD THE SAVED BASE MODEL
# ==============================================================================
print("=" * 60)
print("Loading saved FP32 model...")
print("=" * 60)

model = tf.keras.models.load_model(SAVED_BASE_MODEL_PATH)
#model.summary()

# ==============================================================================
# 3. RECONSTRUCT DATASET SPLIT  (mirrors the base-model training script
#    EXACTLY, including the tf.data-level shuffle, so the test set matches)
# ==============================================================================
random.seed(SEED)

nonDef_dir = os.path.join(DATASET_DIR, "0_DefectFree")
def_dir    = os.path.join(DATASET_DIR, "1_Defective")

nonDef_files = [os.path.join(nonDef_dir, f) for f in os.listdir(nonDef_dir)]
def_files    = [os.path.join(def_dir, f) for f in os.listdir(def_dir)]

nonDef_files = random.sample(nonDef_files, 3992)
def_files    = random.sample(def_files, 3992)

selected_files  = nonDef_files + def_files
selected_labels = [0] * 3992 + [1] * 3992   # 0 = DefectFree, 1 = Defective

combined = list(zip(selected_files, selected_labels))
random.shuffle(combined)
selected_files, selected_labels = zip(*combined)

# Computed dynamically instead of hardcoded (3992 * 2 = 7984, not 8000).
dataset_size = len(selected_files)
train_size   = int(0.65 * dataset_size)
val_size     = int(0.15 * dataset_size)
test_size    = dataset_size - train_size - val_size

# Same tf.data pipeline as the training script — this second shuffle (on top
# of the Python-level one above) is what actually determines which files end
# up in train/val/test, so it must be reproduced exactly.
full_dataset = tf.data.Dataset.from_tensor_slices(
    (list(selected_files), list(selected_labels))
)
full_dataset = full_dataset.shuffle(dataset_size, seed=SEED, reshuffle_each_iteration=False)

train_ds  = full_dataset.take(train_size)
remaining = full_dataset.skip(train_size)
val_ds    = remaining.take(val_size)
test_ds   = remaining.skip(val_size)


def extract_paths_and_labels(ds, limit=None):
    paths, labels = [], []
    iterator = ds.take(limit) if limit is not None else ds
    for path_tensor, label_tensor in iterator:
        paths.append(path_tensor.numpy().decode("utf-8"))
        labels.append(int(label_tensor.numpy()))
    return paths, labels


test_files, test_labels = extract_paths_and_labels(test_ds)
# Calibration images come from the TRAIN split, never from test — using test
# images for calibration would leak test data into the quantization ranges.
calib_files, _ = extract_paths_and_labels(train_ds, limit=NUM_CALIBRATION_SAMPLES)

print(f"\n  Train split : {train_size} images")
print(f"  Val split   : {val_size} images")
print(f"  Test split  : {len(test_files)} images")
print(f"  Calib set   : {len(calib_files)} images (from train split)")


# ==============================================================================
# 4. IMAGE LOADING UTILITY
# ==============================================================================
def load_image(path: str) -> np.ndarray:
    """Read, decode, resize, and normalise one image exactly as the base
    model was trained (mobilenet_v2.preprocess_input, scales to [-1, 1])."""
    raw   = tf.io.read_file(path)
    image = tf.image.decode_bmp(raw, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image.numpy()


# ==============================================================================
# 5. REPRESENTATIVE DATASET GENERATOR
# ==============================================================================
def representative_dataset_gen():
    """
    Yields individual float32 batches of shape [1, H, W, C].
    The TFLite converter calls this to observe activation ranges
    across every layer — this drives INT8 calibration.
    """
    for fpath in calib_files:
        try:
            img = load_image(fpath)                   # (224, 224, 3)
            img = np.expand_dims(img, axis=0)         # (1, 224, 224, 3)
            yield [img.astype(np.float32)]
        except Exception as exc:
            print(f"  [SKIP] {os.path.basename(fpath)}: {exc}")
            continue


# ==============================================================================
# 6. INT8 POST-TRAINING QUANTIZATION
# ==============================================================================
print("\n" + "=" * 60)
print("Starting INT8 Post-Training Quantization...")
print("=" * 60)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# DEFAULT triggers quantization of both weights and activations
# using the representative dataset to calibrate activation ranges.
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen

# Force full INT8: every supported op runs in INT8.
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Set model I/O tensors to INT8 (eliminates float cast at the boundary).
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8

print("\nConverting model — this may take a few minutes...")
tflite_model_int8 = converter.convert()


# ==============================================================================
# 7. SAVE THE QUANTIZED MODEL
# ==============================================================================
with open(QUANTIZED_MODEL_PATH, "wb") as f:
    f.write(tflite_model_int8)

print(f"\nINT8 quantized model saved to:\n  {QUANTIZED_MODEL_PATH}")


# ==============================================================================
# 8. MODEL SIZE COMPARISON
# ==============================================================================
print("\n" + "=" * 60)
print("Model Size Comparison")
print("=" * 60)

original_size     = os.path.getsize(SAVED_BASE_MODEL_PATH) / (1024 ** 2)
quantized_size    = os.path.getsize(QUANTIZED_MODEL_PATH)  / (1024 ** 2)
reduction_pct     = (original_size - quantized_size) / original_size * 100
compression_ratio = original_size / quantized_size

print(f"  FP32 Keras model  : {original_size:.2f} MB")
print(f"  INT8 TFLite model : {quantized_size:.2f} MB")
print(f"  Size reduction    : {reduction_pct:.1f}%")
print(f"  Compression ratio : {compression_ratio:.2f}x smaller")


# ==============================================================================
# 9. INSPECT QUANTIZED MODEL TENSORS
# ==============================================================================
print("\n" + "=" * 60)
print("Quantized Model — Tensor Details")
print("=" * 60)

interpreter = tf.lite.Interpreter(model_path=QUANTIZED_MODEL_PATH)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nInput tensor:")
print(f"  Name         : {input_details[0]['name']}")
print(f"  Shape        : {input_details[0]['shape']}")
print(f"  Dtype        : {input_details[0]['dtype']}")
print(f"  Quantization : scale={input_details[0]['quantization'][0]:.6f}, "
      f"zero_point={input_details[0]['quantization'][1]}")

print("\nOutput tensor:")
print(f"  Name         : {output_details[0]['name']}")
print(f"  Shape        : {output_details[0]['shape']}")
print(f"  Dtype        : {output_details[0]['dtype']}")
print(f"  Quantization : scale={output_details[0]['quantization'][0]:.6f}, "
      f"zero_point={output_details[0]['quantization'][1]}")

input_scale       = input_details[0]['quantization'][0]
input_zero_point  = input_details[0]['quantization'][1]
output_scale      = output_details[0]['quantization'][0]
output_zero_point = output_details[0]['quantization'][1]


# ==============================================================================
# 10. EVALUATE ON THE RECONSTRUCTED TEST SET
# ==============================================================================
print("\n" + "=" * 60)
print(f"Evaluating INT8 model on {len(test_files)} test images...")
print("=" * 60 + "\n")

correct = 0
total   = 0

for fpath, true_label in zip(test_files, test_labels):
    try:
        img_f32 = load_image(fpath)                           # float32
        img_f32 = np.expand_dims(img_f32, axis=0)             # (1, 224, 224, 3)

        # Quantise input: float32 → int8
        # Formula: q = round(f / scale) + zero_point, clipped to [-128, 127]
        img_int8 = np.clip(
            np.round(img_f32 / input_scale) + input_zero_point,
            -128, 127
        ).astype(np.int8)

        interpreter.set_tensor(input_details[0]['index'], img_int8)
        interpreter.invoke()

        out_int8 = interpreter.get_tensor(output_details[0]['index'])  # int8

        # Dequantise output: int8 → float32
        out_f32    = (out_int8.astype(np.float32) - output_zero_point) * output_scale
        pred_label = 1 if out_f32[0][0] >= 0.5 else 0

        correct += int(pred_label == true_label)
        total   += 1

        if total % 100 == 0:
            print(f"  {total} / {len(test_files)} images evaluated "
                  f"— running accuracy: {correct / total * 100:.1f}%")

    except Exception as exc:
        print(f"  [SKIP] {os.path.basename(fpath)}: {exc}")
        continue

print("\n" + "=" * 60)
print("Final Results")
print("=" * 60)
print(f"  Total images evaluated : {total}")
print(f"  Correct predictions    : {correct}")
if total > 0:
    print(f"  Test Accuracy (INT8)   : {correct / total * 100:.2f}%")
else:
    print("  No images were successfully evaluated.")

# ==============================================================================
# 11. SAVE METRICS  (test_accuracy stored as a 0-1 fraction, matching the
#     scale used by every other stage in this pipeline. test_loss is left
#     blank since this script never computes an actual loss value.)
# ==============================================================================
import json
metrics = {
    "test_loss": "",
    "test_accuracy": float(correct / total) if total > 0 else "",
}
with open("last_run_metrics.json", "w") as f:
    json.dump(metrics, f)


print("\nINT8 quantization and evaluation complete.")