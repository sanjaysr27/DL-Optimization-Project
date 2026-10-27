# -*- coding: utf-8 -*-
"""
Post-Training Quantization (PTQ) — INT8
Loads the saved MobileNetV2 Keras model (FP32) and converts it to a
full INT8 TFLite model using a calibration dataset, then evaluates
accuracy on the reconstructed 1600-image test split.
"""

import os
import random
import numpy as np
import tensorflow as tf

# ==============================================================================
# 1. PATHS & CONFIGURATION
# ==============================================================================
DATASET_DIR             = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\DATASET\PetImages"
SAVED_BASE_MODEL_PATH   = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\mobilenetv2_base.keras"
QUANTIZED_MODEL_PATH    = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\mobilenetv2_int16.tflite"

IMG_SIZE                = 224
SEED                    = 42
DATASET_SIZE            = 8000
NUM_CALIBRATION_SAMPLES = 200

# ==============================================================================
# 2. LOAD THE SAVED BASE MODEL
# ==============================================================================
print("=" * 60)
print("Loading saved FP32 model...")
print("=" * 60)

model = tf.keras.models.load_model(SAVED_BASE_MODEL_PATH)
model.summary()

# ==============================================================================
# 3. RECONSTRUCT DATASET SPLIT & BUILD CALIBRATION LIST
# ==============================================================================
# Mirror the exact seed, sampling, and shuffle from the training script
# so that test_files/test_labels match the held-out test set used during training.

random.seed(SEED)

cats_dir = os.path.join(DATASET_DIR, "Cat")
dogs_dir = os.path.join(DATASET_DIR, "Dog")

cat_files_all = [os.path.join(cats_dir, f) for f in os.listdir(cats_dir)]
dog_files_all = [os.path.join(dogs_dir, f) for f in os.listdir(dogs_dir)]

# Same 4000-per-class sampling as the training script
cat_files = random.sample(cat_files_all, 4000)
dog_files = random.sample(dog_files_all, 4000)

selected_files  = cat_files + dog_files
selected_labels = [0] * 4000 + [1] * 4000   # 0 = Cat, 1 = Dog

combined = list(zip(selected_files, selected_labels))
random.shuffle(combined)
selected_files, selected_labels = zip(*combined)

# Mirror the exact same split sizes as the training script
train_size = int(0.65 * DATASET_SIZE)                   # 5200
val_size   = int(0.15 * DATASET_SIZE)                   # 1200
test_size  = DATASET_SIZE - train_size - val_size        # 1600

test_files  = list(selected_files[train_size + val_size:])
test_labels = list(selected_labels[train_size + val_size:])

# Use the first N samples from the TRAINING split for calibration.
# Never use test images for calibration — that leaks test data.
calib_files = list(selected_files[:NUM_CALIBRATION_SAMPLES])

print(f"\n  Train split : {train_size} images")
print(f"  Val split   : {val_size} images")
print(f"  Test split  : {test_size} images")
print(f"  Calib set   : {len(calib_files)} images (from train split)")


# ==============================================================================
# 4. IMAGE LOADING UTILITY
# ==============================================================================
def load_image(path: str) -> np.ndarray:
    """Read, decode, resize, and normalise one image to float32 in [0, 1]."""
    raw   = tf.io.read_file(path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
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

# Calibration dataset
converter.representative_dataset = representative_dataset_gen

# Force full INT8: every supported op runs in INT8.
converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
                                       tf.lite.OpsSet.TFLITE_BUILTINS
                                       ]

print("\nConverting model — this may take a few minutes...")
tflite_model_int16 = converter.convert()


# ==============================================================================
# 7. SAVE THE QUANTIZED MODEL
# ==============================================================================
with open(QUANTIZED_MODEL_PATH, "wb") as f:
    f.write(tflite_model_int16)

print(f"\nINT16 quantized model saved to:\n  {QUANTIZED_MODEL_PATH}")


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
print(f"  INT16 TFLite model : {quantized_size:.2f} MB")
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


print("\nOutput tensor:")
print(f"  Name         : {output_details[0]['name']}")
print(f"  Shape        : {output_details[0]['shape']}")
print(f"  Dtype        : {output_details[0]['dtype']}")





# ==============================================================================
# 10. EVALUATE ON FULL 1600-IMAGE TEST SET
# ==============================================================================
print("\n" + "=" * 60)
print(f"Evaluating INT16 model on {test_size} test images...")
print("=" * 60 + "\n")

correct = 0
total   = 0

for fpath, true_label in zip(test_files, test_labels):
    try:
        img_f32 = load_image(fpath)                           # float32
        img_f32 = np.expand_dims(img_f32, axis=0)             # (1, 224, 224, 3)


        interpreter.set_tensor(input_details[0]['index'], img_f32)
        interpreter.invoke()


        # Dequantise output: int8 → float32
        out_f32    = interpreter.get_tensor(output_details[0]['index'])
        pred_label = 1 if out_f32[0][0] >= 0.5 else 0

        correct += int(pred_label == true_label)
        total   += 1

        if total % 100 == 0:
            print(f"  {total} / {test_size} images evaluated "
                  f"— running accuracy: {correct / total * 100:.1f}%")

    except Exception as exc:
        print(f"  [SKIP] {os.path.basename(fpath)}: {exc}")
        continue

print("\n" + "=" * 60)
print("Final Results")
print("=" * 60)
print(f"  Total images evaluated : {total}")
print(f"  Correct predictions    : {correct}")
print(f"  Test Accuracy (INT16)   : {correct / total * 100:.2f}%")
print("\nINT16 quantization and evaluation complete.")