# -*- coding: utf-8 -*-
"""
VGG16 — INT8 Post-Training Quantization (PTQ) + Fast Evaluation — Leather
==========================================================================

"""

import os
import random
import numpy as np
import tensorflow as tf

# ==============================================================================
# 1.  CONFIGURATION
# ==============================================================================
SAVED_BASE_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\base_applications_VGG16_Full_Train.keras"
QUANTIZED_MODEL_PATH  = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\vgg16_quant_int8.tflite"
DATASET_DIR           = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\DATASET\CSIR_Datasets"

IMG_SIZE    = 224
SEED        = 42
EVAL_BATCH  = 16
NUM_THREADS = os.cpu_count() or 4
N_PER_CLASS = 3992

# ==============================================================================
# 2.  DATASET  (leather defect / non-defect — CSIR_Datasets, same structure
#     as the rest of this pipeline)
# ==============================================================================
random.seed(SEED)

nonDef_dir = os.path.join(DATASET_DIR, "0_DefectFree")
def_dir    = os.path.join(DATASET_DIR, "1_Defective")

nonDef_files = [os.path.join(nonDef_dir, f) for f in os.listdir(nonDef_dir)]
def_files    = [os.path.join(def_dir, f) for f in os.listdir(def_dir)]

assert len(nonDef_files) >= N_PER_CLASS, f"Only {len(nonDef_files)} defect-free images found, need {N_PER_CLASS}"
assert len(def_files) >= N_PER_CLASS, f"Only {len(def_files)} defective images found, need {N_PER_CLASS}"

nonDef_files = random.sample(nonDef_files, N_PER_CLASS)
def_files    = random.sample(def_files, N_PER_CLASS)

selected_files  = nonDef_files + def_files
selected_labels = [0] * N_PER_CLASS + [1] * N_PER_CLASS   # 0=non-defective, 1=defective

combined = list(zip(selected_files, selected_labels))
random.shuffle(combined)
selected_files, selected_labels = zip(*combined)

dataset_size = len(selected_files)
train_size   = int(0.65 * dataset_size)
val_size     = int(0.15 * dataset_size)
test_size    = dataset_size - train_size - val_size


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_bmp(image, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    # >>> Must match the preprocessing used by the VGG16 base training
    # >>> script. Change this line if that script used something else.
    image = tf.keras.applications.vgg16.preprocess_input(image)
    return image, label


full_dataset = (
    tf.data.Dataset.from_tensor_slices((list(selected_files), list(selected_labels)))
    .shuffle(dataset_size, seed=SEED, reshuffle_each_iteration=False)
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
)

remaining    = full_dataset.skip(train_size)
val_dataset  = remaining.take(val_size).batch(EVAL_BATCH).prefetch(tf.data.AUTOTUNE)
test_dataset = remaining.skip(val_size).batch(EVAL_BATCH).prefetch(tf.data.AUTOTUNE)

print(f"Train: {train_size}  |  Val: {val_size}  |  Test: {test_size}")

# ==============================================================================
# 3.  LOAD BASE MODEL + QUANTIZE  — skipped if .tflite already exists
# ==============================================================================
fp32_size = os.path.getsize(SAVED_BASE_MODEL_PATH) / (1024 * 1024)

if os.path.exists(QUANTIZED_MODEL_PATH):
    # ── Fast path: .tflite already on disk, skip the conversion ──
    int8_size = os.path.getsize(QUANTIZED_MODEL_PATH) / (1024 * 1024)
    print(f"\nFound existing TFLite model at: {QUANTIZED_MODEL_PATH}")
    print("Skipping quantization. Delete the file to re-quantize.")
    print(f"Original FP32 size : {fp32_size:.2f} MB")
    print(f"INT8 TFLite size   : {int8_size:.2f} MB  "
          f"(~{fp32_size / int8_size:.1f}x compression)")
else:
    # ── Slow path: first run — load model, quantize, save .tflite ──
    print(f"\nLoading pre-trained base model from: {SAVED_BASE_MODEL_PATH}")
    model = tf.keras.models.load_model(SAVED_BASE_MODEL_PATH)
    print("Base model loaded. Starting INT8 quantization.")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_data_gen():
        for image_batch, _ in val_dataset.unbatch().batch(1).take(100):
            yield [image_batch]

    converter.representative_dataset = representative_data_gen
    # Full INT8 only — no float fallback, so the whole graph is quantized.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_quant_model = converter.convert()

    with open(QUANTIZED_MODEL_PATH, "wb") as f:
        f.write(tflite_quant_model)

    int8_size = os.path.getsize(QUANTIZED_MODEL_PATH) / (1024 * 1024)
    print(f"Quantized INT8 model saved to: {QUANTIZED_MODEL_PATH}")
    print(f"Original FP32 size : {fp32_size:.2f} MB")
    print(f"INT8 TFLite size   : {int8_size:.2f} MB  "
          f"(~{fp32_size / int8_size:.1f}x compression)")

# ==============================================================================
# 4.  FAST BATCHED EVALUATION
# ==============================================================================
print(f"\nEvaluating INT8 model — {test_size} images  |  "
      f"batch {EVAL_BATCH}  |  {NUM_THREADS} threads")

interpreter = tf.lite.Interpreter(
    model_path=QUANTIZED_MODEL_PATH,
    num_threads=NUM_THREADS
)

input_detail  = interpreter.get_input_details()[0]
output_detail = interpreter.get_output_details()[0]
input_idx  = input_detail["index"]
output_idx = output_detail["index"]

in_scale, in_zero_point = input_detail["quantization"]
out_scale, out_zero_point = output_detail["quantization"]

interpreter.resize_tensor_input(input_idx, [EVAL_BATCH, IMG_SIZE, IMG_SIZE, 3])
interpreter.allocate_tensors()
current_batch_size = EVAL_BATCH

# Pre-collect: decode all test images once into RAM, then eval loop is pure
# NumPy slicing — no tf.data overhead inside the hot path
print("Pre-collecting test images into RAM ...")
all_images_list = []
all_labels_list = []
for image_batch, label_batch in test_dataset:
    all_images_list.append(image_batch.numpy())
    all_labels_list.append(label_batch.numpy())
all_images = np.concatenate(all_images_list, axis=0).astype(np.float32)
all_labels = np.concatenate(all_labels_list, axis=0)
del all_images_list, all_labels_list
total_samples = len(all_images)
print(f"Collected {total_samples} images. Running inference ...\n")

correct = 0
for start in range(0, total_samples, EVAL_BATCH):
    imgs   = all_images[start : start + EVAL_BATCH]
    labels = all_labels[start : start + EVAL_BATCH]
    n      = imgs.shape[0]

    if n != current_batch_size:
        interpreter.resize_tensor_input(input_idx, [n, IMG_SIZE, IMG_SIZE, 3])
        interpreter.allocate_tensors()
        current_batch_size = n

    # Quantize float inputs to int8 using the model's own input scale/zero-point
    imgs_q = np.round(imgs / in_scale + in_zero_point).astype(np.int8)
    imgs_q = np.ascontiguousarray(imgs_q)

    interpreter.set_tensor(input_idx, imgs_q)
    interpreter.invoke()

    raw_out = interpreter.get_tensor(output_idx)[:, 0].astype(np.float32)
    probs   = (raw_out - out_zero_point) * out_scale  # dequantize back to [0, 1]

    preds   = (probs >= 0.5).astype(np.int32)
    correct += int(np.sum(preds == labels))
    done    = start + n

    print(f"  [{done:>5} / {total_samples}]  "
          f"Running accuracy: {correct / done * 100:.2f}%")

test_accuracy = correct / total_samples

print(f"\nFinal Quantized INT8 Test Accuracy : {test_accuracy * 100:.2f}%")
print(f"Original FP32 size                 : {fp32_size:.2f} MB")
print(f"INT8 TFLite size                   : {int8_size:.2f} MB")
print(f"Compression                        : ~{fp32_size / int8_size:.1f}x")
print("\nDone.")