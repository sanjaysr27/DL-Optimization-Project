# -*- coding: utf-8 -*-
"""
MobileNetV2 — INT4-Equivalent Quantization via Manual Weight Clustering
========================================================================
Dependencies: tensorflow, numpy, os, random  (NO sklearn, NO tfmot)

K-means is implemented from scratch using NumPy only.


PIPELINE
--------
  1. Load      : mobilenetv2_base.keras
  2. Evaluate  : Baseline FP32 accuracy on test set
  3. Cluster   : Apply 16-centroid K-means to Conv2D/DepthwiseConv2D/Dense
  4. Evaluate  : Clustered Keras model accuracy
  5. Convert   : TFLite INT8 PTQ on top of clustered weights
  6. Evaluate  : TFLite INT8 accuracy — batched, fast
  7. Save      : Clustered Keras model + TFLite binary
"""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers.schedules import CosineDecay

# ==============================================================================
# 1.  CONFIGURATION
# ==============================================================================
SAVED_BASE_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\mobilenetv2_base.keras"
CLUSTERED_MODEL_PATH  = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\mobilenetv2_clustered_int4eq.keras"
TFLITE_MODEL_PATH     = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\mobilenetv2_clustered_int4.tflite"
DATASET_DIR           = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\DATASET\PetImages"

IMG_SIZE   = 224
BATCH_SIZE = 8
SEED       = 42

# 16 clusters = 2^4 = INT4-equivalent representational capacity
NUMBER_OF_CLUSTERS = 16

# Batch size for TFLite evaluation only
TFLITE_EVAL_BATCH_SIZE = 32

# Fine-tuning parameters (done BEFORE clustering on the base model)
FINETUNE_EPOCHS   = 20
FINETUNE_PATIENCE = 5
FINETUNE_LR       = 1e-5

# ==============================================================================
# 2.  DATASET
# ==============================================================================
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

cats_dir = os.path.join(DATASET_DIR, "Cat")
dogs_dir = os.path.join(DATASET_DIR, "Dog")

cat_files = [os.path.join(cats_dir, f) for f in os.listdir(cats_dir)]
dog_files = [os.path.join(dogs_dir, f) for f in os.listdir(dogs_dir)]

cat_files = random.sample(cat_files, 4000)
dog_files = random.sample(dog_files, 4000)

selected_files  = cat_files + dog_files
selected_labels = [0] * 4000 + [1] * 4000   # 0 = Cat, 1 = Dog

combined = list(zip(selected_files, selected_labels))
random.shuffle(combined)
selected_files, selected_labels = zip(*combined)

dataset_size = len(selected_files)
train_size   = int(0.65 * dataset_size)
val_size     = int(0.15 * dataset_size)
test_size    = dataset_size - train_size - val_size


def safe_load_image(path, label):
    """
    Loads and decodes a JPEG, returning a zero image + label -1 for any
    corrupt file. Corrupt samples are dropped by the .filter() step below.
    """
    def _try_load(p, l):
        try:
            img = tf.io.read_file(p)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
            img = tf.cast(img, tf.float32) / 255.0
            return img, l
        except Exception:
            return (
                tf.zeros([IMG_SIZE, IMG_SIZE, 3], dtype=tf.float32),
                tf.constant(-1, dtype=l.dtype),
            )

    image, lbl = tf.py_function(
        func=lambda p, l: _try_load(p.numpy().decode(), l),
        inp=[path, label],
        Tout=[tf.float32, tf.int32],
    )
    image.set_shape([IMG_SIZE, IMG_SIZE, 3])
    lbl.set_shape([])
    return image, lbl


full_dataset = tf.data.Dataset.from_tensor_slices(
    (list(selected_files), list(selected_labels))
)
full_dataset = full_dataset.map(safe_load_image,
                                num_parallel_calls=tf.data.AUTOTUNE)
full_dataset = full_dataset.filter(lambda img, lbl: lbl >= 0)
full_dataset = full_dataset.shuffle(
    dataset_size, seed=SEED, reshuffle_each_iteration=False
)

train_dataset = full_dataset.take(train_size)
remaining     = full_dataset.skip(train_size)
val_dataset   = remaining.take(val_size)
test_dataset  = remaining.skip(val_size)

train_dataset = (train_dataset
                 .shuffle(1000)
                 .batch(BATCH_SIZE)
                 .prefetch(tf.data.AUTOTUNE))

val_dataset   = (val_dataset
                 .batch(BATCH_SIZE)
                 .prefetch(tf.data.AUTOTUNE))

test_dataset  = (test_dataset
                 .batch(BATCH_SIZE)
                 .prefetch(tf.data.AUTOTUNE))

print(f"Train: {train_size}  |  Validation: {val_size}  |  Test: {test_size}")

# ==============================================================================
# 3.  LOAD BASE MODEL AND EVALUATE BASELINE
# ==============================================================================
print(f"\nLoading base model from: {SAVED_BASE_MODEL_PATH}")
model = tf.keras.models.load_model(SAVED_BASE_MODEL_PATH)
print("Base model loaded successfully.")
model.summary()

model.compile(
    optimizer=Adam(),
    loss=BinaryCrossentropy(),
    metrics=['accuracy']
)
base_loss, base_acc = model.evaluate(test_dataset, verbose=1)
print(f"\nBase model test accuracy (FP32): {base_acc * 100:.2f}%")

# ==============================================================================
# 4.  NUMPY-ONLY K-MEANS IMPLEMENTATION
# ==============================================================================

def _kmeans_plus_plus_init(data: np.ndarray,
                            n_clusters: int,
                            rng: np.random.RandomState) -> np.ndarray:
    n = len(data)
    first_idx = rng.randint(0, n)
    centroids = [data[first_idx]]

    for _ in range(1, n_clusters):
        c_arr    = np.array(centroids)
        diffs    = data[:, np.newaxis] - c_arr[np.newaxis, :]
        sq_dists = diffs ** 2
        min_sq_d = sq_dists.min(axis=1)

        total = min_sq_d.sum()
        probs = np.ones(n) / n if total == 0 else min_sq_d / total

        next_idx = rng.choice(n, p=probs)
        centroids.append(data[next_idx])

    return np.array(centroids, dtype=np.float32)


def numpy_kmeans(data: np.ndarray,
                 n_clusters: int,
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 random_state: int = 42) -> np.ndarray:
    rng  = np.random.RandomState(random_state)
    data = data.astype(np.float32)

    centroids = _kmeans_plus_plus_init(data, n_clusters, rng)

    for _ in range(max_iter):
        diffs    = data[:, np.newaxis] - centroids[np.newaxis, :]
        sq_dists = diffs ** 2
        labels   = sq_dists.argmin(axis=1)

        counts = np.bincount(labels, minlength=n_clusters).astype(np.float32)
        sums   = np.bincount(labels, weights=data.astype(np.float64),
                             minlength=n_clusters).astype(np.float32)

        empty_mask = (counts == 0)
        if empty_mask.any():
            farthest = sq_dists.min(axis=1).argmax()
            for k in np.where(empty_mask)[0]:
                sums[k]   = data[farthest]
                counts[k] = 1.0

        new_centroids = sums / counts
        max_shift     = np.abs(new_centroids - centroids).max()
        centroids     = new_centroids
        if max_shift < tol:
            break

    quantized = centroids[labels]
    return quantized.astype(np.float32)


# ==============================================================================
# 5.  APPLY CLUSTERING TO MODEL
# ==============================================================================
_CLUSTER_TYPES = (
    tf.keras.layers.Conv2D,
    tf.keras.layers.DepthwiseConv2D,
    tf.keras.layers.Dense,
)


def apply_clustering_to_model(model: tf.keras.Model,
                               n_clusters: int = 16) -> int:
    import time
    n_clustered  = 0
    total_layers = sum(
        1 for layer in model.layers
        if isinstance(layer, _CLUSTER_TYPES) and layer.get_weights()
    )

    for layer in model.layers:
        if not isinstance(layer, _CLUSTER_TYPES):
            continue
        layer_weights = layer.get_weights()
        if not layer_weights:
            continue

        kernel         = layer_weights[0]
        original_shape = kernel.shape
        n_clustered   += 1

        print(f"  [{n_clustered:>2}/{total_layers}] Clustering '{layer.name}'  "
              f"shape: {original_shape}  weights: {kernel.size:,}")

        t0               = time.time()
        quantized_flat   = numpy_kmeans(kernel.flatten(), n_clusters)
        quantized_kernel = quantized_flat.reshape(original_shape)
        print(f"           done in {time.time() - t0:.1f}s")

        new_weights    = list(layer_weights)
        new_weights[0] = quantized_kernel
        layer.set_weights(new_weights)

    return n_clustered


print(f"\nApplying NumPy K-means clustering "
      f"({NUMBER_OF_CLUSTERS} clusters = INT4-equivalent) ...")
print("Layers targeted : Conv2D, DepthwiseConv2D, Dense (kernel only)")
print("Layers skipped  : BatchNormalization, all bias vectors\n")

n_clustered = apply_clustering_to_model(model, n_clusters=NUMBER_OF_CLUSTERS)
print(f"\nClustering complete — {n_clustered} layers quantized.")

# ==============================================================================
# 6.  EVALUATE CLUSTERED KERAS MODEL
# ==============================================================================
model.compile(
    optimizer=Adam(),
    loss=BinaryCrossentropy(),
    metrics=['accuracy']
)

clustered_loss, clustered_acc = model.evaluate(test_dataset, verbose=1)
print(f"\nBase FP32 accuracy                         : {base_acc * 100:.2f}%")
print(f"Clustered model accuracy (INT4-equivalent) : {clustered_acc * 100:.2f}%")

model.save(CLUSTERED_MODEL_PATH)
print(f"\nClustered Keras model saved to: {CLUSTERED_MODEL_PATH}")

# ==============================================================================
# 7.  TFLITE INT8 CONVERSION ON TOP OF CLUSTERED WEIGHTS
# ==============================================================================
print("\nConverting clustered model to TFLite INT8 ...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]


def representative_data_gen():
    for image_batch, _ in val_dataset.unbatch().batch(1).take(100):
        yield [tf.cast(image_batch, tf.float32)]


converter.representative_dataset  = representative_data_gen
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS
]

tflite_model = converter.convert()

with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved to: {TFLITE_MODEL_PATH}")

# ==============================================================================
# 8.  FILE SIZE COMPARISON
# ==============================================================================
fp32_size      = os.path.getsize(SAVED_BASE_MODEL_PATH) / (1024 * 1024)
clustered_size = os.path.getsize(CLUSTERED_MODEL_PATH)  / (1024 * 1024)
tflite_size    = os.path.getsize(TFLITE_MODEL_PATH)     / (1024 * 1024)

print("\nFile size comparison:")
print(f"  Original FP32 model          : {fp32_size:.2f} MB")
print(f"  Clustered Keras model        : {clustered_size:.2f} MB")
print(f"  TFLite INT8 (clustered)      : {tflite_size:.2f} MB")
print(f"  Compression vs FP32          : ~{fp32_size / tflite_size:.1f}x")

# ==============================================================================
# 9.  EVALUATE TFLITE INT8 MODEL ON TEST SET
#
#     FIX: resize_input_tensor was removed in newer TF versions.
#     The correct approach is to run the interpreter one sample at a time,
#     which works on ALL TF versions and avoids the batch-resize API entirely.
#     Speed is still acceptable because the heavy work (invoke) is in C.
# ==============================================================================
print("\nPre-collecting test data into numpy arrays ...")

all_images = []
all_labels = []

for image_batch, label_batch in test_dataset:
    all_images.append(image_batch.numpy())
    all_labels.append(label_batch.numpy())

all_images = np.concatenate(all_images, axis=0).astype(np.float32)
all_labels = np.concatenate(all_labels, axis=0)

print(f"Test data collected: {all_images.shape}  labels: {all_labels.shape}")

print("\nEvaluating TFLite INT8 (clustered) model — one sample at a time ...")

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

correct_predictions = 0
total_samples       = len(all_images)

for i in range(total_samples):
    # Add batch dimension: (H, W, C) -> (1, H, W, C)
    input_data = all_images[i : i + 1]

    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()

    output     = interpreter.get_tensor(output_details['index'])  # (1, 1)
    pred_prob  = float(output[0][0])
    pred_label = 1 if pred_prob >= 0.5 else 0

    if pred_label == int(all_labels[i]):
        correct_predictions += 1

    if (i + 1) % 200 == 0 or (i + 1) == total_samples:
        running_acc = correct_predictions / (i + 1)
        print(f"  [{i + 1:>4} / {total_samples}]  "
              f"Running accuracy: {running_acc * 100:.2f}%")

tflite_accuracy = correct_predictions / total_samples

# ==============================================================================
# 10.  FINAL SUMMARY
# ==============================================================================
print("\nResults summary:")
print(f"  Base FP32 model accuracy              : {base_acc * 100:.2f}%")
print(f"  Clustered model accuracy (Keras)      : {clustered_acc * 100:.2f}%")
print(f"  TFLite INT8 (clustered) accuracy      : {tflite_accuracy * 100:.2f}%")
print(f"\n  Original FP32 size                    : {fp32_size:.2f} MB")
print(f"  TFLite INT8 (clustered) size          : {tflite_size:.2f} MB")

print("\nDone.")