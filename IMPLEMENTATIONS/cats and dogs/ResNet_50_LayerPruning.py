# -*- coding: utf-8 -*-
"""
Structural Layer (Block) Pruning — ResNet50 (Cat vs Dog)
Compatible with resnet50_base.keras produced by the ResNet50 base-model
training script (resnet50_base_training.py).

PRUNING STRATEGY
----------------
Just like the MobileNetV2 layer-pruning script, this removes ENTIRE
bottleneck blocks from the network depth-wise, rather than shrinking the
width of every block (channel pruning).

This is only structurally safe for blocks that satisfy BOTH:
    stride == 1  AND  input_channels == output_channels (f3)
because these are exactly the blocks that use an IDENTITY shortcut
(conv_shortcut=False) in the original architecture — i.e. every block
except the first block of each stage. Removing such a block simply means
the network "skips" that transformation — the tensor shape flowing into
the next block never changes, since the block's own output is added
directly to its own (unchanged-shape) input.

The first block of every stage (conv_shortcut=True) always changes either
the channel count (going from the previous stage's width into this
stage's width) or the spatial resolution (stride 2), or both — it can
therefore NEVER be removed this way and is excluded from the candidate
set entirely.

IMPORTANCE CRITERION
---------------------
For each removable (identity-shortcut) block we measure how close the
block's transformation is to the identity function it would become if
pruned:

    relative_change = || block_output - block_input ||_2 / || block_input ||_2

block_input  = the tensor entering the block (== the identity shortcut)
block_output = the tensor leaving the block, AFTER the final ReLU
               (stage{s}_block{b}_out_relu), since that is the actual
               value passed on to the next block in this architecture.

This is computed over a sample of real images, per block, with the model
in inference mode (so the augmentation layer is inactive). A LOW score
means the block is barely changing its input (i.e. it is close to a no-op
already) — these are the most redundant blocks and the best candidates
for removal. We remove the N lowest-scoring blocks.

After the slimmer-depth architecture is rebuilt, weights for every
surviving block are copied across UNCHANGED (no slicing is needed, since
layer pruning does not touch channel widths) and the model is fine-tuned
to recover any accuracy lost from the removed blocks.
"""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

# ==============================================================================
# 1.  CONFIGURATION  — edit these to match your environment
# ==============================================================================
SAVED_BASE_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\resnet50_base.keras"
LAYER_PRUNED_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\resnet50_4layerpruned.keras"
DATASET_DIR = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\DATASET\PetImages"

IMG_SIZE = 224
BATCH_SIZE = 8
SEED = 42

# How many identity-shortcut (removable) blocks to prune out of the
# network. There are 12 candidate blocks total (see STAGE_CFG below /
# candidate_blocks printout); this must be less than that count.
# Increase for a smaller/faster model, decrease to preserve more accuracy.
NUM_BLOCKS_TO_REMOVE = 4

# Number of batches sampled from the training set to score block importance
IMPORTANCE_PROBE_BATCHES = 20

# Fine-tuning hyper-parameters
FINETUNE_LR = 1e-4
FINETUNE_EPOCHS = 50
FINETUNE_PATIENCE = 5

# ==============================================================================
# 2.  LOAD BASE MODEL
# ==============================================================================

base_model = tf.keras.models.load_model(SAVED_BASE_MODEL_PATH)
print("Base model loaded successfully.\n")
base_model.summary()

# ==============================================================================
# 3.  DATASET  (identical pipeline to base-model training)
# ==============================================================================

random.seed(SEED)

cats_dir = os.path.join(DATASET_DIR, "Cat")
dogs_dir = os.path.join(DATASET_DIR, "Dog")

cat_files = [os.path.join(cats_dir, f) for f in os.listdir(cats_dir)]
dog_files = [os.path.join(dogs_dir, f) for f in os.listdir(dogs_dir)]

cat_files = random.sample(cat_files, 4000)
dog_files = random.sample(dog_files, 4000)

selected_files = cat_files + dog_files
selected_labels = [0] * 4000 + [1] * 4000   # 0 = cat, 1 = dog

combined = list(zip(selected_files, selected_labels))
random.shuffle(combined)
selected_files, selected_labels = zip(*combined)


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


full_dataset = tf.data.Dataset.from_tensor_slices(
    (list(selected_files), list(selected_labels))
)
full_dataset = full_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

dataset_size = len(selected_files)                   # 8000
train_size = int(0.65 * dataset_size)                # 5200
val_size = int(0.15 * dataset_size)                  # 1200
test_size = dataset_size - train_size - val_size     # 1600

full_dataset = full_dataset.shuffle(
    dataset_size, seed=SEED, reshuffle_each_iteration=False
)

train_dataset = full_dataset.take(train_size)
remaining = full_dataset.skip(train_size)
val_dataset = remaining.take(val_size)
test_dataset = remaining.skip(val_size)

train_dataset = (train_dataset.shuffle(1000)
                              .batch(BATCH_SIZE)
                              .prefetch(tf.data.AUTOTUNE))
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"  Train: {train_size}  |  Validation: {val_size}  |  Test: {test_size}")

# ==============================================================================
# 4.  RESNET50 STAGE / BLOCK CONFIGURATION  (must match the base model exactly)
# ==============================================================================
# (f1, f2, f3, num_blocks, first_block_stride)
#   f1 = channels of the 1x1 "reduce" conv      (bottleneck width)
#   f2 = channels of the 3x3 conv                (bottleneck width)
#   f3 = channels of the 1x1 "expand" conv       (block OUTPUT width)
STAGE_CFG = [
    (64,  64,  256,  3, 1),   # stage 1 -> output 56x56x256
    (128, 128, 512,  4, 2),   # stage 2 -> output 28x28x512
    (256, 256, 1024, 6, 2),   # stage 3 -> output 14x14x1024
    (512, 512, 2048, 3, 2),   # stage 4 -> output  7x7x2048
]

# Flatten the stage config into a single ordered list of blocks, each with
# a unique global id (1..16), and the shape metadata needed both to build
# the network and to decide which blocks carry an identity shortcut.
BLOCK_CFG = []
_gid = 1
for _stage_idx, (_f1, _f2, _f3, _num_blocks, _first_stride) in enumerate(STAGE_CFG, start=1):
    for _block_idx in range(1, _num_blocks + 1):
        _stride = _first_stride if _block_idx == 1 else 1
        _conv_shortcut = (_block_idx == 1)
        BLOCK_CFG.append({
            "gid": _gid,
            "stage": _stage_idx,
            "block": _block_idx,
            "f1": _f1, "f2": _f2, "f3": _f3,
            "stride": _stride,
            "conv_shortcut": _conv_shortcut,
        })
        _gid += 1

NUM_BLOCKS = len(BLOCK_CFG)   # 16 for the standard ResNet50 (3,4,6,3)

# ==============================================================================
# 5.  IDENTIFY REMOVABLE (IDENTITY-SHORTCUT) BLOCKS
# ==============================================================================
# A block is removable only if its input channel count equals its output
# channel count (f3) AND stride == 1 — i.e. exactly the blocks that use an
# identity shortcut in the original architecture. We track channel widths
# sequentially, starting from the stem's 64 output channels (stem_conv +
# stem_pool do not change channel count).

candidate_blocks = []
in_channels = 64   # stem output

for entry in BLOCK_CFG:
    if entry["stride"] == 1 and in_channels == entry["f3"]:
        candidate_blocks.append(entry["gid"])
    in_channels = entry["f3"]   # output of this block becomes input of the next

print(f"\n  Removable (identity-shortcut) candidate blocks: {candidate_blocks}")
print(f"  Total candidates: {len(candidate_blocks)}")

assert NUM_BLOCKS_TO_REMOVE <= len(candidate_blocks), (
    f"NUM_BLOCKS_TO_REMOVE ({NUM_BLOCKS_TO_REMOVE}) exceeds the number of "
    f"removable candidate blocks ({len(candidate_blocks)})."
)

BLOCK_BY_GID = {entry["gid"]: entry for entry in BLOCK_CFG}


def layer_base_name(gid):
    entry = BLOCK_BY_GID[gid]
    return f"stage{entry['stage']}_block{entry['block']}"


# ==============================================================================
# 6.  SCORE BLOCK IMPORTANCE  (relative input→output change, lower = prune first)
# ==============================================================================

probe_outputs = []
for gid in candidate_blocks:
    name_base = layer_base_name(gid)
    probe_outputs.append(base_model.get_layer(f"{name_base}_conv1").input)     # block input
    probe_outputs.append(base_model.get_layer(f"{name_base}_out_relu").output)  # block output

probe_model = Model(inputs=base_model.input, outputs=probe_outputs)

score_sums = {gid: 0.0 for gid in candidate_blocks}
score_counts = {gid: 0 for gid in candidate_blocks}

print("\n  Probing block importance on sample batches...")
for batch_idx, (images, _labels) in enumerate(train_dataset.take(IMPORTANCE_PROBE_BATCHES)):
    outputs = probe_model(images, training=False)

    for i, gid in enumerate(candidate_blocks):
        block_in = outputs[2 * i]
        block_out = outputs[2 * i + 1]

        diff_norm = tf.norm(
            tf.reshape(block_out - block_in, (tf.shape(block_out)[0], -1)),
            axis=1
        )
        inp_norm = tf.norm(
            tf.reshape(block_in, (tf.shape(block_in)[0], -1)),
            axis=1
        )
        rel_change = diff_norm / (inp_norm + 1e-8)

        score_sums[gid] += tf.reduce_sum(rel_change).numpy()
        score_counts[gid] += rel_change.shape[0]

block_scores = {
    gid: score_sums[gid] / score_counts[gid] for gid in candidate_blocks
}

print(f"\n  {'Block':>18}  {'Importance (rel. change)':>26}")
print("  " + "-" * 46)
for gid in sorted(block_scores, key=lambda b: block_scores[b]):
    print(f"{layer_base_name(gid):>18}  {block_scores[gid]:>26.4f}")

# Lowest relative change = most redundant = removed first
ranked_blocks = sorted(block_scores, key=lambda b: block_scores[b])
blocks_to_remove = set(ranked_blocks[:NUM_BLOCKS_TO_REMOVE])

print(f"\n  Blocks selected for REMOVAL : {sorted(layer_base_name(g) for g in blocks_to_remove)}")
print(f"  Remaining depth              : {NUM_BLOCKS - len(blocks_to_remove)} / {NUM_BLOCKS} blocks")

# ==============================================================================
# 7.  BUILD LAYER-PRUNED MODEL ARCHITECTURE
# ==============================================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="data_augmentation")


def bottleneck_block(x, f1, f2, f3, stage, block, stride, conv_shortcut):
    """
    One ResNet50 bottleneck block:
        1x1 reduce -> 3x3 -> 1x1 expand   (+ identity or projection shortcut)

    Identical to the block builder used in the base-model training script.
    conv_shortcut=True is only ever used for the FIRST block of a stage
    (never removed), so every block reachable here with conv_shortcut=False
    is a plain identity-shortcut block that is safe to skip entirely.
    """
    name_base = f"stage{stage}_block{block}"

    shortcut = x
    if conv_shortcut:
        shortcut = layers.Conv2D(
            f3, 1, strides=stride, padding='same', use_bias=False,
            name=f"{name_base}_shortcut_conv"
        )(x)
        shortcut = layers.BatchNormalization(
            name=f"{name_base}_shortcut_bn"
        )(shortcut)

    # 1x1 reduce
    x = layers.Conv2D(
        f1, 1, padding='same', use_bias=False,
        name=f"{name_base}_conv1"
    )(x)
    x = layers.BatchNormalization(name=f"{name_base}_bn1")(x)
    x = layers.ReLU(name=f"{name_base}_relu1")(x)

    # 3x3 (stride lives here)
    x = layers.Conv2D(
        f2, 3, strides=stride, padding='same', use_bias=False,
        name=f"{name_base}_conv2"
    )(x)
    x = layers.BatchNormalization(name=f"{name_base}_bn2")(x)
    x = layers.ReLU(name=f"{name_base}_relu2")(x)

    # 1x1 expand
    x = layers.Conv2D(
        f3, 1, padding='same', use_bias=False,
        name=f"{name_base}_conv3"
    )(x)
    x = layers.BatchNormalization(name=f"{name_base}_bn3")(x)

    x = layers.Add(name=f"{name_base}_add")([shortcut, x])
    x = layers.ReLU(name=f"{name_base}_out_relu")(x)

    return x


def build_layer_pruned_resnet50(removed_gids):

    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_layer")
    x = data_augmentation(inp)

    # Stem (unchanged)
    x = layers.Conv2D(
        64, 7, strides=2, padding='same', use_bias=False, name="stem_conv"
    )(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same', name="stem_pool")(x)

    # Bottleneck blocks — skip any block whose global id is in
    # `removed_gids` entirely. This is safe because every removable block
    # has input_channels == f3 and stride == 1 (identity shortcut), so
    # skipping it changes neither the spatial resolution nor the channel
    # width flowing into the next block.
    for entry in BLOCK_CFG:
        if entry["gid"] in removed_gids:
            continue
        x = bottleneck_block(
            x, entry["f1"], entry["f2"], entry["f3"],
            stage=entry["stage"], block=entry["block"],
            stride=entry["stride"], conv_shortcut=entry["conv_shortcut"]
        )

    # Classifier head (unchanged)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(128, activation='relu', name="fc1")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    out = layers.Dense(1, activation='sigmoid', name="classifier")(x)

    return Model(inp, out)


layer_pruned_model = build_layer_pruned_resnet50(blocks_to_remove)
layer_pruned_model.summary()

print("\nBase params         :", base_model.count_params())
print("Layer-pruned params :", layer_pruned_model.count_params())

# ==============================================================================
# 8.  TRANSFER WEIGHTS  (base model  →  layer-pruned model)
# ==============================================================================
# No slicing is required here — every surviving block keeps its original
# channel widths unchanged, so weights are copied across layer-for-layer.

for lname in ("stem_conv", "stem_bn"):
    layer_pruned_model.get_layer(lname).set_weights(
        base_model.get_layer(lname).get_weights()
    )

surviving_gids = [entry["gid"] for entry in BLOCK_CFG if entry["gid"] not in blocks_to_remove]

for gid in surviving_gids:
    entry = BLOCK_BY_GID[gid]
    name_base = layer_base_name(gid)

    sublayers = [f"{name_base}_conv1", f"{name_base}_bn1",
                 f"{name_base}_conv2", f"{name_base}_bn2",
                 f"{name_base}_conv3", f"{name_base}_bn3"]
    if entry["conv_shortcut"]:
        sublayers += [f"{name_base}_shortcut_conv", f"{name_base}_shortcut_bn"]

    for lname in sublayers:
        layer_pruned_model.get_layer(lname).set_weights(
            base_model.get_layer(lname).get_weights()
        )

for lname in ("fc1", "classifier"):
    layer_pruned_model.get_layer(lname).set_weights(
        base_model.get_layer(lname).get_weights()
    )

print("  All weights transplanted successfully.")

# ==============================================================================
# 9.  FINE-TUNE THE LAYER-PRUNED MODEL
# ==============================================================================

layer_pruned_model.compile(
    optimizer=Adam(learning_rate=FINETUNE_LR),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

tensorboard_layer_pruned = TensorBoard(log_dir='logs\\ResNet50_4layerpruned')

es_layer_pruned = EarlyStopping(
    monitor='val_loss',
    patience=FINETUNE_PATIENCE,
    restore_best_weights=True,
    verbose=1
)

history_layer_pruned = layer_pruned_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=FINETUNE_EPOCHS,
    callbacks=[es_layer_pruned, tensorboard_layer_pruned],
    verbose=1
)

# ==============================================================================
# 10.  FINAL EVALUATION  (after fine-tuning)
# ==============================================================================

loss_after, acc_after = layer_pruned_model.evaluate(test_dataset, verbose=1)

print(f"\n  Test loss after fine-tuning     : {loss_after:.4f}")
print(f"  Test accuracy after fine-tuning : {acc_after:.4f}")
print(f"  Best validation accuracy        : "
      f"{max(history_layer_pruned.history['val_accuracy']):.4f}")

# ==============================================================================
# 11.  MODEL COMPARISON SUMMARY
# ==============================================================================

base_params = base_model.count_params()
pruned_params = layer_pruned_model.count_params()

print(f"\n  Blocks removed           : {sorted(layer_base_name(g) for g in blocks_to_remove)}")
print(f"  Depth                    : {NUM_BLOCKS - len(blocks_to_remove)} / {NUM_BLOCKS} blocks kept")
print(f"  Base   model parameters  : {base_params:>12,}")
print(f"  Pruned model parameters  : {pruned_params:>12,}")
print(f"  Parameters removed       : {base_params - pruned_params:>12,}")
print(f"  Parameter reduction      : {100*(1 - pruned_params/base_params):.1f} %")
print(f"  Layer-pruned model test accuracy (after FT) : {acc_after:.4f}")

# ==============================================================================
# 12.  SAVE LAYER-PRUNED MODEL
# ==============================================================================

layer_pruned_model.save(LAYER_PRUNED_MODEL_PATH)
print(f"\n  Layer-pruned model saved to:\n  {LAYER_PRUNED_MODEL_PATH}")

# ==============================================================================
# 13.  QUICK PREDICTION SANITY CHECK
# ==============================================================================

for images, labels in test_dataset.take(1):
    preds = layer_pruned_model.predict(images, verbose=0)
    pred_classes = (preds.flatten() > 0.5).astype(int)
    actual = labels.numpy()

    print(f"\n  Raw sigmoid outputs  : {preds[:10].flatten().round(4)}")
    print(f"  Predicted classes    : {pred_classes[:10]}")
    print(f"  Actual labels        : {actual[:10]}")

    batch_acc = np.mean(pred_classes == actual)
    print(f"  Batch accuracy       : {batch_acc:.4f}")
    break

print("\nDone.")