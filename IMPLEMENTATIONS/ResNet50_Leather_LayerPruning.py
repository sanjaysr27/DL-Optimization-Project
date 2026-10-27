# -*- coding: utf-8 -*-
"""
Structural Layer (Block) Pruning — ResNet50 leather
Compatible with base_applications_ResNet50_Full_Train.keras produced by the
ResNet50 "applications" base-model training script (frozen head -> full
fine-tune, tf.keras.applications.ResNet50 backbone used as a nested
sub-model, called functionally as `x = base_model(x)`).

This mirrors the design of the MobileNetV2 layer-pruning script in this same
pipeline, adapted to ResNet50's very different block structure:

WHY RESNET50 IS DIFFERENT FROM MOBILENETV2 HERE
-------------------------------------------------
MobileNetV2 blocks are inverted-residual blocks that are individually
removable whenever stride == 1 and in_channels == out_channels. ResNet50 is
organized into 4 stages ("conv2".."conv5"), each made of one "conv block"
(bottleneck block WITH a 1x1 projection shortcut, used to change the
channel count / spatial stride at the start of the stage) followed by
several "identity blocks" (bottleneck blocks whose shortcut is a bare
identity add, i.e. stride == 1 and in_channels == out_channels by
construction). Exactly like the MobileNetV2 script, only the
shape-preserving (here: identity) blocks are structurally safe to remove —
removing one just means the stage "skips" that transformation.

  Stage   base filters   #blocks   removable (identity) blocks
  conv2   64             3         block2, block3                (2)
  conv3   128            4         block2, block3, block4         (3)
  conv4   256            6         block2, block3, block4,
                                    block5, block6                 (5)
  conv5   512            3         block2, block3                 (2)
  --------------------------------------------------------------------
  Total blocks: 16          Total removable candidates: 12

Block 1 of every stage is never a removal candidate (it carries the
projection shortcut that changes channels/stride) and is always rebuilt
with its shortcut conv/bn, exactly as MobileNetV2's block 1 is always kept.

The base model here is always the nested-submodel layout (ResNet50 is used
as `x = base_model(x)` inside the outer functional model), so — unlike the
MobileNetV2 script — there is no "flat_custom" layout to detect. The script
still auto-locates the nested backbone rather than hard-coding its name/
position, so it keeps working if the outer model's layer ordering changes.

IMPORTANCE CRITERION (same as the MobileNetV2 script)
-------------------------------------------------------
For each removable (identity) block we measure how close the block's
transformation is to the identity function it would become if pruned:

    relative_change = || block_output - block_input ||_2 / || block_input ||_2

computed over sampled batches with the model in inference mode. LOW score
= block barely changes its input = most redundant = removed first.
"""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

# ==============================================================================
# 1.  CONFIGURATION
# ==============================================================================
SAVED_BASE_MODEL_PATH   = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\base_applications_ResNet50_Full_Train.keras"
LAYER_PRUNED_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\resnet50_layer_pruned_2blocks.keras"
dataset_dir            = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\DATASET\CSIR_Datasets"

IMG_SIZE   = 224
BATCH_SIZE = 16          # matches the ResNet50 applications base script
SEED       = 42
NUM_BLOCKS = 16          # conv2(3) + conv3(4) + conv4(6) + conv5(3)

NUM_BLOCKS_TO_REMOVE = 2
IMPORTANCE_PROBE_BATCHES = 20

FINETUNE_LR        = 1e-4
FINETUNE_EPOCHS    = 50
FINETUNE_PATIENCE  = 5

# ==============================================================================
# 2.  LOAD BASE MODEL  +  LOCATE THE NESTED RESNET50 BACKBONE
# ==============================================================================
base_model = tf.keras.models.load_model(SAVED_BASE_MODEL_PATH)
print("Base model loaded successfully.\n")
base_model.summary()


def try_get_layer(model, name):
    try:
        return model.get_layer(name)
    except ValueError:
        return None


def find_nested_backbone(model):
    """The ResNet50 applications script always builds the backbone as a
    nested sub-model (`x = base_model(x)`), so pick the largest nested
    tf.keras.Model inside the outer model rather than hard-coding a name."""
    candidates = [l for l in model.layers if isinstance(l, tf.keras.Model)]
    if not candidates:
        raise ValueError(
            "Could not find a nested ResNet50 backbone sub-model inside the "
            f"loaded model. Top-level layer names were: {[l.name for l in model.layers]}"
        )
    return max(candidates, key=lambda l: len(l.layers))


backbone = find_nested_backbone(base_model)
print(f"\n>>> Nested backbone name: '{backbone.name}' ({len(backbone.layers)} layers)")


def get_layer_checked(model, name):
    layer = try_get_layer(model, name)
    if layer is None:
        raise ValueError(
            f"Layer '{name}' not found in '{model.name}'. "
            f"Available layer names: {[l.name for l in model.layers]}"
        )
    return layer


def find_head_dense_layers(model, backbone):
    """Locate the 128-unit hidden Dense and the 1-unit classifier Dense by
    unit count, since the ResNet50 base script does not name them explicitly."""
    top_level = [l for l in model.layers if l is not backbone]
    dense_layers = [l for l in top_level if isinstance(l, layers.Dense)]
    hidden = next((l for l in dense_layers if l.units == 128), None)
    classifier = next((l for l in dense_layers if l.units == 1), None)
    if hidden is None or classifier is None:
        raise ValueError(
            "Could not find head Dense layers (128-unit hidden, 1-unit "
            f"classifier). Found Dense layers: {[(l.name, l.units) for l in dense_layers]}"
        )
    return hidden, classifier


head_hidden_layer, head_classifier_layer = find_head_dense_layers(base_model, backbone)
print(f">>> Head layers: hidden='{head_hidden_layer.name}' (128u), "
      f"classifier='{head_classifier_layer.name}' (1u)")

# ==============================================================================
# 3.  DATASET  (leather defect / non-defect — CSIR_Datasets, same structure
#     and preprocessing family as the ResNet50 base script in this pipeline)
# ==============================================================================
random.seed(SEED)

nonDef_dir = os.path.join(dataset_dir, "0_DefectFree")
def_dir = os.path.join(dataset_dir, "1_Defective")

nonDef_files = [os.path.join(nonDef_dir, f) for f in os.listdir(nonDef_dir)]
def_files = [os.path.join(def_dir, f) for f in os.listdir(def_dir)]

N_PER_CLASS = 3992
assert len(nonDef_files) >= N_PER_CLASS, (
    f"Only {len(nonDef_files)} defect-free images found, need {N_PER_CLASS}"
)
assert len(def_files) >= N_PER_CLASS, (
    f"Only {len(def_files)} defective images found, need {N_PER_CLASS}"
)

nonDef_files = random.sample(nonDef_files, N_PER_CLASS)
def_files = random.sample(def_files, N_PER_CLASS)

selected_files = nonDef_files + def_files
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
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image, label


full_dataset = tf.data.Dataset.from_tensor_slices(
    (list(selected_files), list(selected_labels))
)
full_dataset = full_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
full_dataset = full_dataset.shuffle(dataset_size, seed=SEED, reshuffle_each_iteration=False)

train_dataset = full_dataset.take(train_size)
remaining     = full_dataset.skip(train_size)
val_dataset   = remaining.take(val_size)
test_dataset  = remaining.skip(val_size)

train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset   = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset  = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"  Train: {train_size}  |  Validation: {val_size}  |  Test: {test_size}")

# ==============================================================================
# 4.  RESNET50 STAGE CONFIGURATION
#     (stage_name, base_filters, num_blocks_in_stage, stride_of_block1)
#     f1 = f2 = base_filters, f3 = 4 * base_filters (standard bottleneck)
# ==============================================================================
STAGE_CFG = [
    ("conv2", 64,  3, 1),
    ("conv3", 128, 4, 2),
    ("conv4", 256, 6, 2),
    ("conv5", 512, 3, 2),
]

# ==============================================================================
# 5.  IDENTIFY REMOVABLE (IDENTITY) BLOCKS
#     Block 1 of every stage carries the projection shortcut (channel/stride
#     change) and is never a candidate. Blocks 2..N of each stage are plain
#     identity blocks (stride 1, in_channels == out_channels by construction)
#     and are structurally safe to remove.
# ==============================================================================
candidate_blocks = []
for stage_name, base_filters, num_blocks, stride1 in STAGE_CFG:
    for b in range(2, num_blocks + 1):
        candidate_blocks.append((stage_name, b))

print(f"\n  Removable (identity) candidate blocks: {candidate_blocks}")
print(f"  Total candidates: {len(candidate_blocks)}")

assert NUM_BLOCKS_TO_REMOVE <= len(candidate_blocks), (
    f"NUM_BLOCKS_TO_REMOVE ({NUM_BLOCKS_TO_REMOVE}) exceeds the number of "
    f"removable candidate blocks ({len(candidate_blocks)})."
)

# ==============================================================================
# 6.  SCORE BLOCK IMPORTANCE
#     block_in  = input to the block's first 1x1 conv (== previous block's
#                 post-ReLU output, since identity blocks feed x into both
#                 the main path and the shortcut)
#     block_out = output of the block's final ReLU ("{stage}_block{b}_out")
# ==============================================================================
probe_outputs = []
for stage_name, b in candidate_blocks:
    in_layer = get_layer_checked(backbone, f"{stage_name}_block{b}_1_conv")
    out_layer = get_layer_checked(backbone, f"{stage_name}_block{b}_out")
    probe_outputs.append(in_layer.input)
    probe_outputs.append(out_layer.output)

probe_model = Model(inputs=backbone.input, outputs=probe_outputs)

score_sums = {blk: 0.0 for blk in candidate_blocks}
score_counts = {blk: 0 for blk in candidate_blocks}

print("\n  Probing block importance on sample batches...")
for batch_idx, (images, _labels) in enumerate(train_dataset.take(IMPORTANCE_PROBE_BATCHES)):
    outputs = probe_model(images, training=False)

    for i, blk in enumerate(candidate_blocks):
        block_in = outputs[2 * i]
        block_out = outputs[2 * i + 1]

        diff_norm = tf.norm(
            tf.reshape(block_out - block_in, (tf.shape(block_out)[0], -1)), axis=1
        )
        inp_norm = tf.norm(
            tf.reshape(block_in, (tf.shape(block_in)[0], -1)), axis=1
        )
        rel_change = diff_norm / (inp_norm + 1e-8)

        score_sums[blk] += tf.reduce_sum(rel_change).numpy()
        score_counts[blk] += rel_change.shape[0]

block_scores = {blk: score_sums[blk] / score_counts[blk] for blk in candidate_blocks}

print(f"\n  {'Block':>16}  {'Importance (rel. change)':>26}")
print("  " + "-" * 44)
for blk in sorted(block_scores, key=lambda b: block_scores[b]):
    label = f"{blk[0]}_block{blk[1]}"
    print(f"{label:>16}  {block_scores[blk]:>26.4f}")

ranked_blocks = sorted(block_scores, key=lambda b: block_scores[b])
blocks_to_remove = set(ranked_blocks[:NUM_BLOCKS_TO_REMOVE])

print(f"\n  Blocks selected for REMOVAL : {sorted(blocks_to_remove)}")
print(f"  Remaining depth              : {NUM_BLOCKS - len(blocks_to_remove)} / {NUM_BLOCKS} blocks")

# ==============================================================================
# 7.  BUILD LAYER-PRUNED MODEL ARCHITECTURE
#     Layer names match tf.keras.applications.ResNet50's own naming
#     convention exactly (conv1_*, "{stage}_block{n}_*") so weights can be
#     transplanted directly by name in step 8.
# ==============================================================================
BN_EPS = 1.001e-5  # matches tf.keras.applications.resnet's BatchNormalization epsilon

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="data_augmentation")


def conv_block(x, base_filters, stage_name, block_num, stride):
    """Bottleneck block WITH a 1x1 projection shortcut (used as block1 of
    every stage, to change channel count and/or spatial resolution)."""
    f1 = f2 = base_filters
    f3 = base_filters * 4
    name_base = f"{stage_name}_block{block_num}"

    shortcut = layers.Conv2D(f3, 1, strides=stride, name=f"{name_base}_0_conv")(x)
    shortcut = layers.BatchNormalization(epsilon=BN_EPS, name=f"{name_base}_0_bn")(shortcut)

    y = layers.Conv2D(f1, 1, strides=stride, name=f"{name_base}_1_conv")(x)
    y = layers.BatchNormalization(epsilon=BN_EPS, name=f"{name_base}_1_bn")(y)
    y = layers.Activation('relu', name=f"{name_base}_1_relu")(y)

    y = layers.Conv2D(f2, 3, padding='same', name=f"{name_base}_2_conv")(y)
    y = layers.BatchNormalization(epsilon=BN_EPS, name=f"{name_base}_2_bn")(y)
    y = layers.Activation('relu', name=f"{name_base}_2_relu")(y)

    y = layers.Conv2D(f3, 1, name=f"{name_base}_3_conv")(y)
    y = layers.BatchNormalization(epsilon=BN_EPS, name=f"{name_base}_3_bn")(y)

    y = layers.Add(name=f"{name_base}_add")([shortcut, y])
    y = layers.Activation('relu', name=f"{name_base}_out")(y)
    return y


def identity_block(x, base_filters, stage_name, block_num):
    """Bottleneck block whose shortcut is a bare identity (stride 1,
    in_channels == out_channels). These are the removable candidates."""
    f1 = f2 = base_filters
    f3 = base_filters * 4
    name_base = f"{stage_name}_block{block_num}"

    y = layers.Conv2D(f1, 1, name=f"{name_base}_1_conv")(x)
    y = layers.BatchNormalization(epsilon=BN_EPS, name=f"{name_base}_1_bn")(y)
    y = layers.Activation('relu', name=f"{name_base}_1_relu")(y)

    y = layers.Conv2D(f2, 3, padding='same', name=f"{name_base}_2_conv")(y)
    y = layers.BatchNormalization(epsilon=BN_EPS, name=f"{name_base}_2_bn")(y)
    y = layers.Activation('relu', name=f"{name_base}_2_relu")(y)

    y = layers.Conv2D(f3, 1, name=f"{name_base}_3_conv")(y)
    y = layers.BatchNormalization(epsilon=BN_EPS, name=f"{name_base}_3_bn")(y)

    y = layers.Add(name=f"{name_base}_add")([x, y])
    y = layers.Activation('relu', name=f"{name_base}_out")(y)
    return y


def build_layer_pruned_resnet50(removed_blocks):
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_layer")
    x = data_augmentation(inp)

    # Stem (matches tf.keras.applications.ResNet50's conv1/pool1 exactly)
    x = layers.ZeroPadding2D(padding=(3, 3), name="conv1_pad")(x)
    x = layers.Conv2D(64, 7, strides=2, name="conv1_conv")(x)
    x = layers.BatchNormalization(epsilon=BN_EPS, name="conv1_bn")(x)
    x = layers.Activation('relu', name="conv1_relu")(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name="pool1_pad")(x)
    x = layers.MaxPooling2D(3, strides=2, name="pool1_pool")(x)

    for stage_name, base_filters, num_blocks, stride1 in STAGE_CFG:
        x = conv_block(x, base_filters, stage_name, 1, stride1)  # block1: never removed
        for b in range(2, num_blocks + 1):
            if (stage_name, b) in removed_blocks:
                continue
            x = identity_block(x, base_filters, stage_name, b)

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(128, activation='relu', name="fc1")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    out = layers.Dense(1, activation='sigmoid', name="classifier")(x)

    return Model(inp, out)


layer_pruned_model = build_layer_pruned_resnet50(blocks_to_remove)
layer_pruned_model.summary()

print("\nBase params         :", base_model.count_params())
print("Layer-pruned params :", layer_pruned_model.count_params())

# ==============================================================================
# 8.  TRANSFER WEIGHTS  (base model -> layer-pruned model, by resolved name)
# ==============================================================================
for name in ["conv1_conv", "conv1_bn"]:
    layer_pruned_model.get_layer(name).set_weights(
        get_layer_checked(backbone, name).get_weights()
    )

for stage_name, base_filters, num_blocks, stride1 in STAGE_CFG:
    # block1 of every stage is always kept, and always carries the shortcut conv/bn
    for suffix in ["1_conv", "1_bn", "2_conv", "2_bn", "3_conv", "3_bn", "0_conv", "0_bn"]:
        layer_name = f"{stage_name}_block1_{suffix}"
        layer_pruned_model.get_layer(layer_name).set_weights(
            get_layer_checked(backbone, layer_name).get_weights()
        )

    for b in range(2, num_blocks + 1):
        if (stage_name, b) in blocks_to_remove:
            continue
        for suffix in ["1_conv", "1_bn", "2_conv", "2_bn", "3_conv", "3_bn"]:
            layer_name = f"{stage_name}_block{b}_{suffix}"
            layer_pruned_model.get_layer(layer_name).set_weights(
                get_layer_checked(backbone, layer_name).get_weights()
            )

layer_pruned_model.get_layer("fc1").set_weights(head_hidden_layer.get_weights())
layer_pruned_model.get_layer("classifier").set_weights(head_classifier_layer.get_weights())

print("  All weights transplanted successfully.")

# ==============================================================================
# 9.  FINE-TUNE
# ==============================================================================
layer_pruned_model.compile(
    optimizer=Adam(learning_rate=FINETUNE_LR),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

TENSORBOARD_LOG_DIR_LAYERPRUNED = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\TRAINING\logs\resnet50_layer_pruned_2blocks"
tensorboard_layer_pruned = TensorBoard(log_dir=TENSORBOARD_LOG_DIR_LAYERPRUNED)

es_layer_pruned = EarlyStopping(
    monitor='val_loss', patience=FINETUNE_PATIENCE, restore_best_weights=True, verbose=1
)

history_layer_pruned = layer_pruned_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=FINETUNE_EPOCHS,
    callbacks=[es_layer_pruned, tensorboard_layer_pruned],
    verbose=1
)

# ==============================================================================
# 10.  FINAL EVALUATION
# ==============================================================================
loss_after, acc_after = layer_pruned_model.evaluate(test_dataset, verbose=1)
print(f"\n  Test loss after fine-tuning     : {loss_after:.4f}")
print(f"  Test accuracy after fine-tuning : {acc_after:.4f}")
print(f"  Best validation accuracy        : {max(history_layer_pruned.history['val_accuracy']):.4f}")

# ==============================================================================
# 11.  MODEL COMPARISON SUMMARY
# ==============================================================================
base_params = base_model.count_params()
pruned_params = layer_pruned_model.count_params()

print(f"\n  Blocks removed           : {sorted(blocks_to_remove)}")
print(f"  Depth                    : {NUM_BLOCKS - len(blocks_to_remove)} / {NUM_BLOCKS} blocks kept")
print(f"  Base   model parameters  : {base_params:>12,}")
print(f"  Pruned model parameters  : {pruned_params:>12,}")
print(f"  Parameters removed       : {base_params - pruned_params:>12,}")
print(f"  Parameter reduction      : {100*(1 - pruned_params/base_params):.1f} %")
print(f"  Layer-pruned model test accuracy (after FT) : {acc_after:.4f}")

# ==============================================================================
# 12.  SAVE
# ==============================================================================
layer_pruned_model.save(LAYER_PRUNED_MODEL_PATH)
print(f"\n  Layer-pruned model saved to:\n  {LAYER_PRUNED_MODEL_PATH}")


import json
metrics = {"test_loss": float(loss_after), "test_accuracy": float(acc_after)}
with open("last_run_metrics.json", "w") as f:
    json.dump(metrics, f)

print("\nDone.")