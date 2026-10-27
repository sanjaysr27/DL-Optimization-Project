# -*- coding: utf-8 -*-
"""
Structural Layer (Block) Pruning — MobileNetV2 leather
Compatible with mobilenetv2_base.keras produced by the base-model training script.

FIXES APPLIED vs. the original draft
-------------------------------------
1. `dataset_dir` previously pointed at a PetImages (Cat/Dog) folder while the
   loading code still looked for "0_DefectFree"/"1_Defective" subfolders and
   decoded ".jpg" — a leftover copy-paste from a different project. That
   combination would raise FileNotFoundError immediately (PetImages has no
   "0_DefectFree" folder), and even if the folder names were fixed to
   Cat/Dog, it would silently train on the wrong dataset entirely, breaking
   compatibility with the base / channel-pruning / dropout scripts in this
   same pipeline. This script now points at the SAME leather dataset
   (CSIR_Datasets, 0_DefectFree/1_Defective, .bmp) as every other script in
   the pipeline.
2. `base_model.get_layer(f"block_{bid}_expand")` (used both when probing
   block importance and when transferring weights) assumes a FLAT model
   with exact names "block_N_expand" for every N. If SAVED_BASE_MODEL_PATH
   is actually the standard `x = tf.keras.applications.MobileNetV2(...)(x)`
   construction, the backbone is a single NESTED layer and those names
   don't exist at the top level — plus block 1 has no expand conv at all
   in the real architecture. Both steps now auto-detect the layout
   (`detect_base_layout`, same approach as the other fixed scripts in this
   pipeline) and resolve the correct real layer names before use.
3. The new layer-pruned model always rebuilt block 1 WITH an expand conv.
   Block 1 is never a removal candidate (it's excluded automatically since
   it changes channel count), so it's always present in the output model —
   if the source base model has no block-1 expand conv (nested layout),
   that layer would be left with random weights after "transfer". Block 1
   is now built with or without an expand conv to match the detected base
   layout, exactly like the channel-pruning and dropout scripts.
4. Head Dense(128)/Dense(1) layers are located by `units` as a fallback if
   they aren't literally named "fc1"/"classifier".
5. The `tensorboard_layer_pruned` line previously constructed the
   TensorBoard callback and set its log_dir inline. The pipeline driver
   edits scripts by rewriting a whole "VAR = ..." line, so pointing it at
   `tensorboard_layer_pruned` would have overwritten the TensorBoard object
   itself with a plain string, crashing `model.fit()`. The log directory is
   now its own variable (`TENSORBOARD_LOG_DIR_LAYERPRUNED`) that the driver
   can safely edit.

PRUNING STRATEGY (unchanged from your design)
----------------------------------------------
Layer pruning removes ENTIRE inverted-residual blocks depth-wise. Only
blocks with stride == 1 AND input_channels == output_channels (i.e. blocks
that originally carried a residual/skip connection) are structurally safe
to remove — removing one just means the network "skips" that
transformation, so downstream tensor shapes never change.

IMPORTANCE CRITERION
---------------------
For each removable (residual) block we measure how close the block's
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
SAVED_BASE_MODEL_PATH   = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\base_applications_MobileNetV2_Full_Train.keras"
LAYER_PRUNED_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\MobileNetV2_layer_pruned_1blocks.keras"
dataset_dir            = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\DATASET\CSIR_Datasets"

IMG_SIZE   = 224
BATCH_SIZE = 8
SEED       = 42
NUM_BLOCKS = 17

NUM_BLOCKS_TO_REMOVE = 1
IMPORTANCE_PROBE_BATCHES = 20

FINETUNE_LR        = 1e-4
FINETUNE_EPOCHS    = 50
FINETUNE_PATIENCE  = 5

# ==============================================================================
# 2.  LOAD BASE MODEL  +  DETECT ITS LAYOUT
# ==============================================================================
base_model = tf.keras.models.load_model(SAVED_BASE_MODEL_PATH)
print("Base model loaded successfully.\n")
#base_model.summary()


def try_get_layer(model, name):
    try:
        return model.get_layer(name)
    except ValueError:
        return None


def find_nested_backbone(model):
    candidates = [l for l in model.layers if isinstance(l, tf.keras.Model)]
    if not candidates:
        return None
    return max(candidates, key=lambda l: len(l.layers))


def detect_base_layout(model):
    """
    "flat_custom"  - top-level layers named stem_conv/block_N_expand/... exactly,
                      including a real block_1_expand.
    "nested_keras" - nested tf.keras.applications.MobileNetV2 sub-model, real
                      Keras names, block 1 has no expand conv.
    """
    if try_get_layer(model, "block_1_expand") is not None:
        return "flat_custom", model
    nested = find_nested_backbone(model)
    if nested is not None:
        return "nested_keras", nested
    raise ValueError(
        "Could not determine base model layout: no top-level 'block_1_expand' "
        "layer found, and no nested backbone sub-model found either. "
        f"Top-level layer names were: {[l.name for l in model.layers]}"
    )


BASE_MODE, backbone = detect_base_layout(base_model)
print(f"\n>>> Detected base-model layout: {BASE_MODE}")
if BASE_MODE == "nested_keras":
    print(f">>> Nested backbone name: '{backbone.name}' ({len(backbone.layers)} layers)")


def block_layer_names(bid, mode):
    """Resolve this script's block numbering (1..17) to actual layer names."""
    if mode == "flat_custom":
        return dict(
            expand=f"block_{bid}_expand", expand_bn=f"block_{bid}_expand_bn",
            depthwise=f"block_{bid}_depthwise", depthwise_bn=f"block_{bid}_depthwise_bn",
            project=f"block_{bid}_project", project_bn=f"block_{bid}_project_bn",
            add=f"block_{bid}_add",
            has_expand=True,
        )
    if bid == 1:
        return dict(
            expand=None, expand_bn=None,
            depthwise="expanded_conv_depthwise", depthwise_bn="expanded_conv_depthwise_BN",
            project="expanded_conv_project", project_bn="expanded_conv_project_BN",
            add=None,  # block 1 has no residual (32 -> 16 channels)
            has_expand=False,
        )
    i = bid - 1
    return dict(
        expand=f"block_{i}_expand", expand_bn=f"block_{i}_expand_BN",
        depthwise=f"block_{i}_depthwise", depthwise_bn=f"block_{i}_depthwise_BN",
        project=f"block_{i}_project", project_bn=f"block_{i}_project_BN",
        add=f"block_{i}_add",
        has_expand=True,
    )


STEM_CONV_NAME, STEM_BN_NAME = (
    ("stem_conv", "stem_bn") if BASE_MODE == "flat_custom" else ("Conv1", "bn_Conv1")
)
FINAL_CONV_NAME, FINAL_BN_NAME = (
    ("final_conv", "final_bn") if BASE_MODE == "flat_custom" else ("Conv_1", "Conv_1_bn")
)


def get_layer_checked(model, name):
    layer = try_get_layer(model, name)
    if layer is None:
        raise ValueError(
            f"Layer '{name}' not found in '{model.name}'. "
            f"Available layer names: {[l.name for l in model.layers]}"
        )
    return layer


def find_head_dense_layers(model, backbone):
    top_level = [l for l in model.layers if l is not backbone]
    hidden = try_get_layer(model, "fc1")
    classifier = try_get_layer(model, "classifier")
    if hidden is not None and classifier is not None:
        return hidden, classifier
    dense_layers = [l for l in top_level if isinstance(l, layers.Dense)]
    hidden = hidden or next((l for l in dense_layers if l.units == 128), None)
    classifier = classifier or next((l for l in dense_layers if l.units == 1), None)
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
#     as the base / channel-pruning / dropout scripts in this pipeline)
# ==============================================================================
random.seed(SEED)

# Paths to each class
nonDef_dir = os.path.join(dataset_dir, "0_DefectFree")
def_dir = os.path.join(dataset_dir, "1_Defective")

# Get image filenames
nonDef_files = [os.path.join(nonDef_dir, f) for f in os.listdir(nonDef_dir)]
def_files = [os.path.join(def_dir, f) for f in os.listdir(def_dir)]

# Randomly select 3000 from each class
nonDef_files = random.sample(nonDef_files, 3992)
def_files = random.sample(def_files, 3992)

# Combine and shuffle
selected_files = nonDef_files + def_files
selected_labels = [0] * 3992 + [1] * 3992   # 0=non-defective, 1=defective

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
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
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
# 4.  MOBILENETV2 BLOCK CONFIGURATION
# ==============================================================================
BLOCK_CFG = [
    (1,  16, 1), (6, 24, 2), (6, 24, 1), (6, 32, 2), (6, 32, 1), (6, 32, 1),
    (6,  64, 2), (6, 64, 1), (6, 64, 1), (6, 64, 1), (6, 96, 1), (6, 96, 1),
    (6,  96, 1), (6, 160, 2), (6, 160, 1), (6, 160, 1), (6, 320, 1),
]

# ==============================================================================
# 5.  IDENTIFY REMOVABLE (RESIDUAL) BLOCKS
# ==============================================================================
candidate_blocks = []
in_channels = 32   # stem output

for bid, (exp_factor, filt, stride) in enumerate(BLOCK_CFG, start=1):
    if stride == 1 and in_channels == filt:
        candidate_blocks.append(bid)
    in_channels = filt

print(f"\n  Removable (residual) candidate blocks: {candidate_blocks}")
print(f"  Total candidates: {len(candidate_blocks)}")

assert NUM_BLOCKS_TO_REMOVE <= len(candidate_blocks), (
    f"NUM_BLOCKS_TO_REMOVE ({NUM_BLOCKS_TO_REMOVE}) exceeds the number of "
    f"removable candidate blocks ({len(candidate_blocks)})."
)

# ==============================================================================
# 6.  SCORE BLOCK IMPORTANCE
#     (candidate blocks are never block 1, so they always have an expand
#     conv regardless of base layout — no special-casing needed here)
# ==============================================================================
probe_outputs = []
for bid in candidate_blocks:
    names = block_layer_names(bid, BASE_MODE)
    expand_layer = get_layer_checked(backbone, names["expand"])
    add_layer = get_layer_checked(backbone, names["add"])
    probe_outputs.append(expand_layer.input)
    probe_outputs.append(add_layer.output)

probe_input = backbone.input if BASE_MODE == "nested_keras" else base_model.input
probe_model = Model(inputs=probe_input, outputs=probe_outputs)

score_sums = {bid: 0.0 for bid in candidate_blocks}
score_counts = {bid: 0 for bid in candidate_blocks}

print("\n  Probing block importance on sample batches...")
for batch_idx, (images, _labels) in enumerate(train_dataset.take(IMPORTANCE_PROBE_BATCHES)):
    outputs = probe_model(images, training=False)

    for i, bid in enumerate(candidate_blocks):
        block_in = outputs[2 * i]
        block_out = outputs[2 * i + 1]

        diff_norm = tf.norm(
            tf.reshape(block_out - block_in, (tf.shape(block_out)[0], -1)), axis=1
        )
        inp_norm = tf.norm(
            tf.reshape(block_in, (tf.shape(block_in)[0], -1)), axis=1
        )
        rel_change = diff_norm / (inp_norm + 1e-8)

        score_sums[bid] += tf.reduce_sum(rel_change).numpy()
        score_counts[bid] += rel_change.shape[0]

block_scores = {bid: score_sums[bid] / score_counts[bid] for bid in candidate_blocks}

print(f"\n  {'Block':>6}  {'Importance (rel. change)':>26}")
print("  " + "-" * 34)
for bid in sorted(block_scores, key=lambda b: block_scores[b]):
    print(f"{bid:>6}  {block_scores[bid]:>26.4f}")

ranked_blocks = sorted(block_scores, key=lambda b: block_scores[b])
blocks_to_remove = set(ranked_blocks[:NUM_BLOCKS_TO_REMOVE])

print(f"\n  Blocks selected for REMOVAL : {sorted(blocks_to_remove)}")
print(f"  Remaining depth              : {NUM_BLOCKS - len(blocks_to_remove)} / {NUM_BLOCKS} blocks")

# ==============================================================================
# 7.  BUILD LAYER-PRUNED MODEL ARCHITECTURE
#     Block 1 is built with/without an expand conv to match the detected
#     base layout (it's never removed, so it must always be transferable).
# ==============================================================================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="sequential_7")


def block_with_expand(x, expansion, filters, stride, block_id):
    inp = x
    input_channels = x.shape[-1]

    x = layers.Conv2D(input_channels * expansion, 1, padding='same', use_bias=False,
                       name=f"block_{block_id}_expand")(x)
    x = layers.BatchNormalization(name=f"block_{block_id}_expand_bn")(x)
    x = layers.ReLU(max_value=6, name=f"block_{block_id}_expand_relu")(x)

    x = layers.DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False,
                                name=f"block_{block_id}_depthwise")(x)
    x = layers.BatchNormalization(name=f"block_{block_id}_depthwise_bn")(x)
    x = layers.ReLU(max_value=6, name=f"block_{block_id}_depthwise_relu")(x)

    x = layers.Conv2D(filters, 1, padding='same', use_bias=False,
                       name=f"block_{block_id}_project")(x)
    x = layers.BatchNormalization(name=f"block_{block_id}_project_bn")(x)

    if stride == 1 and input_channels == filters:
        x = layers.Add(name=f"block_{block_id}_add")([inp, x])
    return x


def block_no_expand(x, filters, stride, block_id):
    inp = x
    input_channels = x.shape[-1]

    x = layers.DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False,
                                name=f"block_{block_id}_depthwise")(x)
    x = layers.BatchNormalization(name=f"block_{block_id}_depthwise_bn")(x)
    x = layers.ReLU(max_value=6, name=f"block_{block_id}_depthwise_relu")(x)

    x = layers.Conv2D(filters, 1, padding='same', use_bias=False,
                       name=f"block_{block_id}_project")(x)
    x = layers.BatchNormalization(name=f"block_{block_id}_project_bn")(x)

    if stride == 1 and input_channels == filters:
        x = layers.Add(name=f"block_{block_id}_add")([inp, x])
    return x


def build_layer_pruned_mobilenetv2(removed_blocks, base_mode):
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_layer")
    x = data_augmentation(inp)

    x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False, name="stem_conv")(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(max_value=6, name="stem_relu")(x)

    for bid, (exp_factor, filt, stride) in enumerate(BLOCK_CFG, start=1):
        if bid in removed_blocks:
            continue
        has_expand = block_layer_names(bid, base_mode)["has_expand"]
        if has_expand:
            x = block_with_expand(x, exp_factor, filt, stride, bid)
        else:
            x = block_no_expand(x, filt, stride, bid)

    x = layers.Conv2D(1280, 1, padding='same', use_bias=False, name="final_conv")(x)
    x = layers.BatchNormalization(name="final_bn")(x)
    x = layers.ReLU(max_value=6, name="final_relu")(x)

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(128, activation='relu', name="fc1")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    out = layers.Dense(1, activation='sigmoid', name="classifier")(x)

    return Model(inp, out)


layer_pruned_model = build_layer_pruned_mobilenetv2(blocks_to_remove, BASE_MODE)
#layer_pruned_model.summary()

print("\nBase params         :", base_model.count_params())
print("Layer-pruned params :", layer_pruned_model.count_params())

# ==============================================================================
# 8.  TRANSFER WEIGHTS  (base model -> layer-pruned model, by resolved name)
# ==============================================================================
layer_pruned_model.get_layer("stem_conv").set_weights(
    get_layer_checked(backbone, STEM_CONV_NAME).get_weights()
)
layer_pruned_model.get_layer("stem_bn").set_weights(
    get_layer_checked(backbone, STEM_BN_NAME).get_weights()
)

surviving_blocks = [bid for bid in range(1, NUM_BLOCKS + 1) if bid not in blocks_to_remove]

for bid in surviving_blocks:
    names = block_layer_names(bid, BASE_MODE)

    if names["has_expand"]:
        layer_pruned_model.get_layer(f"block_{bid}_expand").set_weights(
            get_layer_checked(backbone, names["expand"]).get_weights()
        )
        layer_pruned_model.get_layer(f"block_{bid}_expand_bn").set_weights(
            get_layer_checked(backbone, names["expand_bn"]).get_weights()
        )

    layer_pruned_model.get_layer(f"block_{bid}_depthwise").set_weights(
        get_layer_checked(backbone, names["depthwise"]).get_weights()
    )
    layer_pruned_model.get_layer(f"block_{bid}_depthwise_bn").set_weights(
        get_layer_checked(backbone, names["depthwise_bn"]).get_weights()
    )
    layer_pruned_model.get_layer(f"block_{bid}_project").set_weights(
        get_layer_checked(backbone, names["project"]).get_weights()
    )
    layer_pruned_model.get_layer(f"block_{bid}_project_bn").set_weights(
        get_layer_checked(backbone, names["project_bn"]).get_weights()
    )

layer_pruned_model.get_layer("final_conv").set_weights(
    get_layer_checked(backbone, FINAL_CONV_NAME).get_weights()
)
layer_pruned_model.get_layer("final_bn").set_weights(
    get_layer_checked(backbone, FINAL_BN_NAME).get_weights()
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

TENSORBOARD_LOG_DIR_LAYERPRUNED = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\TRAINING\logs\MobileNetV2_layer_pruned_1blocks"
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