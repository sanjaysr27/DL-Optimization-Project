# -*- coding: utf-8 -*-
"""
Structural Channel Pruning — MobileNetV2 (Defect / Non-Defect)
Compatible with base_applications_MobileNetV2_Full_Train.keras produced by
the base-model training script.

FIXES APPLIED vs. the original draft
-------------------------------------
1. `dataset_dir` -> `DATASET_DIR` (was undefined, would NameError).
2. The base model was saved as  Model(inp, base_model(x), ...) i.e. the
   MobileNetV2 backbone is a SINGLE NESTED LAYER inside the saved model,
   not a flat set of top-level layers. `base_model.get_layer("block_3_expand")`
   on the outer model would raise "No such layer". We now locate the nested
   MobileNetV2 sub-model automatically (`find_backbone`).
3. Real Keras `MobileNetV2` layer names were wrong:
     - stem:  "Conv1" / "bn_Conv1" / "Conv1_relu"   (not "stem_conv"/"stem_bn")
     - final: "Conv_1" / "Conv_1_bn" / "out_relu"    (not "final_conv"/"final_bn")
     - block 1 (expansion factor = 1) has NO expand conv at all -> Keras
       names it "expanded_conv_depthwise" / "expanded_conv_project", not
       "block_1_expand"/"block_1_depthwise". Blocks 2..17 in this script's
       numbering correspond to Keras's "block_1".."block_16".
   A resolver function `block_layer_names()` now maps this script's
   1..17 block numbering to the *actual* names Keras assigned.
4. The classifier head Dense layers were never given `name=` in the
   base-model script, so they were NOT called "fc1"/"classifier" in the
   saved model (Keras auto-named them, e.g. "dense"/"dense_1"). We now find
   the 128-unit Dense and the 1-unit Dense head layers by their `units`
   attribute instead of guessing a name.
5. Added assertions that fail loudly (listing available layer names) if a
   lookup ever misses, instead of silently pruning the wrong tensor.
6. The `tensorboard_pruned` line previously constructed the TensorBoard
   callback and set its log_dir inline (`tensorboard_pruned = TensorBoard(
   log_dir='...')`). The pipeline driver script edits scripts by rewriting
   a whole "VAR = ..." line, so pointing it at `tensorboard_pruned` would
   have overwritten the TensorBoard object itself with a plain string,
   crashing `model.fit()`. The log directory is now its own variable
   (`TENSORBOARD_LOG_DIR_PRUNED`) that the driver can safely edit, and the
   TensorBoard callback is built from it on a separate line.

PRUNING STRATEGY (unchanged from your design)
----------------------------------------------
  1. Expansion channels inside every inverted-residual block are pruned by
     L1 norm of output filters. Block OUTPUT width (project conv) is never
     changed, so skip-connections stay valid.
  2. The final 1280-channel conv ("Conv_1") is pruned the same way, and the
     new width is propagated into the first Dense layer of the head.
  3. Block 1 (expansion factor 1, no expand conv) is left completely
     unpruned, matching your original intent — there is no expand tensor
     to rank importance on without pruning the depthwise/project pair
     directly, which is out of scope here.
"""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
SAVED_BASE_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\base_applications_MobileNetV2_Full_Train.keras"
PRUNED_MODEL_PATH     = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\MobileNetV2_channel_pruned_0.3.keras"
DATASET_DIR           = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\DATASET\CSIR_Datasets"

IMG_SIZE   = 224
BATCH_SIZE = 8
SEED       = 42
NUM_BLOCKS = 17   # this script's own 1..17 numbering (see resolver below)

PRUNING_RATIO = 0.3

FINETUNE_LR        = 1e-4
FINETUNE_EPOCHS    = 50
FINETUNE_PATIENCE  = 5

# ==============================================================================
# 2. LOAD BASE MODEL  +  LOCATE THE NESTED MOBILENETV2 BACKBONE
# ==============================================================================

full_model = tf.keras.models.load_model(SAVED_BASE_MODEL_PATH)
print("Base model loaded successfully.\n")
#full_model.summary()


def find_backbone(model):
    """
    The base-model script built the network as:
        x = base_model(x)   # base_model = MobileNetV2(weights=None, include_top=False, ...)
    which makes the entire MobileNetV2 a single nested layer inside the
    saved model, rather than a flat stack of top-level layers. Find it.
    """
    candidates = [l for l in model.layers if isinstance(l, tf.keras.Model)]
    if not candidates:
        raise ValueError(
            "Could not find a nested MobileNetV2 sub-model inside the loaded "
            "model. Top-level layer names were: "
            f"{[l.name for l in model.layers]}"
        )
    # If there are multiple nested models (unlikely here), pick the one with
    # the most layers - that's the backbone, not e.g. the augmentation Sequential.
    backbone = max(candidates, key=lambda l: len(l.layers))
    print(f">>> Found nested backbone: '{backbone.name}' "
          f"({len(backbone.layers)} layers)")
    return backbone


base_model = find_backbone(full_model)


def find_head_dense_layers(model, backbone):
    """
    Locate the two head Dense layers by their `units`, not by name, since
    the base-model script created them without explicit names.
    Returns (hidden_dense_layer, dropout_layer, classifier_dense_layer).
    """
    top_level = [l for l in model.layers if l is not backbone]
    dense_layers = [l for l in top_level if isinstance(l, layers.Dense)]
    hidden = next((l for l in dense_layers if l.units == 128), None)
    classifier = next((l for l in dense_layers if l.units == 1), None)
    dropout = next((l for l in top_level if isinstance(l, layers.Dropout)), None)
    if hidden is None or classifier is None:
        raise ValueError(
            "Could not find expected head Dense layers (128-unit hidden, "
            f"1-unit classifier). Found Dense layers: "
            f"{[(l.name, l.units) for l in dense_layers]}"
        )
    print(f">>> Head layers found: hidden='{hidden.name}' (128u), "
          f"dropout='{dropout.name if dropout else None}', "
          f"classifier='{classifier.name}' (1u)")
    return hidden, dropout, classifier


head_hidden_layer, head_dropout_layer, head_classifier_layer = find_head_dense_layers(
    full_model, base_model
)

# ==============================================================================
# 3. DATASET  (identical pipeline to base-model training)
# ==============================================================================
random.seed(SEED)

nonDef_dir = os.path.join(DATASET_DIR, "0_DefectFree")
def_dir    = os.path.join(DATASET_DIR, "1_Defective")

nonDef_files = [os.path.join(nonDef_dir, f) for f in os.listdir(nonDef_dir)]
def_files    = [os.path.join(def_dir, f) for f in os.listdir(def_dir)]

nonDef_files = random.sample(nonDef_files, 3992)
def_files    = random.sample(def_files, 3992)

selected_files = nonDef_files + def_files
selected_labels = [0] * 3992 + [1] * 3992

combined = list(zip(selected_files, selected_labels))
random.shuffle(combined)
selected_files, selected_labels = zip(*combined)


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

dataset_size = len(selected_files)
train_size   = int(0.65 * dataset_size)
val_size     = int(0.15 * dataset_size)
test_size    = dataset_size - train_size - val_size

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
# 4. BLOCK-NAME RESOLVER  (this script's 1..17 numbering -> real Keras names)
# ==============================================================================

STRIDE2_BLOCKS = {2, 4, 7, 14}   # this script's numbering for the 4 stride-2 blocks


def block_layer_names(block_id):
    """
    Map this script's block numbering (1..17) to the real layer names that
    tf.keras.applications.MobileNetV2 assigns internally.

    Block 1 (expansion factor 1) has NO expand conv - Keras names its
    depthwise/project pair "expanded_conv_depthwise"/"expanded_conv_project".
    Blocks 2..17 here correspond to Keras "block_1".."block_16".
    """
    if block_id == 1:
        return dict(
            expand=None, expand_bn=None, expand_relu=None,
            depthwise="expanded_conv_depthwise",
            depthwise_bn="expanded_conv_depthwise_BN",
            depthwise_relu="expanded_conv_depthwise_relu",
            project="expanded_conv_project",
            project_bn="expanded_conv_project_BN",
            add=None,
        )
    i = block_id - 1  # real Keras block index, 1..16
    return dict(
        expand=f"block_{i}_expand",
        expand_bn=f"block_{i}_expand_BN",
        expand_relu=f"block_{i}_expand_relu",
        depthwise=f"block_{i}_depthwise",
        depthwise_bn=f"block_{i}_depthwise_BN",
        depthwise_relu=f"block_{i}_depthwise_relu",
        project=f"block_{i}_project",
        project_bn=f"block_{i}_project_BN",
        add=f"block_{i}_add",
    )


def get_layer_checked(model, name):
    try:
        return model.get_layer(name)
    except ValueError:
        raise ValueError(
            f"Layer '{name}' not found in backbone '{model.name}'. "
            f"Available layer names: {[l.name for l in model.layers]}"
        )


FINAL_CONV_NAME  = "Conv_1"
FINAL_BN_NAME    = "Conv_1_bn"
STEM_CONV_NAME   = "Conv1"
STEM_BN_NAME     = "bn_Conv1"

# ==============================================================================
# 5. COMPUTE CHANNEL IMPORTANCE -> KEEP INDICES
# ==============================================================================


def l1_keep_indices(conv_layer, pruning_ratio):
    kernel = conv_layer.get_weights()[0]                 # (kH, kW, in_ch, out_ch)
    importance = np.sum(np.abs(kernel), axis=(0, 1, 2))  # (out_ch,)
    n_total = len(importance)
    n_keep = max(1, int(round(n_total * (1.0 - pruning_ratio))))
    keep_idx = np.sort(np.argsort(importance)[::-1][:n_keep])
    return keep_idx


expansion_keep = {}  # {script_block_id: np.ndarray of kept expand-channel indices}

print(f"\n  {'Block':>6}  {'Orig':>6}  {'Kept':>6}  {'Removed':>8}")
print("  " + "-" * 34)

for bid in range(1, NUM_BLOCKS + 1):
    names = block_layer_names(bid)

    if bid == 1:
        # No expand conv exists for this block; nothing to rank importance
        # on, so we keep every channel. orig_ch = depthwise input channels.
        dw_layer = get_layer_checked(base_model, names["depthwise"])
        orig_ch = dw_layer.get_weights()[0].shape[-2]   # (kH,kW,channels,mult)
        keep_idx = np.arange(orig_ch)
    else:
        exp_layer = get_layer_checked(base_model, names["expand"])
        orig_ch = exp_layer.get_weights()[0].shape[-1]
        keep_idx = l1_keep_indices(exp_layer, PRUNING_RATIO)

    expansion_keep[bid] = keep_idx
    removed = orig_ch - len(keep_idx)
    print(f"{bid:>6}  {orig_ch:>6}  {len(keep_idx):>6}  {removed:>8}")

final_conv_layer = get_layer_checked(base_model, FINAL_CONV_NAME)
final_keep = l1_keep_indices(final_conv_layer, PRUNING_RATIO)
orig_final_ch = final_conv_layer.get_weights()[0].shape[-1]
print(f"\n  {'Final':>6}  {orig_final_ch:>6}  {len(final_keep):>6}  "
      f"{orig_final_ch - len(final_keep):>8}  ({FINAL_CONV_NAME})")

# ==============================================================================
# 6. WEIGHT-EXTRACTION HELPERS
# ==============================================================================


def extract_conv2d_kernel(layer, in_keep=None, out_keep=None):
    k = layer.get_weights()[0]
    if in_keep is not None:
        k = k[:, :, in_keep, :]
    if out_keep is not None:
        k = k[:, :, :, out_keep]
    return [k]


def extract_depthwise_kernel(layer, ch_keep):
    k = layer.get_weights()[0]   # (kH, kW, channels, depth_multiplier)
    return [k[:, :, ch_keep, :]]


def extract_batchnorm_weights(layer, ch_keep):
    gamma, beta, mm, mv = layer.get_weights()
    return [gamma[ch_keep], beta[ch_keep], mm[ch_keep], mv[ch_keep]]


def extract_dense_prune_input(layer, in_keep):
    k, b = layer.get_weights()
    return [k[in_keep, :], b]


# ==============================================================================
# 7. BUILD PRUNED MODEL ARCHITECTURE
# ==============================================================================

# (expansion_factor, projection_filters, stride) - this script's own numbering,
# matches the standard MobileNetV2 spec 1:1.
BLOCK_CFG = [
    (1,  16, 1),   # block  1
    (6,  24, 2),   # block  2
    (6,  24, 1),   # block  3
    (6,  32, 2),   # block  4
    (6,  32, 1),   # block  5
    (6,  32, 1),   # block  6
    (6,  64, 2),   # block  7
    (6,  64, 1),   # block  8
    (6,  64, 1),   # block  9
    (6,  64, 1),   # block 10
    (6,  96, 1),   # block 11
    (6,  96, 1),   # block 12
    (6,  96, 1),   # block 13
    (6, 160, 2),   # block 14
    (6, 160, 1),   # block 15
    (6, 160, 1),   # block 16
    (6, 320, 1),   # block 17
]

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="pruned_data_augmentation")


def build_pruned_block(x, filters, stride, block_id, pruned_exp_size):
    """One pruned inverted-residual block, for blocks 2..17 (has expansion)."""
    inp = x
    input_channels = x.shape[-1]

    x = layers.Conv2D(pruned_exp_size, 1, padding='same', use_bias=False,
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


def build_pruned_block1(x, filters, stride, block_id):
    """Block 1 has no expand conv (expansion factor 1) - depthwise runs
    directly on the stem's output channels, all of which are kept."""
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


def build_pruned_mobilenetv2(expansion_keep_dict, final_conv_keep):
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_layer")
    x = data_augmentation(inp)

    x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False, name="stem_conv")(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(max_value=6, name="stem_relu")(x)

    for bid, (_exp_factor, filt, stride) in enumerate(BLOCK_CFG, start=1):
        if bid == 1:
            x = build_pruned_block1(x, filt, stride, bid)
        else:
            pruned_exp = len(expansion_keep_dict[bid])
            x = build_pruned_block(x, filt, stride, bid, pruned_exp)

    x = layers.Conv2D(len(final_conv_keep), 1, padding='same', use_bias=False, name="final_conv")(x)
    x = layers.BatchNormalization(name="final_bn")(x)
    x = layers.ReLU(max_value=6, name="final_relu")(x)

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(128, activation='relu', name="fc1")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    out = layers.Dense(1, activation='sigmoid', name="classifier")(x)

    return Model(inp, out)


pruned_model = build_pruned_mobilenetv2(expansion_keep, final_keep)
#pruned_model.summary()

print("\nBase params  :", full_model.count_params())
print("Pruned params:", pruned_model.count_params())

# ==============================================================================
# 8. TRANSFER PRUNED WEIGHTS  (base model -> pruned model)
# ==============================================================================

# Stem
pruned_model.get_layer("stem_conv").set_weights(
    get_layer_checked(base_model, STEM_CONV_NAME).get_weights()
)
pruned_model.get_layer("stem_bn").set_weights(
    get_layer_checked(base_model, STEM_BN_NAME).get_weights()
)

for bid in range(1, NUM_BLOCKS + 1):
    names = block_layer_names(bid)
    keep = expansion_keep[bid]

    if bid != 1:
        pruned_model.get_layer(f"block_{bid}_expand").set_weights(
            extract_conv2d_kernel(get_layer_checked(base_model, names["expand"]), out_keep=keep)
        )
        pruned_model.get_layer(f"block_{bid}_expand_bn").set_weights(
            extract_batchnorm_weights(get_layer_checked(base_model, names["expand_bn"]), keep)
        )

    pruned_model.get_layer(f"block_{bid}_depthwise").set_weights(
        extract_depthwise_kernel(get_layer_checked(base_model, names["depthwise"]), keep)
    )
    pruned_model.get_layer(f"block_{bid}_depthwise_bn").set_weights(
        extract_batchnorm_weights(get_layer_checked(base_model, names["depthwise_bn"]), keep)
    )

    pruned_model.get_layer(f"block_{bid}_project").set_weights(
        extract_conv2d_kernel(get_layer_checked(base_model, names["project"]), in_keep=keep)
    )
    # Project BN output width is unchanged - copy as-is.
    pruned_model.get_layer(f"block_{bid}_project_bn").set_weights(
        get_layer_checked(base_model, names["project_bn"]).get_weights()
    )

# Final conv
pruned_model.get_layer("final_conv").set_weights(
    extract_conv2d_kernel(final_conv_layer, out_keep=final_keep)
)
pruned_model.get_layer("final_bn").set_weights(
    extract_batchnorm_weights(get_layer_checked(base_model, FINAL_BN_NAME), final_keep)
)

# Classifier head
pruned_model.get_layer("fc1").set_weights(
    extract_dense_prune_input(head_hidden_layer, final_keep)
)
pruned_model.get_layer("classifier").set_weights(
    head_classifier_layer.get_weights()
)

print("  All weights transplanted successfully.")

# ==============================================================================
# 9. FINE-TUNE
# ==============================================================================

pruned_model.compile(
    optimizer=Adam(learning_rate=FINETUNE_LR),
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

TENSORBOARD_LOG_DIR_PRUNED = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\TRAINING\logs\MobileNetV2_channel_pruned_0.3"
tensorboard_pruned = TensorBoard(log_dir=TENSORBOARD_LOG_DIR_PRUNED)
es_pruned = EarlyStopping(monitor='val_loss', patience=FINETUNE_PATIENCE,
                           restore_best_weights=True, verbose=1)

history_pruned = pruned_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=FINETUNE_EPOCHS,
    callbacks=[es_pruned, tensorboard_pruned],
    verbose=1,
)

# ==============================================================================
# 10. FINAL EVALUATION
# ==============================================================================

loss_after, acc_after = pruned_model.evaluate(test_dataset, verbose=1)
print(f"\n  Test loss after fine-tuning     : {loss_after:.4f}")
print(f"  Test accuracy after fine-tuning : {acc_after:.4f}")
print(f"  Best validation accuracy        : {max(history_pruned.history['val_accuracy']):.4f}")

# ==============================================================================
# 11. MODEL COMPARISON SUMMARY
# ==============================================================================

base_params = full_model.count_params()
pruned_params = pruned_model.count_params()

print(f"\n  Pruning ratio applied   : {PRUNING_RATIO * 100:.0f} %")
print(f"  Base   model parameters : {base_params:>12,}")
print(f"  Pruned model parameters : {pruned_params:>12,}")
print(f"  Parameters removed      : {base_params - pruned_params:>12,}")
print(f"  Parameter reduction     : {100*(1 - pruned_params/base_params):.1f} %")
print(f"  Pruned model test accuracy (after FT)      : {acc_after:.4f}")

# ==============================================================================
# 12. SAVE PRUNED MODEL
# ==============================================================================

pruned_model.save(PRUNED_MODEL_PATH)
print(f"\n  Pruned model saved to:\n  {PRUNED_MODEL_PATH}")


import json
metrics = {"test_loss": float(loss_after), "test_accuracy": float(acc_after)}
with open("last_run_metrics.json", "w") as f:
    json.dump(metrics, f)

print("\nDone.")