# -*- coding: utf-8 -*-
"""
Structural Channel Pruning — ResNet50 (Defect / Non-Defect)
Compatible with base_applications_ResNet50_Full_Train.keras produced by
the ResNet50 base-model training script.

WHY THIS SCRIPT LOOKS DIFFERENT FROM THE MOBILENETV2 VERSION
--------------------------------------------------------------
MobileNetV2's inverted-residual block has ONE prunable width per block
(the expansion channels) and a fixed, hand-writable set of Keras layer
names ("block_N_expand", "block_N_depthwise", ...) that could just be
hardcoded once. ResNet50's bottleneck block is structurally different in
ways that make hardcoding risky:

  1. TWO independent prunable widths per block: the first 1x1 conv
     ("_1_conv") and the 3x3 conv ("_2_conv") each have their own output
     channel count. The third 1x1 conv ("_3_conv") is the block's OUTPUT
     width and is left unpruned for the same reason MobileNetV2's project
     conv is left unpruned: it must keep matching the residual shortcut,
     or the Add() breaks.
  2. Only the FIRST block of each stage ("conv2","conv3","conv4","conv5")
     has a projection shortcut ("_0_conv"/"_0_bn"); every other block uses
     a plain identity shortcut. Blocks-per-stage (3,4,6,3 for ResNet50) is
     a fact about the specific architecture, not something to hand-type
     from memory that could silently drift if a different TF version
     names things slightly differently.
  3. Conv2D layers inside tf.keras.applications.ResNet50 (kernel_size,
     strides, padding, use_bias) are NOT uniformly configured the way
     MobileNetV2's "use_bias=False everywhere" convention is — the exact
     values matter for a correct rebuild (e.g. stride lives on the FIRST
     1x1 conv and the shortcut conv, not on the 3x3 conv, in this
     architecture).

To avoid silently building an architecture that's subtly wrong (same
failure mode called out in the MobileNetV2 script's fix notes — it would
run to completion and still report a number), this script:
  - Locates the nested ResNet50 backbone the same way the MobileNetV2
    script does (`find_backbone`), since the base-training script also
    wraps it as `x = base_model(x)`.
  - DISCOVERS stage/block structure at runtime by probing for
    "{stage}_block{n}_1_conv" until a stage runs out of blocks, rather
    than hardcoding 3/4/6/3.
  - DISCOVERS whether a block has a projection shortcut by checking for
    "{stage}_block{n}_0_conv" directly, rather than assuming "only
    block 1".
  - CLONES each conv/BN/activation/padding/pooling layer's hyperparameters
    (kernel_size, strides, padding, use_bias, momentum, epsilon, ...) from
    the actual loaded layer via get_config(), only overriding the filter
    count where pruning changes it. Stride, padding, and bias behavior are
    therefore always correct by construction instead of by memory.
  - Fails loudly (listing available layer names) if any expected layer is
    missing, instead of silently pruning the wrong tensor.

PRUNING STRATEGY (mirrors the MobileNetV2 script's design)
-------------------------------------------------------------
  1. Inside every bottleneck block, the two internal convs ("_1_conv" —
     1x1 reduce, "_2_conv" — 3x3) are pruned independently by L1 norm of
     their output filters.
  2. The block's OUTPUT width ("_3_conv", i.e. 4x the base filter count)
     is never changed, so residual adds stay valid and — since ResNet50's
     last stage output (2048 channels) is therefore also unchanged — the
     GlobalAveragePooling2D output width feeding the classifier head is
     unchanged too. Unlike the MobileNetV2 script, there is no separate
     "final conv" layer whose pruned width needs propagating into the head
     Dense layer: the head Dense(128) input dimension is untouched, so its
     weights are copied straight across.
  3. The stem ("conv1_conv", 64 channels) and every projection shortcut
     ("_0_conv"/"_0_bn") are left completely unpruned and copied as-is,
     matching the MobileNetV2 script's treatment of its stem and its
     block-1 project conv.
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
SAVED_BASE_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\base_applications_ResNet50_Full_Train.keras"
PRUNED_MODEL_PATH     = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\resnet50_channel_pruned_0.3.keras"
DATASET_DIR           = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\DATASET\CSIR_Datasets"

IMG_SIZE   = 224
BATCH_SIZE = 8
SEED       = 42

PRUNING_RATIO = 0.3

FINETUNE_LR        = 1e-4
FINETUNE_EPOCHS    = 50
FINETUNE_PATIENCE  = 5

STAGE_NAMES = ["conv2", "conv3", "conv4", "conv5"]

# ==============================================================================
# 2. LOAD BASE MODEL  +  LOCATE THE NESTED RESNET50 BACKBONE
# ==============================================================================

full_model = tf.keras.models.load_model(SAVED_BASE_MODEL_PATH)
print("Base model loaded successfully.\n")
full_model.summary()


def find_backbone(model):
    """
    The base-model script built the network as:
        x = base_model(x)   # base_model = ResNet50(weights=None, include_top=False, ...)
    which makes the entire ResNet50 a single nested layer inside the saved
    model, rather than a flat stack of top-level layers. Find it.
    """
    candidates = [l for l in model.layers if isinstance(l, tf.keras.Model)]
    if not candidates:
        raise ValueError(
            "Could not find a nested ResNet50 sub-model inside the loaded "
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
DROPOUT_RATE = head_dropout_layer.rate if head_dropout_layer is not None else 0.5

# ==============================================================================
# 3. DATASET  (identical pipeline to the ResNet50 base-model training script)
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
    image = tf.keras.applications.resnet50.preprocess_input(image)
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
# 4. LAYER-LOOKUP + STRUCTURE-DISCOVERY HELPERS
# ==============================================================================


def get_layer_checked(model, name):
    try:
        return model.get_layer(name)
    except ValueError:
        raise ValueError(
            f"Layer '{name}' not found in backbone '{model.name}'. "
            f"Available layer names: {[l.name for l in model.layers]}"
        )


def layer_exists(model, name):
    try:
        model.get_layer(name)
        return True
    except ValueError:
        return False


def discover_blocks(model, stage_name):
    """Count blocks in a stage by probing for '{stage}_block{n}_1_conv'
    until one is missing, rather than hardcoding 3/4/6/3."""
    n = 0
    idx = 1
    while layer_exists(model, f"{stage_name}_block{idx}_1_conv"):
        n += 1
        idx += 1
    if n == 0:
        raise ValueError(
            f"Stage '{stage_name}' has no blocks — expected at least "
            f"'{stage_name}_block1_1_conv' to exist. Available layer names: "
            f"{[l.name for l in model.layers]}"
        )
    return n


STAGE_BLOCK_COUNTS = {s: discover_blocks(base_model, s) for s in STAGE_NAMES}
NUM_BLOCKS = sum(STAGE_BLOCK_COUNTS.values())
print(f"\n>>> Discovered stage structure: "
      f"{ {s: STAGE_BLOCK_COUNTS[s] for s in STAGE_NAMES} }  "
      f"({NUM_BLOCKS} bottleneck blocks total)")


def block_has_shortcut(model, stage_name, block_idx):
    return layer_exists(model, f"{stage_name}_block{block_idx}_0_conv")


# ==============================================================================
# 5. LAYER-CLONING HELPERS
#    Hyperparameters (kernel_size, strides, padding, use_bias, momentum,
#    epsilon, activation, pool_size, ...) are read directly off the actual
#    loaded layer, never hand-typed, so stride/bias behavior is always
#    correct regardless of TF-version-specific config differences.
# ==============================================================================


def clone_conv(source_layer, new_filters, name=None):
    cfg = source_layer.get_config()
    return layers.Conv2D(
        filters=new_filters,
        kernel_size=cfg["kernel_size"],
        strides=cfg["strides"],
        padding=cfg["padding"],
        use_bias=cfg["use_bias"],
        activation=cfg.get("activation"),
        name=name or cfg["name"],
    )


def clone_bn(source_layer, name=None):
    cfg = source_layer.get_config()
    return layers.BatchNormalization(
        axis=cfg.get("axis", -1),
        momentum=cfg.get("momentum", 0.99),
        epsilon=cfg.get("epsilon", 1e-3),
        name=name or cfg["name"],
    )


def clone_activation(source_layer, name=None):
    cfg = source_layer.get_config()
    return layers.Activation(cfg["activation"], name=name or cfg["name"])


def clone_zeropad(source_layer, name=None):
    cfg = source_layer.get_config()
    return layers.ZeroPadding2D(padding=cfg["padding"], name=name or cfg["name"])


def clone_maxpool(source_layer, name=None):
    cfg = source_layer.get_config()
    return layers.MaxPooling2D(
        pool_size=cfg["pool_size"],
        strides=cfg["strides"],
        padding=cfg["padding"],
        name=name or cfg["name"],
    )


# ==============================================================================
# 6. WEIGHT-EXTRACTION HELPERS
# ==============================================================================


def l1_keep_indices(conv_layer, pruning_ratio):
    kernel = conv_layer.get_weights()[0]                 # (kH, kW, in_ch, out_ch)
    importance = np.sum(np.abs(kernel), axis=(0, 1, 2))  # (out_ch,)
    n_total = len(importance)
    n_keep = max(1, int(round(n_total * (1.0 - pruning_ratio))))
    keep_idx = np.sort(np.argsort(importance)[::-1][:n_keep])
    return keep_idx


def extract_conv2d_kernel(layer, in_keep=None, out_keep=None):
    weights = layer.get_weights()
    kernel = weights[0]
    if in_keep is not None:
        kernel = kernel[:, :, in_keep, :]
    if out_keep is not None:
        kernel = kernel[:, :, :, out_keep]
    if len(weights) > 1:
        bias = weights[1]
        if out_keep is not None:
            bias = bias[out_keep]
        return [kernel, bias]
    return [kernel]


def extract_batchnorm_weights(layer, ch_keep=None):
    if ch_keep is None:
        return list(layer.get_weights())
    gamma, beta, mm, mv = layer.get_weights()
    return [gamma[ch_keep], beta[ch_keep], mm[ch_keep], mv[ch_keep]]


# ==============================================================================
# 7. COMPUTE CHANNEL IMPORTANCE -> KEEP INDICES  (per block, two widths)
# ==============================================================================

keep_indices = {}   # {(stage, block_idx): (keep1, keep2)}

print(f"\n  {'Stage/Block':>14}  {'1x1 orig->kept':>16}  {'3x3 orig->kept':>16}")
print("  " + "-" * 52)

for stage in STAGE_NAMES:
    for b in range(1, STAGE_BLOCK_COUNTS[stage] + 1):
        prefix = f"{stage}_block{b}"
        conv1 = get_layer_checked(base_model, f"{prefix}_1_conv")
        conv2 = get_layer_checked(base_model, f"{prefix}_2_conv")

        keep1 = l1_keep_indices(conv1, PRUNING_RATIO)
        keep2 = l1_keep_indices(conv2, PRUNING_RATIO)
        keep_indices[(stage, b)] = (keep1, keep2)

        orig1 = conv1.get_weights()[0].shape[-1]
        orig2 = conv2.get_weights()[0].shape[-1]
        print(f"  {prefix:>14}  {orig1:>5} -> {len(keep1):<5}     "
              f"{orig2:>5} -> {len(keep2):<5}")

# ==============================================================================
# 8. BUILD PRUNED MODEL ARCHITECTURE
# ==============================================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="pruned_data_augmentation")

transfer_registry = []   # list of dicts describing how to copy weights in later


def build_pruned_stem(x):
    src_pad = get_layer_checked(base_model, "conv1_pad")
    x = clone_zeropad(src_pad)(x)

    src_conv = get_layer_checked(base_model, "conv1_conv")
    conv_layer = clone_conv(src_conv, src_conv.get_weights()[0].shape[-1])
    x = conv_layer(x)

    src_bn = get_layer_checked(base_model, "conv1_bn")
    bn_layer = clone_bn(src_bn)
    x = bn_layer(x)

    src_relu = get_layer_checked(base_model, "conv1_relu")
    x = clone_activation(src_relu)(x)

    src_poolpad = get_layer_checked(base_model, "pool1_pad")
    x = clone_zeropad(src_poolpad)(x)

    src_pool = get_layer_checked(base_model, "pool1_pool")
    x = clone_maxpool(src_pool)(x)

    transfer_registry.append(dict(kind="stem_conv", layer=conv_layer, src=src_conv))
    transfer_registry.append(dict(kind="stem_bn", layer=bn_layer, src=src_bn))
    return x


def build_pruned_bottleneck(x, stage, block_idx, keep1, keep2, has_shortcut):
    prefix = f"{stage}_block{block_idx}"

    src1      = get_layer_checked(base_model, f"{prefix}_1_conv")
    src1_bn   = get_layer_checked(base_model, f"{prefix}_1_bn")
    src1_relu = get_layer_checked(base_model, f"{prefix}_1_relu")
    src2      = get_layer_checked(base_model, f"{prefix}_2_conv")
    src2_bn   = get_layer_checked(base_model, f"{prefix}_2_bn")
    src2_relu = get_layer_checked(base_model, f"{prefix}_2_relu")
    src3      = get_layer_checked(base_model, f"{prefix}_3_conv")
    src3_bn   = get_layer_checked(base_model, f"{prefix}_3_bn")
    src_out   = get_layer_checked(base_model, f"{prefix}_out")

    filters3 = src3.get_weights()[0].shape[-1]   # block output width - never pruned

    if has_shortcut:
        src_sc      = get_layer_checked(base_model, f"{prefix}_0_conv")
        src_sc_bn   = get_layer_checked(base_model, f"{prefix}_0_bn")
        sc_layer    = clone_conv(src_sc, src_sc.get_weights()[0].shape[-1])
        sc_bn_layer = clone_bn(src_sc_bn)
        shortcut = sc_bn_layer(sc_layer(x))
        transfer_registry.append(dict(kind="conv_full", layer=sc_layer, src=src_sc))
        transfer_registry.append(dict(kind="bn_full", layer=sc_bn_layer, src=src_sc_bn))
    else:
        shortcut = x

    conv1_layer = clone_conv(src1, len(keep1))
    bn1_layer   = clone_bn(src1_bn)
    h = clone_activation(src1_relu)(bn1_layer(conv1_layer(x)))

    conv2_layer = clone_conv(src2, len(keep2))
    bn2_layer   = clone_bn(src2_bn)
    h = clone_activation(src2_relu)(bn2_layer(conv2_layer(h)))

    conv3_layer = clone_conv(src3, filters3)
    bn3_layer   = clone_bn(src3_bn)
    h = bn3_layer(conv3_layer(h))

    added = layers.Add(name=f"{prefix}_add")([shortcut, h])
    out = clone_activation(src_out, name=f"{prefix}_out")(added)

    transfer_registry.append(dict(kind="conv_in_out", layer=conv1_layer, src=src1,
                                   in_keep=None, out_keep=keep1))
    transfer_registry.append(dict(kind="bn_ch", layer=bn1_layer, src=src1_bn, ch_keep=keep1))
    transfer_registry.append(dict(kind="conv_in_out", layer=conv2_layer, src=src2,
                                   in_keep=keep1, out_keep=keep2))
    transfer_registry.append(dict(kind="bn_ch", layer=bn2_layer, src=src2_bn, ch_keep=keep2))
    transfer_registry.append(dict(kind="conv_in_out", layer=conv3_layer, src=src3,
                                   in_keep=keep2, out_keep=None))
    transfer_registry.append(dict(kind="bn_full", layer=bn3_layer, src=src3_bn))

    return out


inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_layer")
x = data_augmentation(inp)
x = build_pruned_stem(x)

for stage in STAGE_NAMES:
    for b in range(1, STAGE_BLOCK_COUNTS[stage] + 1):
        keep1, keep2 = keep_indices[(stage, b)]
        has_sc = block_has_shortcut(base_model, stage, b)
        x = build_pruned_bottleneck(x, stage, b, keep1, keep2, has_sc)

x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
x = layers.Dense(128, activation="relu", name="fc1")(x)
x = layers.Dropout(DROPOUT_RATE, name="dropout")(x)
out = layers.Dense(1, activation="sigmoid", name="classifier")(x)

pruned_model = Model(inp, out)
pruned_model.summary()

print("\nBase params  :", full_model.count_params())
print("Pruned params:", pruned_model.count_params())

# ==============================================================================
# 9. TRANSFER PRUNED WEIGHTS  (base model -> pruned model)
# ==============================================================================

for entry in transfer_registry:
    kind, layer, src = entry["kind"], entry["layer"], entry["src"]
    if kind == "stem_conv" or kind == "conv_full":
        layer.set_weights(src.get_weights())
    elif kind == "stem_bn" or kind == "bn_full":
        layer.set_weights(src.get_weights())
    elif kind == "conv_in_out":
        layer.set_weights(extract_conv2d_kernel(
            src, in_keep=entry["in_keep"], out_keep=entry["out_keep"]
        ))
    elif kind == "bn_ch":
        layer.set_weights(extract_batchnorm_weights(src, entry["ch_keep"]))
    else:
        raise ValueError(f"Unknown transfer kind: {kind}")

# Head: GAP output width is unchanged (block-output / shortcut channels are
# never pruned), so the hidden Dense layer's input dimension matches exactly
# — no channel selection needed, unlike the MobileNetV2 script's final_conv
# propagation into fc1.
pruned_model.get_layer("fc1").set_weights(head_hidden_layer.get_weights())
pruned_model.get_layer("classifier").set_weights(head_classifier_layer.get_weights())

print("  All weights transplanted successfully.")

# ==============================================================================
# 10. FINE-TUNE
# ==============================================================================

pruned_model.compile(
    optimizer=Adam(learning_rate=FINETUNE_LR),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

TENSORBOARD_LOG_DIR_PRUNED = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\TRAINING\logs\resnet50_channel_pruned_0.3"
tensorboard_pruned = TensorBoard(log_dir=TENSORBOARD_LOG_DIR_PRUNED)
es_pruned = EarlyStopping(monitor="val_loss", patience=FINETUNE_PATIENCE,
                           restore_best_weights=True, verbose=1)

history_pruned = pruned_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=FINETUNE_EPOCHS,
    callbacks=[es_pruned, tensorboard_pruned],
    verbose=1,
)

# ==============================================================================
# 11. FINAL EVALUATION
# ==============================================================================

loss_after, acc_after = pruned_model.evaluate(test_dataset, verbose=1)
print(f"\n  Test loss after fine-tuning     : {loss_after:.4f}")
print(f"  Test accuracy after fine-tuning : {acc_after:.4f}")
print(f"  Best validation accuracy        : {max(history_pruned.history['val_accuracy']):.4f}")

# ==============================================================================
# 12. MODEL COMPARISON SUMMARY
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
# 13. SAVE PRUNED MODEL
# ==============================================================================

pruned_model.save(PRUNED_MODEL_PATH)
print(f"\n  Pruned model saved to:\n  {PRUNED_MODEL_PATH}")


import json
metrics = {"test_loss": float(loss_after), "test_accuracy": float(acc_after)}
with open("last_run_metrics.json", "w") as f:
    json.dump(metrics, f)

print("\nDone.")