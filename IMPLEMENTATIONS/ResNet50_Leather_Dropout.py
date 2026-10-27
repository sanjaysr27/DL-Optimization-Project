# -*- coding: utf-8 -*-
"""
ResNet50 — SpatialDropout2D After Every Bottleneck Block leather
===================================================================

WHY THIS SCRIPT'S LAYOUT-HANDLING IS SIMPLER THAN THE MOBILENETV2 VERSION
----------------------------------------------------------------------------
The MobileNetV2 dropout script needed two entirely different name-resolution
tables ("flat_custom" vs "nested_keras") because its own channel-pruning
script invented a hand-written naming scheme ("block_N_expand", "final_conv",
...) that does NOT match the real names
`tf.keras.applications.MobileNetV2` assigns internally ("Conv1", "Conv_1",
"block_N_expand_BN", block 1 has no expand conv at all, ...).

The ResNet50 channel-pruning script in this pipeline was written to reuse
the REAL Keras ResNet50 layer names when it rebuilds the pruned graph
(e.g. "conv2_block1_1_conv"), rather than inventing its own scheme. That
means a saved ResNet50 model has exactly one of two shapes, and the layer
names are IDENTICAL either way:
  - "nested_keras"  : the whole backbone is wrapped as a single nested
                       layer (`x = base_model(x)`, as produced by the
                       base-training script), so e.g. "conv2_block1_1_conv"
                       lives inside that nested sub-model.
  - "flat_top_level": the backbone's layers were rebuilt directly as
                       top-level layers of the saved model (as produced by
                       the channel-pruning script), so e.g.
                       "conv2_block1_1_conv" is found directly on the
                       loaded model.
`find_backbone` below simply returns the nested sub-model if one exists,
or the loaded model itself otherwise — no separate name-resolution table
needed, and no risk of the block-1-has-no-expand-conv trap that MobileNetV2
has, since ResNet50's bottleneck blocks are structurally uniform (every
block has "_1_conv"/"_2_conv"/"_3_conv"; only the FIRST block of each stage
additionally has a "_0_conv" projection shortcut).

Channel widths (including possibly-pruned widths, if SAVED_BASE_MODEL_PATH
happens to point at a pruned model instead of the full-train one) are never
hardcoded: every conv/BN layer this script builds clones its shape and
hyperparameters (kernel_size, strides, padding, use_bias, momentum,
epsilon, activation) directly from the corresponding source layer via
`get_config()` and `get_weights()[0].shape`, exactly as the channel-pruning
script does. That means this script transfers weights correctly regardless
of whether the source model has standard ResNet50 widths or already-pruned
ones — it never assumes 64/128/256/512-style literals.

This script's preprocessing matches the other scripts in this pipeline
(`tf.keras.applications.resnet50.preprocess_input` — caffe-style BGR +
ImageNet mean-subtraction), consistent with how the ResNet50 base model
was actually trained.

WHY DROPOUT PLACEMENT DIFFERS FROM A PLAIN STACK
---------------------------------------------------
Each ResNet50 bottleneck block looks like this:

    inp ──────────────────────────────────────────┐  (shortcut = inp, or a
         1x1 reduce → BN → ReLU                    │   1x1/BN projection
         3x3        → BN → ReLU                    │   only in the first
         1x1 expand → BN ─────────────────── Add ──┘   block of a stage)
                                                │
                                              ReLU  ("_out")

SpatialDropout2D INSIDE the block (between the expand-BN and the Add) would
randomly zero channels on one branch of the Add but not the other,
corrupting the residual signal. The safe location is AFTER the block's
complete output — i.e. after the "_out" ReLU that follows the Add — the
same principle the MobileNetV2 version applies after its Add/project-BN.

DROPOUT RATE GUIDANCE
----------------------
  Block SpatialDropout2D (CONV_DROPOUT_RATE): 0.20-0.30 is the usual sweet
  spot; >0.50 rarely helps. Kept at 0.60 by default here only to mirror the
  MobileNetV2 script's own (aggressive) default — tune down if training
  looks unstable.
  Head Dropout (HEAD_DROPOUT_RATE): the base model used 0.50; keep at or
  below that unless you have a specific reason to go higher.

NOTE ON TENSORBOARD:
  Same fix as the rest of this pipeline: the log directory is its own
  variable (`TENSORBOARD_LOG_DIR_DROPOUT`) rather than being embedded
  inline in the TensorBoard(...) call, since the pipeline driver edits
  scripts by rewriting whole "VAR = ..." lines and would otherwise
  overwrite the TensorBoard object itself with a plain string.

Workflow:
  1. Load base model, locate the backbone (nested vs. flat top-level)
  2. Discover stage/block structure and per-block channel widths at
     runtime (never hardcoded)
  3. Build a new ResNet50 with SpatialDropout2D after every bottleneck
     block's output, matching the detected widths exactly
  4. Transfer all weights by resolved name
  5. Evaluate before fine-tuning (dropout OFF at eval time)
  6. Fine-tune with fresh CosineDecay (dropout ON during training)
  7. Evaluate after fine-tuning
  8. Save to DROPOUT_MODEL_PATH
"""

import os
import random
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers.schedules import CosineDecay

# ==============================================================================
# 1.  CONFIGURATION
# ==============================================================================
SAVED_BASE_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\base_applications_ResNet50_Full_Train.keras"
DROPOUT_MODEL_PATH    = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\resnet50_dropout_0.2.keras"
dataset_dir = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\DATASET\CSIR_Datasets"

IMG_SIZE   = 224
BATCH_SIZE = 8
SEED       = 42

CONV_DROPOUT_RATE = 0.2
HEAD_DROPOUT_RATE = 0.2

FINETUNE_EPOCHS   = 50
FINETUNE_PATIENCE = 15
FINETUNE_INIT_LR  = 5e-4

STAGE_NAMES = ["conv2", "conv3", "conv4", "conv5"]

# ==============================================================================
# 2.  LOAD BASE MODEL  +  LOCATE THE BACKBONE (nested or flat top-level)
# ==============================================================================
base_model = tf.keras.models.load_model(SAVED_BASE_MODEL_PATH)
print("Base model loaded successfully.")
base_model.summary()


def try_get_layer(model, name):
    try:
        return model.get_layer(name)
    except ValueError:
        return None


def find_backbone(model):
    """
    Returns the sub-model that actually holds the ResNet50 layers.
      - If the loaded model wraps ResNet50 as a single nested layer
        (`x = base_model(x)`, as the base-training script does), return
        that nested sub-model.
      - Otherwise (e.g. a model produced by the channel-pruning script,
        which rebuilds the backbone as top-level layers using the SAME
        real ResNet50 layer names), return the loaded model itself.
    Either way, layer names like "conv2_block1_1_conv" resolve the same.
    """
    candidates = [l for l in model.layers if isinstance(l, tf.keras.Model)]
    if candidates:
        backbone = max(candidates, key=lambda l: len(l.layers))
        print(f">>> Detected layout: nested_keras  "
              f"(backbone '{backbone.name}', {len(backbone.layers)} layers)")
        return backbone
    print(">>> Detected layout: flat_top_level "
          "(backbone layers live directly on the loaded model)")
    return model


backbone = find_backbone(base_model)


def get_layer_checked(model, name):
    layer = try_get_layer(model, name)
    if layer is None:
        raise ValueError(
            f"Layer '{name}' not found in '{model.name}'. "
            f"Available layer names: {[l.name for l in model.layers]}"
        )
    return layer


def layer_exists(model, name):
    return try_get_layer(model, name) is not None


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


STAGE_BLOCK_COUNTS = {s: discover_blocks(backbone, s) for s in STAGE_NAMES}
NUM_BLOCKS = sum(STAGE_BLOCK_COUNTS.values())
print(f">>> Discovered stage structure: "
      f"{ {s: STAGE_BLOCK_COUNTS[s] for s in STAGE_NAMES} }  "
      f"({NUM_BLOCKS} bottleneck blocks total)")


def block_has_shortcut(model, stage_name, block_idx):
    return layer_exists(model, f"{stage_name}_block{block_idx}_0_conv")


def find_head_dense_layers(model, backbone):
    """Locate the 128-unit hidden Dense and 1-unit classifier Dense layers
    by name first ("fc1"/"classifier", as produced by the pruning script),
    falling back to matching by `units` (the base-training script never
    named them explicitly)."""
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

# Verify every expected weight-bearing layer is actually present before
# doing anything else, so failures are loud and early.
missing = []
for name in ("conv1_pad", "conv1_conv", "conv1_bn", "conv1_relu",
             "pool1_pad", "pool1_pool"):
    if not layer_exists(backbone, name):
        missing.append(name)
for stage in STAGE_NAMES:
    for b in range(1, STAGE_BLOCK_COUNTS[stage] + 1):
        prefix = f"{stage}_block{b}"
        required = [f"{prefix}_1_conv", f"{prefix}_1_bn", f"{prefix}_1_relu",
                    f"{prefix}_2_conv", f"{prefix}_2_bn", f"{prefix}_2_relu",
                    f"{prefix}_3_conv", f"{prefix}_3_bn", f"{prefix}_out"]
        if block_has_shortcut(backbone, stage, b):
            required += [f"{prefix}_0_conv", f"{prefix}_0_bn"]
        missing += [n for n in required if not layer_exists(backbone, n)]

if missing:
    raise ValueError(
        f"Base model is missing expected layers: {missing}\n"
        "Make sure SAVED_BASE_MODEL_PATH points to the correct file."
    )
print("Layer-name check passed — all expected layers present.\n")

# ==============================================================================
# 3.  DATASET  (pipeline matches the ResNet50 base-training script exactly)
# ==============================================================================
random.seed(SEED)

nonDef_dir = os.path.join(dataset_dir, "0_DefectFree")
def_dir    = os.path.join(dataset_dir, "1_Defective")

nonDef_files = [os.path.join(nonDef_dir, f) for f in os.listdir(nonDef_dir)]
def_files    = [os.path.join(def_dir, f) for f in os.listdir(def_dir)]

nonDef_files = random.sample(nonDef_files, 3992)
def_files    = random.sample(def_files, 3992)

selected_files  = nonDef_files + def_files
selected_labels = [0] * 3992 + [1] * 3992   # 0 = non-defective, 1 = defective

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

print("Train:", train_size)
print("Validation:", val_size)
print("Test:", test_size)

train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset   = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset  = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ==============================================================================
# 4.  LAYER-CLONING HELPERS
#    Hyperparameters AND channel widths are read directly off the actual
#    source layer, never hand-typed, so this works whether the source model
#    has standard ResNet50 widths or already-pruned ones.
# ==============================================================================


def clone_conv(source_layer, name=None):
    cfg = source_layer.get_config()
    filters = source_layer.get_weights()[0].shape[-1]
    return layers.Conv2D(
        filters=filters,
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
# 5.  BUILD NEW RESNET50 WITH SpatialDropout2D AFTER EVERY BLOCK
# ==============================================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="dropout_data_augmentation")

transfer_registry = []   # [(new_layer, source_layer), ...] - all full copies, no pruning


def build_stem(x):
    src_pad = get_layer_checked(backbone, "conv1_pad")
    x = clone_zeropad(src_pad)(x)

    src_conv = get_layer_checked(backbone, "conv1_conv")
    conv_layer = clone_conv(src_conv)
    x = conv_layer(x)

    src_bn = get_layer_checked(backbone, "conv1_bn")
    bn_layer = clone_bn(src_bn)
    x = bn_layer(x)

    src_relu = get_layer_checked(backbone, "conv1_relu")
    x = clone_activation(src_relu)(x)

    src_poolpad = get_layer_checked(backbone, "pool1_pad")
    x = clone_zeropad(src_poolpad)(x)

    src_pool = get_layer_checked(backbone, "pool1_pool")
    x = clone_maxpool(src_pool)(x)

    transfer_registry.append((conv_layer, src_conv))
    transfer_registry.append((bn_layer, src_bn))
    return x


def build_bottleneck_with_dropout(x, stage, block_idx, has_shortcut):
    prefix = f"{stage}_block{block_idx}"

    src1      = get_layer_checked(backbone, f"{prefix}_1_conv")
    src1_bn   = get_layer_checked(backbone, f"{prefix}_1_bn")
    src1_relu = get_layer_checked(backbone, f"{prefix}_1_relu")
    src2      = get_layer_checked(backbone, f"{prefix}_2_conv")
    src2_bn   = get_layer_checked(backbone, f"{prefix}_2_bn")
    src2_relu = get_layer_checked(backbone, f"{prefix}_2_relu")
    src3      = get_layer_checked(backbone, f"{prefix}_3_conv")
    src3_bn   = get_layer_checked(backbone, f"{prefix}_3_bn")
    src_out   = get_layer_checked(backbone, f"{prefix}_out")

    if has_shortcut:
        src_sc      = get_layer_checked(backbone, f"{prefix}_0_conv")
        src_sc_bn   = get_layer_checked(backbone, f"{prefix}_0_bn")
        sc_layer    = clone_conv(src_sc)
        sc_bn_layer = clone_bn(src_sc_bn)
        shortcut = sc_bn_layer(sc_layer(x))
        transfer_registry.append((sc_layer, src_sc))
        transfer_registry.append((sc_bn_layer, src_sc_bn))
    else:
        shortcut = x

    conv1_layer = clone_conv(src1)
    bn1_layer   = clone_bn(src1_bn)
    h = clone_activation(src1_relu)(bn1_layer(conv1_layer(x)))

    conv2_layer = clone_conv(src2)
    bn2_layer   = clone_bn(src2_bn)
    h = clone_activation(src2_relu)(bn2_layer(conv2_layer(h)))

    conv3_layer = clone_conv(src3)
    bn3_layer   = clone_bn(src3_bn)
    h = bn3_layer(conv3_layer(h))

    added = layers.Add(name=f"{prefix}_add")([shortcut, h])
    out = clone_activation(src_out, name=f"{prefix}_out")(added)

    # SpatialDropout2D AFTER the complete block output (post Add + ReLU),
    # never between the expand-BN and the Add - see module docstring.
    out = layers.SpatialDropout2D(CONV_DROPOUT_RATE, name=f"sdrop_{prefix}")(out)

    transfer_registry.append((conv1_layer, src1))
    transfer_registry.append((bn1_layer, src1_bn))
    transfer_registry.append((conv2_layer, src2))
    transfer_registry.append((bn2_layer, src2_bn))
    transfer_registry.append((conv3_layer, src3))
    transfer_registry.append((bn3_layer, src3_bn))

    return out


inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_layer")
x = data_augmentation(inp)
x = build_stem(x)

for stage in STAGE_NAMES:
    for b in range(1, STAGE_BLOCK_COUNTS[stage] + 1):
        has_sc = block_has_shortcut(backbone, stage, b)
        x = build_bottleneck_with_dropout(x, stage, b, has_sc)

x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
x = layers.Dense(128, activation="relu", name="fc1")(x)
x = layers.Dropout(HEAD_DROPOUT_RATE, name="head_dropout")(x)
out = layers.Dense(1, activation="sigmoid", name="classifier")(x)

new_model = Model(inputs=inp, outputs=out)
new_model.summary()

# ==============================================================================
# 6.  TRANSFER WEIGHTS FROM BASE MODEL → NEW MODEL
# ==============================================================================
for new_layer, src_layer in transfer_registry:
    new_layer.set_weights(src_layer.get_weights())

new_model.get_layer("fc1").set_weights(head_hidden_layer.get_weights())
new_model.get_layer("classifier").set_weights(head_classifier_layer.get_weights())

print(f"\nTransferred {len(transfer_registry) + 2} weight-bearing layers "
      f"(SpatialDropout2D/Add/Activation/GAP layers carry no weights and are skipped).")
print("All weight-bearing layers matched successfully.")

# ==============================================================================
# 7.  EVALUATE BEFORE FINE-TUNING
# ==============================================================================
new_model.compile(
    optimizer=Adam(learning_rate=FINETUNE_INIT_LR),
    loss=BinaryCrossentropy(),
    metrics=['accuracy']
)

loss_before, acc_before = new_model.evaluate(test_dataset, verbose=1)
print(f"\nTest loss     (before fine-tuning) : {loss_before:.4f}")
print(f"Test accuracy (before fine-tuning) : {acc_before * 100:.2f}%")
print("SpatialDropout2D and Dropout are both DISABLED at eval time —")
print("result should be close to the base model's reported test accuracy.")

# ==============================================================================
# 8.  FINE-TUNE WITH DROPOUT ACTIVE
# ==============================================================================
steps_per_epoch_ft = train_size // BATCH_SIZE
total_ft_steps     = steps_per_epoch_ft * FINETUNE_EPOCHS

lr_schedule_ft = CosineDecay(
    initial_learning_rate=FINETUNE_INIT_LR,
    decay_steps=total_ft_steps,
    alpha=0.01
)

new_model.compile(
    optimizer=Adam(learning_rate=lr_schedule_ft),
    loss=BinaryCrossentropy(),
    metrics=['accuracy']
)

print(f"\nFine-tune LR  : CosineDecay {FINETUNE_INIT_LR:.0e} → {FINETUNE_INIT_LR * 0.01:.0e}")
print(f"Max epochs    : {FINETUNE_EPOCHS}  (early stopping patience = {FINETUNE_PATIENCE})")
print(f"CONV_DROPOUT_RATE = {CONV_DROPOUT_RATE}  HEAD_DROPOUT_RATE = {HEAD_DROPOUT_RATE}")

TENSORBOARD_LOG_DIR_DROPOUT = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\TRAINING\logs\resnet50_dropout_0.2"
tensorboard_dropout = TensorBoard(log_dir=TENSORBOARD_LOG_DIR_DROPOUT)

es_dropout = EarlyStopping(
    monitor='val_loss',
    patience=FINETUNE_PATIENCE,
    restore_best_weights=True,
    verbose=1
)

history_ft = new_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=FINETUNE_EPOCHS,
    callbacks=[es_dropout, tensorboard_dropout],
    verbose=1
)

# ==============================================================================
# 9.  FINAL EVALUATION
# ==============================================================================
loss_after, acc_after = new_model.evaluate(test_dataset, verbose=1)
print(f"\nTest loss     (after fine-tuning)  : {loss_after:.4f}")
print(f"Test accuracy (after fine-tuning)  : {acc_after * 100:.2f}%")
print(f"Best val accuracy during fine-tune : {max(history_ft.history['val_accuracy']) * 100:.2f}%")

# ==============================================================================
# 10.  SUMMARY
# ==============================================================================
base_params = sum(tf.keras.backend.count_params(w) for w in base_model.trainable_weights)
new_params = sum(tf.keras.backend.count_params(w) for w in new_model.trainable_weights)

print(f"\nBase model parameters  : {base_params:>12,}")
print(f"New  model parameters  : {new_params:>12,}")
print(f"Difference             : {new_params - base_params:>+12,}  (should be 0)")
print(f"\nNew  model test accuracy (before fine-tune) : {acc_before * 100:.2f}%")
print(f"New  model test accuracy (after  fine-tune) : {acc_after  * 100:.2f}%")
print("\nDropout configuration:")
print(f"  Blocks — SpatialDropout2D rate : {CONV_DROPOUT_RATE}  "
      f"(sdrop_conv2_block1 … sdrop_conv5_block{STAGE_BLOCK_COUNTS['conv5']})")
print(f"  Head   — Dropout          rate : {HEAD_DROPOUT_RATE}  (head_dropout, after Dense(128))")

# ==============================================================================
# 11.  SAVE
# ==============================================================================
new_model.save(DROPOUT_MODEL_PATH)
print(f"\nModel saved to: {DROPOUT_MODEL_PATH}")

import json
metrics = {"test_loss": float(loss_after), "test_accuracy": float(acc_after)}
with open("last_run_metrics.json", "w") as f:
    json.dump(metrics, f)
