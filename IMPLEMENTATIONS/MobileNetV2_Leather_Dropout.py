# -*- coding: utf-8 -*-
"""
MobileNetV2 — SpatialDropout2D After Every Inverted Residual Block leather
===================================================================

FIXES APPLIED vs. the original draft
-------------------------------------
The original script hard-assumed the saved base model is a FLAT model with
exact layer names "stem_conv", "block_N_expand" (for every N, including
block 1), "final_conv", "fc1", "classifier". That is only true if
SAVED_BASE_MODEL_PATH points at a model built with that exact hand-written
naming convention (e.g. the output of a pruning script that rebuilds the
graph that way).

If SAVED_BASE_MODEL_PATH instead points at a model built the standard way —
`x = tf.keras.applications.MobileNetV2(weights=None, ...)(x)` — then:
  1. The whole backbone is a single NESTED layer inside the saved model, so
     `base_model.get_layer("block_3_expand")` raises "No such layer".
  2. Real Keras MobileNetV2 names are different: "Conv1"/"bn_Conv1" (stem),
     "Conv_1"/"Conv_1_bn" (final conv), and blocks use "_BN" suffix.
  3. Block 1 (expansion factor 1) has NO expand conv at all in the real
     architecture — Keras names its depthwise/project pair
     "expanded_conv_depthwise"/"expanded_conv_project". The original script
     always built an expand conv for block 1, which would then be left with
     random (untrained) weights after "transfer", silently breaking the
     "before fine-tuning accuracy should match base model" assumption
     described in the header.
  4. The head Dense(128)/Dense(1) layers are only named "fc1"/"classifier"
     if the base script explicitly passed `name=`.

This version DETECTS which layout SAVED_BASE_MODEL_PATH actually has at
load time (`detect_base_layout`) and builds the new dropout-augmented model
to structurally match whichever one is found, so weight transfer is always
complete and no layer silently keeps random initialization. It prints which
layout it detected — check that this matches what you expect.

This script's preprocessing matches the other scripts in this pipeline
(`tf.keras.applications.mobilenet_v2.preprocess_input`, scaling to [-1, 1]),
consistent with how the base model was actually trained.

WHY DROPOUT PLACEMENT DIFFERS FROM VGG16
-----------------------------------------
In MobileNetV2 each inverted residual block looks like this:

    inp ──────────────────────────────────────────┐
         Expand Conv → BN → ReLU6                 │  (skip exists only when
         Depthwise   → BN → ReLU6                 │   stride==1 and
         Project     → BN ─────────────────── Add─┘   channels match)

SpatialDropout2D INSIDE the block (between Project BN and the Add) would
randomly zero channels on one branch of the Add but not the other, corrupting
the residual signal. The safe location is AFTER the complete block output
(after Add when the residual exists, after Project BN when it does not).

DROPOUT RATE GUIDANCE
----------------------
  Block SpatialDropout2D (CONV_DROPOUT_RATE): 0.20-0.30 is the usual sweet
  spot; >0.50 rarely helps.
  Head Dropout (HEAD_DROPOUT_RATE): base model used 0.30; keep at or below.

NOTE ON TENSORBOARD (fixed):
  The original script built the TensorBoard callback inline
  (`tensorboard_dropout = TensorBoard(log_dir='...')`). The pipeline driver
  edits scripts by rewriting a whole "VAR = ..." line, so pointing it at
  `tensorboard_dropout` would have overwritten the TensorBoard object
  itself with a plain string, crashing `model.fit()`. The log directory is
  now its own variable (`TENSORBOARD_LOG_DIR_DROPOUT`) that the driver can
  safely edit.

Workflow:
  1. Load base model, auto-detect its layout (flat custom vs. nested Keras)
  2. Build new MobileNetV2 with SpatialDropout2D after every block, matching
     the detected layout structurally (so block 1 has an expand conv only
     if the base model actually has one)
  3. Transfer all weights by resolved name
  4. Evaluate before fine-tuning (dropout OFF at eval time)
  5. Fine-tune with fresh CosineDecay (dropout ON during training)
  6. Evaluate after fine-tuning
  7. Save to DROPOUT_MODEL_PATH
"""

import os
import random
import tensorflow as tf
import sys
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers.schedules import CosineDecay

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# ==============================================================================
# 1.  CONFIGURATION
# ==============================================================================
SAVED_BASE_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\base_applications_MobileNetV2_Full_Train.keras"
DROPOUT_MODEL_PATH    = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\MobileNetV2_dropout_0.1.keras"
dataset_dir= r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\DATASET\CSIR_Datasets"

IMG_SIZE   = 224
BATCH_SIZE = 8
SEED       = 42

CONV_DROPOUT_RATE = 0.1
HEAD_DROPOUT_RATE = 0.1

FINETUNE_EPOCHS   = 50
FINETUNE_PATIENCE = 5
FINETUNE_INIT_LR  = 5e-4

# ==============================================================================
# 2.  LOAD BASE MODEL  +  DETECT ITS LAYOUT
# ==============================================================================
base_model = tf.keras.models.load_model(SAVED_BASE_MODEL_PATH)
print("Base model loaded successfully.")
#base_model.summary()


def try_get_layer(model, name):
    try:
        return model.get_layer(name)
    except ValueError:
        return None


def find_nested_backbone(model):
    """Return the first top-level layer that is itself a Keras Model
    (i.e. a nested backbone, as produced by `x = MobileNetV2(...)(x)`), or
    None if the model is flat."""
    candidates = [l for l in model.layers if isinstance(l, tf.keras.Model)]
    if not candidates:
        return None
    return max(candidates, key=lambda l: len(l.layers))


def detect_base_layout(model):
    """
    Returns (mode, backbone) where mode is one of:
      "flat_custom"  - top-level layers are named stem_conv/block_N_expand/...
                        exactly, including a real block_1_expand.
      "nested_keras" - a nested tf.keras.applications.MobileNetV2 sub-model,
                        using real Keras layer names, block 1 has no expand.
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
    """Resolve this script's block numbering (1..17) to the actual layer
    names present in the loaded base model, for either layout."""
    if mode == "flat_custom":
        return dict(
            expand=f"block_{bid}_expand", expand_bn=f"block_{bid}_expand_bn",
            depthwise=f"block_{bid}_depthwise", depthwise_bn=f"block_{bid}_depthwise_bn",
            project=f"block_{bid}_project", project_bn=f"block_{bid}_project_bn",
            has_expand=True,
        )
    # nested_keras
    if bid == 1:
        return dict(
            expand=None, expand_bn=None,
            depthwise="expanded_conv_depthwise", depthwise_bn="expanded_conv_depthwise_BN",
            project="expanded_conv_project", project_bn="expanded_conv_project_BN",
            has_expand=False,
        )
    i = bid - 1
    return dict(
        expand=f"block_{i}_expand", expand_bn=f"block_{i}_expand_BN",
        depthwise=f"block_{i}_depthwise", depthwise_bn=f"block_{i}_depthwise_BN",
        project=f"block_{i}_project", project_bn=f"block_{i}_project_BN",
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
    """Locate the 128-unit hidden Dense and 1-unit classifier Dense layers
    by their `units`, working whether or not they're named fc1/classifier."""
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
for bid in range(1, 18):
    names = block_layer_names(bid, BASE_MODE)
    for key in ("depthwise", "depthwise_bn", "project", "project_bn"):
        if try_get_layer(backbone, names[key]) is None:
            missing.append(names[key])
    if names["has_expand"]:
        for key in ("expand", "expand_bn"):
            if try_get_layer(backbone, names[key]) is None:
                missing.append(names[key])
for name in (STEM_CONV_NAME, STEM_BN_NAME, FINAL_CONV_NAME, FINAL_BN_NAME):
    if try_get_layer(backbone, name) is None:
        missing.append(name)

if missing:
    raise ValueError(
        f"Base model ({BASE_MODE}) is missing expected layers: {missing}\n"
        "Make sure SAVED_BASE_MODEL_PATH points to the correct file."
    )
print("Layer-name check passed — all expected layers present.\n")

# ==============================================================================
# 3.  DATASET  (pipeline matches base model exactly)
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


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_bmp(image, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label


full_dataset = tf.data.Dataset.from_tensor_slices(
    (list(selected_files), list(selected_labels))
)

full_dataset = full_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

dataset_size = len(selected_files)               # 8000
train_size = int(0.65 * dataset_size)             # 5200
val_size = int(0.15 * dataset_size)               # 1200
test_size = dataset_size - train_size - val_size  # 1600

full_dataset = full_dataset.shuffle(
    dataset_size,
    seed=SEED,
    reshuffle_each_iteration=False
)

train_dataset = full_dataset.take(train_size)
remaining = full_dataset.skip(train_size)
val_dataset = remaining.take(val_size)
test_dataset = remaining.skip(val_size)

print("Train:", train_size)
print("Validation:", val_size)
print("Test:", test_size)

train_dataset = (
    train_dataset
    .shuffle(1000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    val_dataset
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_dataset = (
    test_dataset
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# ==============================================================================
# 4.  BUILD NEW MobileNetV2 WITH DROPOUT
#     Structurally mirrors the DETECTED base layout: block 1 gets an expand
#     conv only if the actual base model has one.
# ==============================================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])


def block_with_expand(x, filters, stride, block_id, expansion):
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

    x = layers.SpatialDropout2D(CONV_DROPOUT_RATE, name=f"sdrop_block_{block_id}")(x)
    return x


def block_no_expand(x, filters, stride, block_id):
    """Block 1 when the base model has no expand conv (real MobileNetV2
    behavior for expansion factor 1) - depthwise runs directly on input."""
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

    x = layers.SpatialDropout2D(CONV_DROPOUT_RATE, name=f"sdrop_block_{block_id}")(x)
    return x


def MobileNetV2_With_Dropout(base_mode):
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_layer")
    x = data_augmentation(inp)

    x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False, name="stem_conv")(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(max_value=6, name="stem_relu")(x)

    block_config = [
        (1,  16,  1,  1),
        (6,  24,  2,  2),
        (6,  24,  1,  3),
        (6,  32,  2,  4),
        (6,  32,  1,  5),
        (6,  32,  1,  6),
        (6,  64,  2,  7),
        (6,  64,  1,  8),
        (6,  64,  1,  9),
        (6,  64,  1, 10),
        (6,  96,  1, 11),
        (6,  96,  1, 12),
        (6,  96,  1, 13),
        (6, 160,  2, 14),
        (6, 160,  1, 15),
        (6, 160,  1, 16),
        (6, 320,  1, 17),
    ]

    for expansion, filters, stride, block_id in block_config:
        has_expand = block_layer_names(block_id, base_mode)["has_expand"]
        if has_expand:
            x = block_with_expand(x, filters, stride, block_id, expansion)
        else:
            x = block_no_expand(x, filters, stride, block_id)

    x = layers.Conv2D(1280, 1, padding='same', use_bias=False, name="final_conv")(x)
    x = layers.BatchNormalization(name="final_bn")(x)
    x = layers.ReLU(max_value=6, name="final_relu")(x)

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(128, activation='relu', name="fc1")(x)
    x = layers.Dropout(HEAD_DROPOUT_RATE, name="head_dropout")(x)
    out = layers.Dense(1, activation='sigmoid', name="classifier")(x)

    return Model(inputs=inp, outputs=out)


new_model = MobileNetV2_With_Dropout(BASE_MODE)
#new_model.summary()

# ==============================================================================
# 5.  TRANSFER WEIGHTS FROM BASE MODEL → NEW MODEL  (by resolved name)
# ==============================================================================
transferred = []

new_model.get_layer("stem_conv").set_weights(get_layer_checked(backbone, STEM_CONV_NAME).get_weights())
new_model.get_layer("stem_bn").set_weights(get_layer_checked(backbone, STEM_BN_NAME).get_weights())
transferred += ["stem_conv", "stem_bn"]

for bid in range(1, 18):
    names = block_layer_names(bid, BASE_MODE)

    if names["has_expand"]:
        new_model.get_layer(f"block_{bid}_expand").set_weights(
            get_layer_checked(backbone, names["expand"]).get_weights()
        )
        new_model.get_layer(f"block_{bid}_expand_bn").set_weights(
            get_layer_checked(backbone, names["expand_bn"]).get_weights()
        )
        transferred += [f"block_{bid}_expand", f"block_{bid}_expand_bn"]

    new_model.get_layer(f"block_{bid}_depthwise").set_weights(
        get_layer_checked(backbone, names["depthwise"]).get_weights()
    )
    new_model.get_layer(f"block_{bid}_depthwise_bn").set_weights(
        get_layer_checked(backbone, names["depthwise_bn"]).get_weights()
    )
    new_model.get_layer(f"block_{bid}_project").set_weights(
        get_layer_checked(backbone, names["project"]).get_weights()
    )
    new_model.get_layer(f"block_{bid}_project_bn").set_weights(
        get_layer_checked(backbone, names["project_bn"]).get_weights()
    )
    transferred += [f"block_{bid}_depthwise", f"block_{bid}_depthwise_bn",
                     f"block_{bid}_project", f"block_{bid}_project_bn"]

new_model.get_layer("final_conv").set_weights(get_layer_checked(backbone, FINAL_CONV_NAME).get_weights())
new_model.get_layer("final_bn").set_weights(get_layer_checked(backbone, FINAL_BN_NAME).get_weights())
new_model.get_layer("fc1").set_weights(head_hidden_layer.get_weights())
new_model.get_layer("classifier").set_weights(head_classifier_layer.get_weights())
transferred += ["final_conv", "final_bn", "fc1", "classifier"]

print(f"\nTransferred {len(transferred)} weight-bearing layers "
      f"(SpatialDropout2D/Add/ReLU/GAP layers carry no weights and are skipped).")
print("All weight-bearing layers matched successfully.")

# ==============================================================================
# 6.  EVALUATE BEFORE FINE-TUNING
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
# 7.  FINE-TUNE WITH DROPOUT ACTIVE
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

TENSORBOARD_LOG_DIR_DROPOUT = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\TRAINING\logs\MobileNetV2_dropout_0.1"
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
# 8.  FINAL EVALUATION
# ==============================================================================
loss_after, acc_after = new_model.evaluate(test_dataset, verbose=1)
print(f"\nTest loss     (after fine-tuning)  : {loss_after:.4f}")
print(f"Test accuracy (after fine-tuning)  : {acc_after * 100:.2f}%")
print(f"Best val accuracy during fine-tune : {max(history_ft.history['val_accuracy']) * 100:.2f}%")

# ==============================================================================
# 9.  SUMMARY
# ==============================================================================
base_params = sum(tf.keras.backend.count_params(w) for w in base_model.trainable_weights)
new_params = sum(tf.keras.backend.count_params(w) for w in new_model.trainable_weights)

print(f"\nBase model parameters  : {base_params:>12,}")
print(f"New  model parameters  : {new_params:>12,}")
print(f"Difference             : {new_params - base_params:>+12,}  (should be 0)")
print(f"\nNew  model test accuracy (before fine-tune) : {acc_before * 100:.2f}%")
print(f"New  model test accuracy (after  fine-tune) : {acc_after  * 100:.2f}%")
print("\nDropout configuration:")
print(f"  Blocks — SpatialDropout2D rate : {CONV_DROPOUT_RATE}  (sdrop_block_1 … sdrop_block_17)")
print(f"  Head   — Dropout          rate : {HEAD_DROPOUT_RATE}  (head_dropout, after Dense(128))")

# ==============================================================================
# 10.  SAVE
# ==============================================================================
new_model.save(DROPOUT_MODEL_PATH)
print(f"\nModel saved to: {DROPOUT_MODEL_PATH}")

import json
metrics = {"test_loss": float(loss_after), "test_accuracy": float(acc_after)}
with open("last_run_metrics.json", "w") as f:
    json.dump(metrics, f)