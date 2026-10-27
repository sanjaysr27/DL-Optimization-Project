# -*- coding: utf-8 -*-
"""
VGG16 — SpatialDropout2D After Every Conv Layer + Configurable Dense Dropout
=============================================================================

Architecture changes vs. base model:
--------------------------------------
  • SpatialDropout2D(CONV_DROPOUT_RATE) inserted after every real VGG16 conv
    layer (13 insertions, named sdrop_0 ... sdrop_12).
  • Dropout(DENSE_DROPOUT_RATE) replaces Dropout(0.5) in both dense units.
  • Conv layer names (block1_conv1, block1_conv2, ...) match the real
    tf.keras.applications.VGG16 backbone exactly, so weights transfer by name.
  • The base model backbone has NO BatchNorm and uses ReLU baked into Conv2D
    (not a separate LeakyReLU layer) — this script matches that, unlike the
    old version which assumed conv_N/bn_N/leaky_N flat naming.

Workflow:
  1. Load     base model from SAVED_BASE_MODEL_PATH
  2. Reach    into the nested VGG16 submodel and verify expected layers exist
  3. Build    new VGG16 with SpatialDropout2D inserted after each conv
  4. Transfer conv weights by name, dense weights by matching units/order
  5. Evaluate before fine-tuning  (dropout OFF at eval -> comparable to base)
  6. Fine-tune with fresh CosineDecay schedule (dropout ON during training)
  7. Evaluate after fine-tuning
  8. Save     to DROPOUT_MODEL_PATH
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
# NOTE: point this at your REAL working base model (the one with
# tf.keras.applications.VGG16(weights='imagenet', ...) nested inside),
# not the flexible/pruning-architecture model.
SAVED_BASE_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\base_applications_vgg16_Full_Train.keras"
DROPOUT_MODEL_PATH    = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\vgg16_leather_SpatialDropout60.keras"
DATASET_DIR           = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\DATASET\CSIR_Datasets"

IMG_SIZE   = 224
BATCH_SIZE = 8
SEED       = 42

# Dropout rate for ALL SpatialDropout2D layers (one after each of the 13 conv layers)
CONV_DROPOUT_RATE  = 0.60

# Dropout rate for both Dense layers in the classifier head
DENSE_DROPOUT_RATE = 0.60

# Fine-tuning hyper-parameters
FINETUNE_EPOCHS   = 50
FINETUNE_PATIENCE = 5
FINETUNE_INIT_LR  = 1e-5

# ==============================================================================
# 2.  LOAD BASE MODEL
# ==============================================================================
base_model = tf.keras.models.load_model(SAVED_BASE_MODEL_PATH)
print("Base model loaded successfully.")
base_model.summary()

# Reach into the nested VGG16 submodel (base_model wraps it as x = base_model(x))
base_vgg = None
for layer in base_model.layers:
    if isinstance(layer, Model) and "vgg16" in layer.name.lower():
        base_vgg = layer
        break

if base_vgg is None:
    print("\nCould not auto-detect nested VGG16 submodel. Base model layers:")
    for l in base_model.layers:
        print(f"   {l.name} ({type(l).__name__})")
    raise ValueError(
        "Could not find nested VGG16 submodel in base_model. "
        "Check the layer names printed above and adjust the name match."
    )

# Standard tf.keras.applications.VGG16 conv layer names, in order
VGG16_CONV_NAMES = [
    "block1_conv1", "block1_conv2",
    "block2_conv1", "block2_conv2",
    "block3_conv1", "block3_conv2", "block3_conv3",
    "block4_conv1", "block4_conv2", "block4_conv3",
    "block5_conv1", "block5_conv2", "block5_conv3",
]

missing = [n for n in VGG16_CONV_NAMES if not any(l.name == n for l in base_vgg.layers)]
if missing:
    raise ValueError(
        f"Nested VGG16 submodel is missing expected conv layers: {missing}\n"
        "Make sure SAVED_BASE_MODEL_PATH points to a model with a real "
        "tf.keras.applications.VGG16 backbone."
    )
print("Layer-name check passed — all expected VGG16 conv layers present.")

# Locate the classifier head Dense layers by units (they were left unnamed
# in the base model, so match by role instead of name)
base_dense_hidden = [l for l in base_model.layers if isinstance(l, layers.Dense) and l.units != 1]
base_output_layer = [l for l in base_model.layers if isinstance(l, layers.Dense) and l.units == 1]

if len(base_output_layer) != 1:
    raise ValueError(f"Expected exactly one Dense(1) output layer, found {len(base_output_layer)}.")
base_output_layer = base_output_layer[0]

print(f"Found {len(base_dense_hidden)} hidden Dense layer(s) in classifier head "
      f"(units: {[l.units for l in base_dense_hidden]}).")

# ==============================================================================
# 3.  DATASET  (pipeline order matches base model script exactly)
# ==============================================================================
random.seed(SEED)

# Paths to each class
nonDef_dir = os.path.join(DATASET_DIR, "0_DefectFree")
def_dir = os.path.join(DATASET_DIR, "1_Defective")

# Get image filenames
nonDef_files = [os.path.join(nonDef_dir, f) for f in os.listdir(nonDef_dir)]
def_files = [os.path.join(def_dir, f) for f in os.listdir(def_dir)]

# Randomly select 3992 from each class
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
    return image, label

full_dataset = tf.data.Dataset.from_tensor_slices(
    (list(selected_files), list(selected_labels))
)
full_dataset = full_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

dataset_size = len(selected_files)
train_size = int(0.65 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

# Shuffle BEFORE splitting — same order as base model
full_dataset = full_dataset.shuffle(
    dataset_size, seed=SEED, reshuffle_each_iteration=False
)

train_dataset = full_dataset.take(train_size)
remaining     = full_dataset.skip(train_size)
val_dataset   = remaining.take(val_size)
test_dataset  = remaining.skip(val_size)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
])

def preprocess_img(image, label):
    image = tf.keras.applications.vgg16.preprocess_input(image)
    return image, label

train_dataset = train_dataset.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = val_dataset.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ==============================================================================
# 4.  BUILD NEW VGG16 WITH SPATIAL DROPOUT
#     SpatialDropout2D(CONV_DROPOUT_RATE) is placed after EVERY conv layer
#     (13 times total, named sdrop_0 through sdrop_12). Dropout(DENSE_DROPOUT_RATE)
#     follows each hidden Dense layer.
#
#     Conv layer names match the real VGG16 backbone exactly (block1_conv1, ...)
#     so weights transfer cleanly by name lookup. sdrop_N layers carry no
#     weights and are invisible to the weight-transfer loop.
# ==============================================================================
def VGG16_With_Dropout(dense_units_list, augmentation_layer):
    inp = layers.Input(shape=(224, 224, 3), name="input_layer")
    x = augmentation_layer(inp)

    block_filters = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    blocks = [2, 2, 3, 3, 3]
    cfg_idx = 0

    for block_idx, num_layers in enumerate(blocks, start=1):
        for conv_in_block in range(1, num_layers + 1):
            name = f"block{block_idx}_conv{conv_in_block}"
            x = layers.Conv2D(block_filters[cfg_idx], (3, 3), padding='same',
                               activation='relu', name=name)(x)
            x = layers.SpatialDropout2D(CONV_DROPOUT_RATE, name=f"sdrop_{cfg_idx}")(x)
            cfg_idx += 1
        x = layers.MaxPooling2D((2, 2), name=f"block{block_idx}_pool")(x)

    # ── Classifier Head ───────────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    for i, units in enumerate(dense_units_list, start=1):
        x = layers.Dense(units, activation='relu', name=f"dense_{i}")(x)
        x = layers.Dropout(DENSE_DROPOUT_RATE, name=f"drop_{i}")(x)
    out = layers.Dense(1, activation='sigmoid', name="output")(x)

    return Model(inputs=inp, outputs=out)


dense_units_list = [l.units for l in base_dense_hidden]
new_model = VGG16_With_Dropout(dense_units_list, data_augmentation)
new_model.summary()

# ==============================================================================
# 5.  TRANSFER WEIGHTS FROM BASE MODEL → NEW MODEL
# ==============================================================================

# --- Conv layers: match by name (real VGG16 names, identical on both sides) ---
transferred = []
skipped = []

base_conv_by_name = {l.name: l for l in base_vgg.layers if isinstance(l, layers.Conv2D)}
new_conv_by_name = {l.name: l for l in new_model.layers if isinstance(l, layers.Conv2D)}

for name, target_layer in new_conv_by_name.items():
    if name in base_conv_by_name:
        target_layer.set_weights(base_conv_by_name[name].get_weights())
        transferred.append(name)
    else:
        skipped.append(name)

# --- Dense hidden layers: match by order ---
new_dense_hidden = [l for l in new_model.layers if isinstance(l, layers.Dense) and l.units != 1]
for base_layer, target_layer in zip(base_dense_hidden, new_dense_hidden):
    target_layer.set_weights(base_layer.get_weights())
    transferred.append(target_layer.name)

# --- Output layer ---
new_output_layer = [l for l in new_model.layers if isinstance(l, layers.Dense) and l.units == 1][0]
new_output_layer.set_weights(base_output_layer.get_weights())
transferred.append(new_output_layer.name)

print(f"\nWeight-bearing layers transferred: {len(transferred)}")
for name in transferred:
    print(f"   {name}")

if skipped:
    print(f"\n  No base weights found for: {skipped}")
    print("   These layers keep random initialisation.")
else:
    print("\nAll weight-bearing layers matched successfully.")

# ==============================================================================
# 6.  EVALUATE BEFORE FINE-TUNING
#     Compile with a fixed scalar LR for this eval — the LR value is irrelevant
#     during evaluate() but we need a compiled model. A fresh CosineDecay
#     schedule is built separately for fine-tuning below so that no decay
#     steps are consumed here.
# ==============================================================================
new_model.compile(
    optimizer=Adam(learning_rate=FINETUNE_INIT_LR),   # fixed scalar — eval only
    loss=BinaryCrossentropy(),
    metrics=['accuracy']
)

loss_before, acc_before = new_model.evaluate(test_dataset, verbose=1)
print(f"\nTest loss     (before fine-tuning) : {loss_before:.4f}")
print(f"Test accuracy (before fine-tuning) : {acc_before * 100:.2f}%")
print("Expected: close to base model accuracy.")
print("SpatialDropout2D and Dropout are both DISABLED at eval time,")
print("so inference is identical to the original base model.")

# ==============================================================================
# 7.  FINE-TUNE WITH DROPOUT ACTIVE
#     Build a fresh CosineDecay schedule here so decay starts from step 0
#     (not partially consumed by the evaluate() call above).
# ==============================================================================
steps_per_epoch_ft = train_size // BATCH_SIZE
total_ft_steps     = steps_per_epoch_ft * FINETUNE_EPOCHS

lr_schedule_ft = CosineDecay(
    initial_learning_rate=FINETUNE_INIT_LR,
    decay_steps=total_ft_steps,
    alpha=0.01   # floor: FINETUNE_INIT_LR * 0.01 at end of schedule
)

# Recompile with the cosine-decaying LR for training
new_model.compile(
    optimizer=Adam(learning_rate=lr_schedule_ft),
    loss=BinaryCrossentropy(),
    metrics=['accuracy']
)

print(f"NOTE: CONV_DROPOUT_RATE = {CONV_DROPOUT_RATE}.")
tensorboard_dropout = TensorBoard(log_dir='logs\\VGG16_leather_SpatialDropout60')

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
print(f"Best val accuracy during fine-tune : "
      f"{max(history_ft.history['val_accuracy']) * 100:.2f}%")

# ==============================================================================
# 9.  SUMMARY
# ==============================================================================
base_params = sum(
    tf.keras.backend.count_params(w) for w in base_model.trainable_weights
)
new_params = sum(
    tf.keras.backend.count_params(w) for w in new_model.trainable_weights
)

print(f"\nBase model parameters  : {base_params:>12,}")
print(f"New  model parameters  : {new_params:>12,}")
print(f"Difference             : {new_params - base_params:>+12,}  (should be 0 — dropout adds no weights)")
print(f"New  model test accuracy (before fine-tune) : {acc_before * 100:.2f}%")
print(f"New  model test accuracy (after  fine-tune) : {acc_after  * 100:.2f}%")
print("\nDropout configuration:")
print(f"  Conv layers  — SpatialDropout2D rate : {CONV_DROPOUT_RATE}"
      f"  (sdrop_0 ... sdrop_12, one after each of the 13 conv layers)")
print(f"  Dense layers — Dropout          rate : {DENSE_DROPOUT_RATE}"
      f"  (drop_1, drop_2)")

# ==============================================================================
# 10.  SAVE
# ==============================================================================
new_model.save(DROPOUT_MODEL_PATH)
print(f"\nModel saved to: {DROPOUT_MODEL_PATH}")