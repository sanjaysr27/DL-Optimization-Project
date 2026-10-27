# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 14:19:18 2026

@author: PS1-2026
"""
"""
MobileNetV2 — SpatialDropout2D After Every Inverted Residual Block
===================================================================

WHY DROPOUT PLACEMENT DIFFERS FROM VGG16
-----------------------------------------
In VGG16 every conv layer is a straight chain:
    Conv → BN → Activation → [SpatialDropout2D] → next Conv
Inserting SpatialDropout2D anywhere in that chain is safe.

In MobileNetV2 each inverted residual block looks like this:

    inp ──────────────────────────────────────────┐
         Expand Conv → BN → ReLU6                 │  (skip exists only when
         Depthwise   → BN → ReLU6                 │   stride==1 and
         Project     → BN ─────────────────── Add─┘   channels match)

  • SpatialDropout2D INSIDE the block (between Project BN and the Add) would
    randomly zero channels on one branch of the Add but not the other.
    The two tensors being added would have different expected magnitudes at
    training time vs. eval time, corrupting the residual learning signal.

  • The safe location is AFTER the complete block output (after Add when the
    residual exists, after Project BN when it does not).  At that point the
    tensor is a complete feature map — dropping whole channels here pushes
    the next block to learn redundant, robust representations.

  • Block 1 uses expansion=1 and has NO expand conv (input_channels * 1 == input).
    A SpatialDropout2D rate that is too high here (the tensor is only 16 channels
    wide) can aggressively destroy features early. The same CONV_DROPOUT_RATE is
    applied uniformly; reduce if block_1 causes instability.

  • The head Dropout rate is separate and configurable via HEAD_DROPOUT_RATE.

DROPOUT RATE GUIDANCE
----------------------
  Block SpatialDropout2D (CONV_DROPOUT_RATE):
    0.10 – 0.20  : Light regularisation, very stable training
    0.20 – 0.30  : Recommended sweet spot for MobileNetV2 per-block dropout
    0.40 – 0.50  : Heavy; training will be noisier, may need more epochs
    > 0.50       : Rarely beneficial; inverted residual blocks are already
                   parameter-efficient — aggressive spatial dropout often
                   hurts more than it helps

  Head Dropout (HEAD_DROPOUT_RATE):
    Base model uses 0.30.  Keeping it at 0.30 or slightly lower (0.10–0.20)
    is safe.  Since the head is only one Dense(128) layer, increasing it
    much above 0.40 usually harms accuracy without regularisation benefit.

COMPATIBILITY NOTES (vs. base model)
--------------------------------------
  • All weight-bearing layer names are preserved exactly:
      stem_conv, stem_bn, block_N_expand, block_N_expand_bn,
      block_N_depthwise, block_N_depthwise_bn, block_N_project,
      block_N_project_bn, block_N_add (where present),
      final_conv, final_bn, fc1, classifier
  • New SpatialDropout2D layers are named sdrop_block_N — they carry no
    weights and are invisible to the weight-transfer loop.
  • Data augmentation is baked into the model (inside the Input→stem path)
    exactly as in the base model — reproduced identically here.
  • head Dropout name changes from "dropout" → "head_dropout" to avoid
    confusion; since Dropout carries no weights this has no effect on
    weight transfer.



Workflow:
  1. Load base model from SAVED_BASE_MODEL_PATH
  2. Verify all expected weight-bearing layer names are present
  3. Build new MobileNetV2 with SpatialDropout2D after every block
  4. Transfer all weights by layer name
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
SAVED_BASE_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\mobilenetv2_base.keras"
DROPOUT_MODEL_PATH    = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\mobilenetv2_with_dropout60_15P.keras"
DATASET_DIR           = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\DATASET\PetImages"

IMG_SIZE   = 224
BATCH_SIZE = 8
SEED       = 42

# SpatialDropout2D rate applied after every inverted residual block (17 blocks)
# Recommended range: 0.10 – 0.30  (see header notes above)
CONV_DROPOUT_RATE = 0.60

# Dropout rate for the Dense(128) head layer
# Base model used 0.30 — keep at or below that value
HEAD_DROPOUT_RATE = 0.60

# Fine-tuning hyper-parameters
FINETUNE_EPOCHS   = 50
FINETUNE_PATIENCE = 15
FINETUNE_INIT_LR  = 5e-4   # slightly higher than VGG16 fine-tune; MobileNetV2
                             # is a lighter model and benefits from a larger LR

# ==============================================================================
# 2.  LOAD BASE MODEL
# ==============================================================================
base_model = tf.keras.models.load_model(SAVED_BASE_MODEL_PATH)
print("Base model loaded successfully.")
base_model.summary()

# Verify all weight-bearing layer names before doing anything else.
# Prevents silent weight-skip failures if the wrong file is loaded.
expected_names = (
    ["stem_conv", "stem_bn"] +
    [name
     for i in range(1, 18)
     for name in (
         [f"block_{i}_expand",     f"block_{i}_expand_bn",
          f"block_{i}_depthwise",  f"block_{i}_depthwise_bn",
          f"block_{i}_project",    f"block_{i}_project_bn"]
     )] +
    ["final_conv", "final_bn", "fc1", "classifier"]
    # block_N_add layers carry no weights — excluded intentionally
    # dropout layer in base model carries no weights — excluded intentionally
)

missing = [n for n in expected_names
           if not any(l.name == n for l in base_model.layers)]

if missing:
    raise ValueError(
        f"Base model is missing expected layers: {missing}\n"
        "Make sure SAVED_BASE_MODEL_PATH points to the correct file."
    )
print("Layer-name check passed — all expected layers present.")

# ==============================================================================
# 3.  DATASET  (pipeline matches base model exactly)
# ==============================================================================
random.seed(SEED)

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


def load_image(path, label):
    # Normalisation is done here (matches base model) — unlike the VGG16 scripts
    # where normalisation happened in a separate preprocess step.
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


full_dataset = tf.data.Dataset.from_tensor_slices(
    (list(selected_files), list(selected_labels))
)
full_dataset = full_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
full_dataset = full_dataset.shuffle(
    dataset_size, seed=SEED, reshuffle_each_iteration=False
)

train_dataset = full_dataset.take(train_size)
remaining     = full_dataset.skip(train_size)
val_dataset   = remaining.take(val_size)
test_dataset  = remaining.skip(val_size)

# Pipeline order matches base model: shuffle → batch → prefetch for train;
# batch → prefetch for val/test.
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
# 4.  BUILD NEW MobileNetV2 WITH DROPOUT
#
#     Data augmentation is baked inside the model (same as base model).
#     SpatialDropout2D(CONV_DROPOUT_RATE) is added after every complete
#     inverted residual block — after the Add when a residual exists,
#     after the Project BN when it does not.
# ==============================================================================

# Augmentation pipeline — identical to base model
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])


def inverted_residual_block_with_dropout(x,
                                         expansion,
                                         filters,
                                         stride,
                                         block_id):
    """
    Identical to the base model's inverted_residual_block EXCEPT that a
    SpatialDropout2D is appended after the full block output.

    Layer names inside the block are unchanged so weights transfer by name.
    The new SpatialDropout2D is named sdrop_block_N and carries no weights.
    """
    inp = x
    input_channels = x.shape[-1]

    # ── Expansion phase ───────────────────────────────────────────────────────
    # Block 1 has expansion=1; input_channels * 1 == input_channels, so the
    # expand conv is still built (unlike the official MobileNetV2 which skips
    # it) — matches base model behaviour exactly.
    x = layers.Conv2D(
        input_channels * expansion,
        1,
        padding='same',
        use_bias=False,
        name=f"block_{block_id}_expand"
    )(x)
    x = layers.BatchNormalization(name=f"block_{block_id}_expand_bn")(x)
    x = layers.ReLU(max_value=6,          name=f"block_{block_id}_expand_relu")(x)

    # ── Depthwise Convolution ─────────────────────────────────────────────────
    x = layers.DepthwiseConv2D(
        3,
        strides=stride,
        padding='same',
        use_bias=False,
        name=f"block_{block_id}_depthwise"
    )(x)
    x = layers.BatchNormalization(name=f"block_{block_id}_depthwise_bn")(x)
    x = layers.ReLU(max_value=6,          name=f"block_{block_id}_depthwise_relu")(x)

    # ── Projection ────────────────────────────────────────────────────────────
    x = layers.Conv2D(
        filters,
        1,
        padding='same',
        use_bias=False,
        name=f"block_{block_id}_project"
    )(x)
    x = layers.BatchNormalization(name=f"block_{block_id}_project_bn")(x)

    # ── Residual connection (only when stride==1 and channels match) ──────────
    if stride == 1 and input_channels == filters:
        x = layers.Add(name=f"block_{block_id}_add")([inp, x])

    # ── SpatialDropout2D AFTER complete block output ──────────────────────────
    # Safe here: the tensor is a finished feature map; the residual Add (if any)
    # has already been resolved.  Dropout operates on the combined output,
    # not on one branch of an addition.
    x = layers.SpatialDropout2D(
        CONV_DROPOUT_RATE,
        name=f"sdrop_block_{block_id}"
    )(x)

    return x


def MobileNetV2_With_Dropout():

    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_layer")

    # Data augmentation baked in — identical to base model
    x = data_augmentation(inp)

    # ── Stem ──────────────────────────────────────────────────────────────────
    x = layers.Conv2D(32, 3, strides=2, padding='same',
                      use_bias=False, name="stem_conv")(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(max_value=6, name="stem_relu")(x)

    # ── Inverted Residual Blocks ───────────────────────────────────────────────
    # Configuration matches base model exactly:
    #   (expansion, filters, stride, block_id)
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
        x = inverted_residual_block_with_dropout(
            x,
            expansion=expansion,
            filters=filters,
            stride=stride,
            block_id=block_id
        )

    # ── Final Conv ────────────────────────────────────────────────────────────
    x = layers.Conv2D(1280, 1, padding='same',
                      use_bias=False, name="final_conv")(x)
    x = layers.BatchNormalization(name="final_bn")(x)
    x = layers.ReLU(max_value=6, name="final_relu")(x)

    # ── Classifier Head ───────────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(128, activation='relu', name="fc1")(x)
    x = layers.Dropout(HEAD_DROPOUT_RATE, name="head_dropout")(x)
    out = layers.Dense(1, activation='sigmoid', name="classifier")(x)

    return Model(inputs=inp, outputs=out)


new_model = MobileNetV2_With_Dropout()
new_model.summary()

# ==============================================================================
# 5.  TRANSFER WEIGHTS FROM BASE MODEL → NEW MODEL
# ==============================================================================
base_weights = {
    layer.name: layer.get_weights()
    for layer in base_model.layers
    if len(layer.get_weights()) > 0
}

transferred = []
skipped     = []

for layer in new_model.layers:
    if len(layer.get_weights()) == 0:
        continue   # SpatialDropout2D, Add, ReLU, GAP, etc. — no weights
    if layer.name in base_weights:
        layer.set_weights(base_weights[layer.name])
        transferred.append(layer.name)
    else:
        skipped.append(layer.name)

print(f"\nWeight-bearing layers in base model : {len(base_weights)}")
print(f"Transferred to new model            : {len(transferred)}")
for name in transferred:
    print(f"  {name}")

if skipped:
    print(f"\n  No base weights found for: {skipped}")
    print("   These layers keep random initialisation.")
else:
    print("\nAll weight-bearing layers matched successfully.")

# ==============================================================================
# 6.  EVALUATE BEFORE FINE-TUNING
#     Fixed scalar LR for compile — LR has no effect during evaluate().
#     A fresh CosineDecay is built below for fine-tuning so no decay steps
#     are consumed here.
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

print(f"\nFine-tune LR  : CosineDecay {FINETUNE_INIT_LR:.0e} → "
      f"{FINETUNE_INIT_LR * 0.01:.0e}")
print(f"Max epochs    : {FINETUNE_EPOCHS}  "
      f"(early stopping patience = {FINETUNE_PATIENCE})")
print(f"CONV_DROPOUT_RATE = {CONV_DROPOUT_RATE}  "
      f"HEAD_DROPOUT_RATE = {HEAD_DROPOUT_RATE}")

tensorboard_dropout = TensorBoard(log_dir='logs\\MobileNetV2_with_dropout60_15P')

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
print(f"Difference             : {new_params - base_params:>+12,}  (should be 0)")
print("\nBase model test accuracy (reported)         : see base model output")
print(f"New  model test accuracy (before fine-tune) : {acc_before * 100:.2f}%")
print(f"New  model test accuracy (after  fine-tune) : {acc_after  * 100:.2f}%")
print("\nDropout configuration:")
print(f"  Blocks — SpatialDropout2D rate : {CONV_DROPOUT_RATE}"
      f"  (sdrop_block_1 … sdrop_block_17, one after each of the 17 blocks)")
print(f"  Head   — Dropout          rate : {HEAD_DROPOUT_RATE}"
      f"  (head_dropout, after Dense(128))")

# ==============================================================================
# 10.  SAVE
# ==============================================================================
new_model.save(DROPOUT_MODEL_PATH)
print(f"\nModel saved to: {DROPOUT_MODEL_PATH}")

