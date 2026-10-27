# -*- coding: utf-8 -*-
"""
ResNet50 — SpatialDropout2D After Every Bottleneck Block
=========================================================

WHY DROPOUT PLACEMENT DIFFERS FROM VGG16 BUT IS CLEANER THAN MobileNetV2
--------------------------------------------------------------------------
ResNet50 bottleneck block internal structure:

    shortcut ─────────────────────────────────────┐
      1x1 reduce  → BN → ReLU                     │
      3x3         → BN → ReLU                     │
      1x1 expand  → BN ──────────────── Add ──────┘
                                         ↓
                                      out_relu        ← dropout goes HERE

The final activation (out_relu) comes AFTER the Add, so the entire block
output — both branches fully merged and activated — is available as one
clean tensor before we insert SpatialDropout2D.

This is safer than MobileNetV2 where dropout had to go after the Add but
the Add had no following activation, meaning the dropout sat on a
pre-activation tensor.  Here the tensor is post-activation, which is the
textbook correct position for SpatialDropout2D.

For blocks that USE a projection shortcut (conv_shortcut=True, i.e. the
first block of every stage), the shortcut itself is a Conv2D + BN with no
activation, so the Add still resolves both branches cleanly before out_relu
and our dropout.

Total blocks: 3 + 4 + 6 + 3 = 16 blocks → 16 SpatialDropout2D layers
named sdrop_stage{S}_block{B}.

DROPOUT RATE GUIDANCE
----------------------
Block SpatialDropout2D (CONV_DROPOUT_RATE):
  0.10 – 0.20  : Light regularisation, very stable — good starting point
  0.20 – 0.30  : Recommended sweet spot for ResNet50 per-block dropout
  0.30 – 0.40  : Heavy; stage 1 (only 256 channels) may lose too much info
  > 0.50       : Not recommended — ResNet50's skip connections already act
                 as a strong implicit regulariser; aggressive spatial dropout
                 fights against that

Head Dropout (HEAD_DROPOUT_RATE):
  Base model uses 0.30.  Keep at or below that value.
  The head is only Dense(128) → Dense(1) so high rates give minimal benefit.

COMPATIBILITY NOTES (vs. base model)
--------------------------------------
  • All weight-bearing layer names preserved exactly:
      stem_conv, stem_bn
      stage{S}_block{B}_shortcut_conv, stage{S}_block{B}_shortcut_bn
      stage{S}_block{B}_conv1, stage{S}_block{B}_bn1
      stage{S}_block{B}_conv2, stage{S}_block{B}_bn2
      stage{S}_block{B}_conv3, stage{S}_block{B}_bn3
      fc1, classifier
  • Add, ReLU, MaxPool, GAP, Dropout — no weights, irrelevant to transfer.
  • New SpatialDropout2D layers named sdrop_stage{S}_block{B} — no weights.
  • Data augmentation baked inside the model exactly as in the base model.
  • Normalisation done inside load_image (matches base model pipeline).
  • Head dropout renamed "head_dropout" — carries no weights, no effect on
    transfer, just clearer in the summary alongside 16 block dropouts.

BUG FROM ORIGINAL VGG16 SCRIPT AVOIDED
----------------------------------------
  model.compile() for evaluate() uses a fixed scalar LR.
  A fresh CosineDecay is built immediately before model.fit() so no decay
  steps are consumed during evaluate().

Workflow:
  1. Load base model from SAVED_BASE_MODEL_PATH
  2. Verify all expected weight-bearing layer names are present
  3. Build new ResNet50 with SpatialDropout2D after every bottleneck block
  4. Transfer all weights by layer name
  5. Evaluate before fine-tuning (dropout OFF at eval time)
  6. Fine-tune with fresh CosineDecay (dropout ON during training)
  7. Evaluate after fine-tuning
  8. Save to DROPOUT_MODEL_PATH
  9. Prediction sanity check
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
SAVED_BASE_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\resnet50_base.keras"
DROPOUT_MODEL_PATH    = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\resnet50_with_dropout60new.keras"
DATASET_DIR           = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\DATASET\PetImages"

IMG_SIZE   = 224
BATCH_SIZE = 8
SEED       = 42

# SpatialDropout2D rate applied after every bottleneck block (16 blocks total)
# Recommended range: 0.10 – 0.30
CONV_DROPOUT_RATE = 0.60

# Dropout rate for the Dense(128) head layer
# Base model used 0.30 — keep at or below that value
HEAD_DROPOUT_RATE = 0.60

# Stage configuration — must match base model exactly
# (f1, f2, f3, num_blocks, first_block_stride)
STAGE_CFG = [
    (64,  64,  256,  3, 1),   # stage 1 → output 56x56x256
    (128, 128, 512,  4, 2),   # stage 2 → output 28x28x512
    (256, 256, 1024, 6, 2),   # stage 3 → output 14x14x1024
    (512, 512, 2048, 3, 2),   # stage 4 → output  7x7x2048
]

# Fine-tuning hyper-parameters
FINETUNE_EPOCHS   = 50
FINETUNE_PATIENCE = 15
FINETUNE_INIT_LR  = 1e-4

# ==============================================================================
# 2.  LOAD BASE MODEL
# ==============================================================================
base_model = tf.keras.models.load_model(SAVED_BASE_MODEL_PATH)
print("Base model loaded successfully.")
base_model.summary()

# Build the full list of expected weight-bearing layer names from STAGE_CFG
# so this check is always in sync with the architecture — no hardcoding.
expected_names = ["stem_conv", "stem_bn"]

for stage_idx, (f1, f2, f3, num_blocks, first_stride) in enumerate(
    STAGE_CFG, start=1
):
    for block_idx in range(1, num_blocks + 1):
        nb = f"stage{stage_idx}_block{block_idx}"
        if block_idx == 1:
            # First block of every stage has a projection shortcut
            expected_names += [f"{nb}_shortcut_conv", f"{nb}_shortcut_bn"]
        expected_names += [
            f"{nb}_conv1", f"{nb}_bn1",
            f"{nb}_conv2", f"{nb}_bn2",
            f"{nb}_conv3", f"{nb}_bn3",
        ]

expected_names += ["fc1", "classifier"]
# Note: Add, ReLU, MaxPool, GAP, Dropout carry no weights — excluded.

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
    # Normalisation done here — matches base model pipeline exactly
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
# 4.  BUILD NEW ResNet50 WITH DROPOUT
#
#     Data augmentation baked inside the model (same as base model).
#     SpatialDropout2D(CONV_DROPOUT_RATE) inserted after out_relu in every
#     bottleneck block — the post-Add, post-activation output tensor.
#     This is the cleanest possible dropout position in a ResNet block.
# ==============================================================================

# Augmentation pipeline — identical to base model
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="data_augmentation")


def bottleneck_block_with_dropout(x, f1, f2, f3,
                                   stage, block, stride, conv_shortcut):
    """
    Identical to the base model's bottleneck_block EXCEPT that
    SpatialDropout2D(CONV_DROPOUT_RATE) is appended after out_relu.

    Internal layer names are unchanged so weights transfer by name.
    The new sdrop layer carries no weights.

    Dropout position:
        shortcut ──────────────────────────────┐
          conv1 → bn1 → relu1                  │
          conv2 → bn2 → relu2                  │
          conv3 → bn3 ──────────── Add ────────┘
                                    ↓
                                 out_relu
                                    ↓
                          SpatialDropout2D   ← HERE (post-activation)
    """
    nb = f"stage{stage}_block{block}"

    shortcut = x
    if conv_shortcut:
        shortcut = layers.Conv2D(
            f3, 1, strides=stride, padding='same', use_bias=False,
            name=f"{nb}_shortcut_conv"
        )(x)
        shortcut = layers.BatchNormalization(
            name=f"{nb}_shortcut_bn"
        )(shortcut)

    # 1x1 reduce
    x = layers.Conv2D(
        f1, 1, padding='same', use_bias=False,
        name=f"{nb}_conv1"
    )(x)
    x = layers.BatchNormalization(name=f"{nb}_bn1")(x)
    x = layers.ReLU(name=f"{nb}_relu1")(x)

    # 3x3 (stride lives here — ResNet v1.5 convention)
    x = layers.Conv2D(
        f2, 3, strides=stride, padding='same', use_bias=False,
        name=f"{nb}_conv2"
    )(x)
    x = layers.BatchNormalization(name=f"{nb}_bn2")(x)
    x = layers.ReLU(name=f"{nb}_relu2")(x)

    # 1x1 expand
    x = layers.Conv2D(
        f3, 1, padding='same', use_bias=False,
        name=f"{nb}_conv3"
    )(x)
    x = layers.BatchNormalization(name=f"{nb}_bn3")(x)

    # Residual Add + final activation
    x = layers.Add(name=f"{nb}_add")([shortcut, x])
    x = layers.ReLU(name=f"{nb}_out_relu")(x)

    # SpatialDropout2D after the complete, activated block output
    x = layers.SpatialDropout2D(
        CONV_DROPOUT_RATE,
        name=f"sdrop_{nb}"
    )(x)

    return x


def ResNet50_With_Dropout():

    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_layer")

    # Data augmentation baked in — identical to base model
    x = data_augmentation(inp)

    # ── Stem ──────────────────────────────────────────────────────────────────
    x = layers.Conv2D(
        64, 7, strides=2, padding='same', use_bias=False,
        name="stem_conv"
    )(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same', name="stem_pool")(x)

    # ── Stages ────────────────────────────────────────────────────────────────
    for stage_idx, (f1, f2, f3, num_blocks, first_stride) in enumerate(
        STAGE_CFG, start=1
    ):
        for block_idx in range(1, num_blocks + 1):
            x = bottleneck_block_with_dropout(
                x, f1, f2, f3,
                stage=stage_idx,
                block=block_idx,
                stride=first_stride if block_idx == 1 else 1,
                conv_shortcut=(block_idx == 1)
            )

    # ── Classifier Head ───────────────────────────────────────────────────────
    x   = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x   = layers.Dense(128, activation='relu', name="fc1")(x)
    x   = layers.Dropout(HEAD_DROPOUT_RATE, name="head_dropout")(x)
    out = layers.Dense(1, activation='sigmoid', name="classifier")(x)

    return Model(inputs=inp, outputs=out)


new_model = ResNet50_With_Dropout()
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
        continue   # SpatialDropout2D, Add, ReLU, MaxPool, GAP — no weights
    if layer.name in base_weights:
        layer.set_weights(base_weights[layer.name])
        transferred.append(layer.name)
    else:
        skipped.append(layer.name)

print(f"\nWeight-bearing layers in base model : {len(base_weights)}")
print(f"Transferred to new model            : {len(transferred)}")
for name in transferred:
    print(f"   {name}")

if skipped:
    print(f"\n  No base weights found for: {skipped}")
    print("   These layers keep random initialisation.")
else:
    print("\nAll weight-bearing layers matched successfully.")

# ==============================================================================
# 6.  EVALUATE BEFORE FINE-TUNING
#     Fixed scalar LR for compile — irrelevant during evaluate().
#     Fresh CosineDecay is built below so no decay steps are consumed here.
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

tensorboard_dropout = TensorBoard(log_dir='logs\\ResNet50_with_dropout60new')

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
print(f"  Blocks — SpatialDropout2D rate : {CONV_DROPOUT_RATE}")
print("           sdrop_stage1_block1 … sdrop_stage4_block3")
print("           (16 layers total: 3 + 4 + 6 + 3 blocks)")
print(f"  Head   — Dropout          rate : {HEAD_DROPOUT_RATE}"
      f"  (head_dropout, after Dense(128))")

# ==============================================================================
# 10.  SAVE
# ==============================================================================
new_model.save(DROPOUT_MODEL_PATH)
print(f"\nModel saved to: {DROPOUT_MODEL_PATH}")

