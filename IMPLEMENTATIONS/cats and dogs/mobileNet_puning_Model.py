# -*- coding: utf-8 -*-

"""

Structural Channel Pruning — MobileNetV2 (Cat vs Dog)

Compatible with mobilenetv2_base.keras produced by the base-model training script.



PRUNING STRATEGY

----------------

True structural pruning physically removes channels from the model graph.

For MobileNetV2 we prune along two axes independently:



  1. Expansion channels  (inside every inverted-residual block)

        expand conv  ──►  depthwise conv  ──►  project conv

     Pruning here is always safe because these are internal, block-local

     channels. The block's OUTPUT width (project conv filters) is kept

     unchanged, so skip-connections are never disturbed.



  2. Final-conv channels  (the 1280-channel bottleneck before the head)

     After pruning we propagate the new width into the first Dense layer.



Importance criterion: L1 norm of each output filter (sum of |weights|).

Least-important channels are removed; the rest are packed into a new,

smaller model.



After the architecture is rebuilt and pruned weights are transplanted, the

model is fine-tuned on the same dataset to recover any accuracy lost.

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

SAVED_BASE_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\mobilenetv2_base.keras"

PRUNED_MODEL_PATH     = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\mobilenetv2_pruned80.keras"

DATASET_DIR           = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\DATASET\PetImages"



IMG_SIZE      = 224

BATCH_SIZE    = 8

SEED          = 42

NUM_BLOCKS    = 17



# Fraction of channels to REMOVE per layer  (0.0 = no pruning, 1.0 = remove all)

PRUNING_RATIO = 0.80   # 20 % removed → 80 % kept



# Fine-tuning hyper-parameters

FINETUNE_LR     = 1e-4   # lower LR to preserve transferred weights

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



selected_files  = cat_files + dog_files

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



dataset_size = len(selected_files)           # 8000

train_size   = int(0.65 * dataset_size)      # 5200

val_size     = int(0.15 * dataset_size)      # 1200

test_size    = dataset_size - train_size - val_size  # 1600



full_dataset = full_dataset.shuffle(

    dataset_size, seed=SEED, reshuffle_each_iteration=False

)



train_dataset = full_dataset.take(train_size)

remaining     = full_dataset.skip(train_size)

val_dataset   = remaining.take(val_size)

test_dataset  = remaining.skip(val_size)



train_dataset = (train_dataset.shuffle(1000)

                              .batch(BATCH_SIZE)

                              .prefetch(tf.data.AUTOTUNE))

val_dataset   = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset  = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)



print(f"  Train: {train_size}  |  Validation: {val_size}  |  Test: {test_size}")



# ==============================================================================

# 4.  COMPUTE CHANNEL IMPORTANCE  →  KEEP INDICES

# ==============================================================================



def l1_keep_indices(conv_layer, pruning_ratio):

    """

    Rank the output channels of a Conv2D layer by L1 norm

    (sum of absolute weights across kH, kW, in_channels) and

    return sorted indices of the channels to KEEP.



    Parameters

    ----------

    conv_layer   : tf.keras.layers.Conv2D

    pruning_ratio: float  – fraction to remove (e.g. 0.30)



    Returns

    -------

    np.ndarray of shape (n_keep,) with sorted channel indices

    """

    kernel     = conv_layer.get_weights()[0]          # (kH, kW, in_ch, out_ch)

    importance = np.sum(np.abs(kernel), axis=(0, 1, 2))  # (out_ch,)

    n_total    = len(importance)

    n_keep     = max(1, int(round(n_total * (1.0 - pruning_ratio))))

    keep_idx   = np.sort(np.argsort(importance)[::-1][:n_keep])

    return keep_idx





# --- Expansion channels (blocks 1 – 17) ---

expansion_keep = {}   # {block_id: np.ndarray of indices}



print(f"\n  {'Block':>6}  {'Orig':>6}  {'Kept':>6}  {'Removed':>8}")

print("  " + "-" * 34)



for bid in range(1, NUM_BLOCKS + 1):

    layer   = base_model.get_layer(f"block_{bid}_expand")
    orig_ch = layer.get_weights()[0].shape[-1]

    if bid == 1:
        # Keep ALL channels in Block 1
        keep_idx = np.arange(orig_ch)

    else:
        keep_idx = l1_keep_indices(layer, PRUNING_RATIO)

    expansion_keep[bid] = keep_idx

    removed = orig_ch - len(keep_idx)

    print(
        f"{bid:>6}  {orig_ch:>6}  "
        f"{len(keep_idx):>6}  {removed:>8}"
    )


# --- Final conv (1280 channels) ---

final_conv_layer = base_model.get_layer("final_conv")

final_keep       = l1_keep_indices(final_conv_layer, PRUNING_RATIO)

orig_final_ch    = final_conv_layer.get_weights()[0].shape[-1]

print(f"\n  {'Final':>6}  {orig_final_ch:>6}  {len(final_keep):>6}  "

      f"{orig_final_ch - len(final_keep):>8}  (final_conv)")



# ==============================================================================

# 5.  WEIGHT-EXTRACTION HELPERS

# ==============================================================================



def extract_conv2d_kernel(layer, in_keep=None, out_keep=None):

    """

    Return a [kernel] list from a Conv2D layer (no bias, use_bias=False),

    optionally sliced along the input-channel axis (in_keep) and/or the

    output-channel axis (out_keep).

    """

    k = layer.get_weights()[0]            # (kH, kW, in_ch, out_ch)

    if in_keep  is not None:

        k = k[:, :, in_keep, :]

    if out_keep is not None:

        k = k[:, :, :, out_keep]

    return [k]





def extract_depthwise_kernel(layer, ch_keep):

    """

    Return a [kernel] list from a DepthwiseConv2D layer (no bias),

    sliced along the channels axis (axis=2).

    Kernel shape: (kH, kW, channels, depth_multiplier)

    """

    k = layer.get_weights()[0]            # (3, 3, channels, 1)

    return [k[:, :, ch_keep, :]]





def extract_batchnorm_weights(layer, ch_keep):

    """

    Return all four BN tensors [gamma, beta, moving_mean, moving_var]

    sliced to the kept channels.

    """

    gamma, beta, mm, mv = layer.get_weights()

    return [gamma[ch_keep], beta[ch_keep], mm[ch_keep], mv[ch_keep]]





def extract_dense_prune_input(layer, in_keep):

    """

    Return [kernel, bias] from a Dense layer where the input dimension

    is sliced to kept indices (output dimension unchanged).

    """

    k, b = layer.get_weights()            # (in_features, out_features), (out_features,)

    return [k[in_keep, :], b]



# ==============================================================================

# 6.  BUILD PRUNED MODEL ARCHITECTURE

# ==============================================================================


# MobileNetV2 block configuration — must match the base model exactly

# (expansion_factor, projection_filters, stride)

BLOCK_CFG = [

    (1,  16, 1),   # block  1  – no residual (input 32 ≠ output 16)

    (6,  24, 2),   # block  2  – no residual (stride 2)

    (6,  24, 1),   # block  3  – residual ✓

    (6,  32, 2),   # block  4  – no residual (stride 2)

    (6,  32, 1),   # block  5  – residual ✓

    (6,  32, 1),   # block  6  – residual ✓

    (6,  64, 2),   # block  7  – no residual (stride 2)

    (6,  64, 1),   # block  8  – residual ✓

    (6,  64, 1),   # block  9  – residual ✓

    (6,  64, 1),   # block 10  – residual ✓

    (6,  96, 1),   # block 11  – no residual (input 64 ≠ output 96)

    (6,  96, 1),   # block 12  – residual ✓

    (6,  96, 1),   # block 13  – residual ✓

    (6, 160, 2),   # block 14  – no residual (stride 2)

    (6, 160, 1),   # block 15  – residual ✓

    (6, 160, 1),   # block 16  – residual ✓

    (6, 320, 1),   # block 17  – no residual (input 160 ≠ output 320)

]



# Augmentation pipeline (same layers as base model; no weights to transfer)

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip("horizontal"),

    layers.RandomRotation(0.1),

    layers.RandomZoom(0.1),

], name="sequential_7")





def build_pruned_block(x, filters, stride, block_id, pruned_exp_size):

    """

    One pruned inverted-residual block.



    The expansion dimension is shrunk to `pruned_exp_size`.

    The projection output (= `filters`) is NEVER changed so that

    skip-connection shapes are always preserved.

    """

    inp            = x

    input_channels = x.shape[-1]



    # 1x1 Expansion conv  →  pruned_exp_size output channels

    x = layers.Conv2D(

        pruned_exp_size, 1, padding='same', use_bias=False,

        name=f"block_{block_id}_expand")(x)

    x = layers.BatchNormalization(name=f"block_{block_id}_expand_bn")(x)

    x = layers.ReLU(max_value=6,          name=f"block_{block_id}_expand_relu")(x)



    # 3x3 Depthwise conv  →  operates on pruned_exp_size channels

    x = layers.DepthwiseConv2D(

        3, strides=stride, padding='same', use_bias=False,

        name=f"block_{block_id}_depthwise")(x)

    x = layers.BatchNormalization(name=f"block_{block_id}_depthwise_bn")(x)

    x = layers.ReLU(max_value=6,          name=f"block_{block_id}_depthwise_relu")(x)



    # 1x1 Projection conv  →  filters output channels  (UNCHANGED)

    x = layers.Conv2D(

        filters, 1, padding='same', use_bias=False,

        name=f"block_{block_id}_project")(x)

    x = layers.BatchNormalization(name=f"block_{block_id}_project_bn")(x)



    # Skip connection — only when spatial dimensions and channel counts match

    if stride == 1 and input_channels == filters:

        x = layers.Add(name=f"block_{block_id}_add")([inp, x])



    return x





def build_pruned_mobilenetv2(expansion_keep_dict, final_conv_keep):

    """

    Reconstruct the full MobileNetV2 graph with pruned channel widths.



    Parameters

    ----------

    expansion_keep_dict : dict  {block_id -> np.ndarray of kept channel indices}

    final_conv_keep     : np.ndarray  kept channel indices for final_conv

    """

    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_layer")

    x   = data_augmentation(inp)



    # ── Stem (32 channels, unchanged) ─────────────────────────────────────

    x = layers.Conv2D(32, 3, strides=2, padding='same',

                      use_bias=False, name="stem_conv")(x)

    x = layers.BatchNormalization(name="stem_bn")(x)

    x = layers.ReLU(max_value=6, name="stem_relu")(x)



    # ── Inverted-residual blocks ───────────────────────────────────────────

    for bid, (_exp_factor, filt, stride) in enumerate(BLOCK_CFG, start=1):

        pruned_exp = len(expansion_keep_dict[bid])

        x = build_pruned_block(x, filt, stride, bid, pruned_exp)



    # ── Final conv (pruned to len(final_conv_keep) channels) ──────────────

    x = layers.Conv2D(len(final_conv_keep), 1, padding='same',

                      use_bias=False, name="final_conv")(x)

    x = layers.BatchNormalization(name="final_bn")(x)

    x = layers.ReLU(max_value=6, name="final_relu")(x)



    # ── Classifier head ───────────────────────────────────────────────────

    x   = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    x   = layers.Dense(128, activation='relu', name="fc1")(x)

    x   = layers.Dropout(0.3,                  name="dropout")(x)

    out = layers.Dense(1, activation='sigmoid', name="classifier")(x)



    return Model(inp, out)





pruned_model = build_pruned_mobilenetv2(expansion_keep, final_keep)

pruned_model.summary()

print("\nBase params :", base_model.count_params())
print("Pruned params:", pruned_model.count_params())



# ==============================================================================

# 7.  TRANSFER PRUNED WEIGHTS  (base model  →  pruned model)

# ==============================================================================




# ── Stem  (no channel changes) ─────────────────────────────────────────────

for lname in ("stem_conv", "stem_bn"):

    pruned_model.get_layer(lname).set_weights(

        base_model.get_layer(lname).get_weights()

    )



# ── Inverted-residual blocks  1 – 17 ───────────────────────────────────────

for bid in range(1, NUM_BLOCKS + 1):

    keep = expansion_keep[bid]



    # expand conv  →  slice output channels to 'keep'

    pruned_model.get_layer(f"block_{bid}_expand").set_weights(

        extract_conv2d_kernel(

            base_model.get_layer(f"block_{bid}_expand"),

            out_keep=keep

        )

    )



    # expand BN  →  slice all 4 tensors to 'keep'

    pruned_model.get_layer(f"block_{bid}_expand_bn").set_weights(

        extract_batchnorm_weights(

            base_model.get_layer(f"block_{bid}_expand_bn"), keep

        )

    )



    # depthwise conv  →  slice channel axis (axis 2) to 'keep'

    pruned_model.get_layer(f"block_{bid}_depthwise").set_weights(

        extract_depthwise_kernel(

            base_model.get_layer(f"block_{bid}_depthwise"), keep

        )

    )



    # depthwise BN  →  slice to 'keep'

    pruned_model.get_layer(f"block_{bid}_depthwise_bn").set_weights(

        extract_batchnorm_weights(

            base_model.get_layer(f"block_{bid}_depthwise_bn"), keep

        )

    )



    # project conv  →  slice INPUT channels to 'keep'; output channels UNCHANGED

    pruned_model.get_layer(f"block_{bid}_project").set_weights(

        extract_conv2d_kernel(

            base_model.get_layer(f"block_{bid}_project"),

            in_keep=keep

        )

    )



    # project BN  →  output channels unchanged, copy as-is

    pruned_model.get_layer(f"block_{bid}_project_bn").set_weights(

        base_model.get_layer(f"block_{bid}_project_bn").get_weights()

    )



# ── Final conv  →  slice output channels to final_keep ─────────────────────

pruned_model.get_layer("final_conv").set_weights(

    extract_conv2d_kernel(

        base_model.get_layer("final_conv"),

        out_keep=final_keep

    )

)

pruned_model.get_layer("final_bn").set_weights(

    extract_batchnorm_weights(base_model.get_layer("final_bn"), final_keep)

)



# ── Classifier head ─────────────────────────────────────────────────────────

# fc1 receives GAP output of pruned final_conv  →  prune input dimension

pruned_model.get_layer("fc1").set_weights(

    extract_dense_prune_input(base_model.get_layer("fc1"), final_keep)

)



# Final sigmoid Dense(1)  →  shape unchanged, copy directly

pruned_model.get_layer("classifier").set_weights(

    base_model.get_layer("classifier").get_weights()

)



print("  All weights transplanted successfully.")



# ==============================================================================

# 8.  BASELINE EVALUATION  (pruned model, before fine-tuning)

# ==============================================================================


pruned_model.compile(

    optimizer=Adam(learning_rate=FINETUNE_LR),

    loss='binary_crossentropy',

    metrics=['accuracy']

)

tensorboard_pruned = TensorBoard(log_dir='logs\\MobileNetV2_80pruned')



es_pruned = EarlyStopping(

    monitor='val_loss',

    patience=FINETUNE_PATIENCE,

    restore_best_weights=True,

    verbose=1

)



history_pruned = pruned_model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=FINETUNE_EPOCHS,

    callbacks=[es_pruned, tensorboard_pruned],

    verbose=1

)



# ==============================================================================

# 10.  FINAL EVALUATION  (after fine-tuning)

# ==============================================================================


loss_after, acc_after = pruned_model.evaluate(test_dataset, verbose=1)

print(f"\n  Test loss after fine-tuning   : {loss_after:.4f}")

print(f"  Test accuracy after fine-tuning : {acc_after:.4f}")

print(f"  Best validation accuracy        : "

      f"{max(history_pruned.history['val_accuracy']):.4f}")



# ==============================================================================

# 11.  MODEL COMPARISON SUMMARY

# ==============================================================================



base_params = base_model.count_params()
pruned_params = pruned_model.count_params()



print(f"\n  Pruning ratio applied   : {PRUNING_RATIO * 100:.0f} %")

print(f"  Base   model parameters : {base_params:>12,}")

print(f"  Pruned model parameters : {pruned_params:>12,}")

print(f"  Parameters removed      : {base_params - pruned_params:>12,}")

print(f"  Parameter reduction     : {100*(1 - pruned_params/base_params):.1f} %")

print(f"  Pruned model test accuracy (after FT)      : {acc_after:.4f}")



# ==============================================================================

# 12.  SAVE PRUNED MODEL

# ==============================================================================

pruned_model.save(PRUNED_MODEL_PATH)

print(f"\n  Pruned model saved to:\n  {PRUNED_MODEL_PATH}")



# ==============================================================================

# 13.  QUICK PREDICTION SANITY CHECK

# ==============================================================================



for images, labels in test_dataset.take(1):

    preds = pruned_model.predict(images, verbose=0)

    pred_classes = (preds.flatten() > 0.5).astype(int)

    actual       = labels.numpy()



    print(f"\n  Raw sigmoid outputs  : {preds[:10].flatten().round(4)}")

    print(f"  Predicted classes    : {pred_classes[:10]}")

    print(f"  Actual labels        : {actual[:10]}")

    batch_acc = np.mean(pred_classes == actual)

    print(f"  Batch accuracy       : {batch_acc:.4f}")

    break



print("\nDone.")