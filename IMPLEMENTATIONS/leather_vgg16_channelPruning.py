#final flexible channel pruning

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers.schedules import CosineDecay

# ==============================================================================
# 1. DATASET PATHS & LOADING
# ==============================================================================

dataset_dir = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\DATASET\CSIR_Datasets"
'''
structure of PetImages folder
├── Cat/
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
└── Dog/
    ├── 0.jpg
    ├── 1.jpg
    └── ...
'''

SAVED_BASE_MODEL_PATH   = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\base_applications_vgg16_Full_Train.keras"
SAVED_PRUNED_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\80channel_pruned_leather_vgg16.keras"


IMG_SIZE = 224
BATCH_SIZE = 8

SEED = 42


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
    return image, label

full_dataset = tf.data.Dataset.from_tensor_slices(
    (list(selected_files), list(selected_labels))
)

full_dataset = full_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

dataset_size = len(selected_files)

train_size = int(0.65 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

full_dataset = full_dataset.shuffle(
    dataset_size,
    seed=SEED,
    reshuffle_each_iteration=False
)

train_dataset = full_dataset.take(train_size)
remaining = full_dataset.skip(train_size)
val_dataset = remaining.take(val_size)
test_dataset = remaining.skip(val_size)


# ==============================================================================
# 2. DATA AUGMENTATION & PREPROCESSING
# ==============================================================================

# NOTE: matches base model preprocessing (VGG16 preprocess_input, NOT /255.0)
# If your base model actually used /255.0, switch this back accordingly.
from tensorflow.keras.applications.vgg16 import preprocess_input

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
# 3. FLEXIBLE VGG-16 ARCHITECTURE
#    Rebuilt to structurally mirror the actual base model:
#    - no BatchNorm (base VGG16 backbone doesn't have any)
#    - ReLU (matches base head activation)
#    - same dense head: GAP -> Dense(4096) -> Dropout(0.5) -> Dense(4096) -> Dropout(0.5) -> Dense(1, sigmoid)
#    - includes data_augmentation so pruned fine-tuning still gets regularization
# ==============================================================================

def create_flexible_vgg(filter_cfg=None):
    if filter_cfg is None:
        # Default standard VGG-16 channel counts
        filter_cfg = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]

    inp = layers.Input(shape=(224, 224, 3), name="input_layer")
    x = data_augmentation(inp)
    cfg_idx = 0

    blocks = [2, 2, 3, 3, 3]
    for block_idx, num_layers in enumerate(blocks):
        for _ in range(num_layers):
            x = layers.Conv2D(filter_cfg[cfg_idx], (3, 3), padding='same',
                               activation='relu', name=f"conv_{cfg_idx}")(x)
            cfg_idx += 1
        x = layers.MaxPooling2D((2, 2), name=f"pool_{block_idx}")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(4096, activation='relu', name="dense_1")(x)
    x = layers.Dropout(0.5, name="drop_1")(x)
    x = layers.Dense(4096, activation='relu', name="dense_2")(x)
    x = layers.Dropout(0.5, name="drop_2")(x)
    out = layers.Dense(1, activation='sigmoid', name="output")(x)

    return Model(inputs=inp, outputs=out)


# ==============================================================================
# 4. BASELINE HANDLING (Load saved model if available, else train from scratch)
# ==============================================================================

if os.path.exists(SAVED_BASE_MODEL_PATH):
    print(f"\n>>> [FOUND SAVED MODEL] Loading pre-trained base model from: {SAVED_BASE_MODEL_PATH}")
    base_model = tf.keras.models.load_model(SAVED_BASE_MODEL_PATH)
    print(">>> Baseline weights successfully restored from file storage.")
else:
    raise FileNotFoundError(
        f"Base model not found at {SAVED_BASE_MODEL_PATH}. "
        "This pruning script expects your VGG16-transfer-learning base model "
        "(with a nested VGG16 backbone) to already be trained and saved."
    )


# ==============================================================================
# 5. STRUCTURAL CHANNEL PRUNING ALGORITHM (L1-Norm Based)
# ==============================================================================

PRUNE_RATIO = 0.80  # Structurally eliminate 20% of filters per layer

print(f"\n>>> Computing channel structural masks (Pruning {PRUNE_RATIO*100}% of filters)...")

# --- Reach INTO the nested VGG16 submodel ---
# base_model wraps VGG16 as a single nested layer (x = base_model(x) in the base script),
# so Conv2D layers are NOT directly in base_model.layers.
base_vgg = None
for layer in base_model.layers:
    if isinstance(layer, Model) and "vgg16" in layer.name.lower():
        base_vgg = layer
        break

if base_vgg is None:
    print("\n>>> Could not auto-detect nested VGG16 submodel. Base model layers:")
    for l in base_model.layers:
        print(f"   {l.name} ({type(l).__name__})")
    raise ValueError(
        "Could not find nested VGG16 submodel in base_model. "
        "Check the layer names printed above and adjust the name match."
    )

base_convs = [l for l in base_vgg.layers if isinstance(l, layers.Conv2D)]
print(f">>> Found {len(base_convs)} Conv2D layers inside base VGG16 backbone.")

# --- Build new_filter_cfg from actual filter counts ---
new_filter_cfg = [max(1, int(l.filters * (1 - PRUNE_RATIO))) for l in base_convs]

# --- Reconstruct a physically smaller network architecture ---
pruned_model = create_flexible_vgg(filter_cfg=new_filter_cfg)

dummy_input = tf.random.normal((1, 224, 224, 3))
_ = pruned_model(dummy_input)  # build weights

pruned_convs = [l for l in pruned_model.layers if isinstance(l, layers.Conv2D)]

assert len(base_convs) == len(pruned_convs) == len(new_filter_cfg), \
    "Conv layer count mismatch between base VGG16 and flexible architecture."

running_channel_mask = None

# --- Transfer CONV weights (L1-norm filter selection) ---
for base_layer, target_layer, num_surviving in zip(base_convs, pruned_convs, new_filter_cfg):
    weights, biases = base_layer.get_weights()

    # L1-norm across spatial dims + input channels -> filter importance
    l1_norms = np.sum(np.abs(weights), axis=(0, 1, 2))
    surviving_filter_indices = np.argsort(l1_norms)[-num_surviving:]
    surviving_filter_indices.sort()

    # Slice INPUT channels based on what the previous layer kept
    if running_channel_mask is not None:
        weights = weights[:, :, running_channel_mask, :]

    # Slice OUTPUT channels based on this layer's pruning decision
    weights = weights[:, :, :, surviving_filter_indices]
    biases = biases[surviving_filter_indices]

    target_layer.set_weights([weights, biases])
    running_channel_mask = surviving_filter_indices

print(">>> Conv layer weights transferred with structural pruning applied.")

# --- Transfer DENSE weights (hidden 4096-unit layers, matched by order) ---
base_dense_hidden = [l for l in base_model.layers if isinstance(l, layers.Dense) and l.units == 4096]
pruned_dense_hidden = [l for l in pruned_model.layers if isinstance(l, layers.Dense) and l.units == 4096]

assert len(base_dense_hidden) == len(pruned_dense_hidden) == 2, \
    f"Expected exactly 2 hidden Dense(4096) layers, found {len(base_dense_hidden)} (base) " \
    f"and {len(pruned_dense_hidden)} (pruned)."

for i, (base_layer, target_layer) in enumerate(zip(base_dense_hidden, pruned_dense_hidden)):
    weights, biases = base_layer.get_weights()
    if i == 0 and running_channel_mask is not None:
        # dense_1's input dim shrank because GAP output shrank (fewer surviving conv channels)
        weights = weights[running_channel_mask, :]
    target_layer.set_weights([weights, biases])

print(">>> Dense hidden layer weights transferred.")

# --- Transfer OUTPUT layer (units=1, unaffected by channel pruning) ---
base_output_candidates = [l for l in base_model.layers if isinstance(l, layers.Dense) and l.units == 1]
pruned_output_candidates = [l for l in pruned_model.layers if isinstance(l, layers.Dense) and l.units == 1]

assert len(base_output_candidates) == 1 and len(pruned_output_candidates) == 1, \
    "Expected exactly one Dense(1) output layer in each model."

pruned_output_candidates[0].set_weights(base_output_candidates[0].get_weights())
print(">>> Output layer weights transferred.")

print("\n>>> Model structural transformation complete without errors.")
base_model.summary()
print("\n>>> New Pruned Model Summary (Notice physically reduced parameters):")
pruned_model.summary()


# ==============================================================================
# 6. FINE-TUNE THE PRUNED MODEL TO RECOVER ACCURACY
# ==============================================================================

steps_per_epoch = train_size // BATCH_SIZE
lr_schedule = CosineDecay(initial_learning_rate=1e-5, decay_steps=steps_per_epoch * 30, alpha=0.01)

pruned_model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=BinaryCrossentropy(), metrics=['accuracy'])

tensorboardName = 'logs\\VGG16_leather_channelpruned80'
tensorboardNameApp = tensorboardName
tensorboard2 = TensorBoard(log_dir=tensorboardNameApp)

es2 = EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True)

print("\n>>> Fine-tuning the structurally pruned model to recover accuracy drops...")
pruned_model.fit(train_dataset, epochs=30, validation_data=val_dataset, callbacks=[tensorboard2, es2])


# ==============================================================================
# 7. EVALUATION AND EXPORT
# ==============================================================================

print("\n>>> Evaluating structurally lightened model on test dataset...")
test_loss, test_accuracy = pruned_model.evaluate(test_dataset)

print(f"\nFinal Pruned Test Loss: {test_loss:.4f}")
print(f"Final Pruned Test Accuracy: {test_accuracy * 100:.2f}%")

pruned_model.save(SAVED_PRUNED_MODEL_PATH)
print("\n>>> Saved true pruned model to disk.")#final flexible channel pruning
