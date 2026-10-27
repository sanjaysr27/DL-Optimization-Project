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

# 1. DATASET PATHS & LOADING

dataset_dir = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\DATASET\PetImages"
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
IMG_SIZE = 224
BATCH_SIZE = 8
SEED = 42

# Path where the base model is saved/loaded from
SAVED_BASE_MODEL_PATH   = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\flexible_model_base_vgg16.keras"
SAVED_PRUNED_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\20channel_pruned_vgg16.keras"

random.seed(SEED)

# Paths to each class
cats_dir = os.path.join(dataset_dir, "Cat")
dogs_dir = os.path.join(dataset_dir, "Dog")

# Get image filenames
cat_files = [os.path.join(cats_dir, f) for f in os.listdir(cats_dir)]
dog_files = [os.path.join(dogs_dir, f) for f in os.listdir(dogs_dir)]

# Randomly select 4000 from each class
cat_files = random.sample(cat_files, 4000)
dog_files = random.sample(dog_files, 4000)

# Combine and shuffle
selected_files  = cat_files + dog_files
selected_labels = [0] * 4000 + [1] * 4000   # 0=cats, 1=dogs

combined = list(zip(selected_files, selected_labels))
random.shuffle(combined)

selected_files, selected_labels = zip(*combined)

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))    
    return image, label

#creation of the dataset with 4000 images of each cats, dogs
full_dataset = tf.data.Dataset.from_tensor_slices(
    (list(selected_files), list(selected_labels))
)

full_dataset = full_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

dataset_size = len(selected_files)  

#getting the size of training, validation, testing datasets
train_size = int(0.65 * dataset_size)  
val_size = int(0.15 * dataset_size)    
test_size = dataset_size - train_size - val_size  

full_dataset = full_dataset.shuffle(
    dataset_size,
    seed=SEED,
    reshuffle_each_iteration=False
)

#creation of the datasets
train_dataset = full_dataset.take(train_size)

remaining = full_dataset.skip(train_size)

val_dataset = remaining.take(val_size)

test_dataset = remaining.skip(val_size)




# 2. DATA AUGMENTATION & PREPROCESSING

#data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
])
#normalization of values
def preprocess_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

def train_preprocess(image, label):
    image, label = preprocess_img(image, label)
    return data_augmentation(image, training=True), label

#performing all preprocesses on teh datasets and batching them
train_dataset = train_dataset.shuffle(train_size // 5).map(train_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = val_dataset.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)



# 3. FLEXIBLE VGG-16 CONFIGURATION ARCHITECTURE

#creation of a function to create a flexible VGG16 model for pruning where the no of filters of each layer is not certain

'''
uses filter_cfg(list) to store the no of filters of each convolution layer
using blocks(list) to store the size of each block
using for loop to create model with the info in filter_cfg and blocks
every convolution layer is followed by a batch normalization and every block ends with a maxpooling
'''
def create_flexible_vgg(filter_cfg=None):
    if filter_cfg is None:
        # Default standard VGG-16 channel counts
        filter_cfg = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]

    inp = layers.Input(shape=(224, 224, 3), name="input_layer")
    x = inp
    cfg_idx = 0

    # Define Block-Structures mapping standard VGG
    blocks = [2, 2, 3, 3, 3]
    for block_idx, num_layers in enumerate(blocks):
        for _ in range(num_layers):
            x = layers.Conv2D(filter_cfg[cfg_idx], (3, 3), padding='same', name=f"conv_{cfg_idx}")(x)
   
            x = layers.BatchNormalization(name=f"bn_{cfg_idx}")(x)
            x = layers.LeakyReLU(name=f"leaky_{cfg_idx}")(x)
            cfg_idx += 1
        x = layers.MaxPooling2D((2, 2), name=f"pool_{block_idx}")(x)

    #creatio of a fixed dense layer (this is the same for both base and pruned model)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(4096, name="dense_1")(x)
    x = layers.LeakyReLU(name="leaky_d1")(x)
    x = layers.Dropout(0.5, name="drop_1")(x)
    x = layers.Dense(4096, name="dense_2")(x)
    x = layers.LeakyReLU(name="leaky_d2")(x)
    x = layers.Dropout(0.5, name="drop_2")(x)
    out = layers.Dense(1, activation='sigmoid', name="output")(x)

    return Model(inputs=inp, outputs=out)




# 4. BASELINE HANDLING (Load saved model if available, else train from scratch)

# Checking if base model is already saved
if os.path.exists(SAVED_BASE_MODEL_PATH):
    print(f"\n>>> [FOUND SAVED MODEL] Loading pre-trained base model from: {SAVED_BASE_MODEL_PATH}")
    base_model = tf.keras.models.load_model(SAVED_BASE_MODEL_PATH)
    print(">>> Baseline weights successfully restored from file storage.")
else:
    print("\n>>> [NO SAVED MODEL FOUND] Training Dense Baseline Model from scratch...")
    #creation of the base model using the function create_flexible_vgg and compiling it
    base_model = create_flexible_vgg()
    base_model.compile(optimizer=Adam(learning_rate=1e-4), loss=BinaryCrossentropy(), metrics=['accuracy'])

    #creatio of a tensorboard to plot the graphs and store data
    tensorboardName = 'logs\\VGG16_flexible_base_channel'
    tensorboardNameApp = tensorboardName
    tensorboard1 = TensorBoard(log_dir=tensorboardNameApp)

    #defining earlystop and fitting hte model
    es = EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True)
    base_model.fit(train_dataset, epochs=100, validation_data=val_dataset, callbacks=[es, tensorboard1])

    # Save the trained base model for future runs
    print(f">>> Saving base model to {SAVED_BASE_MODEL_PATH}...")
    base_model.save(SAVED_BASE_MODEL_PATH)




# 5. STRUCTURAL CHANNEL PRUNING ALGORITHM (L1-Norm Based)

PRUNE_RATIO = 0.20  # Structurally eliminate 50% of filters per layer

print(f"\n>>> Computing channel structural masks (Pruning {PRUNE_RATIO*100}% of filters)...")

# 1. creatin of new_filter_cfg to pass to create_flexible_vgg tp create the architecture for the pruned model with 30%less filters
new_filter_cfg = []
conv_layer_indices = [i for i, layer in enumerate(base_model.layers) if isinstance(layer, layers.Conv2D)]

for idx in conv_layer_indices:
    total_filters = base_model.layers[idx].filters
    surviving_count = max(1, int(total_filters * (1 - PRUNE_RATIO)))
    new_filter_cfg.append(surviving_count)

# 2. Reconstruct a physically smaller network architecture
pruned_model = create_flexible_vgg(filter_cfg=new_filter_cfg)

#creation of a fake image to pass thorugh the pruned model to create weights and biases in the model
dummy_input = tf.random.normal((1, 224, 224, 3))
_ = pruned_model(dummy_input)

# 3. Create a look-up dictionary to safely map weights by layer name
pruned_layers_dict = {layer.name: layer for layer in pruned_model.layers}

running_channel_mask = None  # Tracks the active channel layout across all layers
conv_cfg_counter = 0

for layer in base_model.layers:
    # Safely find the corresponding layer in the pruned model by name
    if layer.name not in pruned_layers_dict:
        continue
    target_layer = pruned_layers_dict[layer.name]
   
    if isinstance(layer, layers.Conv2D):
        weights, biases = layer.get_weights()
       
        # Compute L1-norm across spatial dimensions and input channels to find filter importance
        l1_norms = np.sum(np.abs(weights), axis=(0, 1, 2))
       
        # Get indices of the top highest-magnitude filters
        num_surviving_filters = new_filter_cfg[conv_cfg_counter]
        surviving_filter_indices = np.argsort(l1_norms)[-num_surviving_filters:]
        surviving_filter_indices.sort()  # Keep indexing ordered sequentially
       
        # STEP 1: Slice INPUT dimensions based on what the previous layer survived
        if running_channel_mask is not None:
            weights = weights[:, :, running_channel_mask, :]
           
        # STEP 2: Slice OUTPUT dimensions based on this layer's current pruning decision
        weights = weights[:, :, :, surviving_filter_indices]
        biases = biases[surviving_filter_indices]
       
        # Assign structural weights into the new model's respective layer
        target_layer.set_weights([weights, biases])
       
        running_channel_mask = surviving_filter_indices
        conv_cfg_counter += 1
       
    #Asigning the values to batchnormalization
    elif isinstance(layer, layers.BatchNormalization):
        gamma, beta, moving_mean, moving_variance = layer.get_weights()
        gamma = gamma[running_channel_mask]
        beta = beta[running_channel_mask]
        moving_mean = moving_mean[running_channel_mask]
        moving_variance = moving_variance[running_channel_mask]
        target_layer.set_weights([gamma, beta, moving_mean, moving_variance])
       
    #assigning the values to the dense layers
    elif isinstance(layer, layers.Dense):
        weights, biases = layer.get_weights()
       
        if layer.name == "dense_1":
            if running_channel_mask is not None:
                weights = weights[running_channel_mask, :]
            # FIX: Reset mask after dense_1 — dense layers do not propagate channel masks
            running_channel_mask = None
       
        target_layer.set_weights([weights, biases])
   
    #assignig the values to other layers like maxpool, dropout, leakyrelu
    else:
        layer_weights = layer.get_weights()
        if layer_weights:
            target_layer.set_weights(layer_weights)

print("\n>>> Model structural transformation complete without errors.")
base_model.summary()
print("\n>>> New Pruned Model Summary (Notice physically reduced parameters):")
pruned_model.summary()




# 6. FINE-TUNE THE PRUNED MODEL TO RECOVER ACCURACY

steps_per_epoch = train_size // BATCH_SIZE
lr_schedule = CosineDecay(initial_learning_rate=1e-5, decay_steps=steps_per_epoch * 30, alpha=0.01)

#trainig the pruned model again to regain the accuracy lost due to pruning
pruned_model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=BinaryCrossentropy(), metrics=['accuracy'])

#using tensorboard to save and plot the data
tensorboardName = 'logs\\VGG16_flexible_channelpruned20'
tensorboardNameApp = tensorboardName
tensorboard2 = TensorBoard(log_dir=tensorboardNameApp)

# FIX: Added EarlyStopping to prevent overfitting during fine-tuning recovery phase
es2 = EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True)

print("\n>>> Fine-tuning the structurally pruned model to recover accuracy drops...")
pruned_model.fit(train_dataset, epochs=30, validation_data=val_dataset, callbacks=[tensorboard2, es2])


# 7. EVALUATION AND EXPORT

print("\n>>> Evaluating structurally lightened model on test dataset...")
test_loss, test_accuracy = pruned_model.evaluate(test_dataset)

print(f"\nFinal Pruned Test Loss: {test_loss:.4f}")
print(f"Final Pruned Test Accuracy: {test_accuracy * 100:.2f}%")

# FIX: Changed from .h5 to .keras format — HDF5 cannot handle custom pruned layer shapes correctly
pruned_model.save(SAVED_PRUNED_MODEL_PATH)
print("\n>>> Saved true pruned model to disk.")