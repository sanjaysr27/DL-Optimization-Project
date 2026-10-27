# VGG16 form applications


import os
import random
import tensorflow as tf
#import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers.schedules import CosineDecay

# ==============================================================================
# 1. LOCAL DATASET PATHS & HYPERPARAMETERS
# ==============================================================================
dataset_dir = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\DATASET\CSIR_Datasets"

IMG_SIZE = 224
BATCH_SIZE = 32  # Keeping your transfer learning batch size
SEED = 42

# ==========================================
# LOAD DATASET FROM LOCAL FOLDER (65/15/20 Split)
# ==========================================
SAVED_BASE_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\base_applications_vgg16.keras"

# ==========================================
# LOAD DATASET
# ==========================================

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

dataset_size = len(selected_files)   # 8000

train_size = int(0.65 * dataset_size)   # 3900
val_size = int(0.15 * dataset_size)     # 900
test_size = dataset_size - train_size - val_size   # 1200

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
# 2. PREPROCESSING PIPELINE
# ==============================================================================
def preprocess_img(image, label):
    image = tf.keras.applications.vgg16.preprocess_input(image)
    return image, label

# Construct optimized execution pipelines
train_dataset = train_dataset.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)

train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ==============================================================================
# 3. MODEL DEFINITION
# ==============================================================================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),      
    layers.RandomRotation(0.10),          
    layers.RandomZoom(0.10),              
])

# Load VGG16 base pre-trained on ImageNet
base_model = tf.keras.applications.VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
base_model.load_weights(r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\TRAINING\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
base_model.trainable = False

inp = layers.Input((224, 224, 3))
x = data_augmentation(inp)    
x = base_model(x)              

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(4096, activation='relu')(x)
x = layers.Dropout(0.5)(x)  
x = layers.Dense(4096, activation='relu')(x)
x = layers.Dropout(0.5)(x)    
output = layers.Dense(1, activation='sigmoid')(x)

model = Model(inputs=inp, outputs=output)
model.summary()

# ==============================================================================
# 4. COMPILE AND TRAIN
# ==============================================================================

EPOCHS = 100
steps_per_epoch = train_size // BATCH_SIZE
total_decay_steps = steps_per_epoch * EPOCHS

lr_schedule = CosineDecay(
    initial_learning_rate=1e-4,
    decay_steps=total_decay_steps,
    alpha=0.01  
)

model.compile(optimizer=Adam(learning_rate= lr_schedule), loss=BinaryCrossentropy(), metrics=['accuracy'])

tensorboardName = 'logs\\leather_base_applications_vgg16'   #change model
tensorboardNameApp = tensorboardName
tensorboard1 = TensorBoard(log_dir=tensorboardNameApp)


es = EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True)

model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, callbacks=[es, tensorboard1])

# ==============================================================================
# 5. TESTING PHASE (Evaluation on Unseen Data)
# ==============================================================================
print("\n>>> Evaluating model on the completely unseen Testing Dataset...")
test_loss, test_accuracy = model.evaluate(test_dataset)

print(f"\nFinal Test Loss: {test_loss:.4f}")
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")

model.save(SAVED_BASE_MODEL_PATH)
print(f"\n>>> Base model saved to: {SAVED_BASE_MODEL_PATH}")




# Load VGG16 base pre-trained on ImageNet
base_model = tf.keras.applications.VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
#base_model.load_weights(r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\TRAINING\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
 # base_model.trainable = False

inp = layers.Input((224, 224, 3))
x = data_augmentation(inp)    
x = base_model(x)              

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(4096, activation='relu')(x)
x = layers.Dropout(0.5)(x)  
x = layers.Dense(4096, activation='relu')(x)
x = layers.Dropout(0.5)(x)    
output = layers.Dense(1, activation='sigmoid')(x)

model = Model(inputs=inp, outputs=output)
model.summary()

model.load_weights(r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\base_applications_vgg16.keras")
model.trainable = True


EPOCHS = 100
steps_per_epoch = train_size // BATCH_SIZE
total_decay_steps = steps_per_epoch * EPOCHS

lr_schedule = CosineDecay(
    initial_learning_rate=1e-4,
    decay_steps=total_decay_steps,
    alpha=0.01  
)

model.compile(optimizer=Adam(learning_rate= lr_schedule), loss=BinaryCrossentropy(), metrics=['accuracy'])

tensorboardName = 'logs\\leather_base_applications_vgg16_full_train'   #change model
tensorboardNameApp = tensorboardName
tensorboard1 = TensorBoard(log_dir=tensorboardNameApp)


es = EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True)

model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, callbacks=[es, tensorboard1])

# ==============================================================================
# 5. TESTING PHASE (Evaluation on Unseen Data)
# ==============================================================================
print("\n>>> Evaluating model on the completely unseen Testing Dataset...")
test_loss, test_accuracy = model.evaluate(test_dataset)

print(f"\nFinal Test Loss: {test_loss:.4f}")
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")


final_SAVED_BASE_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\base_applications_vgg16_Full_Train.keras"
model.save(final_SAVED_BASE_MODEL_PATH)
print(f"\n>>> Base model saved to: {final_SAVED_BASE_MODEL_PATH}")