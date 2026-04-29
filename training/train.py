import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
DATA_DIR = '../data/processed'
MODEL_DIR = '../final_model'
os.makedirs(MODEL_DIR, exist_ok=True)

# Audio and Spectrogram Parameters (These must match your C++ deployment)
SAMPLE_RATE = 16000
FRAME_LENGTH = int(SAMPLE_RATE * 0.030) # 30ms window
FRAME_STEP = int(SAMPLE_RATE * 0.020)   # 20ms stride
NUM_MEL_BINS = 40
EPOCHS = 20
BATCH_SIZE = 32

# --- 2. DATA LOADING ---
print("Loading data...")
# Automatically loads folders as classes and converts to 16kHz
raw_train_ds, raw_val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=DATA_DIR,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    seed=42,
    subset='both',
    output_sequence_length=SAMPLE_RATE
)

class_names = np.array(raw_train_ds.class_names)
print(f"Classes: {class_names}")

# --- 3. FEATURE EXTRACTION (Spectrogram) ---
def squeeze(audio, labels):
    # Removes the extra channel dimension added by the audio_dataset_from_directory
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels

def get_spectrogram(audio):
    # Short-Time Fourier Transform
    stft = tf.signal.stft(audio, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP, fft_length=FRAME_LENGTH)
    spectrogram = tf.abs(stft)
    # Add a channel dimension for the CNN
    spectrogram = tf.expand_dims(spectrogram, -1)
    return spectrogram

def make_spec_ds(ds):
    return ds.map(squeeze, tf.data.AUTOTUNE).map(
        lambda x, y: (get_spectrogram(x), y), num_parallel_calls=tf.data.AUTOTUNE)

train_ds = make_spec_ds(raw_train_ds).cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
val_ds = make_spec_ds(raw_val_ds).cache().prefetch(tf.data.AUTOTUNE)

# Get the input shape for the model dynamically
for example_spectrograms, example_labels in train_ds.take(1):
    input_shape = example_spectrograms.shape[1:]
    print(f"\n[REPORT METRIC] Input Tensor Shape: {input_shape}")

# --- 4. MODEL ARCHITECTURE (Depthwise Separable CNN) ---
num_classes = len(class_names)

model = models.Sequential([
    layers.Input(shape=input_shape),
    # Downsample the image
    layers.Resizing(32, 32), 
    
    # Standard Conv2D
    layers.Conv2D(16, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),
    
    # Depthwise Separable Conv2D (Saves parameters for Microcontrollers)
    layers.SeparableConv2D(32, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),
    
    layers.SeparableConv2D(64, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),
    
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# [REPORT METRIC] Model Parameters
model.summary()
print(f"[REPORT METRIC] Number of Parameters: {model.count_params()}")

# --- 5. TRAINING ---
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[tf.keras.callbacks.EarlyStopping(verbose=1, patience=4, restore_best_weights=True)]
)

# --- 6. SAVE MODEL AND METRICS ---
model.save(os.path.join(MODEL_DIR, 'kws_model.h5'))

# Plot Training Curves
metrics = history.history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(metrics['loss'], label='Training Loss')
plt.plot(metrics['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(metrics['accuracy'], label='Training Accuracy')
plt.plot(metrics['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.savefig(os.path.join(MODEL_DIR, 'training_curves.png'))
print(f"\nTraining complete! Model and curves saved to {MODEL_DIR}")
print(f"[REPORT METRIC] Final Validation Accuracy: {metrics['val_accuracy'][-1]:.4f}")