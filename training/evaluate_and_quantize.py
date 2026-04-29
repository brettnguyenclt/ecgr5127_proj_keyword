import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
DATA_DIR = '../data/processed'
MODEL_DIR = '../final_model'
MODEL_PATH = os.path.join(MODEL_DIR, 'kws_model.h5')
TFLITE_PATH = os.path.join(MODEL_DIR, 'kws_model.tflite')
C_ARRAY_PATH = os.path.join(MODEL_DIR, 'model_data.h')

SAMPLE_RATE = 16000
FRAME_LENGTH = int(SAMPLE_RATE * 0.030)
FRAME_STEP = int(SAMPLE_RATE * 0.020)
BATCH_SIZE = 32

print("Loading model and data...")
model = models.load_model(MODEL_PATH)

# Load the validation dataset (used here as our test set)
raw_train_ds, raw_val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=DATA_DIR,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    seed=42,
    subset='both',
    output_sequence_length=SAMPLE_RATE
)
class_names = np.array(raw_train_ds.class_names)

def squeeze(audio, labels):
    return tf.squeeze(audio, axis=-1), labels

def get_spectrogram(audio):
    stft = tf.signal.stft(audio, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP, fft_length=FRAME_LENGTH)
    return tf.expand_dims(tf.abs(stft), -1)

def make_spec_ds(ds):
    return ds.map(squeeze).map(lambda x, y: (get_spectrogram(x), y))

train_ds = make_spec_ds(raw_train_ds)
val_ds = make_spec_ds(raw_val_ds)

# --- 2. EVALUATION & FRR CALCULATION ---
print("\nEvaluating model to extract Report Metrics...")
y_true = []
y_pred = []

for x, y in val_ds:
    preds = model.predict(x, verbose=0)
    y_true.extend(y.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Calculate Overall Test Accuracy
test_acc = np.sum(y_true == y_pred) / len(y_true)
print(f"[REPORT METRIC] Test Data Accuracy: {test_acc:.4f}")

# Calculate Confusion Matrix and FRR
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# FRR = False Negatives / (True Positives + False Negatives)
for i, class_name in enumerate(class_names):
    if class_name in ['yes', 'positive']:
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        frr = fn / (tp + fn)
        print(f"[REPORT METRIC] FRR for '{class_name}': {frr:.4f} ({frr*100:.2f}%)")

# --- 3. QUANTIZATION (TFLite INT8) ---
print("\nStarting INT8 Quantization...")
def representative_dataset():
    # Take a small subset of training data to calibrate the quantization ranges
    for spec, _ in train_ds.take(100):
        yield [spec]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()

with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_quant_model)
print(f"Quantized TFLite model saved to {TFLITE_PATH}")

# --- 4. C-ARRAY CONVERSION ---
print("\nConverting TFLite model to C-header file...")
def convert_to_c_array(bytes_data, file_name):
    hex_array = [f"0x{b:02x}" for b in bytes_data]
    c_array_content = f"""// Automatically generated C-array for TFLM
#ifndef MODEL_DATA_H
#define MODEL_DATA_H

const unsigned char g_model[] = {{
    {', '.join(hex_array)}
}};
const int g_model_len = {len(bytes_data)};

#endif // MODEL_DATA_H
"""
    with open(file_name, 'w') as f:
        f.write(c_array_content)

convert_to_c_array(tflite_quant_model, C_ARRAY_PATH)
print(f"✅ C-header model saved to {C_ARRAY_PATH}")