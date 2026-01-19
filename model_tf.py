import tensorflow as tf
from tensorflow.keras import layers, models

# --- 1. IMAGE MODELS ---
def create_tf_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10)
    ])
    return model

def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10)
    ])
    return model

# --- 2. TABULAR MODEL (Iris) ---
def create_iris_model():
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(4,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(3) # Logits (SparseCategorical uses from_logits=True)
    ])
    return model

# --- 3. TIME-SERIES MODEL (Sine Wave) ---
def create_rnn_model(seq_length=10):
    model = models.Sequential([
        layers.SimpleRNN(128, input_shape=(seq_length, 1)),
        layers.Dense(1)
    ])
    return model