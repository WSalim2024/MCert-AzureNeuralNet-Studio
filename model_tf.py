import tensorflow as tf


# --- ORIGINAL SIMPLE NN (For MNIST/Fashion) ---
def create_tf_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    return model


# --- NEW CNN ARCHITECTURE (For CIFAR-10) ---
def create_cnn_model():
    model = tf.keras.models.Sequential([
        # Conv Layer 1
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Conv Layer 2
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Flatten & Dense
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    return model