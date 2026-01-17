import tensorflow as tf


def create_tf_model():
    """
    Creates a simple Feedforward Neural Network using TensorFlow/Keras.
    Structure matches the PyTorch 'SimpleNN' for fair comparison.
    """
    model = tf.keras.models.Sequential([
        # Flatten: 28x28 images -> 784 vector
        tf.keras.layers.Flatten(input_shape=(28, 28)),

        # Hidden Layer: 128 neurons, ReLU activation
        tf.keras.layers.Dense(128, activation='relu'),

        # Output Layer: 10 neurons (for 10 digits)
        tf.keras.layers.Dense(10)
    ])
    return model