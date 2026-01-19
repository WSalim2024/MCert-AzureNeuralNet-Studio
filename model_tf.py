import tensorflow as tf
from tensorflow.keras import layers, models


# --- CLASSIFIERS (Existing) ---
def create_tf_model():
    return models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10)
    ])


def create_cnn_model():
    return models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10)
    ])


def create_iris_model():
    return models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(4,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(3)
    ])


def create_rnn_model(seq_length=10):
    return models.Sequential([
        layers.SimpleRNN(128, input_shape=(seq_length, 1)),
        layers.Dense(1)
    ])


# --- GENERATIVE AI MODELS (New for v5.0) ---

def create_gan_generator(latent_dim=100):
    model = models.Sequential([
        layers.Dense(256, input_dim=latent_dim),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(512),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(1024),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(784, activation='tanh'),
        layers.Reshape((28, 28))
    ])
    return model


def create_gan_discriminator():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(512),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(256),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def create_autoencoder():
    # Encoder
    input_img = layers.Input(shape=(28, 28))
    flat = layers.Flatten()(input_img)
    encoded = layers.Dense(128, activation='relu')(flat)
    latent = layers.Dense(64, activation='relu')(encoded)

    # Decoder
    decoded = layers.Dense(128, activation='relu')(latent)
    output = layers.Dense(784, activation='sigmoid')(decoded)
    reshaped = layers.Reshape((28, 28))(output)

    return models.Model(input_img, reshaped)