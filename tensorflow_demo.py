import tensorflow as tf
import os

# Disable GPU warnings for this simple demo (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("--- 🧠 TENSORFLOW IMPLEMENTATION ---")

# 1. Prepare Data (MNIST)
# TensorFlow has this built-in, similar to torchvision
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values (0-255 -> 0.0-1.0)
x_train, x_test = x_train / 255.0, x_test / 255.0

print(f"Data Loaded: {len(x_train)} training images")

# 2. Define the Model (The "Static Graph" approach simplified by Keras)
# Compare this to the 'class SimpleNN(nn.Module)' in your model.py
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Input Layer
    tf.keras.layers.Dense(128, activation='relu'),  # Hidden Layer
    tf.keras.layers.Dense(10)                       # Output Layer
])

# 3. Compile the Model
# This sets up the optimizer and loss function before running
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='sgd',
              loss=loss_fn,
              metrics=['accuracy'])

# 4. Train the Model
print("\n--- 🔥 STARTING TRAINING ---")
model.fit(x_train, y_train, epochs=5)

print("\n--- ✅ EVALUATION ---")
model.evaluate(x_test,  y_test, verbose=2)