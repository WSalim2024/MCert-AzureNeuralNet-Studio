import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# --- IMPORT OUR MODULES ---
from model import SimpleNN  # PyTorch Model
from model_tf import create_tf_model  # TensorFlow Model
from azure_manager import AzureManager

# --- PAGE CONFIG ---
st.set_page_config(page_title="Azure Neural Net Studio v2.1", page_icon="🧠", layout="wide")

# --- CUSTOM TENSORFLOW CALLBACK (For Live UI Updates) ---
import tensorflow as tf


class StreamlitTFCallback(tf.keras.callbacks.Callback):
    def __init__(self, progress_bar, status_text, chart_placeholder, total_epochs):
        super().__init__()
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.chart_placeholder = chart_placeholder
        self.total_epochs = total_epochs
        self.loss_history = []

    def on_epoch_end(self, epoch, logs=None):
        # 1. Update Stats
        loss = logs.get('loss')
        self.loss_history.append(loss)

        # 2. Update Progress Bar
        self.progress_bar.progress((epoch + 1) / self.total_epochs)
        self.status_text.metric(f"Epoch {epoch + 1}/{self.total_epochs}", f"Loss: {loss:.4f}")

        # 3. Update Chart
        fig, ax = plt.subplots()
        ax.plot(self.loss_history, marker='o', color='orange', label='TF Training Loss')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        self.chart_placeholder.pyplot(fig)

        # Small delay to make it visible
        time.sleep(0.5)


# --- SIDEBAR CONFIGURATION ---
st.sidebar.title("⚙️ Model Config")
st.sidebar.markdown("Customize your training run.")

# 1. Dataset Selector
dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ("MNIST (Digits)", "Fashion MNIST (Clothing)")
)

# 2. Optimizer Selector
optimizer_name = st.sidebar.selectbox(
    "Select Optimizer",
    ("SGD", "Adam")
)

st.sidebar.markdown("---")
st.sidebar.title("☁️ Azure Config")
sub_id = st.sidebar.text_input("Subscription ID", type="password")
res_grp = st.sidebar.text_input("Resource Group")
ws_name = st.sidebar.text_input("Workspace Name")

azure_mgr = None
if st.sidebar.button("🔌 Connect to Azure"):
    if sub_id and res_grp and ws_name:
        azure_mgr = AzureManager(sub_id, res_grp, ws_name)
        success, msg = azure_mgr.connect()
        if success:
            st.sidebar.success(msg)
            st.session_state['azure_mgr'] = azure_mgr
        else:
            st.sidebar.error(msg)

# --- MAIN LAYOUT ---
st.title("🧠 Azure Neural Net Studio: v2.1")
st.markdown(f"""
**Current Configuration:** Dataset: `{dataset_name}` | Optimizer: `{optimizer_name}` | Epochs: `10`
""")

tabs = st.tabs(["📊 Data Inspector", "🆚 Code Diff", "🔥 PyTorch Lab", "🟠 TensorFlow Lab", "🚀 Azure Deploy"])

# =========================================
# TAB 1: DATA INSPECTOR
# =========================================
with tabs[0]:
    st.header(f"{dataset_name} Preview")

    if st.button("📥 Load Sample Batch"):
        transform = transforms.ToTensor()

        # DYNAMIC LOADING
        if dataset_name == "MNIST (Digits)":
            data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            labels_map = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
        else:
            data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
            labels_map = {0: 'T-shirt', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
                          5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Boot'}

        loader = torch.utils.data.DataLoader(data, batch_size=10, shuffle=True)
        images, labels = next(iter(loader))

        fig, axes = plt.subplots(1, 10, figsize=(15, 2))
        for i in range(10):
            axes[i].imshow(images[i].squeeze(), cmap='gray')
            axes[i].axis('off')
            lbl_idx = labels[i].item()
            axes[i].set_title(labels_map[lbl_idx])
        st.pyplot(fig)

# =========================================
# TAB 2: ARCHITECTURE COMPARISON
# =========================================
with tabs[1]:
    st.header("Code Comparison: PyTorch vs TensorFlow")
    st.markdown("Both models accept **784 inputs** (28x28) and output **10 classes**.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔥 PyTorch (Object-Oriented)")
        st.code("""
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
        """, language="python")

    with col2:
        st.subheader("🟠 TensorFlow (Declarative)")
        st.code("""
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
        """, language="python")

# =========================================
# TAB 3: PYTORCH LAB
# =========================================
with tabs[2]:
    st.header("🔥 PyTorch Training Loop")

    if st.button("▶️ Start PyTorch Training"):
        # 0. Set Hyperparameters
        EPOCHS = 10
        BATCH_SIZE = 32

        # 1. Dynamic Data Loading
        transform = transforms.ToTensor()
        if dataset_name == "MNIST (Digits)":
            train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        else:
            train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # 2. Init Model
        model = SimpleNN()

        # 3. Dynamic Optimizer
        lr = 0.01 if optimizer_name == "SGD" else 0.001
        if optimizer_name == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=lr)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)

        criterion = nn.CrossEntropyLoss()

        # UI Setup
        progress_bar = st.progress(0)
        status = st.empty()
        chart = st.empty()
        loss_hist = []

        for epoch in range(EPOCHS):
            running_loss = 0.0
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            loss_hist.append(avg_loss)

            # Update UI
            progress_bar.progress((epoch + 1) / EPOCHS)
            status.metric(f"Epoch {epoch + 1}/{EPOCHS}", f"Loss: {avg_loss:.4f}")

            fig, ax = plt.subplots()
            ax.plot(loss_hist, marker='o', color='teal', label=f'PyTorch ({optimizer_name})')
            ax.legend()
            chart.pyplot(fig)

        torch.save(model.state_dict(), "models/simple_nn.pth")
        st.success(f"✅ PyTorch Model ({dataset_name}) Saved!")

# =========================================
# TAB 4: TENSORFLOW LAB
# =========================================
with tabs[3]:
    st.header("🟠 TensorFlow Training Loop")

    if st.button("▶️ Start TensorFlow Training"):
        # 0. Set Hyperparameters
        EPOCHS = 10

        with st.spinner(f"Loading {dataset_name}..."):
            # 1. Dynamic Data Loading
            if dataset_name == "MNIST (Digits)":
                mnist = tf.keras.datasets.mnist
            else:
                mnist = tf.keras.datasets.fashion_mnist

            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train, x_test = x_train / 255.0, x_test / 255.0

            # 2. Create Model
            tf_model = create_tf_model()
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

            # 3. Dynamic Optimizer
            tf_model.compile(optimizer=optimizer_name.lower(), loss=loss_fn, metrics=['accuracy'])

            # UI Setup
            tf_progress = st.progress(0)
            tf_status = st.empty()
            tf_chart = st.empty()

            # Custom Callback
            streamlit_cb = StreamlitTFCallback(tf_progress, tf_status, tf_chart, total_epochs=EPOCHS)

            # Train
            tf_model.fit(x_train, y_train, epochs=EPOCHS, callbacks=[streamlit_cb], verbose=0)

            # --- EVALUATION (The Final Step) ---
            st.divider()
            with st.spinner("Calculating Test Accuracy..."):
                test_loss, test_acc = tf_model.evaluate(x_test, y_test, verbose=0)

            col1, col2 = st.columns(2)
            col1.metric("🏁 Final Test Accuracy", f"{test_acc:.2%}")
            col2.metric("🏁 Final Test Loss", f"{test_loss:.4f}")

            if test_acc > 0.85:
                st.success(f"🚀 Success! You achieved {test_acc:.2%} (Target: >85%)")
            else:
                st.warning(f"Result: {test_acc:.2%}. Try 'Adam' optimizer to improve!")

            # Save
            if not os.path.exists('models'): os.makedirs('models')
            tf_model.save('models/tf_model.h5')
            st.success(f"✅ TensorFlow Model ({dataset_name}) Saved!")

# =========================================
# TAB 5: AZURE DEPLOY
# =========================================
with tabs[4]:
    st.header("🚀 Deploy to Azure")

    model_choice = st.radio("Select Model File:", ["PyTorch (.pth)", "TensorFlow (.h5)"])

    if st.button("☁️ Register Model"):
        if 'azure_mgr' not in st.session_state:
            st.error("Please connect to Azure in the sidebar first!")
        else:
            mgr = st.session_state['azure_mgr']

            if model_choice == "PyTorch (.pth)":
                path = "models/simple_nn.pth"
                name = "mnist-pytorch"
            else:
                path = "models/tf_model.h5"
                name = "mnist-tensorflow"

            if os.path.exists(path):
                with st.spinner(f"Uploading {name}..."):
                    res = mgr.register_model(path, name)
                    st.success(res)
            else:
                st.error(f"File {path} not found. Train the model first.")