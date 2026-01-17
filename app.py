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
from model import SimpleNN  # PyTorch
from model_tf import create_tf_model  # TensorFlow
from azure_manager import AzureManager

# --- PAGE CONFIG ---
st.set_page_config(page_title="Azure Neural Net Studio", page_icon="🧠", layout="wide")

# --- CUSTOM TENSORFLOW CALLBACK (For Live UI Updates) ---
import tensorflow as tf


class StreamlitTFCallback(tf.keras.callbacks.Callback):
    def __init__(self, progress_bar, status_text, chart_placeholder):
        super().__init__()
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.chart_placeholder = chart_placeholder
        self.loss_history = []

    def on_epoch_end(self, epoch, logs=None):
        # 1. Update Stats
        loss = logs.get('loss')
        self.loss_history.append(loss)

        # 2. Update Progress Bar
        # We assume 5 epochs for this demo
        self.progress_bar.progress((epoch + 1) / 5)
        self.status_text.metric(f"Epoch {epoch + 1}/5", f"Loss: {loss:.4f}")

        # 3. Update Chart
        fig, ax = plt.subplots()
        ax.plot(self.loss_history, marker='o', color='orange', label='TF Training Loss')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        self.chart_placeholder.pyplot(fig)

        # Small delay to make it visible
        time.sleep(0.5)


# --- SIDEBAR ---
st.sidebar.title("☁️ Azure Configuration")
st.sidebar.markdown("Connect to your Azure ML Workspace")
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
st.title("🧠 Azure Neural Net Studio: Dual-Engine Edition")
st.markdown("""
Professional Workbench for **PyTorch** and **TensorFlow** experimentation and **Azure** deployment.
""")

# NEW TAB ADDED: "🆚 Framework Showdown"
tabs = st.tabs(["📊 Data Inspector", "⚙️ Architecture", "🔥 PyTorch Lab", "🟠 TensorFlow Lab", "🚀 Azure Deploy"])

# =========================================
# TAB 1: DATA INSPECTOR
# =========================================
with tabs[0]:
    st.header("MNIST Dataset Preview")
    if st.button("📥 Load Sample Batch"):
        transform = transforms.ToTensor()
        mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        loader = torch.utils.data.DataLoader(mnist_data, batch_size=10, shuffle=True)
        images, labels = next(iter(loader))

        fig, axes = plt.subplots(1, 10, figsize=(15, 2))
        for i in range(10):
            axes[i].imshow(images[i].squeeze(), cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f"{labels[i].item()}")
        st.pyplot(fig)

# =========================================
# TAB 2: ARCHITECTURE COMPARISON
# =========================================
with tabs[1]:
    st.header("Code Comparison: PyTorch vs TensorFlow")
    st.markdown("See how the **same** neural network is defined in both frameworks.")

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
        st.info("Explicit control flow. You define the layers and the path.")

    with col2:
        st.subheader("🟠 TensorFlow (Declarative)")
        st.code("""
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
        """, language="python")
        st.info("Configuration style. You stack layers like lego blocks.")

# =========================================
# TAB 3: PYTORCH LAB
# =========================================
with tabs[2]:
    st.header("🔥 PyTorch Training Loop")
    st.markdown("Manually controlling the training steps.")

    if st.button("▶️ Start PyTorch Training"):
        # Setup
        transform = transforms.ToTensor()
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

        model = SimpleNN()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # UI Elements
        progress_bar = st.progress(0)
        status = st.empty()
        chart = st.empty()
        loss_hist = []

        for epoch in range(5):
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
            progress_bar.progress((epoch + 1) / 5)
            status.metric(f"Epoch {epoch + 1}/5", f"Loss: {avg_loss:.4f}")

            fig, ax = plt.subplots()
            ax.plot(loss_hist, marker='o', color='teal')
            chart.pyplot(fig)

        torch.save(model.state_dict(), "models/simple_nn.pth")
        st.success("✅ PyTorch Model Saved!")

# =========================================
# TAB 4: TENSORFLOW LAB (NEW!)
# =========================================
with tabs[3]:
    st.header("🟠 TensorFlow Training Loop")
    st.markdown("Using `model.fit()` with a custom **Streamlit Callback**.")

    if st.button("▶️ Start TensorFlow Training"):
        with st.spinner("Preparing TensorFlow Data..."):
            # Prepare Data (TF style)
            mnist = tf.keras.datasets.mnist
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train = x_train / 255.0  # Normalize

            # Create Model
            tf_model = create_tf_model()
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            tf_model.compile(optimizer='sgd', loss=loss_fn, metrics=['accuracy'])

            # UI Elements
            tf_progress = st.progress(0)
            tf_status = st.empty()
            tf_chart = st.empty()

            # Connect Custom Callback
            streamlit_cb = StreamlitTFCallback(tf_progress, tf_status, tf_chart)

            # TRAIN
            tf_model.fit(x_train, y_train, epochs=5, callbacks=[streamlit_cb], verbose=0)

            # Save
            if not os.path.exists('models'): os.makedirs('models')
            tf_model.save('models/tf_model.h5')
            st.success("✅ TensorFlow Model Saved!")

# =========================================
# TAB 5: AZURE DEPLOY
# =========================================
with tabs[4]:
    st.header("🚀 Deploy to Azure")
    st.markdown("Upload your trained models to the cloud.")

    model_choice = st.radio("Select Model to Upload:", ["PyTorch (.pth)", "TensorFlow (.h5)"])

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
                st.error(f"File {path} not found. Please train the model first.")