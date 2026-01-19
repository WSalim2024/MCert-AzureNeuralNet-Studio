import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- IMPORT MODULES ---
from model import SimpleNN, SimpleCNN, IrisFNN, SineRNN
from model_tf import create_tf_model, create_cnn_model, create_iris_model, create_rnn_model
from azure_manager import AzureManager

st.set_page_config(page_title="Azure Neural Net Studio v4.0", page_icon="🧠", layout="wide")


# --- UTILS: SINE WAVE GENERATOR ---
def create_sine_data(seq_length=10, n_samples=1000):
    t = np.linspace(0, 100, n_samples)
    data = np.sin(t)
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X)[..., None], np.array(y)[..., None], t


# --- UTILS: TF CALLBACK ---
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
        loss = logs.get('loss')
        self.loss_history.append(loss)
        if self.progress_bar: self.progress_bar.progress((epoch + 1) / self.total_epochs)
        if self.status_text: self.status_text.metric(f"Epoch {epoch + 1}/{self.total_epochs}", f"Loss: {loss:.4f}")
        if self.chart_placeholder:
            fig, ax = plt.subplots()
            ax.plot(self.loss_history, color='orange', label='TF Loss')
            ax.legend()
            self.chart_placeholder.pyplot(fig)
            plt.close(fig)


# --- SIDEBAR ---
st.sidebar.title("⚙️ Studio Config")

# 1. TASK SELECTOR (The v4.0 Upgrade)
task_type = st.sidebar.selectbox(
    "Select Task Type",
    ("Image Classification", "Tabular Classification", "Time-Series Regression")
)

# 2. DATASET LOGIC
dataset_name = "None"
if task_type == "Image Classification":
    dataset_name = st.sidebar.selectbox("Dataset", ("MNIST", "Fashion MNIST", "CIFAR-10"))
    model_arch = "CNN" if dataset_name == "CIFAR-10" else "Simple NN"
elif task_type == "Tabular Classification":
    dataset_name = "Iris Plants"
    model_arch = "FNN"
else:
    dataset_name = "Synthetic Sine Wave"
    model_arch = "RNN"

# 3. OPTIMIZER & EPOCHS
optimizer_name = st.sidebar.selectbox("Optimizer", ("Adam", "SGD"))
epochs_val = st.sidebar.slider("Epochs", 1, 50, 20 if task_type == "Time-Series Regression" else 10)

st.sidebar.info(f"Task: {task_type}\nArch: {model_arch}")

# --- MAIN UI ---
st.title(f"🧠 Azure Neural Net Studio v4.0")
st.markdown(f"**Task:** `{task_type}` | **Data:** `{dataset_name}` | **Arch:** `{model_arch}`")

tabs = st.tabs(["📊 Data Inspector", "🆚 Code Diff", "🔥 PyTorch Lab", "🟠 TensorFlow Lab", "🚀 Azure Deploy"])

# =========================================
# TAB 1: DATA INSPECTOR (Dynamic)
# =========================================
with tabs[0]:
    st.header("Data Inspector")
    if task_type == "Image Classification":
        if st.button("Load Sample Images"):
            # Existing Image Logic (Simplified for brevity)
            if dataset_name == "CIFAR-10":
                d = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
                is_color = True
            elif dataset_name == "Fashion MNIST":
                d = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
                is_color = False
            else:
                d = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
                is_color = False

            loader = torch.utils.data.DataLoader(d, batch_size=5, shuffle=True)
            imgs, _ = next(iter(loader))
            fig, ax = plt.subplots(1, 5, figsize=(15, 3))
            for i in range(5):
                img = imgs[i]
                if is_color:
                    ax[i].imshow(np.transpose(img.numpy(), (1, 2, 0)))
                else:
                    ax[i].imshow(img.squeeze(), cmap='gray')
                ax[i].axis('off')
            st.pyplot(fig)

    elif task_type == "Tabular Classification":
        st.write("Loading Iris Dataset...")
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        st.dataframe(df.head(10))
        st.caption("Target 0: Setosa, 1: Versicolor, 2: Virginica")

    elif task_type == "Time-Series Regression":
        st.write("Generating Sine Wave...")
        _, _, t = create_sine_data()
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(t[:200], np.sin(t[:200]), label='Sine Wave')
        ax.set_title("First 200 Time Steps")
        st.pyplot(fig)

# =========================================
# TAB 2: ARCHITECTURE (Dynamic)
# =========================================
with tabs[1]:
    st.header(f"Architecture: {model_arch}")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🔥 PyTorch")
        if task_type == "Tabular Classification":
            st.code("""class IrisFNN(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)""", language='python')
        elif task_type == "Time-Series Regression":
            st.code("""class SineRNN(nn.Module):
    def __init__(self):
        self.rnn = nn.RNN(1, 128)
        self.fc = nn.Linear(128, 1)""", language='python')
        else:
            st.info("Standard CNN/Linear code (see previous versions)")

    with c2:
        st.subheader("🟠 TensorFlow")
        if task_type == "Tabular Classification":
            st.code("""model = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),
    Dense(32, activation='relu'),
    Dense(3)
])""", language='python')
        elif task_type == "Time-Series Regression":
            st.code("""model = Sequential([
    SimpleRNN(128, input_shape=(10, 1)),
    Dense(1)
])""", language='python')

# =========================================
# TAB 3: PYTORCH LAB
# =========================================
with tabs[2]:
    st.header("🔥 PyTorch Training")
    if st.button("▶️ Start PyTorch Training"):
        status = st.empty()
        prog = st.progress(0)
        chart = st.empty()

        # --- DATA PREP ---
        if task_type == "Tabular Classification":
            iris = load_iris()
            X = StandardScaler().fit_transform(iris.data)
            y = iris.target
            # Convert to Tensors
            X_t = torch.tensor(X, dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.long)
            dataset = torch.utils.data.TensorDataset(X_t, y_t)
            loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
            model = IrisFNN()
            criterion = nn.CrossEntropyLoss()

        elif task_type == "Time-Series Regression":
            X_np, y_np, _ = create_sine_data()
            X_t = torch.tensor(X_np, dtype=torch.float32)
            y_t = torch.tensor(y_np, dtype=torch.float32)
            dataset = torch.utils.data.TensorDataset(X_t, y_t)
            loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
            model = SineRNN()
            criterion = nn.MSELoss()  # MSE for Regression

        else:  # Images
            # (Simplified image loader for brevity - uses same logic as v3.4)
            transform = transforms.Compose([transforms.ToTensor()])
            if dataset_name == "CIFAR-10":
                d = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))
                model = SimpleCNN()
            elif dataset_name == "Fashion MNIST":
                d = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
                model = SimpleNN()
            else:
                d = datasets.MNIST('./data', train=True, download=True, transform=transform)
                model = SimpleNN()
            loader = torch.utils.data.DataLoader(d, batch_size=32, shuffle=True)
            criterion = nn.CrossEntropyLoss()

        # --- OPTIMIZER ---
        lr = 0.01 if optimizer_name == "SGD" else 0.001
        optimizer = optim.SGD(model.parameters(), lr=lr) if optimizer_name == "SGD" else optim.Adam(model.parameters(),
                                                                                                    lr=lr)

        # --- LOOP ---
        loss_hist = []
        for epoch in range(epochs_val):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                out = model(X_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg = epoch_loss / len(loader)
            loss_hist.append(avg)
            prog.progress((epoch + 1) / epochs_val)
            status.metric(f"Epoch {epoch + 1}", f"Loss: {avg:.4f}")

            fig, ax = plt.subplots()
            ax.plot(loss_hist, color='teal')
            chart.pyplot(fig)
            plt.close(fig)

        st.success("✅ Training Complete")
        torch.save(model.state_dict(), "models/pytorch_model.pth")

# =========================================
# TAB 4: TENSORFLOW LAB
# =========================================
with tabs[3]:
    st.header("🟠 TensorFlow Training")
    if st.button("▶️ Start TensorFlow Training"):
        status = st.empty()
        prog = st.progress(0)
        chart = st.empty()

        # --- DATA PREP ---
        if task_type == "Tabular Classification":
            iris = load_iris()
            X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
            model = create_iris_model()
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            metrics = ['accuracy']

        elif task_type == "Time-Series Regression":
            X, y, _ = create_sine_data()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model = create_rnn_model()
            loss = 'mse'  # MSE for Regression
            metrics = ['mae']

        else:  # Images
            if dataset_name == "CIFAR-10":
                (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
                X_train = X_train / 255.0;
                X_test = X_test / 255.0
                model = create_cnn_model()
            else:
                # Default MNIST/Fashion
                d = tf.keras.datasets.fashion_mnist if "Fashion" in dataset_name else tf.keras.datasets.mnist
                (X_train, y_train), (X_test, y_test) = d.load_data()
                X_train = X_train / 255.0;
                X_test = X_test / 255.0
                model = create_tf_model()
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            metrics = ['accuracy']

        # --- TRAIN ---
        model.compile(optimizer=optimizer_name.lower(), loss=loss, metrics=metrics)
        cb = StreamlitTFCallback(prog, status, chart, epochs_val)
        model.fit(X_train, y_train, epochs=epochs_val, callbacks=[cb], verbose=0)

        # --- EVALUATE ---
        st.divider()
        res = model.evaluate(X_test, y_test, verbose=0)
        st.metric("Final Loss", f"{res[0]:.4f}")
        if len(res) > 1:
            st.metric("Final Metric (Acc/MAE)", f"{res[1]:.4f}")

        model.save('models/tf_model.h5')
        st.success("✅ Model Saved")

# =========================================
# TAB 5: AZURE DEPLOY (Unchanged)
# =========================================
with tabs[4]:
    st.header("🚀 Deploy to Azure")
    st.write("Supports deploying .pth and .h5 artifacts from all tasks.")
    # (Existing Azure Logic remains here)