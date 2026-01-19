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

# --- IMPORT OUR MODULES ---
from model import SimpleNN, SimpleCNN
from model_tf import create_tf_model, create_cnn_model
from azure_manager import AzureManager

# --- PAGE CONFIG ---
st.set_page_config(page_title="Azure Neural Net Studio v3.4", page_icon="🧠", layout="wide")

# --- CUSTOM TENSORFLOW CALLBACK ---
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

        # 1. UI Updates (Browser)
        if self.progress_bar:
            self.progress_bar.progress((epoch + 1) / self.total_epochs)
        if self.status_text:
            self.status_text.metric(f"Epoch {epoch + 1}/{self.total_epochs}", f"Loss: {loss:.4f}")

        # 2. Terminal Updates (Console)
        print(f"   Epoch {epoch + 1}/{self.total_epochs}: Loss = {loss:.4f}")

        # 3. Chart Updates
        if self.chart_placeholder:
            fig, ax = plt.subplots()
            ax.plot(self.loss_history, marker='o', color='orange', label='TF Training Loss')
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            self.chart_placeholder.pyplot(fig)
            plt.close(fig)

        # --- HELPER: DATA LOADER ---


def load_tf_data(dataset_name):
    if dataset_name == "CIFAR-10 (Objects)":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        return (x_train, y_train), (x_test, y_test), "CNN"
    elif dataset_name == "Fashion MNIST (Clothing)":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        return (x_train, y_train), (x_test, y_test), "Simple NN"
    else:  # MNIST
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        return (x_train, y_train), (x_test, y_test), "Simple NN"


# --- SIDEBAR CONFIGURATION ---
st.sidebar.title("⚙️ Model Config")

# 1. Dataset Selector
dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ("MNIST (Digits)", "Fashion MNIST (Clothing)", "CIFAR-10 (Objects)")
)

# 2. Architecture Logic
if dataset_name == "CIFAR-10 (Objects)":
    model_type = "CNN"
    st.sidebar.info("ℹ️ CIFAR-10 requires CNN architecture.")
else:
    model_type = st.sidebar.radio("Select Architecture", ("Simple NN", "CNN (Experimental)"))

# 3. Optimizer Selector
optimizer_name = st.sidebar.selectbox("Select Optimizer", ("SGD", "Adam"))

# 4. Epochs
epochs_val = st.sidebar.slider("Epochs", 1, 20, 10)

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
st.title("🧠 Azure Neural Net Studio: v3.4")
st.markdown(f"**Config:** `{dataset_name}` | Arch: `{model_type}` | Opt: `{optimizer_name}` | Epochs: `{epochs_val}`")

tabs = st.tabs(["📊 Data Inspector", "🆚 Code Diff", "🔥 PyTorch Lab", "🟠 TensorFlow Lab", "🚀 Azure Deploy"])

# =========================================
# TAB 1: DATA INSPECTOR
# =========================================
with tabs[0]:
    st.header(f"{dataset_name} Preview")
    if st.button("📥 Load Sample Batch"):
        if dataset_name == "CIFAR-10 (Objects)":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            labels_map = {0: 'Plane', 1: 'Car', 2: 'Bird', 3: 'Cat', 4: 'Deer', 5: 'Dog', 6: 'Frog', 7: 'Horse',
                          8: 'Ship', 9: 'Truck'}
            is_color = True
        elif dataset_name == "Fashion MNIST (Clothing)":
            transform = transforms.ToTensor()
            data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
            labels_map = {0: 'T-shirt', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt',
                          7: 'Sneaker', 8: 'Bag', 9: 'Boot'}
            is_color = False
        else:
            transform = transforms.ToTensor()
            data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            labels_map = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
            is_color = False

        loader = torch.utils.data.DataLoader(data, batch_size=10, shuffle=True)
        images, labels = next(iter(loader))

        fig, axes = plt.subplots(1, 10, figsize=(15, 2))
        for i in range(10):
            img = images[i]
            if is_color:
                img = img / 2 + 0.5
                np_img = img.numpy()
                axes[i].imshow(np.transpose(np_img, (1, 2, 0)))
            else:
                axes[i].imshow(img.squeeze(), cmap='gray')
            axes[i].axis('off')
            lbl_idx = labels[i].item()
            axes[i].set_title(labels_map[lbl_idx])
        st.pyplot(fig)

# =========================================
# TAB 2: ARCHITECTURE COMPARISON
# =========================================
with tabs[1]:
    st.header(f"Architecture: {model_type}")
    col1, col2 = st.columns(2)
    if model_type == "CNN":
        with col1:
            st.subheader("🔥 PyTorch CNN")
            st.code("""
class SimpleCNN(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        ...
            """, language="python")
        with col2:
            st.subheader("🟠 TensorFlow CNN")
            st.code("""
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    ...
])
            """, language="python")
    else:
        st.info("Select 'Simple NN' to view FNN code.")

# =========================================
# TAB 3: PYTORCH LAB
# =========================================
with tabs[2]:
    st.header("🔥 PyTorch Training")
    if st.button("▶️ Start PyTorch Training"):
        # Load Data
        if dataset_name == "CIFAR-10 (Objects)":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
            model = SimpleCNN()
        elif dataset_name == "Fashion MNIST (Clothing)":
            transform = transforms.ToTensor()
            train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
            test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
            model = SimpleNN()
        else:
            transform = transforms.ToTensor()
            train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            model = SimpleNN()

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Optimizer
        lr = 0.01 if optimizer_name == "SGD" else 0.001
        if optimizer_name == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=lr)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Training
        progress_bar = st.progress(0)
        status = st.empty()
        chart = st.empty()
        loss_hist = []

        print(f"\n🔥 PyTorch Training Started: {dataset_name} | {optimizer_name}")
        start_time = time.time()

        for epoch in range(epochs_val):
            running_loss = 0.0
            for i, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            loss_hist.append(avg_loss)

            progress_bar.progress((epoch + 1) / epochs_val)
            status.metric(f"Epoch {epoch + 1}/{epochs_val}", f"Loss: {avg_loss:.4f}")
            print(f"   Epoch {epoch + 1}/{epochs_val}: Loss = {avg_loss:.4f}")

            fig, ax = plt.subplots()
            ax.plot(loss_hist, marker='o', color='teal')
            chart.pyplot(fig)
            plt.close(fig)

        end_time = time.time()
        duration = end_time - start_time
        print(f"✅ PyTorch Done in {duration:.2f}s")
        st.info(f"⏱️ Training Time: {duration:.2f} seconds")

        st.divider()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total
        st.success(f"✅ PyTorch Test Accuracy: {acc:.2%}")
        torch.save(model.state_dict(), "models/pytorch_model.pth")

# =========================================
# TAB 4: TENSORFLOW LAB
# =========================================
with tabs[3]:
    st.header("🟠 TensorFlow Training & Benchmark")

    # --- SECTION 1: SINGLE RUN ---
    st.subheader("1️⃣ Single Training Run")
    if st.button("▶️ Start TensorFlow Training"):
        print(f"\n🟠 TensorFlow Training Started: {dataset_name} | {optimizer_name}")
        with st.spinner(f"Loading {dataset_name}..."):
            (x_train, y_train), (x_test, y_test), arch_used = load_tf_data(dataset_name)

            if arch_used == "CNN":
                tf_model = create_cnn_model()
            else:
                tf_model = create_tf_model()

            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            tf_model.compile(optimizer=optimizer_name.lower(), loss=loss_fn, metrics=['accuracy'])

            tf_progress = st.progress(0)
            tf_status = st.empty()
            tf_chart = st.empty()
            cb = StreamlitTFCallback(tf_progress, tf_status, tf_chart, total_epochs=epochs_val)

            start_time = time.time()
            tf_model.fit(x_train, y_train, epochs=epochs_val, callbacks=[cb], verbose=0)
            end_time = time.time()

            duration = end_time - start_time
            print(f"✅ TensorFlow Done in {duration:.2f}s")
            st.info(f"⏱️ Training Time: {duration:.2f} seconds")

            st.divider()
            test_loss, test_acc = tf_model.evaluate(x_test, y_test, verbose=0)

            col1, col2 = st.columns(2)
            col1.metric("Final Accuracy", f"{test_acc:.2%}")
            col2.metric("Final Loss", f"{test_loss:.4f}")
            print(f"🏁 Final Accuracy: {test_acc:.2%}")

            if not os.path.exists('models'): os.makedirs('models')
            tf_model.save('models/tf_model.h5')

    st.markdown("---")

    # --- SECTION 2: BENCHMARK SUITE ---
    st.subheader("2️⃣ ⚡ Full Benchmark Suite (All Combinations)")

    if st.button("🚀 Run Full Benchmark"):
        print("\n════════════════════════════════════════════")
        print("🚀 STARTING FULL BENCHMARK SUITE")
        print("════════════════════════════════════════════\n")

        benchmark_tasks = [
            ("MNIST (Digits)", "SGD"),
            ("MNIST (Digits)", "Adam"),
            ("Fashion MNIST (Clothing)", "SGD"),
            ("Fashion MNIST (Clothing)", "Adam"),
            ("CIFAR-10 (Objects)", "SGD"),
            ("CIFAR-10 (Objects)", "Adam"),
        ]

        results_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_placeholder = st.empty()

        total_tasks = len(benchmark_tasks)

        for i, (d_name, opt_name) in enumerate(benchmark_tasks):
            # Calculate completion percentage for UI
            percent_complete = int((i / total_tasks) * 100)

            # 1. Terminal Update
            print(f"🚀 [{i + 1}/{total_tasks}] Running Benchmark: {d_name} + {opt_name}...")

            # 2. UI Update (With Percentage!)
            status_text.markdown(
                f"### 🏃‍♂️ Running Task {i + 1}/{total_tasks} ({percent_complete}% Complete)\n**Current:** {d_name} + {opt_name}")
            progress_bar.progress((i) / total_tasks)

            # Clear previous session
            tf.keras.backend.clear_session()

            # Load Data & Model
            (bx_train, by_train), (bx_test, by_test), arch = load_tf_data(d_name)

            if arch == "CNN":
                model = create_cnn_model()
            else:
                model = create_tf_model()

            model.compile(
                optimizer=opt_name.lower(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
            )

            # Callback (No chart for speed)
            bench_cb = StreamlitTFCallback(None, None, None, total_epochs=epochs_val)

            # Train & Time
            t_start = time.time()
            model.fit(bx_train, by_train, epochs=epochs_val, callbacks=[bench_cb], verbose=0)
            t_end = time.time()

            # Evaluate
            _, b_acc = model.evaluate(bx_test, by_test, verbose=0)

            duration = t_end - t_start
            print(f"   ✅ Completed in {duration:.2f}s | Accuracy: {b_acc:.2%}")

            results_data.append({
                "Dataset": d_name.split()[0],
                "Optimizer": opt_name,
                "Time (s)": round(duration, 2),
                "Accuracy": round(b_acc * 100, 2)
            })

            results_placeholder.dataframe(pd.DataFrame(results_data))

        print("\n════════════════════════════════════════════")
        print("✅ BENCHMARK COMPLETE")
        print("════════════════════════════════════════════\n")

        progress_bar.progress(1.0)
        status_text.success("✅ Benchmark Complete! (100%)")

        st.subheader("📊 Benchmark Results")
        df_res = pd.DataFrame(results_data)
        df_res["Task"] = df_res["Dataset"] + " + " + df_res["Optimizer"]
        st.bar_chart(df_res, x="Task", y=["Accuracy", "Time (s)"])

# =========================================
# TAB 5: AZURE DEPLOY
# =========================================
with tabs[4]:
    st.header("🚀 Deploy to Azure")
    model_choice = st.radio("Select Model:", ["PyTorch (.pth)", "TensorFlow (.h5)"])
    if st.button("☁️ Register Model"):
        if 'azure_mgr' in st.session_state:
            mgr = st.session_state['azure_mgr']
            path = "models/pytorch_model.pth" if "PyTorch" in model_choice else "models/tf_model.h5"
            name = "cifar-cnn" if dataset_name == "CIFAR-10 (Objects)" else "mnist-nn"
            if os.path.exists(path):
                res = mgr.register_model(path, name)
                st.success(res)
            else:
                st.error("Model file not found. Train first!")
        else:
            st.error("Connect to Azure first!")