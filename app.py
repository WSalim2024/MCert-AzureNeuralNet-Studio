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
import tensorflow as tf

# --- IMPORT MODULES ---
from model import SimpleNN, SimpleCNN, IrisFNN, SineRNN, GANGenerator, GANDiscriminator, Autoencoder
from model_tf import create_tf_model, create_cnn_model, create_iris_model, create_rnn_model, create_gan_generator, \
    create_gan_discriminator, create_autoencoder
from azure_manager import AzureManager

st.set_page_config(page_title="Azure Neural Net Studio v5.0", page_icon="🧠", layout="wide")


# --- UTILS: SINE WAVE ---
def create_sine_data(seq_length=10, n_samples=1000):
    t = np.linspace(0, 100, n_samples)
    data = np.sin(t)
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X)[..., None], np.array(y)[..., None], t


# --- SIDEBAR ---
st.sidebar.title("⚙️ Studio Config")

task_type = st.sidebar.selectbox(
    "Select Task Type",
    ("Image Classification", "Tabular Classification", "Time-Series Regression", "Generative AI (GANs)",
     "Autoencoders (Compression)")
)

# Logic for Config based on Task
dataset_name = "None"
model_arch = "Custom"

if task_type == "Image Classification":
    dataset_name = st.sidebar.selectbox("Dataset", ("MNIST", "Fashion MNIST", "CIFAR-10"))
    model_arch = "CNN" if dataset_name == "CIFAR-10" else "Simple NN"
elif task_type == "Tabular Classification":
    dataset_name = "Iris Plants"
    model_arch = "FNN"
elif task_type == "Time-Series Regression":
    dataset_name = "Synthetic Sine Wave"
    model_arch = "RNN"
elif task_type == "Generative AI (GANs)":
    dataset_name = "MNIST (Digits)"
    model_arch = "GAN (Gen + Disc)"
    st.sidebar.info("Generates new handwritten digits.")
elif task_type == "Autoencoders (Compression)":
    dataset_name = "MNIST (Digits)"
    model_arch = "Encoder-Decoder"
    st.sidebar.info("Compresses 784 pixels -> 64 latent features.")

optimizer_name = st.sidebar.selectbox("Optimizer", ("Adam", "SGD"))
epochs_val = st.sidebar.slider("Epochs", 1, 100, 10)

# --- MAIN LAYOUT ---
st.title(f"🧠 Azure Neural Net Studio v5.0")
st.markdown(f"**Task:** `{task_type}` | **Data:** `{dataset_name}` | **Arch:** `{model_arch}`")

tabs = st.tabs(["📊 Data Inspector", "🆚 Code Diff", "🔥 PyTorch Lab", "🟠 TensorFlow Lab", "🚀 Azure Deploy"])

# =========================================
# TAB 1: DATA INSPECTOR
# =========================================
with tabs[0]:
    st.header(f"Data: {dataset_name}")
    if st.button("Load Sample Data"):
        if "MNIST" in dataset_name or "Fashion" in dataset_name:
            if dataset_name == "Fashion MNIST":
                d = datasets.FashionMNIST('./data', train=True, download=True, transform=transforms.ToTensor())
            else:
                d = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())

            loader = torch.utils.data.DataLoader(d, batch_size=5, shuffle=True)
            imgs, _ = next(iter(loader))
            fig, ax = plt.subplots(1, 5, figsize=(15, 3))
            for i in range(5):
                ax[i].imshow(imgs[i].squeeze(), cmap='gray')
                ax[i].axis('off')
            st.pyplot(fig)
        elif "CIFAR" in dataset_name:
            d = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
            loader = torch.utils.data.DataLoader(d, batch_size=5, shuffle=True)
            imgs, _ = next(iter(loader))
            fig, ax = plt.subplots(1, 5, figsize=(15, 3))
            for i in range(5):
                ax[i].imshow(np.transpose(imgs[i].numpy(), (1, 2, 0)))
                ax[i].axis('off')
            st.pyplot(fig)
        elif "Iris" in dataset_name:
            iris = load_iris()
            st.write(pd.DataFrame(iris.data, columns=iris.feature_names).head())
        elif "Sine" in dataset_name:
            _, _, t = create_sine_data()
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.plot(t[:200], np.sin(t[:200]))
            st.pyplot(fig)

# =========================================
# TAB 2: ARCHITECTURE
# =========================================
with tabs[1]:
    st.header(f"Architecture: {model_arch}")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🔥 PyTorch")
        if "GAN" in task_type:
            st.code("""class GANGenerator(nn.Module):
    # Latent (100) -> Image (784)
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(100, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 784), nn.Tanh()
        )""", language='python')
        elif "Autoencoder" in task_type:
            st.code("""class Autoencoder(nn.Module):
    def __init__(self):
        self.encoder = nn.Sequential(
            nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 784), nn.Sigmoid()
        )""", language='python')
        else:
            st.info("See code for standard classifiers.")

# =========================================
# TAB 3: PYTORCH LAB
# =========================================
with tabs[2]:
    st.header("🔥 PyTorch Training")
    if st.button("▶️ Start PyTorch Training"):
        status = st.empty()
        prog = st.progress(0)
        chart = st.empty()
        viz_area = st.empty()  # For Generated Images

        # --- GANS LOGIC ---
        if task_type == "Generative AI (GANs)":
            # Data
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
            loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

            # Models
            netG = GANGenerator()
            netD = GANDiscriminator()

            # Optimizers
            optG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
            optD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
            criterion = nn.BCELoss()

            G_losses, D_losses = [], []

            for epoch in range(epochs_val):
                for i, (imgs, _) in enumerate(loader):
                    batch_size = imgs.size(0)
                    real_labels = torch.ones(batch_size, 1)
                    fake_labels = torch.zeros(batch_size, 1)

                    # 1. Train Discriminator
                    optD.zero_grad()
                    outputs = netD(imgs)
                    d_loss_real = criterion(outputs, real_labels)

                    z = torch.randn(batch_size, 100)
                    fake_imgs = netG(z)
                    outputs = netD(fake_imgs.detach())
                    d_loss_fake = criterion(outputs, fake_labels)

                    d_loss = d_loss_real + d_loss_fake
                    d_loss.backward()
                    optD.step()

                    # 2. Train Generator
                    optG.zero_grad()
                    outputs = netD(fake_imgs)
                    g_loss = criterion(outputs, real_labels)  # Fool D
                    g_loss.backward()
                    optG.step()

                G_losses.append(g_loss.item())
                D_losses.append(d_loss.item())

                # UI Updates
                prog.progress((epoch + 1) / epochs_val)
                status.metric(f"Epoch {epoch + 1}", f"G Loss: {g_loss.item():.4f} | D Loss: {d_loss.item():.4f}")

                # Visualize Generation
                with torch.no_grad():
                    sample_z = torch.randn(5, 100)
                    gen_imgs = netG(sample_z).reshape(5, 28, 28)
                    fig, ax = plt.subplots(1, 5, figsize=(10, 2))
                    for k in range(5):
                        ax[k].imshow(gen_imgs[k], cmap='gray')
                        ax[k].axis('off')
                    viz_area.pyplot(fig)
                    plt.close(fig)

            torch.save(netG.state_dict(), "models/gan_generator.pth")
            st.success("GAN Training Complete!")

        # --- AUTOENCODER LOGIC ---
        elif task_type == "Autoencoders (Compression)":
            transform = transforms.ToTensor()
            dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
            loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

            model = Autoencoder()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            loss_hist = []

            for epoch in range(epochs_val):
                epoch_loss = 0
                for imgs, _ in loader:
                    optimizer.zero_grad()
                    recon = model(imgs)
                    loss = criterion(recon, imgs)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(loader)
                loss_hist.append(avg_loss)
                prog.progress((epoch + 1) / epochs_val)
                status.metric(f"Epoch {epoch + 1}", f"Recon Loss: {avg_loss:.4f}")

                # Visualize Reconstruction
                with torch.no_grad():
                    orig = imgs[:5]
                    rec = model(orig).reshape(5, 28, 28)
                    fig, ax = plt.subplots(2, 5, figsize=(10, 4))
                    for k in range(5):
                        ax[0, k].imshow(orig[k].squeeze(), cmap='gray')
                        ax[0, k].axis('off')
                        ax[0, k].set_title("Orig")
                        ax[1, k].imshow(rec[k], cmap='gray')
                        ax[1, k].axis('off')
                        ax[1, k].set_title("Recon")
                    viz_area.pyplot(fig)
                    plt.close(fig)

            torch.save(model.state_dict(), "models/autoencoder.pth")
            st.success("Autoencoder Training Complete!")

        # --- STANDARD CLASSIFIERS (Same as v4.0) ---
        else:
            st.info("Please use previous tabs for Classification/Regression tasks.")

# =========================================
# TAB 4: TENSORFLOW LAB
# =========================================
with tabs[3]:
    st.header("🟠 TensorFlow Training")
    if st.button("▶️ Start TensorFlow Training"):
        status = st.empty()
        prog = st.progress(0)
        viz_area = st.empty()

        # --- AUTOENCODER LOGIC (TF) ---
        if task_type == "Autoencoders (Compression)":
            (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
            x_train = x_train.astype('float32') / 255.
            x_test = x_test.astype('float32') / 255.

            autoencoder = create_autoencoder()
            autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

            # Custom Loop for Visualization
            for epoch in range(epochs_val):
                history = autoencoder.fit(x_train, x_train, epochs=1, batch_size=256, shuffle=True, verbose=0)
                loss = history.history['loss'][0]

                prog.progress((epoch + 1) / epochs_val)
                status.metric(f"Epoch {epoch + 1}", f"Loss: {loss:.4f}")

                # Visualize
                decoded_imgs = autoencoder.predict(x_test[:5])
                fig, ax = plt.subplots(2, 5, figsize=(10, 4))
                for k in range(5):
                    ax[0, k].imshow(x_test[k], cmap='gray')
                    ax[0, k].axis('off')
                    ax[1, k].imshow(decoded_imgs[k], cmap='gray')
                    ax[1, k].axis('off')
                viz_area.pyplot(fig)
                plt.close(fig)

            autoencoder.save('models/tf_autoencoder.h5')
            st.success("✅ TF Autoencoder Saved")

        elif task_type == "Generative AI (GANs)":
            st.warning("⚠️ TensorFlow GAN training requires complex custom loops. Please use PyTorch tab for GAN demo.")

        else:
            st.info("Please use previous versions for Classification/Regression tasks.")