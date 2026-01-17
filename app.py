import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

# Import our modules
from model import SimpleNN
from azure_manager import AzureManager

# --- PAGE CONFIG ---
st.set_page_config(page_title="Azure Neural Net Studio", page_icon="🧠", layout="wide")

# --- SIDEBAR: AZURE CONFIG ---
st.sidebar.title("☁️ Azure Configuration")
st.sidebar.markdown("Connect to your Azure ML Workspace")

sub_id = st.sidebar.text_input("Subscription ID", type="password")
res_grp = st.sidebar.text_input("Resource Group")
ws_name = st.sidebar.text_input("Workspace Name")

azure_status = st.sidebar.empty()
azure_mgr = None

if st.sidebar.button("🔌 Connect to Azure"):
    if sub_id and res_grp and ws_name:
        azure_mgr = AzureManager(sub_id, res_grp, ws_name)
        success, msg = azure_mgr.connect()
        if success:
            azure_status.success(msg)
            st.session_state['azure_connected'] = True
            st.session_state['azure_mgr'] = azure_mgr
        else:
            azure_status.error(msg)
    else:
        azure_status.warning("Please fill all Azure fields.")

# --- MAIN LAYOUT ---
st.title("🧠 Azure Neural Net Studio")
st.markdown("""
Implement, Train, and Deploy neural networks using **PyTorch** and **Azure Machine Learning**.
""")

tabs = st.tabs(["📊 Data Inspector", "⚙️ Model Architecture", "🔥 Training Lab", "🚀 Azure Deployment"])

# --- TAB 1: DATA INSPECTOR ---
with tabs[0]:
    st.header("MNIST Dataset Preview")
    st.markdown("The dataset consists of 28x28 grayscale images of handwritten digits.")

    if st.button("📥 Load Random Sample Batch"):
        transform = transforms.ToTensor()
        mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)  # Load MNIST
        loader = torch.utils.data.DataLoader(mnist_data, batch_size=10, shuffle=True)
        dataiter = iter(loader)
        images, labels = next(dataiter)

        fig, axes = plt.subplots(1, 10, figsize=(15, 2))
        for i in range(10):
            axes[i].imshow(images[i].squeeze(), cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f"Label: {labels[i].item()}")
        st.pyplot(fig)

# --- TAB 2: ARCHITECTURE ---
with tabs[1]:
    st.header("Feedforward Network Structure")
    st.markdown("We define a simple network with **One Hidden Layer** and **ReLU Activation**.")

    col1, col2 = st.columns(2)
    with col1:
        st.code("""
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128) 
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
        """, language="python")

    with col2:
        st.info(
            "Input Layer: 784 Neurons (Pixels)\n\n⬇️\n\nHidden Layer: 128 Neurons (ReLU)\n\n⬇️\n\nOutput Layer: 10 Neurons (Classes)")

# --- TAB 3: TRAINING LAB ---
with tabs[2]:
    st.header("Model Training")

    # Hyperparameters
    col1, col2, col3 = st.columns(3)
    epochs = col1.slider("Epochs", 1, 10, 5)  # Training epochs
    lr = col2.number_input("Learning Rate", value=0.01, format="%.4f")
    batch_size = col3.selectbox("Batch Size", [32, 64, 128], index=1)

    start_train = st.button("▶️ Start Training")

    if start_train:
        # Prepare Data
        transform = transforms.ToTensor()
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Init Model
        model = SimpleNN()
        criterion = nn.CrossEntropyLoss()  # Loss function
        optimizer = optim.SGD(model.parameters(), lr=lr)  # Optimizer

        # UI Elements for progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        chart_placeholder = st.empty()
        loss_history = []

        st.spinner("Training in progress...")

        for epoch in range(epochs):
            running_loss = 0.0
            for i, (images, labels) in enumerate(train_loader):
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()  # Backpropagation
                optimizer.step()

                running_loss += loss.item()

            # Update Stats
            avg_loss = running_loss / len(train_loader)
            loss_history.append(avg_loss)

            # Update UI
            progress_bar.progress((epoch + 1) / epochs)
            status_text.metric(f"Epoch {epoch + 1}/{epochs}", f"Loss: {avg_loss:.4f}")

            # Live Chart
            fig, ax = plt.subplots()
            ax.plot(loss_history, marker='o', color='teal', label='Training Loss')
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            chart_placeholder.pyplot(fig)

        st.success("✅ Training Complete!")

        # Save Model locally
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/simple_nn.pth")  # Save model
        st.session_state['model_trained'] = True

# --- TAB 4: AZURE DEPLOYMENT ---
with tabs[3]:
    st.header("Deploy to Azure Cloud")

    if 'model_trained' not in st.session_state:
        st.warning("⚠️ Please train a model in the 'Training Lab' tab first.")
    else:
        st.success("✅ Model file `simple_nn.pth` is ready for upload.")

        model_name = st.text_input("Model Registry Name", "mnist-simple-nn")

        if st.button("☁️ Register Model to Azure"):
            if 'azure_mgr' in st.session_state:
                with st.spinner("Uploading to Azure ML Workspace..."):
                    mgr = st.session_state['azure_mgr']
                    result = mgr.register_model("models/simple_nn.pth", model_name)  # Register step
                    st.info(result)
            else:
                st.error("❌ Please connect to Azure in the Sidebar first.")