<div align="center">

# 🧠☁️ Azure Neural Net Studio: Dual-Engine Edition

### **Version 2.1 — Multi-Dataset & Optimizer Update**

*Compare Frameworks, Datasets, and Optimizers — All in One Dashboard*

---

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Azure](https://img.shields.io/badge/Azure_ML-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

[![GitHub](https://img.shields.io/badge/GitHub-WSalim2024-181717?style=flat-square&logo=github)](https://github.com/WSalim2024)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/waqar-salim/)

<br>

[**Features**](#-key-features) · [**Architecture**](#-technical-architecture) · [**Installation**](#-installation-and-setup) · [**User Guide**](#-user-guide)

<br>

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   🆕 NEW IN v2.1                                                              ║
║   ──────────────                                                              ║
║                                                                               ║
║   👗 Fashion MNIST Support — Train on T-shirts, Sneakers, Dresses            ║
║   ⚡ Adam Optimizer — Compare convergence speed vs SGD                        ║
║   🔟 Extended Training — Default 10 epochs for better visualization          ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

</div>

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Key Features](#-key-features)
3. [What This Project Is About](#-what-this-project-is-about)
4. [What It Does](#-what-it-does)
5. [What Is The Logic](#-what-is-the-logic)
6. [How Does It Work](#-how-does-it-work)
7. [What Are The Requirements](#-what-are-the-requirements)
8. [Technical Architecture](#-technical-architecture)
9. [Model Specifications](#-model-specifications)
10. [Tech Stack](#-tech-stack)
11. [Install Dependencies](#-install-dependencies)
12. [Installation and Setup](#-installation-and-setup)
13. [Launching the Cockpit](#-launching-the-cockpit)
14. [User Guide](#-user-guide)
15. [Restrictions and Limitations](#-restrictions-and-limitations)
16. [Disclaimer](#-disclaimer)
17. [Author](#-author)

---

## 🚀 Overview

**Azure Neural Net Studio: Dual-Engine Edition (v2.1)** is a professional **"Zero to Cloud"** workbench. It enables users to design, train, and deploy Neural Networks using **both PyTorch and TensorFlow** from a single, interactive dashboard.

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      VERSION 2.1 — THE COMPLETE WORKBENCH                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│                         ┌─────────────────────────┐                             │
│                         │    📊 SIDEBAR CONFIG    │                             │
│                         │                         │                             │
│                         │  Dataset:               │                             │
│                         │  ○ MNIST (Digits)       │                             │
│                         │  ● Fashion MNIST 👗     │ ← NEW!                      │
│                         │                         │                             │
│                         │  Optimizer:             │                             │
│                         │  ○ SGD (Slow & Steady)  │                             │
│                         │  ● Adam (Fast) ⚡       │ ← NEW!                      │
│                         │                         │                             │
│                         │  Epochs: [10] 🔟        │ ← Extended!                 │
│                         └───────────┬─────────────┘                             │
│                                     │                                           │
│                    ┌────────────────┴────────────────┐                          │
│                    │                                 │                          │
│                    ▼                                 ▼                          │
│        ┌─────────────────────┐         ┌─────────────────────┐                  │
│        │  🔥 PYTORCH         │         │  🟠 TENSORFLOW      │                  │
│        │                     │         │                     │                  │
│        │  optim.SGD          │         │  'sgd'              │                  │
│        │  optim.Adam ⚡      │         │  'adam' ⚡          │                  │
│        │                     │         │                     │                  │
│        │  Output: .pth       │         │  Output: .h5        │                  │
│        └──────────┬──────────┘         └──────────┬──────────┘                  │
│                   │                               │                             │
│                   └───────────────┬───────────────┘                             │
│                                   │                                             │
│                                   ▼                                             │
│                      ┌─────────────────────────┐                                │
│                      │  ☁️ AZURE ML REGISTRY   │                                │
│                      └─────────────────────────┘                                │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

### What's New in v2.1?

| Feature | v2.0 | v2.1 |
|:--------|:----:|:----:|
| **Datasets** | MNIST (Digits) only | ✅ MNIST + Fashion MNIST |
| **Optimizers** | SGD only | ✅ SGD + Adam |
| **Default Epochs** | 5 | ✅ 10 |
| **Frameworks** | PyTorch + TensorFlow | PyTorch + TensorFlow |
| **Azure Deploy** | ✅ | ✅ |

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🆚 Framework Showdown

Side-by-side **code comparison** of PyTorch (Object-Oriented) vs. TensorFlow (Declarative).

```
┌─────────────────┬─────────────────┐
│    PyTorch      │   TensorFlow    │
├─────────────────┼─────────────────┤
│ class SimpleNN  │ tf.keras.       │
│   (nn.Module):  │   Sequential([  │
│                 │                 │
│   def __init__ │     Dense(128), │
│   def forward  │     Dense(10)   │
│                 │   ])            │
└─────────────────┴─────────────────┘
```

*Compare paradigms in Tab 2*

</td>
<td width="50%">

### 👗 Multi-Dataset Support

Toggle between **MNIST (Digits)** and **Fashion MNIST (Clothing)**.

```
MNIST (Digits)          Fashion MNIST
┌───┬───┬───┐          ┌───┬───┬───┐
│ 0 │ 1 │ 2 │          │👕 │👖 │👗 │
├───┼───┼───┤          ├───┼───┼───┤
│ 3 │ 4 │ 5 │          │👔 │🧥 │👠 │
├───┼───┼───┤          ├───┼───┼───┤
│ 6 │ 7 │ 8 │          │👜 │👟 │🥾 │
└───┴───┴───┘          └───┴───┴───┘
  10 Classes             10 Classes
```

*Same architecture, different domains*

</td>
</tr>
<tr>
<td width="50%">

### ⚡ Dynamic Optimization

Compare convergence speed of **SGD vs. Adam**.

```
Loss
  │
  │\  ← Adam (Fast start)
  │ \____
  │      \____
  │           \
  │\              ← SGD (Slow & steady)
  │  \
  │    \____
  │         \_____
  └───────────────────
           Epochs
```

*Adam often converges faster, but SGD may generalize better*

</td>
<td width="50%">

### 🔥 Dual Training Labs

Real-time visualization with **color-coded loss curves**.

| Framework | Color | Style |
|:----------|:-----:|:------|
| **PyTorch** | 🟢 Teal | Manual loop |
| **TensorFlow** | 🟠 Orange | Keras callbacks |

```
Tab 3: PyTorch Lab
████████████ 100%
Loss: 0.234 ✓

Tab 4: TensorFlow Lab
████████████ 100%
Loss: 0.241 ✓
```

*Train both, compare results*

</td>
</tr>
<tr>
<td colspan="2">

### ☁️ Azure Integration

**One-click deployment** for both `.pth` (PyTorch) and `.h5` (TensorFlow) models to Azure ML Registry.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         AZURE DEPLOYMENT CENTER                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Select Model to Deploy:                                                       │
│                                                                                 │
│   ┌─────────────────────────────┐    ┌─────────────────────────────┐            │
│   │  🔥 PyTorch                 │    │  🟠 TensorFlow              │            │
│   │                             │    │                             │            │
│   │  File: simple_nn.pth        │    │  File: simple_nn.h5         │            │
│   │  Dataset: Fashion MNIST     │    │  Dataset: Fashion MNIST     │            │
│   │  Optimizer: Adam            │    │  Optimizer: Adam            │            │
│   │  Epochs: 10                 │    │  Epochs: 10                 │            │
│   │                             │    │                             │            │
│   │  [ Register to Azure ]      │    │  [ Register to Azure ]      │            │
│   └─────────────────────────────┘    └─────────────────────────────┘            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</td>
</tr>
</table>

---

## 🎓 What This Project Is About

This project is a **masterclass in MLOps and Framework Interoperability**, demonstrating how to handle **multiple data sources** and **training strategies** in a single interface.

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      THE CONFIGURABILITY MATRIX                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│                    FRAMEWORK        DATASET          OPTIMIZER                  │
│                    ─────────        ───────          ─────────                  │
│                                                                                 │
│   Experiment 1:    PyTorch    ×    MNIST      ×       SGD                       │
│   Experiment 2:    PyTorch    ×    MNIST      ×       Adam                      │
│   Experiment 3:    PyTorch    ×    Fashion    ×       SGD                       │
│   Experiment 4:    PyTorch    ×    Fashion    ×       Adam                      │
│   Experiment 5:    TensorFlow ×    MNIST      ×       SGD                       │
│   Experiment 6:    TensorFlow ×    MNIST      ×       Adam                      │
│   Experiment 7:    TensorFlow ×    Fashion    ×       SGD                       │
│   Experiment 8:    TensorFlow ×    Fashion    ×       Adam                      │
│                                                                                 │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                 │
│                    2 Frameworks × 2 Datasets × 2 Optimizers                     │
│                              = 8 COMBINATIONS                                   │
│                                                                                 │
│              All configurable from a single sidebar. No code changes.           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

### Learning Outcomes

| Skill | What You'll Learn |
|:------|:------------------|
| **Framework Flexibility** | Same task in PyTorch vs TensorFlow |
| **Dataset Handling** | Dynamic data loading based on user selection |
| **Optimizer Comparison** | SGD vs Adam convergence behavior |
| **MLOps** | Model versioning and cloud deployment |

---

## ⚡ What It Does

The Dual-Engine Edition v2.1 performs four core operations:

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           CORE CAPABILITIES                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  1️⃣ IMPLEMENT IDENTICAL NETWORKS                                        │   │
│   │                                                                         │   │
│   │  Same Feedforward Network in both PyTorch and TensorFlow                │   │
│   │  → Fair comparison, only framework differs                              │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  2️⃣ TRAIN LIVE IN BROWSER                                               │   │
│   │                                                                         │   │
│   │  10-epoch training with real-time loss curves                           │   │
│   │  → Watch convergence happen before your eyes                            │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  3️⃣ DYNAMICALLY LOAD DIFFERENT DATASETS                                 │   │
│   │                                                                         │   │
│   │  MNIST (Digits 0-9)         vs         Fashion MNIST (Clothing)         │   │
│   │  ┌───────────────────┐                 ┌───────────────────┐            │   │
│   │  │ "Is this a 7?"    │                 │ "Is this a shoe?" │            │   │
│   │  └───────────────────┘                 └───────────────────┘            │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  4️⃣ DEPLOY TO AZURE                                                     │   │
│   │                                                                         │   │
│   │  Upload .pth or .h5 artifacts to Azure ML Model Registry                │   │
│   │  → Production-ready model hosting                                       │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

---

## 🧮 What Is The Logic

### Dataset Loading

Uses `torchvision` or `tf.keras.datasets` to load either **MNIST** or **Fashion MNIST** based on user selection.

```python
# PyTorch Dataset Loading
if dataset_choice == "MNIST (Digits)":
    train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True)
elif dataset_choice == "Fashion MNIST":
    train_data = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True)

# TensorFlow Dataset Loading
if dataset_choice == "MNIST (Digits)":
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
elif dataset_choice == "Fashion MNIST":
    (x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
```

### Fashion MNIST Classes

| Label | Class Name | Emoji |
|:-----:|:-----------|:-----:|
| 0 | T-shirt/Top | 👕 |
| 1 | Trouser | 👖 |
| 2 | Pullover | 🧥 |
| 3 | Dress | 👗 |
| 4 | Coat | 🧥 |
| 5 | Sandal | 👡 |
| 6 | Shirt | 👔 |
| 7 | Sneaker | 👟 |
| 8 | Bag | 👜 |
| 9 | Ankle Boot | 🥾 |

---

### Optimizer Selection

Dynamically switches between optimizers based on sidebar selection.

<table>
<tr>
<th>PyTorch</th>
<th>TensorFlow</th>
</tr>
<tr>
<td>

```python
if optimizer_choice == "SGD":
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=learning_rate
    )
elif optimizer_choice == "Adam":
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate
    )
```

</td>
<td>

```python
if optimizer_choice == "SGD":
    model.compile(
        optimizer='sgd',
        loss=loss_fn
    )
elif optimizer_choice == "Adam":
    model.compile(
        optimizer='adam',
        loss=loss_fn
    )
```

</td>
</tr>
</table>

### Optimizer Comparison

| Property | SGD | Adam |
|:---------|:----|:-----|
| **Speed** | Slower convergence | Faster convergence |
| **Stability** | More stable | Can overshoot |
| **Memory** | Low | Higher (stores momentum) |
| **Best For** | Generalization | Fast prototyping |

---

### Training Logic

**PyTorch:** Uses a **manual training loop** with `optimizer.step()`.

**TensorFlow:** Uses **`model.fit()`** with a custom Streamlit callback for UI updates.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING COMPARISON                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   PYTORCH (Manual Control)              TENSORFLOW (Keras API)                  │
│   ────────────────────────              ──────────────────────                  │
│                                                                                 │
│   for epoch in range(10):               class StreamlitCallback:                │
│       optimizer.zero_grad()                 def on_epoch_end(self):             │
│       outputs = model(x)                        update_progress()               │
│       loss = criterion(outputs, y)                                              │
│       loss.backward()                   model.fit(x, y,                         │
│       optimizer.step()                      epochs=10,                          │
│       update_ui()                           callbacks=[StreamlitCallback()])    │
│                                                                                 │
│   YOU control the loop                  KERAS controls the loop                 │
│   YOU update the UI manually            YOU hook via callbacks                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ How Does It Work

The user navigates through **5 tabs** with configuration in the **Sidebar**:

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           5-TAB + SIDEBAR WORKFLOW                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌─────────────────┐                                                           │
│   │  📊 SIDEBAR     │                                                           │
│   │                 │                                                           │
│   │  Dataset:       │                                                           │
│   │  [Digits ▼]     │──┐                                                        │
│   │  [Fashion ▼]    │  │                                                        │
│   │                 │  │                                                        │
│   │  Optimizer:     │  │ Applies to ALL tabs                                    │
│   │  [SGD ▼]        │  │                                                        │
│   │  [Adam ▼]       │  │                                                        │
│   │                 │  │                                                        │
│   │  Epochs: [10]   │  │                                                        │
│   └─────────────────┘  │                                                        │
│                        │                                                        │
│   ─────────────────────┴────────────────────────────────────────────────────    │
│                                                                                 │
│   ┌─────────┬─────────┬─────────┬─────────┬─────────┐                           │
│   │📊 Tab 1 │🆚 Tab 2 │🔥 Tab 3 │🟠 Tab 4 │☁️ Tab 5 │                           │
│   │  Data   │  Code   │ PyTorch │  Tensor │  Azure  │                           │
│   │Inspector│  Diff   │   Lab   │ FlowLab │ Deploy  │                           │
│   └────┬────┴────┬────┴────┬────┴────┬────┴────┬────┘                           │
│        │         │         │         │         │                                │
│        ▼         ▼         ▼         ▼         ▼                                │
│   ┌─────────┐┌─────────┐┌─────────┐┌─────────┐┌─────────┐                       │
│   │ Shows   ││ Compare ││ Train   ││ Train   ││ Upload  │                       │
│   │ 👕 or 5 ││ PyTorch ││ 10 eps  ││ 10 eps  ││ .pth or │                       │
│   │ based   ││ vs TF   ││ with    ││ with    ││ .h5 to  │                       │
│   │ on      ││ code    ││ SGD/Adam││ SGD/Adam││ Azure   │                       │
│   │ dataset ││         ││ 🟢 Teal ││ 🟠Orange││         │                       │
│   └─────────┘└─────────┘└─────────┘└─────────┘└─────────┘                       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

### Tab Responsibilities

| Tab | Name | What It Shows |
|:---:|:-----|:--------------|
| 1 | **Data Inspector** | Sample images — digits OR clothing items |
| 2 | **Code Diff** | Side-by-side PyTorch vs TensorFlow code |
| 3 | **PyTorch Lab** | 10-epoch training with teal 🟢 loss curve |
| 4 | **TensorFlow Lab** | 10-epoch training with orange 🟠 loss curve |
| 5 | **Azure Deployment** | Upload trained `.pth` or `.h5` to cloud |

---

## 📦 What Are The Requirements

### System Requirements

| Requirement | Specification |
|:------------|:--------------|
| **Python** | 3.10 or higher |
| **OS** | Windows, macOS, or Linux |
| **RAM** | 4GB minimum (8GB recommended) |
| **Internet** | Required (dataset download + Azure) |

### Library Requirements

| Library | Purpose |
|:--------|:--------|
| `torch` | PyTorch deep learning |
| `torchvision` | MNIST & Fashion MNIST (PyTorch) |
| `tensorflow` | TensorFlow/Keras deep learning |
| `streamlit` | Interactive dashboard |
| `azureml-core` | Azure ML SDK |
| `matplotlib` | Loss curve visualization |
| `numpy`, `pandas` | Data handling |

---

## 🏗️ Technical Architecture

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          SYSTEM ARCHITECTURE v2.1                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                      STREAMLIT FRONTEND (app.py)                        │   │
│   │                                                                         │   │
│   │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│   │   │                         SIDEBAR                                 │   │   │
│   │   │   Dataset: [MNIST ▼] [Fashion ▼]                                │   │   │
│   │   │   Optimizer: [SGD ▼] [Adam ▼]                                   │   │   │
│   │   │   Epochs: [10]                                                  │   │   │
│   │   └─────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                         │   │
│   │   ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐                    │   │
│   │   │ Tab 1 │ │ Tab 2 │ │ Tab 3 │ │ Tab 4 │ │ Tab 5 │                    │   │
│   │   └───────┘ └───────┘ └───────┘ └───────┘ └───────┘                    │   │
│   └───────────────────────────────┬─────────────────────────────────────────┘   │
│                                   │                                             │
│               ┌───────────────────┴───────────────────┐                         │
│               │                                       │                         │
│               ▼                                       ▼                         │
│   ┌───────────────────────────┐       ┌───────────────────────────┐             │
│   │       model.py            │       │       model_tf.py         │             │
│   │                           │       │                           │             │
│   │  🔥 PyTorch Engine        │       │  🟠 TensorFlow Engine     │             │
│   │                           │       │                           │             │
│   │  • SimpleNN class         │       │  • create_model()         │             │
│   │  • torch.optim.SGD        │       │  • optimizer='sgd'        │             │
│   │  • torch.optim.Adam ⚡    │       │  • optimizer='adam' ⚡    │             │
│   │  • Manual training loop   │       │  • model.fit() + callback │             │
│   │                           │       │                           │             │
│   │  Datasets:                │       │  Datasets:                │             │
│   │  • torchvision.MNIST      │       │  • keras.datasets.mnist   │             │
│   │  • torchvision.FashionMNIST│      │  • keras.datasets.        │             │
│   │                           │       │      fashion_mnist        │             │
│   │                           │       │                           │             │
│   │  Output: simple_nn.pth    │       │  Output: simple_nn.h5     │             │
│   └─────────────┬─────────────┘       └─────────────┬─────────────┘             │
│                 │                                   │                           │
│                 └─────────────┬─────────────────────┘                           │
│                               │                                                 │
│                               ▼                                                 │
│                    ┌───────────────────────────┐                                │
│                    │     azure_manager.py      │                                │
│                    │                           │                                │
│                    │  ☁️ Azure SDK Wrapper     │                                │
│                    │  • Model.register()       │                                │
│                    │  • Supports .pth & .h5    │                                │
│                    └───────────────────────────┘                                │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

---

## 🤖 Model Specifications

### Architecture

| Property | Specification |
|:---------|:--------------|
| **Type** | Feedforward Neural Network |
| **Input** | 784 dimensions (flattened 28×28) |
| **Hidden** | 128 units, ReLU activation |
| **Output** | 10 units (logits) |
| **Parameters** | ~101,770 |

### Dataset Compatibility

| Dataset | Classes | Examples |
|:--------|:-------:|:---------|
| **MNIST** | 10 | Digits 0-9 |
| **Fashion MNIST** | 10 | Clothing items (T-shirt, Trouser, etc.) |

Both datasets have **identical dimensions** (28×28 grayscale), making them interchangeable without architecture changes.

### Training Configuration

| Property | v2.0 | v2.1 |
|:---------|:----:|:----:|
| **Default Epochs** | 5 | **10** |
| **Optimizers** | SGD | **SGD + Adam** |
| **Learning Rate** | Configurable | Configurable |

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology | Version | Purpose |
|:-----:|:----------:|:-------:|:--------|
| 🐍 | **Python** | 3.10+ | Core runtime |
| 🔥 | **PyTorch** | Latest | Deep learning (Engine 1) |
| 🟠 | **TensorFlow** | 2.x | Deep learning (Engine 2) |
| ☁️ | **Azure ML SDK** | azureml-core | Cloud deployment |
| 🖥️ | **Streamlit** | Latest | Interactive dashboard |
| 📊 | **Matplotlib** | Latest | Loss visualization |
| 🔢 | **NumPy** | Latest | Array operations |
| 📋 | **Pandas** | Latest | Data handling |

</div>

---

## 📥 Install Dependencies

Create a `requirements.txt` file:

```
streamlit
torch
torchvision
azureml-core
matplotlib
numpy
pandas
tensorflow
```

Or install directly:

```bash
pip install streamlit torch torchvision azureml-core matplotlib numpy pandas tensorflow
```

---

## 🔧 Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/WSalim2024/Azure-Neural-Net-Studio-v2.1.git
```

### Step 2: Navigate to Project Directory

```bash
cd Azure-Neural-Net-Studio-v2.1
```

### Step 3: Create Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate Environment

<table>
<tr>
<th>🪟 Windows</th>
<th>🐧 Linux / 🍎 macOS</th>
</tr>
<tr>
<td>

```bash
venv\Scripts\activate
```

</td>
<td>

```bash
source venv/bin/activate
```

</td>
</tr>
</table>

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Launching the Cockpit

### Start the Dashboard

```bash
streamlit run app.py
```

### Access in Browser

```
Local URL: http://localhost:8501
```

---

## 📖 User Guide

### Recommended Workflow: Fashion MNIST + Adam

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           USER WORKFLOW                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   STEP 1: CONFIGURE                                                             │
│   ─────────────────                                                             │
│                                                                                 │
│   In the Sidebar:                                                               │
│   • Select "Fashion MNIST" 👗                                                   │
│   • Select "Adam" ⚡                                                            │
│   • Leave Epochs at 10                                                          │
│                                                                                 │
│   ─────────────────────────────────────────────────────────────────────────     │
│                                                                                 │
│   STEP 2: EXPLORE                                                               │
│   ───────────────                                                               │
│                                                                                 │
│   Go to Tab 1 (Data Inspector)                                                  │
│   • See clothing images: T-shirts 👕, Sneakers 👟, Bags 👜                      │
│   • Confirm Fashion MNIST is loaded                                             │
│                                                                                 │
│   ─────────────────────────────────────────────────────────────────────────     │
│                                                                                 │
│   STEP 3: TRAIN                                                                 │
│   ─────────────                                                                 │
│                                                                                 │
│   Tab 3 (PyTorch): Click "Start Training"                                       │
│   • Watch 10 epochs with teal 🟢 loss curve                                     │
│   • Adam converges faster than SGD!                                             │
│                                                                                 │
│   Tab 4 (TensorFlow): Click "Start Training"                                    │
│   • Watch 10 epochs with orange 🟠 loss curve                                   │
│   • Compare convergence patterns                                                │
│                                                                                 │
│   ─────────────────────────────────────────────────────────────────────────     │
│                                                                                 │
│   STEP 4: DEPLOY                                                                │
│   ──────────────                                                                │
│                                                                                 │
│   Go to Tab 5 (Azure Deployment)                                                │
│   • Select your preferred model (.pth or .h5)                                   │
│   • Click "Register to Azure"                                                   │
│   • Verify upload success ✅                                                    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

### Experiment Ideas

| Experiment | Config | What to Observe |
|:-----------|:-------|:----------------|
| **SGD vs Adam** | Same dataset, different optimizers | Adam converges faster |
| **Digits vs Fashion** | Same optimizer, different datasets | Fashion is harder to classify |
| **PyTorch vs TensorFlow** | Same settings for both | Similar results, different code |
| **Low vs High Epochs** | 5 vs 10 epochs | More epochs = lower loss |

---

## ⚠️ Restrictions and Limitations

| Limitation | Description | Reason |
|:-----------|:------------|:-------|
| **Compute** | Runs on local CPU only | Optimized for small datasets |
| **Persistence** | Session resets on refresh | Saved models persist on disk |
| **TensorFlow** | Requires version 2.x+ | Uses Keras API |
| **Datasets** | MNIST and Fashion MNIST only | Fixed input shape (28×28) |

---

## 📜 Disclaimer

<div align="center">

---

**🎓 EDUCATIONAL USE ONLY**

---

</div>

This is an **educational tool** demonstrating framework interoperability and MLOps practices.

⚠️ **Azure costs are the user's responsibility.**

---

## 👨‍💻 Author

<div align="center">

### **Waqar Salim**

*Master's Student & IT Professional*

---

[![GitHub](https://img.shields.io/badge/GitHub-WSalim2024-181717?style=for-the-badge&logo=github)](https://github.com/WSalim2024)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/waqar-salim/)

---

**Built with 🔥 PyTorch, 🟠 TensorFlow, 👗 Fashion, and ⚡ Adam**

*Azure Neural Net Studio v2.1 — Now with More Choices*

---

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   "Why choose one framework when you can master both?                         ║
║    Why use one dataset when you can compare two?                              ║
║    Why stick with SGD when Adam exists?"                                      ║
║                                                                               ║
║                        — v2.1: The Update That Asked "Why Not?"               ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

</div>
