<div align="center">

# 🧠☁️ Azure Neural Net Studio: Dual-Engine Edition

### **The Ultimate Framework Showdown**

*Design, Train, and Deploy Neural Networks with PyTorch AND TensorFlow — Side by Side*

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
║   "Two frameworks. One dashboard. Zero excuses."                              ║
║                                                                               ║
║   PyTorch vs TensorFlow — the debate ends here. Now you can run both.        ║
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

**Azure Neural Net Studio: Dual-Engine Edition** is a professional **"Zero to Cloud"** workbench that unifies the two giants of Deep Learning. It enables users to design, train, and deploy Neural Networks using **both PyTorch and TensorFlow** from a single, interactive dashboard.

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      THE DUAL-ENGINE ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│                            📊 MNIST DATA                                        │
│                                  │                                              │
│                    ┌─────────────┴─────────────┐                                │
│                    │                           │                                │
│                    ▼                           ▼                                │
│        ┌─────────────────────┐     ┌─────────────────────┐                      │
│        │  🔥 PYTORCH         │     │  🟠 TENSORFLOW      │                      │
│        │                     │     │                     │                      │
│        │  Object-Oriented    │     │  Declarative        │                      │
│        │  nn.Module Class    │     │  keras.Sequential   │                      │
│        │                     │     │                     │                      │
│        │  Manual Loop:       │     │  Keras API:         │                      │
│        │  optimizer.step()   │     │  model.fit()        │                      │
│        │                     │     │                     │                      │
│        │  Output: .pth       │     │  Output: .h5        │                      │
│        └──────────┬──────────┘     └──────────┬──────────┘                      │
│                   │                           │                                 │
│                   └─────────────┬─────────────┘                                 │
│                                 │                                               │
│                                 ▼                                               │
│                    ┌─────────────────────────┐                                  │
│                    │  ☁️ AZURE ML REGISTRY   │                                  │
│                    │                         │                                  │
│                    │  Register either:       │                                  │
│                    │  • simple_nn.pth        │                                  │
│                    │  • simple_nn.h5         │                                  │
│                    └─────────────────────────┘                                  │
│                                                                                 │
│                    ONE DASHBOARD. TWO FRAMEWORKS. CLOUD READY.                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

### Why Dual-Engine?

| Question | Answer |
|:---------|:-------|
| "Which framework should I learn?" | **Both** — see them side by side |
| "Which is faster?" | Train both and compare live |
| "Which deploys easier?" | Same Azure workflow for both |
| "Which code is cleaner?" | View the Code Diff tab |

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

*See exactly how the same network looks in different paradigms*

</td>
<td width="50%">

### 🔥 Dual Training Labs

Real-time training visualization for **both engines**:

```
PyTorch Loss (Teal)     TensorFlow Loss (Orange)
   │\                      │\
   │ \                     │ \
   │  \                    │  \
   │   \_____              │   \_____
   └──────────             └──────────
     Epochs                  Epochs
```

*PyTorch: Manual Loop | TensorFlow: Custom Callbacks*

</td>
</tr>
<tr>
<td width="50%">

### 📊 Data Inspector

Interactive preview of the **MNIST dataset** — shared by both frameworks.

```
┌─────────────────────────────────┐
│  Shared Data Source             │
│                                 │
│    ┌───┐  ┌───┐  ┌───┐         │
│    │ 5 │  │ 0 │  │ 4 │         │
│    └───┘  └───┘  └───┘         │
│                                 │
│  Same preprocessing for both    │
│  → Fair comparison guaranteed   │
└─────────────────────────────────┘
```

</td>
<td width="50%">

### ☁️ Azure Integration

**One-click connection** to Azure ML Workspace to register models from **either framework**.

```
┌─────────────────────────────────┐
│  Azure Deployment Center        │
│                                 │
│  Select Model:                  │
│  ○ PyTorch (.pth)               │
│  ○ TensorFlow (.h5)             │
│                                 │
│  [ Register to Azure ]          │
│                                 │
│  ✅ Model uploaded!             │
└─────────────────────────────────┘
```

*Same cloud workflow regardless of framework*

</td>
</tr>
</table>

---

## 🎓 What This Project Is About

This project is a **masterclass in MLOps and Framework Interoperability**. It demonstrates how to build production-grade deep learning workflows regardless of the underlying library.

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       FRAMEWORK INTEROPERABILITY                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   THE OLD WAY                              THE DUAL-ENGINE WAY                  │
│   ───────────                              ───────────────────                  │
│                                                                                 │
│   Pick PyTorch OR TensorFlow               Use BOTH in parallel                 │
│        │                                            │                           │
│        ▼                                            ▼                           │
│   Learn one paradigm only                  Compare paradigms live               │
│        │                                            │                           │
│        ▼                                            ▼                           │
│   Separate deployment scripts              Unified Azure workflow               │
│        │                                            │                           │
│        ▼                                            ▼                           │
│   Framework lock-in                        Framework agnostic                   │
│                                                                                 │
│   😵 "Which do I choose?"                 😊 "I understand both!"               │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

### Learning Outcomes

| Skill | What You'll Learn |
|:------|:------------------|
| **PyTorch Fundamentals** | Custom training loops, `nn.Module`, `autograd` |
| **TensorFlow/Keras** | `model.fit()`, custom callbacks, `Sequential` API |
| **MLOps** | Model versioning, cloud deployment, artifact management |
| **Comparative Analysis** | Same task, different approaches, same result |

---

## ⚡ What It Does

The Dual-Engine Edition performs four core operations:

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           CORE CAPABILITIES                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  1️⃣ IMPLEMENT IDENTICAL NETWORKS                                        │   │
│   │                                                                         │   │
│   │  The EXACT SAME Feedforward Neural Network in both frameworks:          │   │
│   │                                                                         │   │
│   │  PyTorch:               TensorFlow:                                     │   │
│   │  class SimpleNN         tf.keras.Sequential([                           │   │
│   │    fc1: 784 → 128         Dense(128, 'relu'),                           │   │
│   │    fc2: 128 → 10          Dense(10)                                     │   │
│   │                         ])                                              │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  2️⃣ TRAIN LIVE IN BROWSER                                               │   │
│   │                                                                         │   │
│   │  Both models train inside Streamlit with real-time loss visualization   │   │
│   │                                                                         │   │
│   │  PyTorch: Teal curve 🟢                                                 │   │
│   │  TensorFlow: Orange curve 🟠                                            │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  3️⃣ VISUALIZE ARCHITECTURAL DIFFERENCES                                 │   │
│   │                                                                         │   │
│   │  Imperative (PyTorch)          vs          Symbolic (TensorFlow)        │   │
│   │  "Define-by-Run"                           "Define-then-Run"            │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  4️⃣ DEPLOY TO AZURE CLOUD                                               │   │
│   │                                                                         │   │
│   │  Upload trained artifacts to Azure ML Registry:                         │   │
│   │  • PyTorch: models/simple_nn.pth                                        │   │
│   │  • TensorFlow: models/simple_nn.h5                                      │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

---

## 🧮 What Is The Logic

Each framework uses a fundamentally different training paradigm:

### 🔥 PyTorch Engine — Imperative / Object-Oriented

Uses a **custom training loop** with manual gradient zeroing and stepping.

```python
# PyTorch: Full control over every step
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Manual training loop
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
for epoch in range(epochs):
    optimizer.zero_grad()           # ← Manual gradient reset
    outputs = model(x_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()                 # ← Manual backprop
    optimizer.step()                # ← Manual weight update
```

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         PYTORCH TRAINING FLOW                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│   │  zero_   │───►│  forward │───►│  loss    │───►│  back    │───►│  step    │ │
│   │  grad()  │    │   pass   │    │  compute │    │  ward()  │    │   ()     │ │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│                                                                                 │
│   YOU control every step — maximum flexibility, maximum responsibility          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

---

### 🟠 TensorFlow Engine — Declarative / Keras API

Uses the **`model.fit()` API** hooked into a **custom StreamlitCallback** to update the UI in real-time.

```python
# TensorFlow: Declarative and concise
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(
    optimizer='sgd',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

# Custom callback for Streamlit UI updates
class StreamlitCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        loss_history.append(logs['loss'])
        progress_bar.progress((epoch + 1) / epochs)

# One-liner training
model.fit(x_train, y_train, epochs=epochs, callbacks=[StreamlitCallback()])
```

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       TENSORFLOW TRAINING FLOW                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │                           model.fit()                                    │  │
│   │                                                                          │  │
│   │   Internally handles:  forward → loss → backward → update                │  │
│   │                                                                          │  │
│   │   You hook in via:     ┌─────────────────────┐                           │  │
│   │                        │ StreamlitCallback   │                           │  │
│   │                        │ on_epoch_end()      │                           │  │
│   │                        └─────────────────────┘                           │  │
│   └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│   KERAS handles the loop — you configure and observe                            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

---

### ☁️ Azure Integration

Uses `azureml-core` to authenticate and register model files to the cloud registry.

```python
from azureml.core import Workspace, Model

ws = Workspace.from_config()

# Register PyTorch model
Model.register(ws, model_path="models/simple_nn.pth", model_name="pytorch_mnist")

# Register TensorFlow model
Model.register(ws, model_path="models/simple_nn.h5", model_name="tensorflow_mnist")
```

---

## ⚙️ How Does It Work

The user navigates through **5 tabs** in the Streamlit UI:

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           5-TAB WORKFLOW                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌───────────┬───────────┬───────────┬───────────┬───────────┐                 │
│   │ 📊 Tab 1  │ 🆚 Tab 2  │ 🔥 Tab 3  │ 🟠 Tab 4  │ ☁️ Tab 5  │                 │
│   │   Data    │   Code    │  PyTorch  │TensorFlow │   Azure   │                 │
│   │ Inspector │   Diff    │   Lab     │   Lab     │  Deploy   │                 │
│   └─────┬─────┴─────┬─────┴─────┬─────┴─────┬─────┴─────┬─────┘                 │
│         │           │           │           │           │                       │
│         ▼           ▼           ▼           ▼           ▼                       │
│   ┌───────────┐┌───────────┐┌───────────┐┌───────────┐┌───────────┐             │
│   │  View     ││  Compare  ││  Manual   ││  Keras    ││  Select   │             │
│   │  MNIST    ││  PyTorch  ││  training ││  model.   ││  .pth or  │             │
│   │  samples  ││  vs TF    ││  loop     ││  fit()    ││  .h5 file │             │
│   │           ││  code     ││  with     ││  with     ││           │             │
│   │           ││  side by  ││  teal     ││  orange   ││  Upload   │             │
│   │           ││  side     ││  loss     ││  loss     ││  to Azure │             │
│   │           ││           ││  curve    ││  curve    ││           │             │
│   └───────────┘└───────────┘└───────────┘└───────────┘└───────────┘             │
│                                                                                 │
│        EXPLORE → UNDERSTAND → TRAIN PYTORCH → TRAIN TF → DEPLOY                 │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

### Tab Responsibilities

| Tab | Name | Purpose |
|:---:|:-----|:--------|
| 1 | **Data Inspector** | View random MNIST samples, understand the input |
| 2 | **Code Diff** | Compare PyTorch vs TensorFlow implementations |
| 3 | **PyTorch Lab** | Train with manual loop, see teal loss curve |
| 4 | **TensorFlow Lab** | Train with Keras API, see orange loss curve |
| 5 | **Azure Deployment** | Upload `.pth` or `.h5` to cloud registry |

---

## 📦 What Are The Requirements

### System Requirements

| Requirement | Specification |
|:------------|:--------------|
| **Python** | 3.9 or 3.10 |
| **OS** | Windows, macOS, or Linux |
| **RAM** | 4GB minimum (8GB recommended) |
| **Internet** | Required (MNIST download + Azure) |

### Library Requirements

| Library | Purpose |
|:--------|:--------|
| `torch` | PyTorch deep learning framework |
| `torchvision` | MNIST dataset for PyTorch |
| `tensorflow` | TensorFlow deep learning framework |
| `streamlit` | Interactive dashboard |
| `azureml-core` | Azure ML SDK |
| `matplotlib` | Visualization |
| `numpy`, `pandas` | Data handling |

### Cloud Requirements (Optional)

| Requirement | Description |
|:------------|:------------|
| **Azure Subscription** | Active Microsoft Azure account |
| **Azure ML Workspace** | Pre-configured workspace |
| **Permissions** | Contributor role on workspace |

---

## 🏗️ Technical Architecture

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          SYSTEM ARCHITECTURE                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                      STREAMLIT FRONTEND                                 │   │
│   │                         (app.py)                                        │   │
│   │                                                                         │   │
│   │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │   │
│   │   │  Tab 1  │ │  Tab 2  │ │  Tab 3  │ │  Tab 4  │ │  Tab 5  │          │   │
│   │   │  Data   │ │  Diff   │ │ PyTorch │ │   TF    │ │  Azure  │          │   │
│   │   └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘          │   │
│   └───────────────────────────────┬─────────────────────────────────────────┘   │
│                                   │                                             │
│               ┌───────────────────┴───────────────────┐                         │
│               │                                       │                         │
│               ▼                                       ▼                         │
│   ┌───────────────────────────┐       ┌───────────────────────────┐             │
│   │       model.py            │       │       model_tf.py         │             │
│   │                           │       │                           │             │
│   │  🔥 PyTorch Class         │       │  🟠 TensorFlow Function   │             │
│   │                           │       │                           │             │
│   │  class SimpleNN:          │       │  def create_model():      │             │
│   │    nn.Module              │       │    keras.Sequential       │             │
│   │    forward()              │       │    compile()              │             │
│   │                           │       │                           │             │
│   │  Training:                │       │  Training:                │             │
│   │    Manual loop            │       │    model.fit()            │             │
│   │    optimizer.step()       │       │    callbacks              │             │
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
│                    │                           │                                │
│                    │  • Authentication         │                                │
│                    │  • Model.register()       │                                │
│                    │  • Supports .pth & .h5    │                                │
│                    └─────────────┬─────────────┘                                │
│                                  │                                              │
│                                  ▼                                              │
│                    ┌───────────────────────────┐                                │
│                    │  ☁️ AZURE ML REGISTRY     │                                │
│                    │                           │                                │
│                    │  Stores both:             │                                │
│                    │  • pytorch_mnist (.pth)   │                                │
│                    │  • tensorflow_mnist (.h5) │                                │
│                    └───────────────────────────┘                                │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

### Component Responsibilities

| Component | File | Responsibility |
|:----------|:-----|:---------------|
| **Frontend** | `app.py` | Streamlit UI, tab navigation, visualization |
| **PyTorch Model** | `model.py` | `nn.Module` class, manual training loop |
| **TensorFlow Model** | `model_tf.py` | Keras Sequential, `model.fit()` with callbacks |
| **Cloud Manager** | `azure_manager.py` | Azure authentication, model upload |

---

## 🤖 Model Specifications

### Identical Architecture

Both frameworks implement the **exact same network** for fair comparison:

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    IDENTICAL FEEDFORWARD NETWORK                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   INPUT                    HIDDEN                    OUTPUT                     │
│   ─────                    ──────                    ──────                     │
│                                                                                 │
│   28×28 Image              128 Units                 10 Classes                 │
│   (Flattened)              (ReLU)                    (Logits)                   │
│                                                                                 │
│   ┌─────────┐            ┌─────────┐              ┌─────────┐                   │
│   │   784   │───────────►│   128   │─────────────►│   10    │                   │
│   │ neurons │  Linear    │ neurons │   Linear     │ neurons │                   │
│   └─────────┘            └─────────┘              └─────────┘                   │
│                             ReLU                                                │
│                                                                                 │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                 │
│   PYTORCH SYNTAX                       TENSORFLOW SYNTAX                        │
│   ──────────────                       ─────────────────                        │
│                                                                                 │
│   self.fc1 = nn.Linear(784, 128)       Dense(128, activation='relu',           │
│   self.fc2 = nn.Linear(128, 10)              input_shape=(784,))               │
│   x = F.relu(self.fc1(x))              Dense(10)                               │
│   x = self.fc2(x)                                                              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

### Specifications Table

| Property | Specification |
|:---------|:--------------|
| **Architecture** | Feedforward Neural Network (FNN) |
| **Input** | 784 dimensions (flattened 28×28 image) |
| **Hidden Layers** | 1 layer with **128 units**, ReLU activation |
| **Output** | 10 units (Linear logits) — digits 0-9 |
| **Parameters** | ~101,770 (identical in both frameworks) |

### Framework-Specific Implementation

| Aspect | PyTorch | TensorFlow |
|:-------|:--------|:-----------|
| **Style** | `nn.Module` class | `keras.Sequential` list |
| **Definition** | Object-Oriented | Declarative/Functional |
| **Training** | Manual loop | `model.fit()` |
| **Save Format** | `.pth` (state dict) | `.h5` (HDF5) |
| **Optimizer** | `torch.optim.SGD` | `tf.keras.optimizers.SGD` |
| **Loss** | `nn.CrossEntropyLoss()` | `SparseCategoricalCrossentropy` |

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology | Version | Purpose |
|:-----:|:----------:|:-------:|:--------|
| 🐍 | **Python** | 3.10+ | Core runtime |
| 🔥 | **PyTorch** | Latest | Deep learning (Engine 1) |
| 🟠 | **TensorFlow** | 2.x | Deep learning (Engine 2) |
| 🖼️ | **Torchvision** | Latest | MNIST dataset (PyTorch) |
| ☁️ | **Azure ML SDK** | azureml-core | Cloud deployment |
| 🖥️ | **Streamlit** | Latest | Interactive dashboard |
| 🔢 | **NumPy** | Latest | Array operations |
| 📊 | **Matplotlib** | Latest | Loss curve visualization |
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
git clone https://github.com/WSalim2024/Azure-Neural-Net-Studio-Dual-Engine.git
```

### Step 2: Navigate to Project Directory

```bash
cd Azure-Neural-Net-Studio-Dual-Engine
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

### Step 6: Verify Installation

```bash
python -c "
import torch
import tensorflow as tf
import streamlit

print('✅ Dual-Engine Ready!')
print(f'   PyTorch: {torch.__version__}')
print(f'   TensorFlow: {tf.__version__}')
"
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
Network URL: http://192.168.x.x:8501
```

---

## 📖 User Guide

### Recommended Workflow

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           USER WORKFLOW                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   STEP 1: COMPARE                                                               │
│   ───────────────                                                               │
│                                                                                 │
│   Go to Tab 2 (Code Diff) to read the architectural differences                 │
│   between PyTorch and TensorFlow implementations.                               │
│                                                                                 │
│   ─────────────────────────────────────────────────────────────────────────     │
│                                                                                 │
│   STEP 2: TRAIN PYTORCH                                                         │
│   ─────────────────────                                                         │
│                                                                                 │
│   • Go to Tab 3 (PyTorch Lab)                                                   │
│   • Set epochs and learning rate                                                │
│   • Click "Start Training"                                                      │
│   • Watch the TEAL 🟢 loss curve descend                                        │
│   • Model saves to: models/simple_nn.pth                                        │
│                                                                                 │
│   ─────────────────────────────────────────────────────────────────────────     │
│                                                                                 │
│   STEP 3: TRAIN TENSORFLOW                                                      │
│   ────────────────────────                                                      │
│                                                                                 │
│   • Go to Tab 4 (TensorFlow Lab)                                                │
│   • Use same epochs and learning rate for fair comparison                       │
│   • Click "Start Training"                                                      │
│   • Watch the ORANGE 🟠 loss curve descend                                      │
│   • Model saves to: models/simple_nn.h5                                         │
│                                                                                 │
│   ─────────────────────────────────────────────────────────────────────────     │
│                                                                                 │
│   STEP 4: DEPLOY                                                                │
│   ──────────────                                                                │
│                                                                                 │
│   • Go to Tab 5 (Azure Deployment)                                              │
│   • Select which model file to upload:                                          │
│     ○ PyTorch: simple_nn.pth                                                    │
│     ○ TensorFlow: simple_nn.h5                                                  │
│   • Click "Register to Azure"                                                   │
│   • Verify upload success                                                       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

### Color Legend

| Color | Framework | Training Style |
|:------|:----------|:---------------|
| 🟢 **Teal** | PyTorch | Manual loop with `optimizer.step()` |
| 🟠 **Orange** | TensorFlow | Keras `model.fit()` with callbacks |

---

## ⚠️ Restrictions and Limitations

| Limitation | Description | Reason |
|:-----------|:------------|:-------|
| **Compute** | Runs on local CPU only | Optimized for small datasets like MNIST |
| **Session** | Data not persisted on refresh | Except for saved model files (.pth, .h5) |
| **TF Version** | Designed for TensorFlow 2.x | Uses Keras API extensively |
| **Dataset** | Hardcoded to MNIST | Demonstration purposes |
| **Azure Auth** | Interactive authentication | May require browser popup |

### Framework Compatibility Note

> ⚠️ **TensorFlow 2.x Required:** This project uses the `tf.keras` API. TensorFlow 1.x is not supported.

---

## 📜 Disclaimer

<div align="center">

---

**🎓 EDUCATIONAL USE ONLY**

---

</div>

This is an **educational tool** demonstrating PyTorch + TensorFlow interoperability with Azure ML.

⚠️ **Azure Usage Warning:**
- Azure services may incur costs depending on your subscription plan
- The author is **not responsible** for any cloud charges
- Monitor your usage in the Azure Portal

---

## 👨‍💻 Author

<div align="center">

### **Waqar Salim**

*Master's Student & IT Professional*

---

[![GitHub](https://img.shields.io/badge/GitHub-WSalim2024-181717?style=for-the-badge&logo=github)](https://github.com/WSalim2024)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/waqar-salim/)

---

**Built with 🔥 PyTorch, 🟠 TensorFlow, ☁️ Azure, and 🆚 Competitive Spirit**

*Azure Neural Net Studio: Dual-Engine Edition — Why Choose When You Can Compare?*

---

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   "The best framework is the one you understand.                              ║
║    With this studio, you'll understand both."                                 ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

</div>
