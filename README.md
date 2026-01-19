<div align="center">

# 🧠☁️ Azure Neural Net Studio v3.4

### **The Benchmark Edition**

*Automated Performance Testing • Visual Telemetry • Production-Grade Observability*

---

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Azure](https://img.shields.io/badge/Azure_ML-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

[![GitHub](https://img.shields.io/badge/GitHub-WSalim2024-181717?style=flat-square&logo=github)](https://github.com/WSalim2024)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/waqar-salim/)

<br>

[**Features**](#-key-features) · [**Architecture**](#-technical-architecture) · [**Installation**](#-installation-and-setup) · [**User Guide**](#-user-guide)

<br>

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   🆕 NEW IN v3.4 — THE BENCHMARK EDITION                                      ║
║   ───────────────────────────────────────                                     ║
║                                                                               ║
║   ⚡ AUTOMATED BENCHMARKING — One-click test of 6 model combinations          ║
║   📊 VISUAL TELEMETRY — Live charts + "33% Complete" progress bars            ║
║   🖥️  TERMINAL LOGGING — Epoch-by-epoch telemetry for headless monitoring     ║
║   📈 LEADERBOARD — Real-time Time vs Accuracy comparison table                ║
║                                                                               ║
║   "Don't guess which model is best. Benchmark them all."                      ║
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

**Azure Neural Net Studio v3.4** is the ultimate Dual-Engine workbench for deep learning experimentation. Compare **PyTorch vs TensorFlow**, **SimpleNN vs CNN**, and **SGD vs Adam** — all from a single dashboard. Now featuring a full **Automated Benchmark Suite** that tests all combinations and builds a performance leaderboard in real-time.

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      VERSION 3.4 — THE BENCHMARK EDITION                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│                         ⚡ AUTOMATED BENCHMARK SUITE                            │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                        BENCHMARK TASK QUEUE                             │   │
│   │                                                                         │   │
│   │   Task 1: MNIST      + SGD   → SimpleNN    ████████████████░░ 83%      │   │
│   │   Task 2: MNIST      + Adam  → SimpleNN    ████████████████░░ 83%      │   │
│   │   Task 3: Fashion    + SGD   → SimpleNN    ████████████████░░ 83%      │   │
│   │   Task 4: Fashion    + Adam  → SimpleNN    ████████████████░░ 83%      │   │
│   │   Task 5: CIFAR-10   + SGD   → CNN         ████████░░░░░░░░░░ 50%      │   │
│   │   Task 6: CIFAR-10   + Adam  → CNN         ░░░░░░░░░░░░░░░░░░ 0%       │   │
│   │                                                                         │   │
│   │   Overall Progress: ████████████░░░░░░░░ 67% (4/6 Complete)            │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│                                     │                                           │
│                                     ▼                                           │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                      📊 PERFORMANCE LEADERBOARD                         │   │
│   │                                                                         │   │
│   │   Rank │ Dataset    │ Optimizer │ Time (s) │ Accuracy │                 │   │
│   │   ─────┼────────────┼───────────┼──────────┼──────────┤                 │   │
│   │    1   │ MNIST      │ Adam      │   12.3   │  97.8%   │ 🏆              │   │
│   │    2   │ MNIST      │ SGD       │   11.9   │  96.2%   │                 │   │
│   │    3   │ Fashion    │ Adam      │   14.1   │  89.4%   │                 │   │
│   │    4   │ Fashion    │ SGD       │   13.8   │  87.1%   │                 │   │
│   │    5   │ CIFAR-10   │ Adam      │   48.2   │  72.3%   │                 │   │
│   │    6   │ CIFAR-10   │ SGD       │   47.5   │  68.9%   │                 │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

### Version Evolution

| Version | Key Feature | Focus |
|:--------|:------------|:------|
| **v2.1** | Multi-Dataset, Adam Optimizer | Flexibility |
| **v3.0** | CIFAR-10, CNN Architecture | Visual Learning |
| **v3.4** | **Automated Benchmark Suite** | **Observability** |

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### ⚡ Automated Benchmarking

**One-click performance test** of all dataset/optimizer combinations.

```
┌─────────────────────────────────┐
│  🏁 BENCHMARK SUITE             │
│                                 │
│  [ Run Full Benchmark ]         │
│                                 │
│  Tasks: 6 combinations          │
│  Time: ~3-5 minutes (CPU)       │
│                                 │
│  Output:                        │
│  • Time (seconds)               │
│  • Accuracy (%)                 │
│  • Ranked Leaderboard           │
└─────────────────────────────────┘
```

**Benchmark Task List:**

| # | Dataset | Optimizer | Architecture |
|:-:|:--------|:----------|:-------------|
| 1 | MNIST | SGD | SimpleNN |
| 2 | MNIST | Adam | SimpleNN |
| 3 | Fashion | SGD | SimpleNN |
| 4 | Fashion | Adam | SimpleNN |
| 5 | CIFAR-10 | SGD | CNN |
| 6 | CIFAR-10 | Adam | CNN |

</td>
<td width="50%">

### 👁️ Visual Telemetry

**Live charts, progress bars with % completion**, and terminal logs.

```
┌─────────────────────────────────┐
│  📊 TELEMETRY DASHBOARD         │
│                                 │
│  UI Progress:                   │
│  ████████████░░░░ 67% Complete  │
│                                 │
│  Current Task: Fashion + Adam   │
│  Epoch: 7/10                    │
│  Loss: 0.342 ↓                  │
│                                 │
│  ─────────────────────────────  │
│                                 │
│  Terminal Output:               │
│  [INFO] Task 3/6 started        │
│  [EPOCH 7] loss=0.342 acc=87.1% │
│  [INFO] ETA: 45 seconds         │
└─────────────────────────────────┘
```

*Monitor from UI or terminal — your choice*

</td>
</tr>
<tr>
<td width="50%">

### 🌈 Multi-Modal Support

**Grayscale and Color** image support with automatic preprocessing.

| Mode | Dataset | Dimensions | Channels |
|:-----|:--------|:-----------|:--------:|
| Grayscale | MNIST | 28×28 | 1 |
| Grayscale | Fashion | 28×28 | 1 |
| **Color** | CIFAR-10 | 32×32 | **3 (RGB)** |

```
Grayscale (MNIST):       Color (CIFAR-10):
┌─────────────┐          ┌─────────────┐
│  ░░███░░    │          │  🔴🟢🔵    │
│  ░░███░░    │          │  RGB layers │
│  ░░███░░    │          │  32×32×3    │
└─────────────┘          └─────────────┘
   1 channel               3 channels
```

</td>
<td width="50%">

### 🏗️ Dual Architectures

**Automatic switching** between SimpleNN and CNN based on data type.

```
IF dataset == "CIFAR-10":
    architecture = CNN        # Conv2D layers
ELSE:
    architecture = SimpleNN   # Dense layers
```

| Dataset | Auto-Selected | Why |
|:--------|:--------------|:----|
| MNIST | SimpleNN | 28×28, simple patterns |
| Fashion | SimpleNN | 28×28, grayscale |
| CIFAR-10 | **CNN** | 32×32 RGB, spatial features |

*No manual configuration needed — the app is smart.*

</td>
</tr>
</table>

---

## 🎓 What This Project Is About

This project is a masterclass in building **production-grade ML tools** that focus on **Observability** and **Interoperability**. It demonstrates how to create dashboards that don't just train models — they **measure, compare, and report** on them systematically.

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      THE OBSERVABILITY PHILOSOPHY                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   TRADITIONAL ML WORKFLOW                   v3.4 BENCHMARK WORKFLOW             │
│   ───────────────────────                   ───────────────────────             │
│                                                                                 │
│   1. Pick a model                           1. Define ALL models                │
│   2. Train it                               2. Run automated benchmark          │
│   3. Check results                          3. Compare with leaderboard         │
│   4. Manually try another                   4. Deploy the winner                │
│   5. Repeat (tedious)                                                           │
│                                                                                 │
│   😵 "Which model is best?"                 📊 "The data shows Model X wins."   │
│                                                                                 │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                 │
│                         OBSERVABILITY = CONFIDENCE                              │
│                                                                                 │
│   • Visual Progress Bars → Know exactly where you are                           │
│   • Terminal Telemetry → Monitor headlessly (SSH, CI/CD)                        │
│   • Pandas DataFrame → Export results for further analysis                      │
│   • Leaderboard → Instant comparison, no guesswork                              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

### Engineering Principles in v3.4

| Principle | Implementation |
|:----------|:---------------|
| **Observability** | Progress bars, terminal logs, live charts |
| **Automation** | One-click benchmark of 6 combinations |
| **Interoperability** | PyTorch + TensorFlow in same workflow |
| **Reproducibility** | Consistent task list, comparable results |
| **Scalability** | Session clearing prevents RAM overflow |

---

## ⚡ What It Does

Azure Neural Net Studio v3.4 performs **three core functions**:

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           CORE CAPABILITIES v3.4                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  1️⃣ TRAIN MODELS LIVE IN THE BROWSER                                    │   │
│   │                                                                         │   │
│   │  • PyTorch Tab: Manual training loop with optimizer.step()              │   │
│   │  • TensorFlow Tab: model.fit() with StreamlitCallback                   │   │
│   │  • Real-time loss curves update as training progresses                  │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  2️⃣ MEASURE AND COMPARE PERFORMANCE                                     │   │
│   │                                                                         │   │
│   │  Metrics Captured:                                                      │   │
│   │  ┌────────────────┬────────────────┬────────────────┐                   │   │
│   │  │ Training Time  │    Accuracy    │   Loss Curve   │                   │   │
│   │  │   (seconds)    │      (%)       │   (history)    │                   │   │
│   │  └────────────────┴────────────────┴────────────────┘                   │   │
│   │                                                                         │   │
│   │  Comparison Output:                                                     │   │
│   │  • Pandas DataFrame with all results                                    │   │
│   │  • Ranked leaderboard (best accuracy first)                             │   │
│   │  • Time vs Accuracy scatter plot                                        │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  3️⃣ DEPLOY ARTIFACTS TO AZURE                                           │   │
│   │                                                                         │   │
│   │  • Register trained models (.pth or .h5) to Azure ML Registry           │   │
│   │  • Works for both SimpleNN and CNN architectures                        │   │
│   │  • One-click deployment from Tab 5                                      │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

---

## 🧮 What Is The Logic

### Smart Architecture Selection

The system **automatically assigns** the correct architecture based on input data dimensions.

```python
def select_architecture(dataset_name):
    """Smart selection based on data characteristics"""
    if dataset_name == "CIFAR-10":
        # 32×32 RGB images need spatial feature extraction
        return "CNN"  # Conv2D layers
    else:
        # 28×28 grayscale (MNIST, Fashion) work well with dense layers
        return "SimpleNN"  # Flatten → Dense
```

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      SMART ARCHITECTURE SELECTION                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   INPUT                          DECISION                        OUTPUT         │
│   ─────                          ────────                        ──────         │
│                                                                                 │
│   Dataset: MNIST                                                                │
│   Shape: 28×28×1        ───►     "Grayscale, small"     ───►    SimpleNN       │
│   Channels: 1                    Use Dense layers                               │
│                                                                                 │
│   Dataset: Fashion MNIST                                                        │
│   Shape: 28×28×1        ───►     "Grayscale, patterns"  ───►    SimpleNN       │
│   Channels: 1                    Use Dense layers                               │
│                                                                                 │
│   Dataset: CIFAR-10                                                             │
│   Shape: 32×32×3        ───►     "Color, spatial"       ───►    CNN            │
│   Channels: 3 (RGB)              Use Conv2D layers                              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

---

### Benchmarking Loop Logic

The benchmark engine **iterates through a task list**, trains each model, and aggregates results.

```python
def run_full_benchmark():
    """Execute all 6 benchmark tasks sequentially"""
    
    task_list = [
        {"dataset": "MNIST",    "optimizer": "SGD",  "arch": "SimpleNN"},
        {"dataset": "MNIST",    "optimizer": "Adam", "arch": "SimpleNN"},
        {"dataset": "Fashion",  "optimizer": "SGD",  "arch": "SimpleNN"},
        {"dataset": "Fashion",  "optimizer": "Adam", "arch": "SimpleNN"},
        {"dataset": "CIFAR-10", "optimizer": "SGD",  "arch": "CNN"},
        {"dataset": "CIFAR-10", "optimizer": "Adam", "arch": "CNN"},
    ]
    
    results = []
    
    for i, task in enumerate(task_list):
        # Update UI progress
        progress = (i / len(task_list)) * 100
        st.progress(progress, text=f"{progress:.0f}% Complete")
        
        # Clear TF session to free RAM
        tf.keras.backend.clear_session()
        
        # Train and measure
        start_time = time.time()
        accuracy = train_model(task)
        elapsed = time.time() - start_time
        
        # Log to terminal
        print(f"[INFO] Task {i+1}/6: {task['dataset']} + {task['optimizer']}")
        print(f"[RESULT] Time: {elapsed:.1f}s | Accuracy: {accuracy:.1f}%")
        
        results.append({
            "Dataset": task["dataset"],
            "Optimizer": task["optimizer"],
            "Time (s)": round(elapsed, 1),
            "Accuracy (%)": round(accuracy, 1)
        })
    
    # Create leaderboard
    df = pd.DataFrame(results)
    df = df.sort_values("Accuracy (%)", ascending=False)
    return df
```

---

### Terminal Telemetry

Real-time **epoch-by-epoch logging** for headless monitoring (SSH, CI/CD pipelines).

```
$ streamlit run app.py

[INFO] Azure Neural Net Studio v3.4 - Benchmark Mode
[INFO] ═══════════════════════════════════════════════════
[INFO] Task 1/6: MNIST + SGD (SimpleNN)
[EPOCH 1/10] loss=2.142 acc=45.2% time=1.2s
[EPOCH 2/10] loss=0.891 acc=72.3% time=1.1s
[EPOCH 3/10] loss=0.534 acc=84.1% time=1.1s
...
[EPOCH 10/10] loss=0.198 acc=96.2% time=1.0s
[RESULT] MNIST + SGD: 11.9s | 96.2% ✓

[INFO] Task 2/6: MNIST + Adam (SimpleNN)
[EPOCH 1/10] loss=1.823 acc=52.1% time=1.3s
...
[RESULT] MNIST + Adam: 12.3s | 97.8% ✓

[INFO] ═══════════════════════════════════════════════════
[INFO] BENCHMARK COMPLETE
[INFO] Best Model: MNIST + Adam (97.8% accuracy)
[INFO] Results exported to: benchmark_results.csv
```

---

## ⚙️ How Does It Work

The application provides **5 tabs** with specialized functions:

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           TAB STRUCTURE v3.4                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌─────────┬─────────┬─────────┬─────────┬─────────┐                           │
│   │📊 Tab 1 │🆚 Tab 2 │🔥 Tab 3 │🟠 Tab 4 │☁️ Tab 5 │                           │
│   │  Data   │  Code   │ PyTorch │TensorFlow│  Azure  │                           │
│   │Inspector│  Diff   │   Lab   │   Lab   │ Deploy  │                           │
│   └────┬────┴────┬────┴────┬────┴────┬────┴────┬────┘                           │
│        │         │         │         │         │                                │
│        ▼         ▼         ▼         ▼         ▼                                │
│   ┌─────────┐┌─────────┐┌─────────┐┌─────────┐┌─────────┐                       │
│   │ Un-     ││ Compare ││ Manual  ││ TWO     ││ Upload  │                       │
│   │ normalize││ PyTorch ││ training││ MODES:  ││ models  │                       │
│   │ CIFAR   ││ vs TF   ││ loop    ││         ││ to      │                       │
│   │ images  ││ code    ││ with    ││ • Single││ Azure   │                       │
│   │ for     ││         ││ time    ││   Run   ││ ML      │                       │
│   │ viewing ││         ││ .time() ││ • Full  ││ Registry│                       │
│   │         ││         ││ tracking││ Benchmark││        │                       │
│   └─────────┘└─────────┘└─────────┘└─────────┘└─────────┘                       │
│                                                                                 │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                 │
│   TAB 4 DETAIL — TENSORFLOW LAB                                                 │
│   ─────────────────────────────                                                 │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                         │   │
│   │   MODE SELECTOR                                                         │   │
│   │   ──────────────                                                        │   │
│   │                                                                         │   │
│   │   ○ Single Run        Train one model with current sidebar settings     │   │
│   │   ● Full Benchmark    Run all 6 combinations automatically 🆕          │   │
│   │                                                                         │   │
│   │   ─────────────────────────────────────────────────────────────────     │   │
│   │                                                                         │   │
│   │   [ Start Training ]   [ Run Full Benchmark ]                           │   │
│   │                                                                         │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

### Tab 1: Data Inspector

**Un-normalizes CIFAR images** for proper viewing (reverses the -1 to 1 normalization back to 0-255 RGB).

```python
def display_cifar_image(normalized_image):
    """Convert normalized tensor back to viewable image"""
    # Reverse normalization: (x * 0.5) + 0.5 → 0 to 1 range
    image = (normalized_image * 0.5) + 0.5
    # Convert to 0-255 range
    image = (image * 255).astype(np.uint8)
    return image
```

### Tab 3: PyTorch Lab

**Manual training loop** with `time.time()` tracking for precise duration measurement.

```python
start_time = time.time()
for epoch in range(epochs):
    # Training loop
    optimizer.zero_grad()
    outputs = model(x_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()
    
elapsed = time.time() - start_time
print(f"Training completed in {elapsed:.1f} seconds")
```

### Tab 4: TensorFlow Lab

Contains **two modes**: Single Run (manual) and Full Benchmark (automated).

---

## 📦 What Are The Requirements

### System Requirements

| Requirement | Specification |
|:------------|:--------------|
| **Python** | 3.10 or higher |
| **OS** | Windows, macOS, or Linux |
| **RAM** | 8GB recommended (benchmark runs 6 models) |
| **Internet** | Required (CIFAR-10 download ~160MB) |

### Library Requirements

| Library | Purpose |
|:--------|:--------|
| `torch` | PyTorch training engine |
| `torchvision` | Dataset loading (PyTorch) |
| `tensorflow` | TensorFlow training engine |
| `streamlit` | Interactive dashboard |
| `pandas` | Benchmark results aggregation |
| `matplotlib` | Loss curves and charts |
| `azureml-core` | Azure ML deployment |
| `numpy` | Array operations |

---

## 🏗️ Technical Architecture

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          SYSTEM ARCHITECTURE v3.4                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                      STREAMLIT UI (app.py)                              │   │
│   │                                                                         │   │
│   │   ┌──────────────────────────────────────────────────────────────────┐  │   │
│   │   │   SIDEBAR: Dataset | Optimizer | Epochs | Architecture           │  │   │
│   │   └──────────────────────────────────────────────────────────────────┘  │   │
│   │                                                                         │   │
│   │   ┌──────────────────────────────────────────────────────────────────┐  │   │
│   │   │   TELEMETRY PANEL                                                │  │   │
│   │   │   • Progress Bar: ████████░░░░ 67%                               │  │   │
│   │   │   • Current Task: Fashion + Adam                                 │  │   │
│   │   │   • ETA: 45 seconds                                              │  │   │
│   │   └──────────────────────────────────────────────────────────────────┘  │   │
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
│   │    🔥 PYTORCH ENGINE      │       │   🟠 TENSORFLOW ENGINE    │             │
│   │       (model.py)          │       │      (model_tf.py)        │             │
│   │                           │       │                           │             │
│   │  SimpleNN:                │       │  SimpleNN:                │             │
│   │    nn.Linear(784, 128)    │       │    Dense(128)             │             │
│   │    nn.Linear(128, 10)     │       │    Dense(10)              │             │
│   │                           │       │                           │             │
│   │  CNN:                     │       │  CNN:                     │             │
│   │    nn.Conv2d(3, 32)       │       │    Conv2D(32)             │             │
│   │    nn.Conv2d(32, 64)      │       │    Conv2D(64)             │             │
│   │    nn.Linear(64*8*8, 128) │       │    Dense(128)             │             │
│   │                           │       │                           │             │
│   │  + time.time() tracking   │       │  + StreamlitCallback      │             │
│   │  + Terminal telemetry     │       │  + Benchmark engine 🆕    │             │
│   └─────────────┬─────────────┘       └─────────────┬─────────────┘             │
│                 │                                   │                           │
│                 └───────────────┬───────────────────┘                           │
│                                 │                                               │
│                                 ▼                                               │
│                    ┌───────────────────────────┐                                │
│                    │   📊 BENCHMARK ENGINE     │                                │
│                    │                           │                                │
│                    │  • Task Queue (6 tasks)   │                                │
│                    │  • tf.keras.backend.      │                                │
│                    │      clear_session()      │                                │
│                    │  • Pandas DataFrame       │                                │
│                    │  • Leaderboard sorting    │                                │
│                    └─────────────┬─────────────┘                                │
│                                  │                                              │
│                                  ▼                                              │
│                    ┌───────────────────────────┐                                │
│                    │     azure_manager.py      │                                │
│                    │                           │                                │
│                    │  ☁️ Azure ML Registry     │                                │
│                    │  • Model.register()       │                                │
│                    │  • Supports all model types│                               │
│                    └───────────────────────────┘                                │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

---

## 🤖 Model Specifications

### SimpleNN (Feedforward Neural Network)

For **MNIST** and **Fashion MNIST** (28×28 grayscale).

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         SimpleNN ARCHITECTURE                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌─────────┐            ┌─────────┐              ┌─────────┐                   │
│   │ Flatten │───────────►│ Dense   │─────────────►│ Dense   │                   │
│   │ 28×28→  │            │  128    │              │   10    │                   │
│   │  784    │            │  ReLU   │              │ Output  │                   │
│   └─────────┘            └─────────┘              └─────────┘                   │
│                                                                                 │
│   Parameters: ~101,770                                                          │
│   Best for: Digit recognition, simple patterns                                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### CNN (Convolutional Neural Network)

For **CIFAR-10** (32×32 RGB color).

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            CNN ARCHITECTURE                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐          │
│   │ Input   │──►│ Conv2D  │──►│ MaxPool │──►│ Conv2D  │──►│ MaxPool │          │
│   │ 32×32×3 │   │   32    │   │  2×2    │   │   64    │   │  2×2    │          │
│   │  (RGB)  │   │  3×3    │   │ 32→16   │   │  3×3    │   │ 16→8    │          │
│   └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘          │
│                                                                 │               │
│                                                                 ▼               │
│                                              ┌─────────┐   ┌─────────┐          │
│                                              │ Dense   │──►│ Dense   │          │
│                                              │  128    │   │   10    │          │
│                                              │  ReLU   │   │ Output  │          │
│                                              └─────────┘   └─────────┘          │
│                                                                                 │
│   Parameters: ~122,570                                                          │
│   Best for: Object recognition, spatial features, color images                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology | Version | Purpose |
|:-----:|:----------:|:-------:|:--------|
| 🐍 | **Python** | 3.10+ | Core runtime |
| 🔥 | **PyTorch** | Latest | Training engine 1 |
| 🟠 | **TensorFlow** | 2.x | Training engine 2 + Benchmark |
| ☁️ | **Azure ML SDK** | azureml-core | Cloud deployment |
| 🖥️ | **Streamlit** | Latest | Interactive dashboard |
| 📊 | **Pandas** | Latest | Benchmark results aggregation |
| 📈 | **Matplotlib** | Latest | Loss curves & charts |
| 🔢 | **NumPy** | Latest | Array operations |

</div>

---

## 📥 Install Dependencies

Create a `requirements.txt` file:

```
streamlit
torch
torchvision
tensorflow
azureml-core
matplotlib
numpy
pandas
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 🔧 Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/WSalim2024/Azure-Neural-Net-Studio-v3.4.git
```

### Step 2: Navigate to Project Directory

```bash
cd Azure-Neural-Net-Studio-v3.4
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

### Monitor in Terminal

Watch real-time telemetry in your console while the benchmark runs.

---

## 📖 User Guide

### Mode A: Learning Mode (Manual Training)

For users who want to **experiment one model at a time**.

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MODE A: LEARNING                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   STEP 1: Configure in Sidebar                                                  │
│   ────────────────────────────                                                  │
│   • Select Dataset: MNIST / Fashion / CIFAR-10                                  │
│   • Select Optimizer: SGD / Adam                                                │
│   • Set Epochs: 5-20                                                            │
│                                                                                 │
│   STEP 2: Choose Your Engine                                                    │
│   ──────────────────────────                                                    │
│   • Tab 3 (PyTorch): See the manual training loop                               │
│   • Tab 4 (TensorFlow): See the Keras model.fit() approach                      │
│                                                                                 │
│   STEP 3: Train                                                                 │
│   ─────────────                                                                 │
│   • Click "Start Training"                                                      │
│   • Watch the loss curve descend                                                │
│   • Note the training time                                                      │
│                                                                                 │
│   STEP 4: Deploy (Optional)                                                     │
│   ────────────────────────                                                      │
│   • Go to Tab 5                                                                 │
│   • Upload your trained model to Azure                                          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

---

### Mode B: Power User Mode (Full Benchmark)

For users who want to **test all combinations automatically**.

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MODE B: POWER USER                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   STEP 1: Go to Tab 4 (TensorFlow Lab)                                          │
│   ─────────────────────────────────────                                         │
│   This tab contains the Benchmark Engine                                        │
│                                                                                 │
│   STEP 2: Click "Run Full Benchmark"                                            │
│   ────────────────────────────────────                                          │
│   • 6 model combinations will run automatically                                 │
│   • Watch the progress bar: "33% Complete"                                      │
│   • Monitor terminal for epoch-by-epoch logs                                    │
│                                                                                 │
│   STEP 3: Watch the Leaderboard Build                                           │
│   ─────────────────────────────────────                                         │
│   • Results appear row by row                                                   │
│   • Sorted by accuracy (best first)                                             │
│   • Time and accuracy for each combination                                      │
│                                                                                 │
│   STEP 4: Analyze Results                                                       │
│   ───────────────────────                                                       │
│   • Compare: Which optimizer wins?                                              │
│   • Compare: Which dataset is hardest?                                          │
│   • Compare: Time vs accuracy trade-off                                         │
│                                                                                 │
│   EXPECTED OUTPUT:                                                              │
│   ────────────────                                                              │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │   Rank │ Dataset    │ Optimizer │ Time (s) │ Accuracy │                 │   │
│   │   ─────┼────────────┼───────────┼──────────┼──────────┤                 │   │
│   │    1   │ MNIST      │ Adam      │   12.3   │  97.8%   │ 🏆              │   │
│   │    2   │ MNIST      │ SGD       │   11.9   │  96.2%   │                 │   │
│   │    3   │ Fashion    │ Adam      │   14.1   │  89.4%   │                 │   │
│   │    4   │ Fashion    │ SGD       │   13.8   │  87.1%   │                 │   │
│   │    5   │ CIFAR-10   │ Adam      │   48.2   │  72.3%   │                 │   │
│   │    6   │ CIFAR-10   │ SGD       │   47.5   │  68.9%   │                 │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

---

## ⚠️ Restrictions and Limitations

| Limitation | Description | Impact |
|:-----------|:------------|:-------|
| **Benchmark Duration** | Full benchmark takes **~3-5 minutes** on CPU | Be patient during automated runs |
| **CIFAR-10 Download** | Initial download is **~160MB** | First run takes longer |
| **Memory Usage** | 6 sequential model trainings | Session clearing mitigates this |
| **CPU Only** | No GPU acceleration | CNN training is slower |
| **TensorFlow Version** | Requires **TensorFlow 2.x** | Uses Keras API |

### Performance Expectations

| Task | Approximate Time (CPU) |
|:-----|:----------------------:|
| MNIST + SGD | ~12 seconds |
| MNIST + Adam | ~12 seconds |
| Fashion + SGD | ~14 seconds |
| Fashion + Adam | ~14 seconds |
| CIFAR-10 + SGD (CNN) | ~50 seconds |
| CIFAR-10 + Adam (CNN) | ~50 seconds |
| **Full Benchmark** | **~3-5 minutes** |

---

## 📜 Disclaimer

<div align="center">

---

**🎓 EDUCATIONAL USE ONLY**

---

</div>

This is an **educational tool** demonstrating ML benchmarking and observability practices.

- Results may vary based on hardware
- Azure usage may incur costs
- The author is not responsible for cloud charges

---

## 👨‍💻 Author

<div align="center">

### **Waqar Salim**

*Master's Student & IT Professional*

---

[![GitHub](https://img.shields.io/badge/GitHub-WSalim2024-181717?style=for-the-badge&logo=github)](https://github.com/WSalim2024)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/waqar-salim/)

---

**Built with ⚡ Benchmarks, 📊 Telemetry, 🔥 PyTorch, and 🟠 TensorFlow**

*Azure Neural Net Studio v3.4 — The Benchmark Edition*

---

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   "Don't guess which model is best.                                           ║
║    Benchmark them all. Let the data decide."                                  ║
║                                                                               ║
║                        — v3.4: Observability Matters                          ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

</div>
