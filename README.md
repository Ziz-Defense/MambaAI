# 🐍 MambaAI: Neuro-Symbolic Acoustic DOA Estimation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: Modal](https://img.shields.io/badge/Framework-Modal-green.svg)](https://modal.com)
[![Model: Mamba](https://img.shields.io/badge/Backbone-Mamba_S4-blue.svg)](https://github.com/state-spaces/mamba)
[![Status: Active](https://img.shields.io/badge/Status-Training-orange.svg)]()

**MambaAI** is a groundbreaking Direction-of-Arrival (DOA) estimation system designed for **sub-degree precision** in tracking airborne acoustic sources (e.g., drones). It bridges the gap between classical signal processing and modern deep learning by combining a **Physics-Based Simulation Engine** with a **State-Space Model (SSM)** backbone.

---

## 📑 Table of Contents
- [Abstract](#-abstract)
- [Scientific Foundation](#-scientific-foundation)
    - [The Mamba-Unfolding Architecture](#the-mamba-unfolding-architecture)
    - [Why Not CNNs or Transformers?](#why-not-cnns-or-transformers)
- [The Physics Engine: PySKIacoustics](#-the-physics-engine-pyskiacoustics)
- [Training Methodology](#-training-methodology)
    - [Adaptive Curriculum Learning](#adaptive-curriculum-learning)
    - [The "Hive Mind" Consensus](#the-hive-mind-consensus)
- [Installation & Usage](#-installation--usage)
- [Configuration](#-configuration)
- [Roadmap](#-roadmap)

---

## 🎯 Abstract
Traditional DOA estimation methods like MUSIC and ESPRIT are mathematically elegant but brittle in real-world conditions (reverberation, wind noise, non-stationary sources). Deep Learning approaches (CNNs/CRNNs) are robust but require massive labeled datasets which are impossible to collect at scale for 3D aerial tracking.

**MambaAI solves this via two key innovations:**
1.  **Infinite Synthetic Data:** We do not collect data; we grow it. Our `PySKIacoustics` engine generates physically accurate training samples on-the-fly, modeling atmospheric absorption, Doppler shifts, and ground impedance.
2.  **Mamba Frequency Unfolding:** We treat the DOA problem as a sequence modeling task across the frequency domain. By feeding the covariance matrix sequence into a Mamba S4 backbone, we capture long-range dependencies in the phase information that CNNs miss.

---

## 🔬 Scientific Foundation

### The Mamba-Unfolding Architecture
The core model is defined in `mamba_unfolding_net.py`. It operates on the **Short-Time Fourier Transform (STFT) Covariance Matrix**.

1.  **Input Representation:**
    Given a multichannel signal $X \in \mathbb{C}^{M \times T}$, we compute the narrowband covariance matrix $R(f)$ for each frequency bin $f$.
    $$R(f) = \frac{1}{T} \sum_{t} X(f,t) X(f,t)^H$$
    This results in a sequence of matrices $R \in \mathbb{C}^{F \times M \times M}$, where $F$ is the number of frequency bins (seq_len) and $M$ is the number of microphones.

2.  **Mamba Backbone:**
    Instead of 2D convolutions over the spectrogram, we flatten the covariance matrices and feed them as a sequence into a **Mamba Block**.
    *   **Selective State Spaces:** Mamba allows the model to selectively propagate relevant phase information across the entire frequency spectrum, effectively performing "beamforming" in the latent space.
    *   **Linear Complexity:** Unlike Transformers ($O(N^2)$), Mamba scales linearly ($O(N)$), allowing us to process high-resolution spectral data (48kHz audio) efficiently.

3.  **Angular Output Head:**
    The network outputs a 720-point "Angular Spectrum" (similar to MUSIC's pseudospectrum), which is then regressed to a 3D unit vector $(x, y, z)$. We optimize using **Angular Loss** ($1 - \text{CosineSimilarity}$) to strictly penalize directional error.

### Why Not CNNs or Transformers?
*   **CNNs** are translation invariant, which is bad for frequency data where pitch matters.
*   **Transformers** are too heavy for high-resolution audio sequences.
*   **Mamba** offers the perfect trade-off: global receptive field (like attention) with the speed of a CNN.

---

## 🌪 The Physics Engine: PySKIacoustics
Located in `pyskiacoustics.py`, this engine is the heart of our Sim-to-Real strategy. It implements:

*   **ISO 9613-1 Atmospheric Absorption:**
    Calculates signal attenuation based on Temperature (20°C), Humidity (50%), and Pressure. High frequencies degrade faster over distance, a crucial cue for depth estimation.
*   **Delany-Bazley Ground Impedance:**
    Models the ground not as a hard surface, but as a porous medium (flow resistivity $\sigma = 200 kPas/m^2$ for grass). This creates accurate "Ground Dip" interference patterns.
*   **Doppler Shift:**
    Simulates the frequency compression/expansion caused by a drone moving at velocity $v$ relative to the array.
    $$f_{obs} = f_{src} \left( \frac{c}{c - v_{radial}} \right)$$

---

## 🥋 Training Methodology

### Adaptive Curriculum Learning
We do not train on random data. The model must "earn" complexity. The training loop monitors the Validation Loss in real-time and advances the difficulty tier automatically.

| Phase | Name | Distance | Wind | Elevation | Loss Threshold |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **I** | **The Nursery** | 5-30m | 0 m/s | 30°-80° | Start |
| **II** | **The Playground** | 10-60m | 0-2 m/s | 20°-80° | < 0.15 |
| **III** | **"The Wild"** | 5-200m | 0-8 m/s | 0°-90° | < 0.10 |

### The "Hive Mind" Consensus
Training is distributed across **4x NVIDIA A100 GPUs** hosted on different Modal workspaces. To prevent model divergence, we implement a **Consensus Mechanism**:

1.  **Workers** train independently on their own generated data streams.
2.  **Coordinator** (`consensus_bridge.py`) polls all workers every 5 minutes.
3.  **Sync:** If Worker A finds a model with significantly lower MAE than the others, the Coordinator extracts its weights and **hot-swaps** them into Workers B, C, and D.
4.  **Result:** The entire swarm effectively trains as one giant batch, converging on the global optimum faster.

---

## 💻 Installation & Usage

### Prerequisites
*   Python 3.10+
*   [`modal`](https://modal.com) CLI installed and authenticated.
*   NVIDIA GPU (optional for inference, required for training).

### 1. Launch Distributed Training
This command spins up the entire fleet (4 GPUS + 16 CPU Generator Nodes):
```bash
sh launch_all_parallel.sh
```

### 2. Start the Hive Mind
In a separate terminal, launch the bridge to enable cross-workspace weight synchronization:
```bash
python3 consensus_bridge.py
```

### 3. Monitoring
Use the Modal Dashboard to view live logs. You will see metrics like:
`Ep 12 (Playground) Loss: 0.0842 | MAE: 6.4°`

---

## 🛠 Configuration
Key parameters are defined in `train_sim_gcp_native.py` and can be overridden via environment variables:

*   `EPOCHS_SIM`: Total simulation epochs (Default: 20).
*   `BATCH_SIZE`: Training batch size per GPU (Default: 64).
*   `N_SAMPLES`: Samples generated per epoch (Default: 300,000).

---

## 🗺 Roadmap
- [x] **Phase 1:** Mamba Backbone & Physics Engine.
- [x] **Phase 2:** Distributed "Hive Mind" Training.
- [ ] **Phase 3:** Real-Time Inference on Edge Devices (Jetson Orin).
- [ ] **Phase 4:** Multi-Drone Tracking (Source Separation).

---

**Maintained by Ziz-Defense**  
*Building the future of airspace security.*
