# MambaAI: High-Precision Sim-to-Real DOA Estimation

## Overview
**MambaAI** is a next-generation acoustic Direction-of-Arrival (DOA) system designed for sub-degree accuracy in complex outdoor environments (e.g., drone tracking). 

Unlike traditional methods that rely on fixed datasets or pure signal processing (MUSIC/ESPRIT), MambaAI utilizes a **Hybrid Neuro-Symbolic** approach:
1.  **Physics-Based Simulation:** A custom acoustic propagation engine (**PySKIacoustics**) generates infinite, physically accurate training data on-the-fly.
2.  **State-Space Modeling:** A **Mamba (S4)** backbone processes frequency-domain covariance sequences to capture long-range temporal dependencies in the signal.
3.  **Sim-to-Real Transfer:** A robust curriculum learning strategy and "Hive Mind" distributed training ensure the model generalizes from simulation to real-world deployment.

---

## 🏗 Architecture

### 1. The Model: `MambaDOANet`
The core model replaces traditional CNNs/RNNs with a **Mamba State-Space Model**.
*   **Input:** Covariance Matrix Sequence (calculated via STFT).
*   **Backbone:** Mamba layers unfold the signal in the angular spectrum domain, treating DOA estimation as a sequence modeling problem over frequency bins.
*   **Output:** 3D Unit Vector $(x, y, z)$ pointing to the source.
*   **Loss Function:** **Angular Loss** ($1 - \text{CosineSimilarity}$), directly optimizing for directional precision rather than Euclidean distance.

### 2. The Engine: `PySKIacoustics`
A custom physics engine that simulates outdoor acoustic propagation:
*   **ISO 9613-1** Atmospheric Absorption (Temperature, Humidity).
*   **Doppler Shift** for moving sources (Drones).
*   **Ground Impedance** (Delany-Bazley model) & Reflection (Ground Dip).
*   **Sensor Imperfections:** Gain mismatch, thermal noise, and position jitter.

---

## 🧠 Training Strategy

### Adaptive Curriculum Learning ("Self-Paced")
Instead of training on hard data immediately, the model follows a progressive difficulty schedule based on its own performance (Validation Loss).

| Phase | Name | Conditions | Trigger (Loss) |
| :--- | :--- | :--- | :--- |
| **1** | **"The Nursery"** | Close range (5-30m), No Wind, High Elevation. | Start |
| **2** | **"The Playground"** | Medium range (60m), Mild Wind (2m/s). | < 0.15 |
| **3** | **"The Wild"** | Full Range (200m+), Heavy Wind (8m/s), Ground Reflections. | < 0.10 |

### "Hive Mind" Distributed Consensus
To overcome the limits of single-GPU training, MambaAI runs on **Modal** using a custom distributed architecture:
*   **Quad-Parallel Launch:** Training runs simultaneously on 4+ isolated workspaces (e.g., `sounds`, `sounds2`), each with an NVIDIA A100.
*   **Consensus Bridge:** A local orchestrator (`consensus_bridge.py`) monitors all running jobs.
*   **Hot-Swapping:** If one "Worker" discovers a better model (lower MAE), the Bridge extracts its weights and **hot-swaps** them into all other workers in real-time. This prevents divergence and ensures the entire swarm converges on the global optimum.

---

## 🚀 Usage

### 1. Installation
The stack is designed to run on **Modal** (Cloud) with a local orchestrator.
```bash
pip install modal torch numpy
# Ensure you have credentials for multiple Modal profiles if using Quad-Launch
```

### 2. Launching the Swarm
To start the distributed training fleet (4x A100s + 16 CPU Data Generators):
```bash
# Launches detached jobs on 'sounds', 'sounds2', 'sounds3', 'sounds4'
sh launch_all_parallel.sh
```

### 3. Activating the Hive Mind
Once the jobs are running, start the Consensus Bridge in a separate terminal to enable weight syncing:
```bash
python3 consensus_bridge.py
```

### 4. Monitoring
*   **Logs:** Stream checkpoints and MAE metrics via `modal volume get` or the Modal Dashboard.
*   **Manifest:** A live `manifest.csv` tracks every single generated training sample for full data traceability.

---

## 📂 File Structure
*   `train_sim_gcp_native.py`: **Core Training Script.** Contains the Mamba model, Training Loop, and Curriculum Logic.
*   `mamba_unfolding_net.py`: **Model Definition.** The Mamba architecture.
*   `pyskiacoustics.py`: **Physics Engine.** Handles atmospheric/ground simulation.
*   `consensus_bridge.py`: **Orchestrator.** Syncs weights between isolated Modal workspaces.
*   `launch_all_parallel.sh`: **Launcher.** Bash script to fire all Modal jobs.

---
*Maintained by Ziz-Defense*
