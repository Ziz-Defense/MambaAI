# MambaAI - High-Precision DOA Training Pipeline

This repository contains the full stack for training Mamba-based Direction-of-Arrival (DOA) models using synthetic and real-world acoustic data.

## Key Features
- **Streaming Synthetic Data Generation:** Uses parallel CPU workers to generate physics simulations on-the-fly, eliminating data-loading bottlenecks.
- **Mamba Unfolding Network:** High-precision hybrid architecture for acoustic localization.
- **Quad-Parallel Modal Launch:** Orchestration scripts to run multiple A100 training jobs concurrently across workspaces.
- **Live Manifest Tracking:** Real-time logging of generated shards for dataset traceability.

## File Guide
- `train_sim_gcp_native.py`: Core training logic with `LiveSyntheticDataset`.
- `mamba_unfolding_net.py`: Mamba model architecture.
- `pyskiacoustics.py`: Physics simulation engine.
- `train_sim_modal_final.py`: Modal Entrypoint for A100 training.
- `launch_all_parallel.sh`: Bulk orchestrator for multi-workspace training.

## How to Run
1. Configure your `modal.toml` profiles.
2. Run `sh launch_all_parallel.sh` to start parallel training jobs.
3. Monitor logs and checkpoints in your Modal volume.

---
*Maintained by Ziz-Defense*
