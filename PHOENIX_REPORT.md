# 🦅 PHOENIX PROTOCOL: System Status & Fix Report
**Date:** 2026-01-06
**Status:** ACTIVE TRAINING (Workspace: `sound`)

## 🚨 The Challenge: The 35° MAE Stall
Our training was hit by a "Dead Gradient" floor where the MAE refused to drop below 35 degrees. This was diagnosed as a combination of three failures:
1. **Coordinate Misalignment:** The label vectors were fighting a static Z-offset in the physics engine.
2. **Trace-Normalization Suicide:** Normalizing by the trace of the covariance matrix was destroying amplitude information that the model needed for distance/depth estimation.
3. **Mamba Stiffening:** Initial weights for the SSM discretization ($\Delta$) were too low, effectively blocking learning.

## 🛠️ The Fixes (Implemented)
- **[FIX 1] Coordinate Sanity:** Labels are now pure relative bearing vectors, decoupled from absolute world height.
- **[FIX 2] Robust Scaling:** Replaced trace-normalization with static `1/100` scaling. This preserves "energy" patterns across frequencies.
- **[FIX 3] Bi-Mamba (Bidirectional):** Implemented forward and backward frequency scans. Since audio spectrum is not "causal," this allows the model to capture non-local spectral dependencies (Issue #10).
- **[FIX 4] Nursery Curriculum:** Softened the starting phase to 0.5m - 3.0m distance with no physics (ground reflection disabled) to guarantee an early "win" for the model.
- **[FIX 5] Epsilon-Loss:** Stabilized `AngularLoss` with an epsilon term and a norm-penalty to prevent predictions from collapsing to zero.

## 📈 Current Progress
Training has been migrated to the fresh `sound` workspace due to billing limits on `shleiby` and `sounds`. 
- **Current Run:** `ap-WTOZPRzobw6jttyu0DK6tu`
- **Latest MAE:** ~69° (Starting from scratch with actual learning, not the false 35° floor).
- **Target:** Break below 10° within 1-2 hours of Nursery training.

## 🕒 Why the Startup Delay?
It currently takes **~3 minutes** to start a session because:
1. **Provisioning:** Modal takes ~60s to pull the image and assign an A100.
2. **Data Unpacking:** Every worker must unpack the 165MB `oren_data.tar` from the Volume to local NVMe for fast random access.
3. **Physics Warmup:** The first batch requires a warm-up of the JIT-compiled physics engine.

## 🚀 Repository Sync
This repository now contains the absolute latest state of the production pipeline, including the `consensus_bridge.py` hive-mind orchestrator.
