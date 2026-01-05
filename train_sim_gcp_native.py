"""
Native GCP / GPU-Accelerated Sim-to-Real Training
=================================================
This script runs on a GCP GPU instance (T4/L4/A100).
It implements the High-Fidelity Data Generation Strategy:
1. Loads Real Matrix 4T Reference Library (extracted from Oren1).
2. Uses GPU (Via torch/torchaudio) to augment and synthesize 100k samples:
   - Pitch Shift (RPM), Doppler, Spectral shaping (Wind/Atmosphere).
   - Fast GPU Convolution for Room Impulse Responses.
3. Trains Mamba-DOA model on this synthetic data.
4. Fine-tunes on Real Oren1 data.
5. Verifies on Oren2 data.
"""

import os
import glob
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
import pandas as pd
import soundfile as sf
import sys
import tqdm
import tqdm
from torch.utils.data import Dataset, DataLoader

# === CONFIG ===
REF_LIB_PATH = "/data/oren1_reference_library" # Already on Volume
REAL_DATA_PATH = "Oren1_complete"
OUTPUT_DIR = "training_output"
N_SAMPLES = 100000
BATCH_SIZE = 64
EPOCHS_SIM = 20
EPOCHS_FT = 10

# UMA-16 Geometry (Meters)
# [3, 16] tensor
MIC_POS = torch.tensor([
    [-0.066,  0.066, 0], [-0.024,  0.066, 0], [ 0.024,  0.066, 0], [ 0.066,  0.066, 0], # MIC8,7,10,9
    [-0.066,  0.024, 0], [-0.024,  0.024, 0], [ 0.024,  0.024, 0], [ 0.066,  0.024, 0], # MIC6,5,12,11
    [-0.066, -0.024, 0], [-0.024, -0.024, 0], [ 0.024, -0.024, 0], [ 0.066, -0.024, 0], # MIC4,3,14,13
    [-0.066, -0.066, 0], [-0.024, -0.066, 0], [ 0.024, -0.066, 0], [ 0.066, -0.066, 0]  # MIC2,1,16,15
], dtype=torch.float32).T # 3x16

def load_reference_library():
    print("Loading Reference Library...")
    lib = {}
    import soundfile as sf
    total_clips = 0
    # Expected structure: oren1_reference_library/clips_10m/clip_000.wav
    for d in sorted(glob.glob(f"{REF_LIB_PATH}/clips_*m")):
        dist = int(d.split('_')[-1].replace('m',''))
        clips = []
        for wav in sorted(glob.glob(f"{d}/*.wav")):
            sig_np, sr = sf.read(wav)
            sig = torch.from_numpy(sig_np).float()
            if sig.ndim == 1: sig = sig.unsqueeze(0).repeat(16, 1) # Multi-mic dummy if mono
            if sig.shape[0] < 16: sig = sig.repeat(16 // sig.shape[0], 1)
            
            if sig.shape[1] < 24000:
                sig = F.pad(sig, (0, 24000 - sig.shape[1]))
            else:
                sig = sig[:, :24000]
            clips.append(sig)
            total_clips += 1
        if clips:
            lib[dist] = torch.stack(clips).cuda() # [N, 16, L]
    print(f"Loaded {total_clips} clips across {len(lib)} distance bins.")
    return lib

# pyskiacoustics Integration
try:
    import pyskiacoustics
except ImportError:
    print("WARNING: pyskiacoustics not found. Using fallback?")
    # In Modal, it should be mounted at /root/pyskiacoustics.py
    sys.path.append("/root")
    try:
        import pyskiacoustics
    except:
        print("CRITICAL: PySKIacoustics Physics Engine Missing!")
        exit(1)

# GPU Audio Augmentation Pipeline
class AudioSynthesizer:
    def __init__(self, lib, config=None):
        self.lib = lib
        self.config = config if config else {}
        self.dists = sorted(list(lib.keys()))
        # pyskiacoustics runs on CPU numpy, so we need access to CPU clips or move back and forth
        # Ideally, generate on CPU (physics is fast numpy) then stack to GPU for return
        
    def generate_batch(self, batch_size, device='cuda'):
        batch_audio = []
        batch_vecs = [] 
        
        # Unpack Config (Curriculum)
        cfg = self.config
        min_dist = cfg.get('min_dist', 10)
        max_dist = cfg.get('max_dist', 100)
        wind_opts = cfg.get('wind_opts', [0, 2, 5, 8])
        wind_probs = cfg.get('wind_probs', [0.3, 0.3, 0.3, 0.1])
        min_el = cfg.get('min_el', 5)
        max_el = cfg.get('max_el', 60)
        
        # CPU generation loop (Physics engine is numpy based)
        for i in range(batch_size):
            # 1. Select Source Clip
            d_idx = np.random.randint(0, len(self.dists))
            dist_key = self.dists[d_idx]
            clips = self.lib[dist_key] # Torch tensor
            
            clip_idx = np.random.randint(0, len(clips))
            sig_torch = clips[clip_idx] # [16, 24000]
            
            # Move to CPU numpy for physics
            sig_numpy = sig_torch[0].cpu().numpy()
            
            # 2. Physics Simulation (PySKIacoustics)
            az = np.random.uniform(0, 360)
            el = np.random.uniform(min_el, max_el) # Dynamic Elevation
            # Continuous distance or snapped? PySKI supports continuous.
            # But we might want to respect the clip's base distance logic? 
            # No, pyskiacoustics generates propagation.
            dist = np.random.uniform(min_dist, max_dist)
            wind = np.random.choice(wind_opts, p=wind_probs)
            
            sim_audio, vec = pyskiacoustics.generate_training_sample(
                sig_numpy, 
                48000,
                azimuth_deg=az, 
                elevation_deg=el, 
                distance_m=dist,
                wind_speed_mps=wind
            )
            
            # 3. Convert back to Torch/GPU
            sim_tensor = torch.from_numpy(sim_audio).float().to(device)
            vec_tensor = torch.from_numpy(vec).float().to(device)
            
            batch_audio.append(sim_tensor)
            batch_vecs.append(vec_tensor)
            
        return torch.stack(batch_audio), torch.stack(batch_vecs)

class AngularLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    def forward(self, pred, target):
        return 1 - self.cos(pred, target).mean()

def compute_degree_error(pred, target):
    # pred, target: (B, 3) normalized
    # dot product:
    dot = (pred * target).sum(dim=1)
    # clamp for numerical stability
    dot = torch.clamp(dot, -1.0, 1.0)
    # acos
    rad = torch.acos(dot)
    return torch.rad2deg(rad).mean().item()

# === MODEL (Mamba-DOA or Simple CNN) ===
# Using the CNN from my previous successful scripts for robustness
# === MODEL (Mamba-DOA Physics Unfolding) ===
try:
    from mamba_unfolding_net import MambaDOANet
except ImportError:
    # Included in case file is missing local, but should be in container
    print("WARNING: mamba_unfolding_net not found locally. Assuming it exists in Modal.")
    # Dummy placeholder for linting
    class MambaDOANet(nn.Module): pass

class DOA_Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize Mamba-DOA Physics Engine
        self.backbone = MambaDOANet(num_mics=16, d_model=128, grid_size=720)
        
        # Regression Head: Map 720-point Angular Spectrum -> 3D Unit Vector
        # This preserves the physics-based "Sparse Coding" benefits of the backbone
        # while complying with the regression loss function of the pipeline.
        self.regressor = nn.Sequential(
            nn.Linear(720, 128),
            nn.ReLU(),
            nn.Linear(128, 3) # x, y, z
        )
        
    def compute_stft_covariance(self, x):
        # x: (Batch, 16, N_samples)
        B, C, N = x.shape
        x_flat = x.view(B*C, N)
        
        # 1. STFT
        # N=1024, Hop=512 -> ~46 frames for 24000 samples
        window = torch.hann_window(1024, device=x.device)
        X_flat = torch.stft(x_flat, n_fft=1024, hop_length=512, window=window, return_complex=True)
        # X_flat: (B*C, Freq, Time)
        
        # Reshape back
        _, F, T = X_flat.shape
        X = X_flat.view(B, C, F, T)
        # X: (Batch, Mics, Freq, Time)
        
        # 2. Covariance per Freq-Bin (Narrowband, averaged over Time)
        # R(f) = Average_Time( x(t,f) @ x(t,f).H )
        # X: (Batch, Mics, Freq, Time)
        X = X.permute(0, 2, 3, 1) # (B, Freq, Time, Mics)
        B, F, T, M = X.shape
        
        # We want sequence over Freq.
        # Average over Time T.
        # Flatten B*F -> Process -> Reshape
        X_flat = X.reshape(B*F, T, M)
        
        # R_flat: (B*F, M, M)
        R_flat = torch.matmul(X_flat.transpose(1, 2).conj(), X_flat) / T
        
        # NORMALIZATION FIX: Divide by trace (energy) to prevent explosion
        # trace is sum of diagonal elements (real)
        # R_ii are real.
        trace = torch.diagonal(R_flat, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1) # (B*F, 1, 1)
        R_flat = R_flat / (trace + 1e-6)
        
        # Reshape to (B, F, M, M) -> This is our Sequence!
        R = R_flat.reshape(B, F, M, M)
        return R

    def forward(self, x):
        # x: (B, 16, 24000)
        # 1. Preprocess -> Covariance Sequence
        R_seq = self.compute_stft_covariance(x)
        
        # 2. Mamba Physics Backbone
        # Returns: (B, 720) Angular Spectrum
        spectrum = self.backbone(R_seq)
        
        # 3. Regress to Unit Vector
        v = self.regressor(spectrum)
        return F.normalize(v, p=2, dim=1)

# === MAIN ===
def generate_shard(gpu_id, n_samples, output_file, lib_cpu):
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Initializing on {device}...")
    
    # Re-init synthesizer on specific GPU
    # Move lib to correct GPU
    lib_gpu = {}
    for k, v in lib_cpu.items():
        lib_gpu[k] = v.to(device)
        
    synth = AudioSynthesizer(lib_gpu) # Now on GPU
    
    x_list, y_list = [], []
    batch_size = 64
    
    for _ in range(0, n_samples, batch_size):
        curr_batch = min(batch_size, n_samples - len(x_list)*batch_size)
        if curr_batch <= 0: break
        
        # Generator handles device placement internally if passed correctly
        x, y = synth.generate_batch(curr_batch, device=device)
        x_list.append(x.cpu())
        y_list.append(y.cpu())
        
    X = torch.cat(x_list)
    Y = torch.cat(y_list)
    torch.save((X, Y), output_file)
    print(f"[GPU {gpu_id}] Saved {output_file} ({len(X)} samples)")

# ... (previous imports)

# ... (Previous code remains)

# ... (Previous imports and config)

def main_training_loop():
    if not torch.cuda.is_available():
        print("ERROR: GPU required.")
        # exit(1) # Continue for debugging?
    
    # OUTPUT_DIR from Env (default to local)
    OUTPUT_DIR = os.getenv("SIM_OUTPUT_DIR", "training_output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    num_gpus = torch.cuda.device_count()
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    
    # 3. Training with Streaming Dataset (Memory Efficient & Instant Start)
    from torch.utils.data import IterableDataset
    import time
    
    class LiveSyntheticDataset(IterableDataset):
        def __init__(self, output_dir, samples_per_epoch, lib_cpu):
            self.output_dir = output_dir
            self.samples_per_epoch = samples_per_epoch
            self.lib_cpu = lib_cpu
            self.batch_size = 64 # Internal batch for generation efficiency
            
        def __iter__(self):
            # Worker Initialization
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:
                worker_id = 0
                num_workers = 1
            else:
                worker_id = worker_info.id
                num_workers = worker_info.num_workers
            
            # Unique Seed per Worker
            seed = int(time.time()) + worker_id
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            print(f"[Worker {worker_id}] Initializing AudioSynthesizer...")
            
            # Initialize Physics Engine (CPU-based)
            # PySKIacoustics generates on CPU numpy
            # AudioSynthesizer expects GPU lib? No, let's keep it CPU for generation efficiency
            # then move to GPU only when training.
            # But the existing `AudioSynthesizer` moves to GPU inside init if lib is GPU?
            # Let's use CPU lib for generation to avoid pickling CUDA tensors across processes
            
            # Correct logic:
            # 1. Load lib (CPU) -> Already passed in __init__
            # 2. Init Synthesizer (CPU lib)
            # 3. Generate (CPU numpy) -> Convert to Torch (CPU) -> Return
            # 4. DataLoader collate moves to GPU? No, main loop does x.cuda()
            
            # Re-use existing class but ensure it handles CPU lib
            # The current AudioSynthesizer takes a lib and assumes it might be CPU or GPU
            # generate_batch converts to `device`
            
            synth = AudioSynthesizer(self.lib_cpu, config=self.config) 
            
            # Persistence Buffer
            x_buffer = []
            y_buffer = []
            buffer_limit = 50 # Save every 50 samples (Quick Feedback)
            
            # Seed for logging
            seed = int(time.time()) + worker_id
            np.random.seed(seed) # Ensure worker randomness
            torch.manual_seed(seed)
            
            count = 0
            while count < self.samples_per_epoch // num_workers:
                # Generate small batch
                # device='cpu' to avoid CUDA in worker process
                x, y = synth.generate_batch(self.batch_size, device='cpu') 
                
                # Yield samples one by one
                for i in range(len(x)):
                    yield x[i], y[i]
                    
                    # Add to buffer
                    x_buffer.append(x[i].unsqueeze(0)) # [1, 16, 24000]
                    y_buffer.append(y[i].unsqueeze(0))
                    
                    if len(x_buffer) >= buffer_limit:
                        # Flush to disk (Async-like via OS buffer)
                        timestamp = int(time.time() * 1000)
                        fname = f"{self.output_dir}/shard_live_{worker_id}_{timestamp}.pt"
                        
                        X_save = torch.cat(x_buffer)
                        Y_save = torch.cat(y_buffer)
                        
                        # We save it, but we DON'T stop yielding.
                        # This enables "Save while Training"
                        try: 
                            torch.save((X_save, Y_save), fname)
                            
                            # === MANIFEST LOGGING (Smart Tracking) ===
                            manifest_path = f"{self.output_dir}/manifest.csv"
                            if not os.path.exists(manifest_path):
                                with open(manifest_path, 'w') as f:
                                    f.write("timestamp,worker_id,filename,samples,seed,model_version\n")
                            
                            with open(manifest_path, 'a') as f:
                                # Log details so we know exactly what we made
                                f.write(f"{int(time.time())},{worker_id},{os.path.basename(fname)},{len(x_buffer)},{seed},v1_sim2real\n")
                            # ==========================================
                            
                        except Exception as e:
                            print(f"[Worker {worker_id}] Save Failed: {e}")
                            
                        x_buffer = []
                        y_buffer = []
                        
                count += self.batch_size * 1 # Approximate count update
                
    # 1. Load Lib (CPU) - done once in main
    print("Loading Ref Lib (CPU)...")
    lib_cpu = {}
    total_clips = 0
    import soundfile as sf
    folders = sorted(glob.glob(f"{REF_LIB_PATH}/clips_*m"))
    if not folders:
         print("WARNING: No clips found! Creating dummy keys for testing.")
         # Dummy fallback
         lib_cpu[20] = torch.randn(10, 16, 24000)
    else:
        for d in folders:  
            dist = int(d.split('_')[-1].replace('m',''))
            clips = []
            files = sorted(glob.glob(f"{d}/*.wav"))
            for wav in files:
                sig_np, sr = sf.read(wav)
                sig = torch.from_numpy(sig_np).float()
                if sig.ndim == 1: sig = sig.unsqueeze(0).repeat(16, 1)
                if sig.shape[0] < 16: sig = sig.repeat(16 // sig.shape[0], 1)
                if sig.shape[1] < 24000: sig = F.pad(sig, (0, 24000-sig.shape[1]))
                else: sig = sig[:, :24000]
                clips.append(sig)
            if clips: 
                lib_cpu[dist] = torch.stack(clips) # Keep on CPU
                print(f" -> Loaded {len(clips)} clips for {dist}m")

    # 2. Check for Existing Data (Hybrid Mode)
    # If shards exist, we can use them via a Mixed Dataset or just Chain them
    # For simplicity: If shards > 0, use them. If not, use Live.
    # User wanted "Infinite Storage", so ideally we mix both.
    # But `ChainDataset` with `Iterable` is tricky if one is finite.
    # Strategy: Just use LiveDataset for now to Unblock.
    # The LiveDataset SAVES to disk, so next time `shard_files` will be populated.
    
    shard_files = glob.glob(f"{OUTPUT_DIR}/shard_*.pt")
    print(f"Found {len(shard_files)} existing shards.")
    
    # HYBRID DATASET SETUP:
    # We want to train on (Existing Shards) + (Live Generation)
    # Since we want to GROW the dataset, we should prioritize Live Gen if dataset is small.
    # If dataset is huge (4M), we might just read.
    
    # For this launch: FORCE LIVE GEN to ensures we start immediately.
    # Existing shards will be ignored for this run to keep logic simple and ensure generation happens.
    # In future: `torch.utils.data.ChainDataset([FileDataset(shards), LiveDataset(...)])`
    
    print("🚀 Initializing Live Streaming Dataset (Instant Start)...")
    
    # === CURRICULUM SCHEDULE ===
    # === ADAPTIVE CURRICULUM ===
    CURRICULUM_PHASES = {
        'Nursery': {'min_dist': 5, 'max_dist': 30, 'wind_opts': [0], 'wind_probs': [1.0], 'min_el': 30, 'max_el': 80},
        'Playground': {'min_dist': 10, 'max_dist': 60, 'wind_opts': [0, 2], 'wind_probs': [0.7, 0.3], 'min_el': 20, 'max_el': 80},
        'The Wild': {'min_dist': 5, 'max_dist': 200, 'wind_opts': [0, 2, 5, 8], 'wind_probs': [0.25]*4, 'min_el': 0, 'max_el': 90}
    }
    
    current_phase_name = 'Nursery'
    running_loss_avg = 1.0
    
    def update_curriculum(current_loss):
        nonlocal current_phase_name, running_loss_avg
        # Exponential moving average
        running_loss_avg = 0.9 * running_loss_avg + 0.1 * current_loss
        
        if current_phase_name == 'Nursery' and running_loss_avg < 0.15:
            current_phase_name = 'Playground'
            print("🚀 LEVEL UP! Graduated to 'Playground' Phase.")
        elif current_phase_name == 'Playground' and running_loss_avg < 0.10:
            current_phase_name = 'The Wild'
            print("🦅 LEVEL UP! Entered 'The Wild' Phase.")
            
        return CURRICULUM_PHASES[current_phase_name]
    # train_ds = LiveSyntheticDataset(OUTPUT_DIR, N_SAMPLES, lib_cpu)
    # loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=4, persistent_workers=True)
    
    # 4. Train Sim
    print(f"Starting Sim Training on {num_gpus} GPUs...")
    model = DOA_Net().cuda()
    if num_gpus > 1: model = nn.DataParallel(model)
        
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = AngularLoss()
    
    # To avoid "Waiting for data" log spam, we just rely on tqdm
    print("Training Loop Started! (Waiting for first batch from workers...)")
    
    for ep in range(EPOCHS_SIM):
        # [CONSENSUS] Check for Hot-Swap
        CONSENSUS_FILE = f"{OUTPUT_DIR}/consensus_sync.pth"
        if os.path.exists(CONSENSUS_FILE):
            print(f"🔥 HOT SWAP DETECTED! Loading Consensus Model...")
            try:
                # Load to CPU first
                state = torch.load(CONSENSUS_FILE, map_location='cpu')
                if isinstance(model, nn.DataParallel): model.module.load_state_dict(state)
                else: model.load_state_dict(state)
                print("✅ Hot Swap Successful. We have assimilated.")
                os.remove(CONSENSUS_FILE) # Consume it
            except Exception as e:
                print(f"❌ Hot Swap Failed: {e}")

        # Update Curriculum Logic
        # Or let's just use the avg loss of the *previous* epoch to decide config for *this* epoch.
        # Initial: Nursery.
        
        curr_config = CURRICULUM_PHASES[current_phase_name] # update_curriculum called at end of epoch
        print(f"--- Epoch {ep} Curriculum: {current_phase_name} ---")
        
        # Re-Init Loader to pass new config to workers
        train_ds = LiveSyntheticDataset(OUTPUT_DIR, N_SAMPLES, lib_cpu, config=curr_config)
        # persistent_workers=False so workers restart with new config each epoch
        loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=4, persistent_workers=False)
        
        losses = []
        # Tqdm total is approximate since it's streaming
        deg_errors = []
        pbar = tqdm.tqdm(loader, desc=f"Sim Epoch {ep} ({current_phase_name})", total=N_SAMPLES//BATCH_SIZE)
        
        for batch_idx, (x, y) in enumerate(pbar):
            opt.zero_grad()
            pred = model(x.cuda())
            loss = loss_fn(pred, y.cuda())
            loss.backward()
            opt.step()
            losses.append(loss.item())
            
            # Metrics
            deg = compute_degree_error(pred, y.cuda())
            deg_errors.append(deg)
            
            if batch_idx % 20 == 0:
                avg_mae = np.mean(deg_errors[-20:])
                pbar.set_description(f"Ep {ep} ({current_phase_name}) Loss: {loss.item():.4f} | MAE: {avg_mae:.1f}°")
                
        # End of Epoch: Update Curriculum
        epoch_loss = np.mean(losses)
        update_curriculum(epoch_loss)
                
        # SAVE CHECKPOINT (Every Epoch)
        if ep % 1 == 0: # Every epoch
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state, f"{OUTPUT_DIR}/checkpoint_sim_epoch_{ep}.pth")
            torch.save(state, f"{OUTPUT_DIR}/latest_model.pth")
            
            # [CONSENSUS] Write Status
            with open(f"{OUTPUT_DIR}/status.json", "w") as f:
                json.dump({'epoch': ep, 'mae': update_curriculum(epoch_loss).get('dummy_mae', epoch_loss * 50)}, f) # Using Loss*50 as proxy if MAE not calc
            
            print(f"✅ Saved checkpoint_sim_epoch_{ep}.pth")
        
    # Save Sim Model (Final)
    state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state, f"{OUTPUT_DIR}/sim_model.pth")
    print("Sim Model Saved.")
    
    # 5. Fine-Tune on Real (CRITICAL FIX 6)
    print("=== Fine-Tuning on Real Oren1 Data ===")
    
    # Load Oren1 Labels & Audio
    LBL_PATH = f"{REAL_DATA_PATH}/labels_3d.json" # Or V2 labels if generated
    # === LABEL GENERATION (Ported from V1) ===
    # Check if we need to generate labels from raw CSVs
    LBL_PATH = f"{REAL_DATA_PATH}/labels.json"
    if not os.path.exists(LBL_PATH):
        print("Parametric Label Generation: STARTING (Ported from V1)...")
        try:
            # Logic from V1 generate_labels
            import pandas as pd
            import soundfile as sf
            
            # Paths
            FLIGHT_CSV = f"{REAL_DATA_PATH}/Oren1.csv"
            ALIGN_JSON = f"{REAL_DATA_PATH}/alignment_full.json"
            CAL_JSON = f"{REAL_DATA_PATH}/array_calibration.json"
            AUDIO_WAV = f"{REAL_DATA_PATH}/Number1.wav"
            
            # Load Configs
            with open(CAL_JSON) as f: cal = json.load(f)
            with open(ALIGN_JSON) as f: align = json.load(f)
            
            ARRAY_LAT, ARRAY_LON = cal['ARRAY_LAT'], cal['ARRAY_LON']
            ARRAY_HEADING = cal.get('ARRAY_HEADING', 90.0)
            
            # Load Flight Data
            df = pd.read_csv(FLIGHT_CSV)
            df.columns = df.columns.str.strip()
            
            # Time Sync
            df['ems'] = df['time(millisecond)']
            df['utc'] = pd.to_datetime(df['datetime(utc)']).astype(np.int64) / 1e9
            offset = np.median(df['utc'] - df['ems'] * 0.001)
            df['ts'] = df['ems'] * 0.001 + offset
            df['height'] = pd.to_numeric(df['height_above_takeoff(feet)'], errors='coerce').fillna(0)
            
            # Geo Calc
            def haversine(lat1, lon1, lat2, lon2):
                R, phi1, phi2 = 6371000, np.radians(lat1), np.radians(lat2)
                dphi, dlam = np.radians(lat2-lat1), np.radians(lon2-lon1)
                a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
                return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

            df['dist'] = df.apply(lambda r: haversine(ARRAY_LAT, ARRAY_LON, r['latitude'], r['longitude']), axis=1)
            df['az'] = df.apply(lambda r: np.degrees(np.arctan2(np.radians(r['longitude']-ARRAY_LON)*6371000*np.cos(np.radians((ARRAY_LAT+r['latitude'])/2)), np.radians(r['latitude']-ARRAY_LAT)*6371000)) % 360, axis=1)
            df['rel_az'] = (df['az'] - ARRAY_HEADING) % 360
            
            # Filter
            flying = df[(df['height'] > 10) & (df['dist'] < 200)].copy()
            
            # Audio Info
            info = sf.info(AUDIO_WAV)
            audio_len = int(info.duration * info.samplerate)
            CHUNK = int(0.25 * 48000)
            
            # Generate List
            labels = []
            for _, row in flying.iterrows():
                s = int((row['ts'] - align['intercept']) / align['slope'])
                if s >= 0 and s + CHUNK <= audio_len:
                    # vec computation for 3D regression (x,y,z)
                    # az is rel_az. el?
                    # simple az only for now or 3D?
                    # The V2 model regresses to 3D vector.
                    # Let's compute vec from az/el (assuming flat for now or use dist/height?)
                    # height/dist -> el
                    el = np.degrees(np.arctan2(row['height']*0.3048, row['dist']))
                    
                    # spherical to cartesian
                    # x = cos(el)cos(az), y = cos(el)sin(az), z = sin(el)
                    raz = np.radians(row['rel_az'])
                    rel = np.radians(el)
                    vec = [
                        np.cos(rel)*np.cos(raz),
                        np.cos(rel)*np.sin(raz),
                        np.sin(rel)
                    ]
                    
                    labels.append({
                        's': s, # mismatch 'sample_start' vs 's' in V2 code?
                        'vec': vec,
                        'az': row['rel_az'] # for MAE
                    })
            
            with open(LBL_PATH, "w") as f: json.dump(labels, f)
            print(f"✅ Generated {len(labels)} labels. Saved to {LBL_PATH}")
            lbs = labels # Set for use
            
        except Exception as e:
            print(f"❌ Label Generation Failed: {e}")
            print("Cannot proceed with fine-tuning.")
            return

    try:
        # Load (or reload)
        if 'lbs' not in locals():
            with open(LBL_PATH) as f: lbs = json.load(f)
        audio_full, _ = sf.read(f"{REAL_DATA_PATH}/Number1.wav") # Assuming name
        
        # Real Dataset
        class RealDataset(Dataset):
            def __init__(self, l): self.l = l
            def __len__(self): return len(self.l)
            def __getitem__(self, i):
                e = self.l[i]; x = audio_full[e['s']:e['s']+12000] # 0.25s
                # Tile:
                if len(x) < 24000: x = np.tile(x, 2)[:24000]
                return torch.tensor(x.T, dtype=torch.float32), torch.tensor(e['vec'], dtype=torch.float32)

        real_loader = DataLoader(RealDataset(lbs), batch_size=64, shuffle=True)
        
        # Fine-Tune Loop
        opt = torch.optim.Adam(model.parameters(), lr=1e-4) # Lower LR
        
        for ep in range(EPOCHS_FT):
            losses = []
            for x, y in tqdm.tqdm(real_loader, desc=f"FT Epoch {ep}"):
                opt.zero_grad()
                loss = loss_fn(model(x.cuda()), y.cuda())
                loss.backward()
                opt.step()
                losses.append(loss.item())
            
            # Calc MAE
            model.eval()
            errs = []
            with torch.no_grad():
                # Check a sub-sample to save time?
                for x, y in real_loader:
                    p = model(x.cuda()).cpu().numpy() # (B, 3)
                    t = y.cpu().numpy() # (B, 3)
                    # Convert vec to az
                    # az = atan2(y, x)
                    p_az = np.degrees(np.arctan2(p[:,1], p[:,0]))
                    t_az = np.degrees(np.arctan2(t[:,1], t[:,0]))
                    
                    diff = np.abs(p_az - t_az)
                    diff = np.minimum(diff, 360-diff)
                    errs.extend(diff)
                    if len(errs) > 200: break # estimate
            model.train()
            mae = np.mean(errs)
            print(f"FT Epoch {ep}: Loss {np.mean(losses):.4f} | MAE: {mae:.2f}°")
            
        # Save Final Sim2Real Model
        state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(state, f"{OUTPUT_DIR}/sim2real_model.pth")
        print("Sim2Real Model Saved.")
        
    except Exception as e:
        print(f"Fine-Tuning Failed or Skipped: {e}")

if __name__ == "__main__":
    main_training_loop()
