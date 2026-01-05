import modal
import os
import torch
import numpy as np
import soundfile as sf
import time
import glob
import sys
import torch.nn.functional as F

# === CONFIG ===
APP_NAME = "oren-data-swarm"
N_WORKERS = 100
SAMPLES_PER_WORKER = 40000  # Total 4M
REF_LIB_PATH = "/data/oren1_reference_library"
OUTPUT_DIR = "/data/training_output/swarm_4m"

# === IMAGE ===
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "torchaudio", 
        "numpy",
        "scipy",
        "soundfile",
        "pandas",
        "pyroomacoustics"
    )
    .add_local_file("pyskiacoustics.py", "/root/pyskiacoustics.py")
)

# === VOLUME ===
vol = modal.Volume.from_name("oren-sim-data")

app = modal.App(APP_NAME)

# === WORKER FUNCTION ===
@app.function(
    image=image,
    cpu=1.0, 
    memory=2048,
    timeout=3600, # 1 hour max
    volumes={"/data": vol}
)
def generate_shard_swarm(worker_id):
    print(f"[Worker {worker_id}] Starting...")
    sys.path.append("/root")
    try:
        import pyskiacoustics
    except ImportError:
        print("CRITICAL: PySKIacoustics not found!")
        return

    # 1. Load Reference Library (CPU)
    print(f"[Worker {worker_id}] Loading Library...")
    lib_cpu = {}
    folders = sorted(glob.glob(f"{REF_LIB_PATH}/clips_*m"))
    if not folders:
        print("Error: Reference Library not found!")
        return
        
    for d in folders:
        dist = int(d.split('_')[-1].replace('m',''))
        clips = []
        for wav in glob.glob(f"{d}/*.wav"):
            sig_np, sr = sf.read(wav)
            if len(sig_np.shape) == 1: sig_np = sig_np[None, :] # unsqueeze
            if sig_np.shape[0] < 16: # channel check
                 sig_np = np.pad(sig_np, ((0, 16 - sig_np.shape[0]), (0, 0)))
            
            # Length fix
            if sig_np.shape[1] < 24000:
                sig_np = np.pad(sig_np, ((0,0), (0, 24000 - sig_np.shape[1])))
            else:
                sig_np = sig_np[:, :24000]
                
            clips.append(torch.from_numpy(sig_np).float())
            
        if clips:
            lib_cpu[dist] = torch.stack(clips)
            
    print(f"[Worker {worker_id}] Lib Loaded. Keys: {list(lib_cpu.keys())}")
    dists = sorted(list(lib_cpu.keys()))
    
    # 2. Generation Loop
    # We re-implement specific generation logic here to avoid importing the whole training script
    # and its GPU dependencies.
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    x_list, y_list = [], []
    batch_size = 64
    
    # Unique Seed
    seed = int(time.time()) + worker_id * 999
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    for i in range(0, SAMPLES_PER_WORKER, batch_size):
        curr_batch = min(batch_size, SAMPLES_PER_WORKER - i)
        
        batch_audio = []
        batch_vecs = []
        
        for _ in range(curr_batch):
            # A. Select Clip
            d_idx = np.random.randint(0, len(dists))
            dist_key = dists[d_idx]
            clips = lib_cpu[dist_key]
            clip_idx = np.random.randint(0, len(clips))
            sig_torch = clips[clip_idx]
            sig_numpy = sig_torch[0].numpy() # Use Ch0 as source
            
            # B. Physics Sim
            az = np.random.uniform(0, 360)
            el = np.random.uniform(5, 60)
            dist = np.random.uniform(10, 100)
            wind = np.random.choice([0, 2, 5, 8], p=[0.3, 0.3, 0.3, 0.1])
            
            sim_audio, vec = pyskiacoustics.generate_training_sample(
                sig_numpy, 48000,
                azimuth_deg=az, elevation_deg=el, distance_m=dist, wind_speed_mps=wind
            )
            
            batch_audio.append(torch.from_numpy(sim_audio).float())
            batch_vecs.append(torch.from_numpy(vec).float())
            
        # Buffer
        x_list.append(torch.stack(batch_audio))
        y_list.append(torch.stack(batch_vecs))
        
        # Periodic Flush (every 1000 samples) to avoid RAM OOM
        if len(x_list) * 64 >= 1000:
            # We aggregate locally but we only save ONE big file at the end 
            # OR simple incremental saves?
            # 40k samples is ~8GB? No.
            # 40k * 16 * 24000 * 4 bytes = 60 GB! Too big for RAM.
            # We MUST save incrementally.
            pass
            
    # CRITICAL: We need to save efficiently.
    # 40,000 samples is HUGE. 100 workers x 40k = 4M.
    # Let's chunk it. 1000 samples per file = 1.5GB.
    # So each worker saves 40 files? That's 4000 files total. Manageable.
    
    # Refined Loop with Saving
    pass # (Re-written below properly)
    
    return f"Worker {worker_id} Done"

# === RE-WRITTEN WORKER WITH PROPER SAVING ===
@app.function(
    image=image,
    cpu=1.0, 
    memory=4096, # Bump memory slightly
    timeout=3600,
    volumes={"/data": vol}
)
def generate_shard_swarm_safe(worker_id):
    import pyskiacoustics
    import shutil
    
    # ... Lib loading (same as above) ...
    print(f"[Worker {worker_id}] Loading Library...")
    lib_cpu = {}
    folders = sorted(glob.glob(f"{REF_LIB_PATH}/clips_*m"))
    if not folders: return "No Data"
    for d in folders:
        dist = int(d.split('_')[-1].replace('m',''))
        clips = []
        for wav in glob.glob(f"{d}/*.wav"):
            sig_np, sr = sf.read(wav)
            if len(sig_np.shape) == 1: sig_np = sig_np[None, :] 
            if sig_np.shape[0] < 16: sig_np = np.pad(sig_np, ((0, 16 - sig_np.shape[0]), (0, 0)))
            if sig_np.shape[1] < 24000: sig_np = np.pad(sig_np, ((0,0), (0, 24000 - sig_np.shape[1])))
            else: sig_np = sig_np[:, :24000]
            clips.append(torch.from_numpy(sig_np).float())
        if clips: lib_cpu[dist] = torch.stack(clips)
    dists = sorted(list(lib_cpu.keys()))
    
    output_path = f"{OUTPUT_DIR}/worker_{worker_id}"
    os.makedirs(output_path, exist_ok=True)
    
    seed = int(time.time()) + worker_id * 9999
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    samples_generated = 0
    buffer_x = []
    buffer_y = []
    buffer_limit = 500 # Save every 500 samples (~750MB file)
    
    while samples_generated < SAMPLES_PER_WORKER:
        # Generate ONE sample (simplest loop)
        d_idx = np.random.randint(0, len(dists))
        clips = lib_cpu[dists[d_idx]]
        sig = clips[np.random.randint(0, len(clips))]
        
        sim, vec = pyskiacoustics.generate_training_sample(
            sig[0].numpy(), 48000,
            azimuth_deg=np.random.uniform(0, 360),
            elevation_deg=np.random.uniform(5, 60),
            distance_m=np.random.uniform(10, 100),
            wind_speed_mps=np.random.choice([0,2,5,8], p=[0.3,0.3,0.3,0.1])
        )
        
        buffer_x.append(torch.from_numpy(sim).float())
        buffer_y.append(torch.from_numpy(vec).float())
        
        if len(buffer_x) >= buffer_limit:
            fname = f"{output_path}/chunk_{samples_generated}.pt"
            torch.save((torch.stack(buffer_x), torch.stack(buffer_y)), fname)
            # vol.commit() # Commit periodically? Too slow. Auto-commit at end is better.
            buffer_x = []
            buffer_y = []
            print(f"[Worker {worker_id}] Saved {fname}")
            
        samples_generated += 1
        
    return f"Worker {worker_id} Generated {samples_generated}"

@app.local_entrypoint()
def main():
    print(f"🚀 Launching swarm of {N_WORKERS} workers...")
    print(f"Target: {N_WORKERS * SAMPLES_PER_WORKER} samples")
    
    # Map over workers
    results = list(generate_shard_swarm_safe.map(range(N_WORKERS)))
    
    print("✅ Swarm Complete!")
    for r in results:
        print(r)
