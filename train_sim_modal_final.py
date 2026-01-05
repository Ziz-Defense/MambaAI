import modal
import os
import sys

# === Lightweight Image (No Data Baked In) ===
image = (
    modal.Image.debian_slim()
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install("torch==2.5.1", "torchaudio==2.5.1", "numpy<2", "pandas", "soundfile", "scipy", "pyroomacoustics", "tqdm", "einops")
    .add_local_file("train_sim_gcp_native.py", "/root/train_sim_gcp_native.py")
    .add_local_file("pyskiacoustics.py", "/root/pyskiacoustics.py")
    .add_local_file("mamba_unfolding_net.py", "/root/mamba_unfolding_net.py")
)

app = modal.App("oren-sim2real-official")
vol = modal.Volume.from_name("oren-sim-data", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100:1",
    timeout=86400,
    volumes={"/data": vol},
    env={"SIM_OUTPUT_DIR": "/data"}
)
def train_on_gpu():
    import subprocess
    import train_sim_gcp_native
    
    # 1. Untar data if needed (first launch only)
    if not os.path.exists("/data/Oren1_complete"):
        if os.path.exists("/data/oren_data.tar"):
            print("📦 Untarring flight data from Volume...")
            subprocess.run(["tar", "-xf", "/data/oren_data.tar", "-C", "/data"])
            vol.commit()
            print("✅ Data Unpacked!")
        else:
            raise RuntimeError("No oren_data.tar found on volume! Run: modal volume put oren-sim-data oren_data.tar /oren_data.tar")
    
    # 2. Symlink for script compatibility
    if not os.path.exists("Oren1_complete"):
        try: os.symlink("/data/Oren1_complete", "Oren1_complete")
        except: pass
    
    print("🚀 Starting Training Loop on A100...")
    train_sim_gcp_native.main_training_loop()

@app.local_entrypoint()
def main():
    print("=== Launching V3 (Volume-Seeded, No Image Data) ===")
    train_on_gpu.remote()
    print("✅ DETACHED JOB LAUNCHED.")
