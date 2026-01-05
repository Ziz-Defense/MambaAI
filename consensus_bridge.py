import os
import time
import subprocess
import glob
import re

# === CONFIG ===
PROFILES = ["sounds", "sounds2", "sounds3", "sounds4", "shleiby"]
VOLUME_NAME = "oren-sim-data"
CHECKPOINT_NAME = "latest_model.pth"
SYNC_FILE_NAME = "consensus_sync.pth"
TEMP_DIR = "consensus_temp"
ARCHIVE_DIR = "Local_Checkpoints"
POLL_INTERVAL = 60 # 1 Minute

def get_best_mae(profile):
    """
    Fetch logs from volume (or infer from partial logs) to find current MAE.
    Since we don't have a direct 'mae.txt', we might need to rely on the training script 
    writing a tiny 'status.json' or just trusting the checkpoint existence for now.
    
    TODO: In `train_sim_gcp_native.py`, write `status.json` with {mae: 12.3, epoch: 5}.
    For now, we'll assume the file `status.json` exists.
    """
    try:
        # Fetch status file
        cmd = f"modal volume get {VOLUME_NAME} status.json {TEMP_DIR}/{profile}_status.json -p {profile}"
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        
        with open(f"{TEMP_DIR}/{profile}_status.json", 'r') as f:
            import json
            data = json.load(f)
            return data.get('mae', 999.9), data.get('epoch', 0)
    except:
        return 999.9, 0

def run_bridge():
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    best_overall_mae = 999.9
    leader_profile = None
    
    print(f"🌉 Consensus Bridge Active! Monitoring {len(PROFILES)} profiles...")
    
    while True:
        print(f"\n--- Polling Cycle {time.strftime('%H:%M:%S')} ---")
        
        current_scores = {}
        
        # 1. Identify Leader
        for p in PROFILES:
            mae, ep = get_best_mae(p)
            print(f"[{p}] Ep {ep} | MAE: {mae:.2f}°")
            current_scores[p] = mae
            
        # Find winner
        winner = min(current_scores, key=current_scores.get)
        win_mae = current_scores[winner]
        
        if win_mae < best_overall_mae - 0.5: # 0.5deg improvement threshold to avoid thrashing
            print(f"👑 NEW LEADER: {winner} ({win_mae:.2f}° vs {best_overall_mae:.2f}°)")
            best_overall_mae = win_mae
            leader_profile = winner
            
            try:
                # 2. Download Leader Model
                print(f"⬇️ Downloading Checkpoint from {winner}...")
                cmd = f"modal volume get {VOLUME_NAME} {CHECKPOINT_NAME} {TEMP_DIR}/leader.pth -p {winner}"
                subprocess.run(cmd, shell=True, check=True)
                
                # 2b. Archive locally for User
                import shutil
                ts = int(time.time())
                archive_name = f"{ARCHIVE_DIR}/best_model_mae{win_mae:.2f}_{ts}.pth"
                shutil.copy(f"{TEMP_DIR}/leader.pth", archive_name)
                print(f"💾 Archived to {archive_name}")
                
                # 3. Propagate to Others
                for p in PROFILES:
                    if p == winner: continue
                    
                    print(f"🚀 Syncing to {p}...")
                    # Upload as tmp then rename to ensure atomic? 
                    # Modal volume put is fairly atomic for files.
                    cmd = f"modal volume put {VOLUME_NAME} {TEMP_DIR}/leader.pth {SYNC_FILE_NAME} -p {p}"
                    subprocess.run(cmd, shell=True)
                    
                print("✅ Consensus Sync Complete.")
            except Exception as e:
                print(f"❌ Sync Failed this cycle: {e}")
        else:
            print(f"💤 No leadership change. Current Best: {best_overall_mae:.2f}° ({leader_profile})")
            
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    run_bridge()
