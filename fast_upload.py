import modal
import os
import shutil

# 1. Define the APP and the VOLUME
app = modal.App("fast-data-upload")
vol = modal.Volume.from_name("oren-sim-data", create_if_missing=True)

# 2. Define a function that mounts the local data
# Mounts are generally more resilient than add_local_dir for large transfers
@app.function(
    image=modal.Image.debian_slim(),
    volumes={"/data": vol},
    mounts=[
        modal.Mount.from_local_dir("oren1_reference_library", remote_path="/root/lib"),
        modal.Mount.from_local_dir("Oren1_complete", remote_path="/root/oren1"),
    ],
    timeout=3600 # 1 hour
)
def upload_data():
    print("🚀 TARGET: /data/oren1_reference_library")
    if not os.path.exists("/data/oren1_reference_library"):
        print("Copying Reference Library...")
        shutil.copytree("/root/lib", "/data/oren1_reference_library")
        print("✅ Library Done.")
    else:
        print("✅ Library already exists in Volume.")

    print("🚀 TARGET: /data/Oren1_complete")
    if not os.path.exists("/data/Oren1_complete"):
        print("Copying Oren1 Flight Data (1.5GB)...")
        # Use a more verbose copy to see progress? shutil is silent.
        # But Modal will show Mount upload progress in the terminal.
        shutil.copytree("/root/oren1", "/data/Oren1_complete")
        print("✅ Oren1 Data Done.")
    else:
        print("✅ Oren1 Data already exists in Volume.")

    print("Finalizing Volume commit...")
    vol.commit()
    print("✨ ALL DATA SYNCED.")

if __name__ == "__main__":
    with app.run():
        upload_data.remote()
