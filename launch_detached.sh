#!/bin/bash
# Launch Detached Mamba Training with Verification

PROFILE=$1
GPU_TYPE=${2:-"a100"} # Default to A100 for speed

if [ -z "$PROFILE" ]; then
    echo "Usage: ./launch_detached.sh <profile> [gpu_type]"
    exit 1
fi

echo "=================================================="
echo "🚀 Launching Mamba Training on Profile: $PROFILE"
echo "🖥️  GPU: $GPU_TYPE"
echo "=================================================="

# ISOLATION: Set the env var for this process only
export MODAL_PROFILE=$PROFILE

# 1. Update Script GPU Config (Hack/Patch)
# NOTE: This file modification is STILL A RACE CONDITION if multiple scripts try to edit `train_sim_modal_final.py` at once.
# Ideally we pass GPU as an arg to the python script, but `train_sim_modal_final.py` has it hardcoded in the decorator.
# Since all 5 jobs are A100, this race is "safe" (they all write the same thing).
# But if we mixed T4 and A100, this would break.
# For now, assumed all A100.
if [ "$GPU_TYPE" == "a100" ]; then
    sed -i '' 's/gpu="T4:8"/gpu="A100:1"/g' train_sim_modal_final.py
    echo "⚡ Configuration set to A100:1"
else
    sed -i '' 's/gpu="A100:1"/gpu="T4:8"/g' train_sim_modal_final.py
    echo "🐢 Configuration set to T4:8"
fi

# 2. Launch Detached
# Capture the Run ID for checking
echo "Submitting job..."
RUN_OUTPUT=$(modal run --detach train_sim_modal_final.py)
echo "$RUN_OUTPUT"

# 3. Wait and Verify
echo "⏳ Waiting 2 minutes to confirm stability..."
sleep 120

# Check status
# We list recent apps and grep for 'oren-sim2real-final'
STATUS=$(modal app list | grep oren-sim2real-final | head -n 1)

if [[ "$STATUS" == *"running"* ]] || [[ "$STATUS" == *"detached"* ]]; then
    echo "✅ CONFIRMED: Job is running stably on $PROFILE."
    echo "Status Line: $STATUS"
else
    echo "⚠️  WARNING: Job might have failed or finished early."
    echo "Current App List:"
    modal app list
fi
