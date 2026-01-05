#!/bin/bash
# Master Parallel Launch Script for Oren1
# Backgrounds all launches to save time.

set -e

# Clear existing oren-sim2real-final processes to start fresh
pkill -f "modal run.*train_sim_modal_final.py" || true

launch_bg() {
    PROFILE=$1
    echo "🚀 Background Launching: $PROFILE"
    # We output to separate logs to avoid terminal clutter
    ./launch_detached.sh $PROFILE a100 > "launch_${PROFILE}.log" 2>&1 &
}

# 1. Start all five concurrently
launch_bg "sounds"
launch_bg "sounds3"
launch_bg "sounds4"
launch_bg "sounds2"
launch_bg "shleiby"

echo "=================================================="
echo "⚡ ALL FIVE WORKSPACES ARE NOW UPLOADING PURELY IN PARALLEL."
echo "I am monitoring the logs for all five."
echo "=================================================="

# Monitor logs for 'Job is running stably'
MAX_TRIALS=20 # 10 minutes max
TRIALS=0

while [ $TRIALS -lt $MAX_TRIALS ]; do
    COMPLETE_COUNT=0
    for p in sounds sounds3 sounds4 sounds2 shleiby; do
        if grep -q "Job is running stably" "launch_${p}.log"; then
            ((COMPLETE_COUNT++))
        fi
    done
    
    if [ $COMPLETE_COUNT -eq 5 ]; then
        echo "🎉 ALL FIVE WORKSPACES CONFIRMED RUNNING!"
        exit 0
    fi
    
    echo "⏳ Progress: $COMPLETE_COUNT/5 workspaces confirmed stable... (Trial $((TRIALS+1))/$MAX_TRIALS)"
    ((TRIALS++))
    sleep 30
done

echo "❌ TIMEOUT: Some jobs failed to stabilize within 10 minutes. Check launch_*.log"
exit 1
