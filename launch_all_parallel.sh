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

# 1. Start all three concurrently
launch_bg "sounds"
launch_bg "sounds3"
launch_bg "sounds4"

echo "=================================================="
echo "⚡ ALL THREE WORKSPACES ARE NOW UPLOADING PURELY IN PARALLEL."
echo "I am monitoring the logs for all three."
echo "=================================================="

# Monitor logs for 'Job is running stably'
while true; do
    COMPLETE_COUNT=0
    for p in sounds sounds2 ws1; do
        if grep -q "Job is running stably" "launch_${p}.log"; then
            ((COMPLETE_COUNT++))
        fi
    done
    
    if [ $COMPLETE_COUNT -eq 3 ]; then
        echo "🎉 ALL THREE WORKSPACES CONFIRMED RUNNING!"
        break
    fi
    
    echo "⏳ Progress: $COMPLETE_COUNT/3 workspaces confirmed stable... (checking logs)"
    sleep 30
done
