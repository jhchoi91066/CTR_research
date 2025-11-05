#!/bin/bash
# Phase 3 Training Monitor Script

LOG_FILE="/Users/jinhochoi/Desktop/dev/Research/results/logs/mdaf_mamba_phase3_seed42.log"

echo "=================================================="
echo "PHASE 3 TRAINING MONITOR"
echo "=================================================="
echo ""

# Check if process is running
if ps aux | grep "train_mdaf_taobao_phase3.py" | grep -v grep > /dev/null; then
    echo "✓ Training process is RUNNING"
    echo ""
else
    echo "✗ Training process is NOT RUNNING"
    echo ""
fi

# Get latest epoch info
echo "Latest Training Progress:"
echo "------------------------------------------------"
grep "Epoch" "$LOG_FILE" | tail -5
echo ""

# Get latest metrics
echo "Latest Metrics:"
echo "------------------------------------------------"
grep -E "(Train -|Val   -|Gap   -|Gate  -)" "$LOG_FILE" | tail -10
echo ""

# Check for errors
echo "Recent Errors/Warnings:"
echo "------------------------------------------------"
grep -i "error\|warning" "$LOG_FILE" | tail -5
if [ $? -ne 0 ]; then
    echo "No errors or warnings found"
fi
echo ""

# File size (indicates progress)
echo "Log File Size: $(du -h "$LOG_FILE" | cut -f1)"
echo "Last Modified: $(date -r "$LOG_FILE")"
echo ""

echo "=================================================="
echo "Use: tail -f $LOG_FILE"
echo "to follow live updates"
echo "=================================================="
