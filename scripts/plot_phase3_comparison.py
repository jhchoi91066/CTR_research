#!/usr/bin/env python3
"""
Plot Phase 1 vs Phase 3 Learning Curves
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Phase 1 data (from log file)
phase1_data = {
    'epochs': [1, 2, 3, 4, 5],
    'train_auc': [0.5410, 0.6643, 0.7723, 0.8368, 0.8848],
    'val_auc': [0.5829, 0.5703, 0.5576, 0.5542, 0.5513],
    'gap': [-0.0419, 0.0940, 0.2147, 0.2826, 0.3335],
    'gate_mean': [0.3003, 0.3799, 0.4315, 0.4473, 0.4059]
}

# Phase 3 data (only Epoch 1)
phase3_data = {
    'epochs': [1],
    'train_auc': [0.5121],
    'val_auc': [0.5698],
    'gap': [-0.0577],
    'gate_mean': [0.1666]
}

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Phase 1 vs Phase 3: Enhanced Regularization Impact', fontsize=16, fontweight='bold')

# Plot 1: Train AUC
ax = axes[0, 0]
ax.plot(phase1_data['epochs'], phase1_data['train_auc'], 'o-', label='Phase 1 (Dropout 0.2)', linewidth=2, markersize=8, color='#e74c3c')
ax.plot(phase3_data['epochs'], phase3_data['train_auc'], 's-', label='Phase 3 (Dropout 0.3)', linewidth=2, markersize=10, color='#3498db')
ax.axhline(y=0.70, color='red', linestyle='--', alpha=0.5, label='Overfitting Threshold')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Train AUC', fontsize=12)
ax.set_title('Training AUC: Regularization Effect', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, 5.5)
ax.set_ylim(0.45, 0.95)

# Plot 2: Validation AUC
ax = axes[0, 1]
ax.plot(phase1_data['epochs'], phase1_data['val_auc'], 'o-', label='Phase 1', linewidth=2, markersize=8, color='#e74c3c')
ax.plot(phase3_data['epochs'], phase3_data['val_auc'], 's-', label='Phase 3', linewidth=2, markersize=10, color='#3498db')
ax.axhline(y=phase1_data['val_auc'][0], color='#e74c3c', linestyle=':', alpha=0.5, label=f'Phase 1 Best: {phase1_data["val_auc"][0]:.4f}')
ax.axhline(y=phase3_data['val_auc'][0], color='#3498db', linestyle=':', alpha=0.5, label=f'Phase 3 Epoch 1: {phase3_data["val_auc"][0]:.4f}')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Validation AUC', fontsize=12)
ax.set_title('Validation AUC: Performance Comparison', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, 5.5)
ax.set_ylim(0.54, 0.59)

# Plot 3: Train-Val Gap
ax = axes[1, 0]
ax.plot(phase1_data['epochs'], phase1_data['gap'], 'o-', label='Phase 1', linewidth=2, markersize=8, color='#e74c3c')
ax.plot(phase3_data['epochs'], phase3_data['gap'], 's-', label='Phase 3', linewidth=2, markersize=10, color='#3498db')
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.axhspan(0.05, 0.12, alpha=0.2, color='green', label='Target Range [0.05, 0.12]')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Train-Val AUC Gap', fontsize=12)
ax.set_title('Train-Val Gap: Overfitting Indicator', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, 5.5)
ax.set_ylim(-0.1, 0.4)

# Annotate Phase 1 overfitting trajectory
for i, gap in enumerate(phase1_data['gap']):
    if gap > 0.15:
        ax.annotate(f'Epoch {i+1}\nSevere\nOverfit', xy=(i+1, gap), xytext=(i+1+0.3, gap+0.05),
                   fontsize=8, color='red', weight='bold',
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

# Plot 4: Gate Mean (Branch Balance)
ax = axes[1, 1]
ax.plot(phase1_data['epochs'], phase1_data['gate_mean'], 'o-', label='Phase 1', linewidth=2, markersize=8, color='#e74c3c')
ax.plot(phase3_data['epochs'], phase3_data['gate_mean'], 's-', label='Phase 3', linewidth=2, markersize=10, color='#3498db')
ax.axhline(y=0.5, color='purple', linestyle='--', alpha=0.5, label='Equal Balance (0.5)')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Gate Mean', fontsize=12)
ax.set_title('Gate Statistics: DCNv3 vs Mamba Branch Balance', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, 5.5)
ax.set_ylim(0.0, 0.6)

# Add text annotation
ax.text(3.5, 0.05, 'Lower gate = More Mamba\nHigher gate = More DCNv3',
        fontsize=9, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()

# Save figure
output_path = Path('/Users/jinhochoi/Desktop/dev/Research/results/phase3_learning_curves.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")

# Create summary statistics table
print("\n" + "="*80)
print("PHASE 1 VS PHASE 3 SUMMARY STATISTICS")
print("="*80)
print(f"\n{'Metric':<30} {'Phase 1 Epoch 1':<20} {'Phase 3 Epoch 1':<20} {'Difference':<15}")
print("-"*80)
print(f"{'Train AUC':<30} {phase1_data['train_auc'][0]:<20.4f} {phase3_data['train_auc'][0]:<20.4f} {phase3_data['train_auc'][0] - phase1_data['train_auc'][0]:<+15.4f}")
print(f"{'Val AUC':<30} {phase1_data['val_auc'][0]:<20.4f} {phase3_data['val_auc'][0]:<20.4f} {phase3_data['val_auc'][0] - phase1_data['val_auc'][0]:<+15.4f}")
print(f"{'Train-Val Gap':<30} {phase1_data['gap'][0]:<20.4f} {phase3_data['gap'][0]:<20.4f} {phase3_data['gap'][0] - phase1_data['gap'][0]:<+15.4f}")
print(f"{'Gate Mean':<30} {phase1_data['gate_mean'][0]:<20.4f} {phase3_data['gate_mean'][0]:<20.4f} {phase3_data['gate_mean'][0] - phase1_data['gate_mean'][0]:<+15.4f}")

print("\n" + "="*80)
print("PHASE 1 TRAJECTORY (5 Epochs)")
print("="*80)
print(f"{'Epoch':<10} {'Train AUC':<15} {'Val AUC':<15} {'Gap':<15} {'Status':<20}")
print("-"*80)
for i in range(5):
    status = "✓ Best" if i == 0 else f"Overfitting (+{phase1_data['gap'][i]:.3f})"
    print(f"{i+1:<10} {phase1_data['train_auc'][i]:<15.4f} {phase1_data['val_auc'][i]:<15.4f} {phase1_data['gap'][i]:<+15.4f} {status:<20}")

print("\n" + "="*80)
print("KEY OBSERVATIONS")
print("="*80)
print("1. Phase 3 shows REDUCED train AUC (-5.3%) indicating regularization is working")
print("2. Phase 3 shows SLIGHTLY LOWER val AUC (-2.25%) - may be over-regularized")
print("3. Phase 1 suffers CATASTROPHIC OVERFITTING (gap grows to +0.33 by Epoch 5)")
print("4. Phase 3 trajectory UNKNOWN (only 1 epoch completed due to infrastructure limits)")
print("5. Gate mean DECREASED 44.5% in Phase 3 (DCNv3 branch underutilized)")
print("="*80)
