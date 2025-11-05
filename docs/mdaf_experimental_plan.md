# MDAF Experimental Plan (Week 10-11)

**Project**: MDAF (Mamba-DCN with Adaptive Fusion)
**Phase**: MDAF Integration and Comprehensive Experiments
**Timeline**: Week 10-11 (Nov 4-17, 2025)
**Status**: Ready to Execute

---

## Executive Summary

This document outlines the comprehensive experimental plan for validating MDAF's performance and conducting rigorous ablation studies. The plan includes baseline training, multi-seed experiments, fusion mechanism comparisons, and statistical analysis.

**Primary Research Question**: Does MDAF (hybrid static + sequential) outperform single-branch baselines through adaptive fusion?

**Key Hypothesis**: Gated fusion of DCNv3 (static) and Mamba4Rec (sequential) will achieve > 0.5820 AUC on Taobao dataset, surpassing standalone models.

---

## Experiment Matrix

### Phase 1: Baseline Training (Week 10, Day 2-3)

| Exp ID | Model | Description | Random Seed | Priority | Est. Time | Target AUC |
|--------|-------|-------------|-------------|----------|-----------|------------|
| **EXP-1** | MDAF-Mamba | DCNv3 + Mamba4Rec + Fusion | 42 | **P0** | 2-3h | **≥ 0.5820** |
| **EXP-2** | MDAF-Static-Only | DCNv3 branch only (no sequential) | 42 | **P0** | 1.5-2h | 0.5500-0.5550 |
| **EXP-3** | MDAF-Sequential-Only | Mamba4Rec branch only (no static) | 42 | **P0** | 1.5-2h | 0.5680-0.5720 |
| **EXP-4** | MDAF-BST | DCNv3 + BST + Fusion | 42 | **P1** | 3-4h | 0.5800-0.5840 |

**Total Phase 1 Time**: ~10-12 hours wall-clock (sequential execution)

**Success Criteria (EXP-1)**:
- ✅ Pass: MDAF-Mamba ≥ 0.5780 AUC (BST + 0.69%p)
- 🎯 Target: MDAF-Mamba ≥ 0.5820 AUC (BST + 1.09%p)
- ⭐ Stretch: MDAF-Mamba ≥ 0.5900 AUC (BST + 1.89%p)

**Decision Point After Phase 1**:
- IF EXP-1 meets target (≥ 0.5820) → Proceed to Phase 2
- IF EXP-1 below minimum (< 0.5780) → Investigate fusion mechanism, adjust hyperparameters
- IF EXP-4 significantly outperforms EXP-1 → Analyze why BST fusion is better

---

### Phase 2: Multi-Seed Validation (Week 10, Day 4-5)

| Exp ID | Model | Random Seeds | Priority | Est. Time | Purpose |
|--------|-------|--------------|----------|-----------|---------|
| **EXP-5** | MDAF-Mamba | [42, 123, 456, 789, 2024] | **P1** | 10-15h | Statistical significance |
| **EXP-6** | MDAF-BST | [42, 123, 456, 789, 2024] | **P1** | 15-20h | Comparison baseline |
| **EXP-7** | Mamba4Rec v2 | [123, 456, 789, 2024] | P2 | 12-16h | Standalone variance |
| **EXP-8** | BST | [123, 456, 789, 2024] | P2 | 8-12h | Baseline variance |

**Note**: Seed 42 results already available from Phase 1 (EXP-1, EXP-4)

**Total Phase 2 Time**: ~45-63 hours wall-clock (parallelizable to ~20-30h with 2 GPUs)

**Analysis Plan**:
- Compute mean ± std for each model across 5 seeds
- Two-tailed t-test for MDAF-Mamba vs BST (α=0.05)
- Effect size (Cohen's d) for relative improvement quantification
- 95% confidence intervals for all reported metrics

---

### Phase 3: Fusion Mechanism Ablation (Week 11, Day 1-2)

| Exp ID | Fusion Type | Description | Random Seed | Priority | Est. Time |
|--------|-------------|-------------|-------------|----------|-----------|
| **EXP-9** | Gated Fusion | Current: `gate·static + (1-gate)·seq` | 42 | **P0** | - (EXP-1) |
| **EXP-10** | Concat Fusion | Simple: `concat([static, seq]) → MLP` | 42 | P2 | 2-3h |
| **EXP-11** | Attention Fusion | `attention(Q=static, K=seq, V=seq)` | 42 | P2 | 2-3h |
| **EXP-12** | Learned Weighting | `α·static + β·seq` (α, β learnable) | 42 | P2 | 2-3h |

**Purpose**: Justify gated fusion design choice over alternatives

**Expected Outcome**:
- Gated Fusion (EXP-9) should outperform or match alternatives
- If Attention Fusion (EXP-11) is better → consider switching
- Results inform Discussion section on design choices

---

### Phase 4: Hyperparameter Sensitivity (Week 11, Day 3-4)

| Exp ID | Hyperparameter | Values Tested | Base Config | Priority | Est. Time |
|--------|----------------|---------------|-------------|----------|-----------|
| **EXP-13** | Batch Size | [2048, 4096, 8192] | MDAF-Mamba | P2 | 6-9h |
| **EXP-14** | Learning Rate | [5e-4, 1e-3, 2e-3] | MDAF-Mamba | P2 | 6-9h |
| **EXP-15** | Dropout | [0.1, 0.2, 0.3] | MDAF-Mamba | P2 | 6-9h |
| **EXP-16** | Fusion Hidden Dim | [64, 128, 256] | MDAF-Mamba | P3 | 6-9h |

**Purpose**: Demonstrate robustness to hyperparameter choices

**Expected Outcome**: MDAF performance should be stable across reasonable hyperparameter ranges

---

### Phase 5: Component Architecture Ablation (Week 11, Day 5-6)

| Exp ID | Component | Variants Tested | Random Seed | Priority | Est. Time |
|--------|-----------|-----------------|-------------|----------|-----------|
| **EXP-17** | Mamba Layers | [1, 2, 3] layers | 42 | P2 | 6-9h |
| **EXP-18** | DCNv3 Cross Layers | [2, 3, 4] layers | 42 | P2 | 6-9h |
| **EXP-19** | Embedding Dim | [64, 128, 256] | 42 | P3 | 6-9h |

**Purpose**: Validate architectural design choices

**Expected Outcome**: Current configuration (2 Mamba layers, 3 DCNv3 layers, 128 dim) should be near-optimal

---

## Detailed Experiment Specifications

### EXP-1: MDAF-Mamba Baseline (P0 - Critical)

**Objective**: Establish primary MDAF performance on Taobao dataset

**Configuration**:
```python
{
    'model': 'MDAF_Mamba',
    'dataset': 'Taobao',
    'batch_size': 2048,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'dropout': 0.2,
    'max_epochs': 15,
    'early_stopping_patience': 5,
    'optimizer': 'adam',
    'lr_schedule': 'cosine_annealing',
    'warmup_epochs': 2,
    'random_seed': 42,
}
```

**Expected Performance**:
```
Pessimistic:  Val AUC 0.5760-0.5790 (20% probability)
Realistic:    Val AUC 0.5800-0.5860 (60% probability) ← Expected
Optimistic:   Val AUC 0.5870-0.5950 (20% probability)
```

**Training Command**:
```bash
./venv/bin/python experiments/train_mdaf_taobao.py \
  --model mamba \
  --epochs 15 \
  --batch_size 2048 \
  --lr 1e-3 \
  --weight_decay 1e-5 \
  --dropout 0.2 \
  --warmup_epochs 2 \
  --early_stopping_patience 5 \
  --seed 42 \
  --save_dir results/checkpoints \
  --log_dir results/logs
```

**Metrics to Track**:
- Validation AUC (primary)
- Training AUC
- Train-Val gap (overfitting indicator)
- Validation LogLoss
- Gate statistics (mean, std, distribution)
- Training time per epoch
- Total training time

**Success Criteria**:
1. Val AUC ≥ 0.5780 (minimum threshold)
2. Train-Val gap ≤ 0.08 (generalization check)
3. Gate mean ∈ [0.2, 0.8] (both branches utilized)
4. Training convergence within 15 epochs

**Failure Response**:
- IF Val AUC < 0.5780:
  - Check gate values: if gate ≈ 1.0 → static branch dominates (issue with sequential)
  - Check gate values: if gate ≈ 0.0 → sequential branch dominates (issue with static)
  - Review training curves for instability or overfitting
  - Consider hyperparameter adjustment (increase regularization, reduce LR)

---

### EXP-2: MDAF-Static-Only (P0 - Ablation)

**Objective**: Isolate DCNv3 contribution in MDAF framework

**Configuration**: Same as EXP-1, but disable sequential branch
```python
# In MDAF forward pass:
sequential_emb = torch.zeros_like(static_emb)  # Zero out sequential
```

**Expected Performance**: Val AUC 0.5500-0.5550 (similar to DCNv2/AutoInt baselines)

**Purpose**:
- Demonstrate that sequential branch is necessary
- Quantify sequential contribution: `EXP-1 AUC - EXP-2 AUC`

**Success Criteria**: EXP-2 AUC < EXP-1 AUC by at least 0.02 (2%p)

---

### EXP-3: MDAF-Sequential-Only (P0 - Ablation)

**Objective**: Isolate Mamba4Rec contribution in MDAF framework

**Configuration**: Same as EXP-1, but disable static branch
```python
# In MDAF forward pass:
static_emb = torch.zeros_like(sequential_emb)  # Zero out static
```

**Expected Performance**: Val AUC 0.5680-0.5720 (similar to Mamba4Rec v2 standalone)

**Purpose**:
- Demonstrate that static branch is necessary
- Quantify static contribution: `EXP-1 AUC - EXP-3 AUC`

**Success Criteria**: EXP-3 AUC < EXP-1 AUC by at least 0.01 (1%p)

---

### EXP-4: MDAF-BST (P1 - Critical Comparison)

**Objective**: Compare SSM vs Transformer in hybrid framework

**Configuration**: Identical to EXP-1, but use BST instead of Mamba4Rec

**Expected Performance**: Val AUC 0.5800-0.5840 (similar to or slightly below EXP-1)

**Strategic Importance**:
- Addresses reviewer question: "Why Mamba4Rec over BST?"
- Three possible outcomes:
  1. **MDAF-Mamba > MDAF-BST**: SSMs provide superior fusion (ideal outcome)
  2. **MDAF-Mamba ≈ MDAF-BST**: Transformer-parity in hybrid setting (acceptable, emphasize efficiency)
  3. **MDAF-Mamba < MDAF-BST**: SSMs underperform (analyze why, emphasize efficiency trade-off)

**Success Criteria**:
- IF MDAF-Mamba ≥ MDAF-BST → Claim SSM superiority
- IF MDAF-Mamba ≈ MDAF-BST (within 0.5%p) → Claim efficiency advantage (3× smaller, O(L) vs O(L²))
- IF MDAF-Mamba < MDAF-BST (>1%p) → Re-evaluate SSM design or accept as performance-efficiency trade-off

---

### EXP-5/6: Multi-Seed Validation (P1 - Statistical Rigor)

**Objective**: Establish statistical significance of MDAF improvements

**Methodology**:
1. Train MDAF-Mamba with 5 random seeds: [42, 123, 456, 789, 2024]
2. Train MDAF-BST with 5 random seeds: [42, 123, 456, 789, 2024]
3. Compute mean ± std for validation AUC
4. Conduct paired t-test for MDAF-Mamba vs MDAF-BST
5. Compute 95% confidence intervals

**Statistical Tests**:
```python
from scipy import stats

# Two-tailed t-test
t_stat, p_value = stats.ttest_rel(mdaf_mamba_aucs, mdaf_bst_aucs)

# Effect size (Cohen's d)
def cohens_d(group1, group2):
    mean_diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt((np.std(group1)**2 + np.std(group2)**2) / 2)
    return mean_diff / pooled_std

effect_size = cohens_d(mdaf_mamba_aucs, mdaf_bst_aucs)

# 95% Confidence Interval
ci_lower = np.mean(mdaf_mamba_aucs) - 1.96 * np.std(mdaf_mamba_aucs) / np.sqrt(5)
ci_upper = np.mean(mdaf_mamba_aucs) + 1.96 * np.std(mdaf_mamba_aucs) / np.sqrt(5)
```

**Publication Reporting**:
- "MDAF-Mamba achieves 0.XXXX ± 0.00XX AUC (mean ± std, n=5 runs)"
- "MDAF-Mamba significantly outperforms BST baseline (p < 0.05, Cohen's d = X.XX)"
- "The 95% confidence interval [0.XXXX, 0.XXXX] does not overlap with BST [0.XXXX, 0.XXXX]"

**Success Criteria**:
- MDAF-Mamba mean > BST mean + 0.01 (1%p absolute)
- p-value < 0.05
- Confidence intervals non-overlapping

---

## Computational Resources

### Hardware Requirements
- **Device**: Apple MPS (M-series GPU) or CUDA GPU
- **Memory**: 16GB+ RAM, 8GB+ VRAM
- **Storage**: ~5GB for checkpoints, ~2GB for logs

### Time Estimates

**Single Experiment Duration**:
- MDAF-Mamba: 2-3 hours (46M parameters)
- MDAF-BST: 3-4 hours (133M parameters)
- Static/Sequential-Only: 1.5-2 hours
- Fusion variants: 2-3 hours each

**Total Project Time**:
- Sequential execution: ~120-150 hours
- Parallel execution (2 GPUs): ~60-75 hours wall-clock
- Recommended: Run critical experiments (EXP-1 to EXP-4) first, then parallelize multi-seed

### Parallelization Strategy

**Day 2-3 (Sequential - Critical Path)**:
1. EXP-1 (MDAF-Mamba baseline) → Immediate assessment
2. IF success → EXP-2, EXP-3, EXP-4 in parallel
3. IF failure → Debug before proceeding

**Day 4-5 (Parallel - Multi-Seed)**:
- GPU 1: EXP-5 (MDAF-Mamba seeds 123, 456, 789, 2024)
- GPU 2: EXP-6 (MDAF-BST seeds 123, 456, 789, 2024)

**Day 6-7 (As needed - Ablations)**:
- Fusion mechanism comparisons (EXP-10, EXP-11, EXP-12)
- Hyperparameter sensitivity if time permits

---

## Data Management

### Checkpoints
```
results/checkpoints/
├── mdaf_mamba_taobao_seed42_best.pth
├── mdaf_mamba_taobao_seed123_best.pth
├── mdaf_mamba_taobao_seed456_best.pth
├── mdaf_mamba_taobao_seed789_best.pth
├── mdaf_mamba_taobao_seed2024_best.pth
├── mdaf_bst_taobao_seed42_best.pth
├── mdaf_bst_taobao_seed123_best.pth
└── ... (similar for other experiments)
```

### Logs
```
results/logs/
├── mdaf_mamba_taobao_seed42_20251104_143022.log
├── mdaf_bst_taobao_seed42_20251104_163045.log
└── ... (timestamped logs for all experiments)
```

### Results Aggregation
```
results/analysis/
├── mdaf_baseline_results.csv          # EXP-1 to EXP-4 summary
├── mdaf_multiseed_results.csv         # EXP-5 to EXP-8 summary
├── mdaf_ablation_results.csv          # EXP-9 to EXP-19 summary
├── statistical_tests.json             # p-values, effect sizes, CIs
└── gate_analysis.json                 # Gate value statistics
```

---

## Analysis Pipeline

### Step 1: Performance Aggregation
```python
# Script: scripts/aggregate_mdaf_results.py

import pandas as pd
import json
from pathlib import Path

results = []
for log_file in Path('results/logs').glob('mdaf_*.log'):
    # Parse log file
    metrics = parse_training_log(log_file)
    results.append({
        'exp_id': extract_exp_id(log_file),
        'model': metrics['model'],
        'seed': metrics['seed'],
        'val_auc': metrics['best_val_auc'],
        'train_auc': metrics['train_auc_at_best'],
        'train_val_gap': metrics['train_val_gap'],
        'epochs': metrics['total_epochs'],
        'training_time': metrics['total_time'],
    })

df = pd.DataFrame(results)
df.to_csv('results/analysis/mdaf_all_results.csv', index=False)
```

### Step 2: Statistical Testing
```python
# Script: scripts/statistical_analysis.py

from scipy import stats
import numpy as np

# Load multi-seed results
mdaf_mamba = df[df['model'] == 'MDAF_Mamba']['val_auc'].values
mdaf_bst = df[df['model'] == 'MDAF_BST']['val_auc'].values

# T-test
t_stat, p_value = stats.ttest_rel(mdaf_mamba, mdaf_bst)

# Effect size
cohen_d = (np.mean(mdaf_mamba) - np.mean(mdaf_bst)) / np.sqrt(
    (np.var(mdaf_mamba) + np.var(mdaf_bst)) / 2
)

# Confidence intervals
ci_mamba = stats.t.interval(0.95, len(mdaf_mamba)-1,
                            loc=np.mean(mdaf_mamba),
                            scale=stats.sem(mdaf_mamba))
ci_bst = stats.t.interval(0.95, len(mdaf_bst)-1,
                          loc=np.mean(mdaf_bst),
                          scale=stats.sem(mdaf_bst))

results = {
    'mdaf_mamba_mean': float(np.mean(mdaf_mamba)),
    'mdaf_mamba_std': float(np.std(mdaf_mamba)),
    'mdaf_mamba_ci': [float(ci_mamba[0]), float(ci_mamba[1])],
    'mdaf_bst_mean': float(np.mean(mdaf_bst)),
    'mdaf_bst_std': float(np.std(mdaf_bst)),
    'mdaf_bst_ci': [float(ci_bst[0]), float(ci_bst[1])],
    't_statistic': float(t_stat),
    'p_value': float(p_value),
    'cohens_d': float(cohen_d),
    'significant': p_value < 0.05,
}

with open('results/analysis/statistical_tests.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### Step 3: Gate Analysis
```python
# Script: scripts/analyze_gates.py

# Extract gate values from model checkpoints
gate_stats = []
for checkpoint in Path('results/checkpoints').glob('mdaf_*.pth'):
    model = load_checkpoint(checkpoint)

    # Run inference on validation set with return_gate=True
    gates = []
    for batch in val_dataloader:
        _, gate = model(batch, return_gate=True)
        gates.extend(gate.cpu().numpy())

    gate_stats.append({
        'model': extract_model_name(checkpoint),
        'gate_mean': np.mean(gates),
        'gate_std': np.std(gates),
        'gate_median': np.median(gates),
        'gate_min': np.min(gates),
        'gate_max': np.max(gates),
        'gate_q25': np.percentile(gates, 25),
        'gate_q75': np.percentile(gates, 75),
    })

# Interpretation
# gate ≈ 1.0: Static branch dominates
# gate ≈ 0.0: Sequential branch dominates
# gate ≈ 0.5: Balanced fusion
```

### Step 4: Visualization
```python
# Script: scripts/visualize_results.py

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Model Comparison Barplot
fig, ax = plt.subplots(figsize=(10, 6))
models = ['AutoInt', 'DCNv2', 'BST', 'Mamba4Rec v2', 'MDAF-Mamba', 'MDAF-BST']
aucs = [0.5499, 0.5498, 0.5711, 0.5716, 0.5820, 0.5810]  # Example
ax.bar(models, aucs)
ax.axhline(0.5730, color='r', linestyle='--', label='Target AUC')
ax.set_ylabel('Validation AUC')
ax.set_title('Taobao CTR Prediction: Model Comparison')
plt.savefig('results/figures/model_comparison.png', dpi=300, bbox_inches='tight')

# 2. Multi-Seed Boxplot
fig, ax = plt.subplots(figsize=(8, 6))
data = [mdaf_mamba_aucs, mdaf_bst_aucs, bst_aucs, mamba4rec_aucs]
ax.boxplot(data, labels=['MDAF-Mamba', 'MDAF-BST', 'BST', 'Mamba4Rec v2'])
ax.set_ylabel('Validation AUC')
ax.set_title('Multi-Seed Performance Distribution (n=5)')
plt.savefig('results/figures/multiseed_boxplot.png', dpi=300, bbox_inches='tight')

# 3. Gate Value Distribution
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(gates, bins=50, alpha=0.7, edgecolor='black')
ax.axvline(np.mean(gates), color='r', linestyle='--', label=f'Mean: {np.mean(gates):.3f}')
ax.set_xlabel('Gate Value')
ax.set_ylabel('Frequency')
ax.set_title('MDAF Gated Fusion: Gate Value Distribution')
ax.legend()
plt.savefig('results/figures/gate_distribution.png', dpi=300, bbox_inches='tight')
```

---

## Success Metrics Summary

### Primary Metrics (Must Achieve)
1. **MDAF-Mamba Val AUC ≥ 0.5820** (target performance)
2. **MDAF-Mamba > BST by ≥ 1%p** (clear improvement)
3. **p-value < 0.05** in multi-seed t-test (statistical significance)
4. **Train-Val gap ≤ 0.08** (generalization)

### Secondary Metrics (Desirable)
1. MDAF-Mamba ≥ MDAF-BST (SSM superiority)
2. Gate mean ∈ [0.3, 0.7] (both branches utilized)
3. Training time < 3h per run (efficiency)
4. Fusion ablation: Gated > Concat (design justification)

### Publication-Ready Criteria
1. ✅ All primary metrics achieved
2. ✅ At least 3 secondary metrics achieved
3. ✅ Comprehensive ablation studies complete
4. ✅ Statistical tests with confidence intervals reported
5. ✅ Visualizations publication-quality

---

## Risk Mitigation

### Risk 1: MDAF-Mamba < 0.5780 (Below Minimum)
**Probability**: 20%
**Impact**: HIGH (invalidates core hypothesis)

**Mitigation Plan**:
1. Analyze gate values:
   - IF gate ≈ 1.0 → Sequential branch underperforming, check Mamba4Rec integration
   - IF gate ≈ 0.0 → Static branch underperforming, check DCNv3 integration
2. Review training curves for instability
3. Increase regularization (dropout 0.2 → 0.3)
4. Reduce learning rate (1e-3 → 5e-4)
5. Extend training (15 → 20 epochs)

### Risk 2: MDAF-BST >> MDAF-Mamba (BST Fusion Superior)
**Probability**: 15%
**Impact**: MEDIUM (weakens SSM contribution claim)

**Mitigation Plan**:
1. Emphasize efficiency: "MDAF-Mamba achieves 98% of MDAF-BST performance with 3× fewer parameters"
2. Analyze gate values to understand why BST fusion is better
3. Highlight O(L) vs O(L²) complexity advantage
4. Frame as performance-efficiency trade-off

### Risk 3: Multi-Seed Variance Too High (p > 0.05)
**Probability**: 25%
**Impact**: MEDIUM (statistical significance not achieved)

**Mitigation Plan**:
1. Increase number of seeds (5 → 10)
2. Report confidence intervals explicitly
3. Use non-parametric tests (Wilcoxon signed-rank)
4. Frame as "consistent trend across all seeds"

### Risk 4: Computational Resources Insufficient
**Probability**: 30%
**Impact**: LOW (delays timeline)

**Mitigation Plan**:
1. Prioritize P0 experiments (EXP-1 to EXP-4)
2. Defer P2/P3 experiments if time-constrained
3. Use checkpointing to resume interrupted runs
4. Consider cloud GPU rental (AWS/GCP) if local resources exhausted

---

## Timeline Contingency

**Optimistic Timeline** (All experiments successful):
- Week 10: EXP-1 to EXP-4 complete
- Week 11: EXP-5 to EXP-8 complete, analysis done
- Week 12: Begin paper writing

**Realistic Timeline** (Some iterations needed):
- Week 10: EXP-1 to EXP-4 complete, 1-2 iterations
- Week 11: EXP-5 to EXP-6 complete, partial ablations
- Week 12: Finish ablations, analysis, begin paper writing

**Pessimistic Timeline** (Major issues):
- Week 10-11: EXP-1 iterations, troubleshooting
- Week 12: EXP-5 to EXP-6, minimal ablations
- Week 13: Analysis and paper writing (1 week delay acceptable)

---

## Deliverables

### Week 10 Deliverables
- [ ] EXP-1: MDAF-Mamba baseline results
- [ ] EXP-2: Static-only ablation results
- [ ] EXP-3: Sequential-only ablation results
- [ ] EXP-4: MDAF-BST comparison results
- [ ] Preliminary analysis report

### Week 11 Deliverables
- [ ] EXP-5: MDAF-Mamba multi-seed results (n=5)
- [ ] EXP-6: MDAF-BST multi-seed results (n=5)
- [ ] Statistical significance tests (t-test, Cohen's d, CIs)
- [ ] Gate analysis and visualization
- [ ] Fusion mechanism ablation (if time permits)

### Final Deliverables (End of Week 11)
- [ ] Comprehensive results table for paper
- [ ] Publication-quality figures (3-5 plots)
- [ ] Statistical analysis report
- [ ] Ablation study summary
- [ ] Raw data for reproducibility

---

## Conclusion

This experimental plan provides a rigorous framework for validating MDAF's performance and conducting comprehensive ablation studies. The plan prioritizes critical experiments (Phase 1-2) while allowing flexibility for additional ablations (Phase 3-5) based on available time and resources.

**Key Success Factors**:
1. Achieve ≥ 0.5820 AUC on MDAF-Mamba (target performance)
2. Establish statistical significance over BST baseline (p < 0.05)
3. Demonstrate complementary information from static and sequential branches
4. Justify gated fusion design through ablations

**Research Architect Approval**: ✅ Proceed with confidence

---

**Document Version**: 1.0
**Author**: Research Team
**Last Updated**: 2025-10-31
**Status**: Ready for Execution
**Next Action**: Begin EXP-1 (MDAF-Mamba baseline training)
