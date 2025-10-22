# BST Performance Analysis Report

## Problem Summary
BST model was significantly underperforming compared to other baseline models:
- **AutoInt**: AUC 0.7802
- **DCNv2**: AUC 0.7722  
- **BST (original)**: AUC 0.5667 ❌

Performance gap: **~0.21 AUC difference**

## Root Cause Investigation

### Issue Discovered
The BST model was **not using category information** at all:
1. Dataset provided `target_category` but BST had comment "현재 미사용" (currently unused)
2. No `category_history` existed in the dataset (only `item_history`)
3. Other models (DeepFM, AutoInt, DCNv2) used all categorical features including categories

### Why This Matters
According to the BST paper (Alibaba, 2019), the model should use **item + category embeddings** (element-wise sum) for both target and history sequences. Category information provides crucial context for recommendation.

## Solution Implemented

### Full Category Integration (Option 2)
1. **Data Preprocessing**: Modified `preprocess_taobao_ads.py`
   - Generated `category_history` alongside `item_history`
   - Maintained proper sequence alignment and padding

2. **Dataset**: Updated `TaobaoDataset` 
   - Added `category_history` to data loading
   - Modified `__getitem__` and `collate_fn`

3. **Model**: Modified BST architecture
   - Item embedding + Category embedding (element-wise sum)
   - Applied to both target and history sequences
   - Proper position encoding and masking

4. **Training**: Updated training pipeline
   - Pass `category_history` to model
   - Both `train_epoch` and `evaluate` functions updated

## Results

### Before (No Category)
```
Validation AUC: 0.5667
```

### After (Full Category Integration)
```
Final Best Validation AUC: 0.5711
Improvement: +0.0044 (0.78%)
```

## Critical Issue: Performance Still Too Low

### Expected vs Actual
- **Expected improvement**: +0.10 to +0.15 (targeting AUC ~0.70-0.75)
- **Actual improvement**: +0.0044 (only 0.78%)
- **Current performance**: Still **0.21 AUC below** other models

### Comparison with Other Models
| Model | AUC | Gap from Best |
|-------|-----|---------------|
| AutoInt | 0.7802 | - |
| DCNv2 | 0.7722 | -0.0080 |
| **BST (with category)** | **0.5711** | **-0.2091** ❌ |

## Potential Root Causes for Low Performance

### 1. **Training Configuration Issues**
- Learning rate may be too high/low for BST
- Batch size (512) may not be optimal  
- Only 5 epochs - may need more training
- Optimizer settings (Adam with default params)

### 2. **Model Architecture Mismatch**
- Current implementation uses 1 Transformer layer (paper recommends b=1 as optimal)
- DNN layers: [256, 128] - may need different configuration
- Embedding dimension: 64 - other models may use different sizes
- Number of attention heads: 2 - may need tuning

### 3. **Data Quality/Feature Engineering**
- Position embedding strategy (learnable vs sinusoidal vs time-based)
- Sequence padding approach
- Feature normalization/scaling issues
- Max sequence length (50) may be too short/long

### 4. **Implementation Bugs**
- Element-wise sum of item + category embeddings may have issues
- Target attention mechanism may not be working correctly
- Padding mask logic
- Position embedding not properly applied

## Next Steps Recommendations

### Immediate Actions
1. **Debug embedding learning**: 
   - Check if category embeddings are actually being updated
   - Verify gradients are flowing through category path
   - Inspect embedding weight magnitudes

2. **Compare with paper implementation**:
   - Verify target item should be prepended to sequence
   - Check if target position output (index 0) is correctly extracted
   - Validate attention mechanism

3. **Training diagnostics**:
   - Plot training/validation curves
   - Check for overfitting/underfitting
   - Monitor gradient norms
   - Try longer training (20+ epochs)

### Hyperparameter Tuning
- Learning rate: Try [0.0001, 0.0005, 0.001, 0.005]
- Batch size: Try [256, 512, 1024]
- Embedding dim: Try [32, 64, 128]
- DNN hidden units: Try [[256, 128, 64], [512, 256], [128, 64]]

### Alternative Approaches
1. Use only target item category (simpler baseline)
2. Try concat instead of sum for item + category
3. Add more features (user features, temporal features)
4. Compare with reference BST implementation

## Files Modified

### Data Processing
- `scripts/preprocess_taobao_ads.py`: Added category_history generation

### Dataset
- `utils/taobao_dataset.py`: Added category_history loading

### Model
- `models/baseline/bst_fixed.py`: Implemented item + category embedding fusion

### Training
- `experiments/train_bst.py`: Updated to pass category_history

## Training Logs
- Original: `results/train_bst_fixed.log` (AUC: 0.5667)
- With categories: `results/train_bst_category.log` (AUC: 0.5711)

## Conclusion

While we successfully implemented the paper-accurate BST architecture with full category integration, the performance improvement was **minimal** (+0.0044 instead of expected +0.10-0.15). 

The BST model is still **significantly underperforming** (AUC 0.5711 vs 0.78 for other models), suggesting there may be:
1. **Fundamental implementation issues** that need debugging
2. **Training configuration problems** that need tuning
3. **Model architecture mismatches** with the dataset characteristics

**Priority**: Deep debugging and comparison with reference implementations is needed before this model can be considered a valid baseline.
