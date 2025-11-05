# MDAF: Multi-branch Dual Attention Fusion

## Overview

MDAF is a hybrid CTR prediction model that combines static feature modeling (DCNv3) with sequential behavioral modeling (Mamba4Rec or BST) through an adaptive gated fusion mechanism.

## Architecture

```
Input Features
├── Static Features (target_item, target_category, user_id, hour, dayofweek)
│   └→ DCNv3 → static_embedding (128-dim)
│
├── Sequential Features (item_history[50], category_history[50])
│   ├→ Mamba4Rec → sequential_embedding (128-dim)  [MDAF-Mamba]
│   └→ BST → sequential_embedding (128-dim)        [MDAF-BST]
│
└→ Gated Fusion → MLP Prediction Head → click_probability
```

## Components

### Core Models
- **`mdaf_mamba.py`**: MDAF with Mamba4Rec sequential branch (46M params)
- **`mdaf_bst.py`**: MDAF with BST sequential branch (133M params)
- **`mdaf_components.py`**: Shared GatedFusion and PredictionHead modules

### Branch Models
- **`dcnv3.py`**: Deep Cross Network v3 for static features (modified with `return_embedding=True`)
- **`mamba4rec.py`**: Mamba-based sequence model (modified with `return_embedding=True`)
- **`bst.py`**: Behavior Sequence Transformer (in `models/baseline/`, modified)

## Usage

### Quick Test
```bash
# Run all validation tests
./scripts/quick_test_mdaf.sh
```

### Training

**MDAF-Mamba:**
```bash
./venv/bin/python experiments/train_mdaf_taobao.py \
    --model mamba \
    --epochs 15 \
    --batch_size 2048 \
    --lr 1e-3
```

**MDAF-BST:**
```bash
./venv/bin/python experiments/train_mdaf_taobao.py \
    --model bst \
    --epochs 15 \
    --batch_size 2048 \
    --lr 1e-3
```

### Python API

```python
from models.mdaf.mdaf_mamba import MDAF_Mamba
from models.mdaf.mdaf_bst import MDAF_BST

# Create MDAF-Mamba model
model = MDAF_Mamba(
    item_vocab_size=335164,
    category_vocab_size=5480,
    user_vocab_size=577482,
    embedding_dim=128
)

# Forward pass
predictions, gates = model(
    target_item=target_items,
    target_category=target_categories,
    item_history=item_histories,
    category_history=category_histories,
    other_features=other_features,
    return_gate=True
)

# Gate interpretation:
# - gate ≈ 1.0: Model relies on static features (DCNv3)
# - gate ≈ 0.0: Model relies on sequential features (Mamba4Rec)
# - gate ≈ 0.5: Model balances both branches
```

## Key Features

1. **Adaptive Fusion**: Learned gating mechanism automatically balances static and sequential information
2. **Fair Comparison**: MDAF-Mamba and MDAF-BST use identical fusion for controlled ablation
3. **Interpretability**: Gate values provide insight into model decision-making
4. **Production-Ready**: Comprehensive logging, checkpointing, and early stopping

## Model Comparison

| Model | Parameters | Embedding Dim | Sequential Branch |
|-------|-----------|---------------|-------------------|
| MDAF-Mamba | 46M | 128 | Mamba4Rec (SSM) |
| MDAF-BST | 133M | 128 | BST (Transformer) |

## Configuration

Default hyperparameters (in `experiments/train_mdaf_taobao.py`):

```python
{
    'batch_size': 2048,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'max_epochs': 15,
    'early_stopping_patience': 5,
    'embedding_dim': 128,
    'prediction_hidden_dims': [128, 64],
    'dropout': 0.2,

    # DCNv3 branch
    'dcnv3_embed_dim': 16,
    'dcnv3_lcn_layers': 3,
    'dcnv3_ecn_layers': 3,

    # Mamba4Rec branch
    'mamba_hidden_dim': 128,
    'mamba_num_layers': 2,

    # BST branch
    'bst_embed_dim': 128,
    'bst_num_transformer_layers': 2,
    'bst_num_heads': 4,
}
```

## Testing

### Unit Tests
```bash
# Test components
python -m models.mdaf.mdaf_components

# Test MDAF-Mamba
python -m models.mdaf.mdaf_mamba

# Test MDAF-BST
python -m models.mdaf.mdaf_bst
```

### Integration Test
```bash
# Test with real Taobao data
python tests/test_mdaf_integration.py
```

## Expected Performance

Based on baseline results (from `docs/project_status.md`):

| Model | Validation AUC | Training Time |
|-------|----------------|---------------|
| BST (baseline) | 0.7295 | ~2h |
| Mamba4Rec v2 (baseline) | 0.7414 | ~1.5h |
| **MDAF-Mamba (target)** | **> 0.75** | ~2-3h |
| **MDAF-BST (target)** | **> 0.75** | ~4-5h |

## File Outputs

- **Checkpoints:** `results/checkpoints/mdaf_{mamba|bst}_taobao_best.pth`
- **Logs:** `results/logs/mdaf_{mamba|bst}_taobao_*.log`

## Implementation Status

- ✅ MDAF-Mamba implementation complete
- ✅ MDAF-BST implementation complete
- ✅ Shared fusion components tested
- ✅ Training script ready
- ✅ Integration tests passing
- ⏳ Training in progress (Week 10, Day 2-3)

## References

- **DCNv3:** [arXiv:2407.13349](https://arxiv.org/abs/2407.13349)
- **Mamba:** Selective State Space Models
- **BST:** Behavior Sequence Transformer (Alibaba, 2019)

## Contact

For questions or issues, refer to `docs/mdaf_implementation_report.md` for detailed implementation notes.
