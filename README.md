# MDAF: Mamba-DCN with Adaptive Fusion for CTR Prediction

Research project implementing a novel hybrid architecture combining DCNv3 (static feature interaction) and Mamba4Rec (sequential modeling) with an adaptive fusion gate for Click-Through Rate (CTR) prediction.

## Overview

**MDAF** (Mamba-DCN with Adaptive Fusion) is a hybrid CTR prediction model that bridges the gap between static feature-based and sequential modeling paradigms. By combining Deep Cross Network v3 (DCNv3) for explicit static feature interactions and Mamba4Rec for efficient sequential modeling, MDAF achieves superior performance with a learnable adaptive fusion mechanism.

## Key Contributions

1. **Novel Hybrid Architecture**: First framework to combine DCNv3 and Mamba4Rec for CTR prediction
2. **Adaptive Fusion Mechanism**: Sample-dependent gating that dynamically weights static and sequential branches
3. **Strong Empirical Results**: Achieved **0.6007 Val AUC** on Taobao dataset (+5.2% vs BST baseline, 3× fewer parameters)
4. **Interpretability**: Gate analysis reveals dataset-specific signal characteristics (83% static, 17% sequential on Taobao)

## Performance Results

### Taobao User Behavior Dataset

| Model | Val AUC | Test AUC | Parameters | Improvement vs BST |
|-------|---------|----------|------------|-------------------|
| BST (Baseline) | 0.5711 | 0.5698 | 130M | — |
| AutoInt | 0.5655 | 0.5648 | 23M | -56bp |
| DCNv2 | 0.5602 | 0.5594 | 23M | -109bp |
| **MDAF (Ours)** | **0.6007** | **0.5992** | **46M** | **+296bp (+5.2%)** |

### Key Findings

- **Hybrid superiority**: Outperforms both static-only (AutoInt, DCNv2) and sequential-only (BST) models
- **Parameter efficiency**: 3× fewer parameters than BST (46M vs 130M)
- **Adaptive fusion benefit**: +239bp improvement over simple concatenation
- **Gate insights**: 83% weight to static features, 17% to sequential (reflecting Taobao's weak sequential signals)

## Model Architecture

```
MDAF Framework:
┌─────────────────────┐
│  Static Features    │
│  (user, item, cat)  │
└──────────┬──────────┘
           │
    ┌──────▼──────┐
    │   DCNv3     │
    │  (LCN+ECN)  │
    └──────┬──────┘
           │
      h_static
           │
           ├──────────────────┐
           │                  │
    ┌──────▼────────┐  ┌─────▼──────┐
    │ Adaptive Gate │  │Sequential  │
    │   MLP + σ     │  │  Features  │
    │      g        │  │(item seq)  │
    └──────┬────────┘  └─────┬──────┘
           │                 │
           │           ┌─────▼──────┐
           │           │ Mamba4Rec  │
           │           │(State SSM) │
           │           └─────┬──────┘
           │                 │
           │             h_seq
           │                 │
           └────────┬────────┘
                    │
         h_fusion = (1-g)*h_static + g*h_seq
                    │
              ┌─────▼─────┐
              │   MLP     │
              │ Predictor │
              └─────┬─────┘
                    │
                 ŷ ∈ [0,1]
```

## Project Structure

```
.
├── models/
│   ├── baselines/           # Baseline implementations
│   │   ├── autoint.py
│   │   ├── dcnv2.py
│   │   └── bst.py
│   └── mdaf/               # MDAF implementation
│       ├── dcnv3.py        # Static branch
│       ├── mamba4rec.py    # Sequential branch
│       └── mdaf_mamba.py   # Full MDAF model
├── experiments/
│   ├── train_autoint_taobao.py
│   ├── train_dcnv2_taobao.py
│   ├── train_bst.py
│   └── train_mdaf_taobao*.py
├── utils/
│   ├── taobao_dataset.py   # Dataset loader
│   └── metrics.py          # Evaluation metrics
├── results/
│   └── checkpoints/        # Model checkpoints
└── docs/
    ├── MDAF_paper_complete.md      # Full paper (English)
    ├── MDAF_paper_complete_KR.md   # Full paper (Korean)
    ├── MDAF_paper_complete_KR.docx # DOCX version
    └── mdaf_results/
        └── MDAF_결과_요약.txt      # Experimental results
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset

**Taobao User Behavior Dataset**
- Source: Alibaba Taobao e-commerce platform
- Features: User ID, item ID, category ID, behavior sequence
- Task: Binary classification (click/no-click)
- Preprocessing: Category filtering applied (473,044 training samples)

## Usage

### Training Baselines

```bash
# Train AutoInt baseline
python experiments/train_autoint_taobao.py

# Train DCNv2 baseline
python experiments/train_dcnv2_taobao.py

# Train BST baseline
python experiments/train_bst.py
```

### Training MDAF

```bash
# Train MDAF with best hyperparameters
python experiments/train_mdaf_taobao_phase3.py
```

## Hyperparameters (Best Configuration)

| Parameter | Value |
|-----------|-------|
| Embedding dim | 32 |
| DCNv3 layers | 3 |
| Mamba layers | 2 |
| Gate dim | 64 |
| Dropout | 0.15 |
| Learning rate | 0.0005 |
| Weight decay | 1e-5 |
| Batch size | 512 |
| Max sequence length | 50 |

## Paper

The complete research paper is available in multiple formats:

- **English**: [docs/MDAF_paper_complete.md](docs/MDAF_paper_complete.md)
- **Korean**: [docs/MDAF_paper_complete_KR.md](docs/MDAF_paper_complete_KR.md)
- **DOCX**: [docs/MDAF_paper_complete_KR.docx](docs/MDAF_paper_complete_KR.docx)

The paper includes:
- Complete methodology and architecture details
- Comprehensive experimental results with 16 tables
- Ablation studies analyzing each component
- Gate analysis and interpretability insights
- Honest discussion of limitations and future work

## Requirements

- Python 3.8+
- PyTorch 2.0+
- pandas
- numpy
- scikit-learn
- tqdm
- deepctr-torch

## Citation

If you use this code or reference this work, please cite:

```bibtex
@article{mdaf2025,
  title={MDAF: Mamba-DCN with Adaptive Fusion for Click-Through Rate Prediction},
  author={Choi, Jinho},
  journal={Research Report},
  year={2025}
}
```

## References

1. **Mamba4Rec**: Chengkai Liu et al. "Mamba4Rec: Towards Efficient Sequential Recommendation with Selective State Space Models" (2024)
2. **DCNv3**: Ruoxi Wang et al. "DCN V2: Improved Deep & Cross Network" (2021)
3. **BST**: Qiwei Chen et al. "Behavior Sequence Transformer for E-commerce Recommendation in Alibaba" (2019)
4. **Mamba**: Albert Gu & Tri Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)

## License

MIT License

## Contact

For questions or collaborations, please open an issue in this repository.

---

**Status**: ✅ Research Complete | Paper Submitted | November 2025
