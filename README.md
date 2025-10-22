# CTR Prediction Research

Click-Through Rate (CTR) prediction research project with focus on Multi-Domain Attention Fusion (MDAF).

## Project Overview

This project implements and evaluates various deep learning models for CTR prediction on the Taobao Ad Click dataset.

## Models Implemented

### Baseline Models
- **DeepFM**: Combines factorization machines with deep neural networks
- **AutoInt**: Automatic feature interaction learning via self-attention
- **DCNv2**: Deep & Cross Network V2
- **BST**: Behavior Sequence Transformer (In progress)

### Target Model
- **MDAF**: Multi-Domain Attention Fusion (To be implemented)

## Dataset

**Taobao Ad Click Dataset**
- Source: Alibaba Taobao advertising platform
- Features: User profiles, item features, behavior sequences
- Task: Binary classification (click/no-click)

## Project Structure

```
.
├── data/
│   ├── raw/              # Raw dataset files
│   └── processed/        # Preprocessed data
├── models/
│   ├── baseline/         # Baseline model implementations
│   └── mdaf/            # MDAF model (to be implemented)
├── utils/
│   ├── taobao_dataset.py    # Dataset loader
│   └── metrics.py           # Evaluation metrics
├── scripts/
│   └── preprocess_taobao_ads.py  # Data preprocessing
├── experiments/
│   ├── train_*.py       # Training scripts
│   └── debug_*.py       # Debugging tools
├── results/
│   └── bst_analysis_report.md   # Analysis reports
└── docs/
    └── research_roadmap.md      # Research plan
```

## Performance Results

| Model | AUC | Status |
|-------|-----|--------|
| AutoInt | 0.7802 | ✅ Complete |
| DCNv2 | 0.7722 | ✅ Complete |
| DeepFM | - | ✅ Complete |
| BST | 0.5711 | 🔧 Debugging |
| MDAF | - | 📋 Planned |

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Preprocessing
```bash
python scripts/preprocess_taobao_ads.py
```

### Training
```bash
# Train specific model
python experiments/train_autoint.py
python experiments/train_dcnv2.py
python experiments/train_bst.py
```

### Debugging
```bash
# Debug BST embeddings
python experiments/debug_bst_embeddings.py
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- pandas
- numpy
- scikit-learn
- tqdm

## Research Progress

See [docs/research_roadmap.md](docs/research_roadmap.md) for detailed research plan and progress.

## Current Issues

- BST model underperforming (~0.21 AUC gap compared to other baselines)
- Investigating category embedding integration
- See [results/bst_analysis_report.md](results/bst_analysis_report.md) for detailed analysis

## License

MIT License

## References

- BST: Chen et al. "Behavior Sequence Transformer for E-commerce Recommendation in Alibaba" (2019)
- AutoInt: Song et al. "AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks" (2019)
- DCNv2: Wang et al. "DCN V2: Improved Deep & Cross Network" (2021)
