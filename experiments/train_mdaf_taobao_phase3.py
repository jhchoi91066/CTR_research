#!/usr/bin/env python3
"""
MDAF Training Script for Taobao Dataset - Phase 3 Enhanced Regularization

Phase 3 adds:
- Configurable dropout, weight_decay, label_smoothing, gradient clipping
- Warmup + Cosine Annealing LR scheduler
- Enhanced logging with per-epoch CSV output
- Full checkpoint saving for learning curve analysis

Usage:
    ./venv/bin/python experiments/train_mdaf_taobao_phase3.py \
        --model mamba \
        --epochs 15 \
        --batch_size 2048 \
        --lr 3e-4 \
        --dropout 0.3 \
        --weight_decay 5e-5 \
        --label_smoothing 0.1 \
        --grad_clip_norm 1.0 \
        --warmup_epochs 2 \
        --seed 42
"""

import argparse
import json
import pickle
import logging
import sys
import csv
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from sklearn.metrics import roc_auc_score, log_loss
import numpy as np

from models.mdaf.mdaf_mamba import MDAF_Mamba
from models.mdaf.mdaf_bst import MDAF_BST
from utils.taobao_dataset import TaobaoDataset, collate_fn


# ==================== Label Smoothing BCE Loss ====================

class BCEWithLabelSmoothingLoss(nn.Module):
    """Binary Cross Entropy with Label Smoothing"""
    def __init__(self, smoothing=0.0):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        """
        pred: (N,) predicted probabilities in [0, 1]
        target: (N,) true labels in {0, 1}
        """
        if self.smoothing > 0:
            # Smooth labels: 1 -> 1-smoothing, 0 -> smoothing
            target = target * (1 - self.smoothing) + 0.5 * self.smoothing

        # BCE loss
        loss = -target * torch.log(pred + 1e-7) - (1 - target) * torch.log(1 - pred + 1e-7)
        return loss.mean()


# ==================== Learning Rate Scheduler with Warmup ====================

def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    """
    Create LR scheduler with linear warmup followed by cosine annealing

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs
        steps_per_epoch: Number of optimization steps per epoch
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine annealing
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ==================== Setup Logging ====================

def setup_logging(model_name, output_dir, phase_name="phase3"):
    """Setup logging to file and console"""
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{model_name}_{phase_name}_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__), log_file


# ==================== Training Functions ====================

def train_epoch(model, dataloader, optimizer, criterion, device, logger, grad_clip_norm=1.0, scheduler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch_idx, batch_data in enumerate(dataloader):
        # Unpack batch
        target_items, target_categories, item_histories, category_histories, other_features, labels = batch_data

        # Move to device
        target_items = target_items.to(device)
        target_categories = target_categories.to(device)
        item_histories = item_histories.to(device)
        category_histories = category_histories.to(device)
        other_features = {k: v.to(device) for k, v in other_features.items()}
        labels = labels.to(device)

        # Forward pass
        predictions = model(
            target_item=target_items,
            target_category=target_categories,
            item_history=item_histories,
            category_history=category_histories,
            other_features=other_features,
            return_gate=False
        )

        # Compute loss
        loss = criterion(predictions, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        optimizer.step()

        # Step scheduler (if per-step scheduling)
        if scheduler is not None:
            scheduler.step()

        # Track metrics
        total_loss += loss.item()
        all_preds.extend(predictions.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Log progress
        if (batch_idx + 1) % 200 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f'   Batch [{batch_idx+1}/{len(dataloader)}] Loss: {loss.item():.4f}, LR: {current_lr:.2e}')

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_preds)
    logloss = log_loss(all_labels, all_preds)

    return avg_loss, auc, logloss


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, logger, return_gates=False):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_gates = []

    for batch_data in dataloader:
        # Unpack batch
        target_items, target_categories, item_histories, category_histories, other_features, labels = batch_data

        # Move to device
        target_items = target_items.to(device)
        target_categories = target_categories.to(device)
        item_histories = item_histories.to(device)
        category_histories = category_histories.to(device)
        other_features = {k: v.to(device) for k, v in other_features.items()}
        labels = labels.to(device)

        # Forward pass
        if return_gates:
            predictions, gates = model(
                target_item=target_items,
                target_category=target_categories,
                item_history=item_histories,
                category_history=category_histories,
                other_features=other_features,
                return_gate=True
            )
            all_gates.extend(gates.cpu().numpy())
        else:
            predictions = model(
                target_item=target_items,
                target_category=target_categories,
                item_history=item_histories,
                category_history=category_histories,
                other_features=other_features,
                return_gate=False
            )

        # Compute loss
        loss = criterion(predictions, labels)

        # Track metrics
        total_loss += loss.item()
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_preds)
    logloss = log_loss(all_labels, all_preds)

    if return_gates:
        gate_stats = {
            'mean': np.mean(all_gates),
            'std': np.std(all_gates),
            'min': np.min(all_gates),
            'max': np.max(all_gates),
            'median': np.median(all_gates)
        }
        return avg_loss, auc, logloss, gate_stats

    return avg_loss, auc, logloss


# ==================== Main Training Loop ====================

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train MDAF models on Taobao dataset - Phase 3')
    parser.add_argument('--model', type=str, required=True, choices=['mamba', 'bst'],
                       help='Sequential branch type: mamba or bst')
    parser.add_argument('--epochs', type=int, default=15, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay (L2 regularization)')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0, help='Gradient clipping max norm')
    parser.add_argument('--warmup_epochs', type=int, default=2, help='Number of warmup epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='auto', help='Device: auto, cpu, cuda, or mps')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Early stopping patience')
    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup paths
    data_dir = Path('data/processed/taobao')
    output_dir = Path('results')
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Model name
    model_name = f'mdaf_{args.model}_taobao'
    phase_name = f'phase3_seed{args.seed}'

    # Setup logging
    logger, log_file = setup_logging(model_name, output_dir, phase_name)
    logger.info("="*80)
    logger.info(f"PHASE 3: ENHANCED REGULARIZATION - {model_name.upper()}")
    logger.info("="*80)
    logger.info(f"\nConfiguration:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  Learning Rate: {args.lr}")
    logger.info(f"  Dropout: {args.dropout}")
    logger.info(f"  Weight Decay: {args.weight_decay}")
    logger.info(f"  Label Smoothing: {args.label_smoothing}")
    logger.info(f"  Gradient Clip Norm: {args.grad_clip_norm}")
    logger.info(f"  Warmup Epochs: {args.warmup_epochs}")
    logger.info(f"  Early Stopping Patience: {args.early_stopping_patience}")
    logger.info(f"  Random Seed: {args.seed}")

    # Load metadata
    logger.info("\n1. Loading Taobao metadata...")
    with open(data_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    logger.info(f"   Vocabulary sizes:")
    logger.info(f"      Items: {metadata['vocab_sizes']['item_id']:,}")
    logger.info(f"      Categories: {metadata['vocab_sizes']['category_id']:,}")
    logger.info(f"      Users: {metadata['vocab_sizes']['user_id']:,}")
    logger.info(f"   Max sequence length: {metadata['max_seq_len']}")

    # Device selection
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    logger.info(f"\n2. Device: {device}")

    # Create dataloaders
    logger.info("\n3. Creating dataloaders...")
    train_dataset = TaobaoDataset(data_dir / 'train.parquet', data_dir / 'metadata.pkl')
    val_dataset = TaobaoDataset(data_dir / 'val.parquet', data_dir / 'metadata.pkl')

    # Use multiprocessing for CPU, single process for MPS to avoid deadlocks
    if device.type == 'cpu':
        num_workers_train = 4
        num_workers_val = 2
        pin_memory = False
        persistent_workers = True
    else:  # mps
        num_workers_train = 0
        num_workers_val = 0
        pin_memory = False
        persistent_workers = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers_train,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers_train > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers_val,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers_val > 0 else False
    )

    logger.info(f"   Train samples: {len(train_dataset):,}")
    logger.info(f"   Val samples: {len(val_dataset):,}")
    logger.info(f"   Train batches: {len(train_loader)}")
    logger.info(f"   Val batches: {len(val_loader)}")

    # Create model
    logger.info(f"\n4. Creating {model_name.upper()} model...")

    # Default architecture config (from Phase 1)
    DEFAULT_CONFIG = {
        # Model architecture
        'embedding_dim': 128,
        'prediction_hidden_dims': [128, 64],

        # DCNv3 branch
        'dcnv3_embed_dim': 16,
        'dcnv3_lcn_layers': 3,
        'dcnv3_ecn_layers': 3,
        'dcnv3_dropout': 0.0,

        # Mamba4Rec branch
        'mamba_item_embed_dim': 64,
        'mamba_category_embed_dim': 32,
        'mamba_static_embed_dim': 16,
        'mamba_hidden_dim': 128,
        'mamba_num_layers': 2,
        'mamba_d_state': 16,
        'mamba_d_conv': 4,
        'mamba_expand': 2,

        # BST branch
        'bst_embed_dim': 128,
        'bst_max_seq_len': 50,
        'bst_num_transformer_layers': 2,
        'bst_num_heads': 4,
        'bst_d_ff': 256,
        'bst_dropout': 0.1,
    }

    if args.model == 'mamba':
        model = MDAF_Mamba(
            item_vocab_size=metadata['vocab_sizes']['item_id'],
            category_vocab_size=metadata['vocab_sizes']['category_id'],
            user_vocab_size=metadata['vocab_sizes']['user_id'],
            dcnv3_embed_dim=DEFAULT_CONFIG['dcnv3_embed_dim'],
            dcnv3_lcn_layers=DEFAULT_CONFIG['dcnv3_lcn_layers'],
            dcnv3_ecn_layers=DEFAULT_CONFIG['dcnv3_ecn_layers'],
            dcnv3_dropout=DEFAULT_CONFIG['dcnv3_dropout'],
            item_embed_dim=DEFAULT_CONFIG['mamba_item_embed_dim'],
            category_embed_dim=DEFAULT_CONFIG['mamba_category_embed_dim'],
            static_embed_dim=DEFAULT_CONFIG['mamba_static_embed_dim'],
            mamba_hidden_dim=DEFAULT_CONFIG['mamba_hidden_dim'],
            mamba_num_layers=DEFAULT_CONFIG['mamba_num_layers'],
            mamba_d_state=DEFAULT_CONFIG['mamba_d_state'],
            mamba_d_conv=DEFAULT_CONFIG['mamba_d_conv'],
            mamba_expand=DEFAULT_CONFIG['mamba_expand'],
            embedding_dim=DEFAULT_CONFIG['embedding_dim'],
            prediction_hidden_dims=DEFAULT_CONFIG['prediction_hidden_dims'],
            dropout=args.dropout  # Use Phase 3 dropout
        )
    else:  # bst
        model = MDAF_BST(
            item_vocab_size=metadata['vocab_sizes']['item_id'],
            category_vocab_size=metadata['vocab_sizes']['category_id'],
            user_vocab_size=metadata['vocab_sizes']['user_id'],
            dcnv3_embed_dim=DEFAULT_CONFIG['dcnv3_embed_dim'],
            dcnv3_lcn_layers=DEFAULT_CONFIG['dcnv3_lcn_layers'],
            dcnv3_ecn_layers=DEFAULT_CONFIG['dcnv3_ecn_layers'],
            dcnv3_dropout=DEFAULT_CONFIG['dcnv3_dropout'],
            bst_embed_dim=DEFAULT_CONFIG['bst_embed_dim'],
            bst_max_seq_len=DEFAULT_CONFIG['bst_max_seq_len'],
            bst_num_transformer_layers=DEFAULT_CONFIG['bst_num_transformer_layers'],
            bst_num_heads=DEFAULT_CONFIG['bst_num_heads'],
            bst_d_ff=DEFAULT_CONFIG['bst_d_ff'],
            bst_dropout=DEFAULT_CONFIG['bst_dropout'],
            embedding_dim=DEFAULT_CONFIG['embedding_dim'],
            prediction_hidden_dims=DEFAULT_CONFIG['prediction_hidden_dims'],
            dropout=args.dropout  # Use Phase 3 dropout
        )

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"   Total parameters: {total_params:,}")
    logger.info(f"   Trainable parameters: {trainable_params:,}")

    # Optimizer with Phase 3 weight decay
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler with warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        steps_per_epoch=len(train_loader)
    )

    # Loss function with label smoothing
    criterion = BCEWithLabelSmoothingLoss(smoothing=args.label_smoothing)

    logger.info(f"\n5. Training components:")
    logger.info(f"   Optimizer: Adam (lr={args.lr}, weight_decay={args.weight_decay})")
    logger.info(f"   LR Scheduler: Warmup({args.warmup_epochs} epochs) + CosineAnnealing")
    logger.info(f"   Loss: BCELoss + LabelSmoothing({args.label_smoothing})")
    logger.info(f"   Gradient Clipping: Max Norm = {args.grad_clip_norm}")

    # Prepare CSV logging
    csv_path = output_dir / f'{phase_name}_metrics.csv'
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'epoch', 'train_loss', 'train_auc', 'train_logloss',
        'val_loss', 'val_auc', 'val_logloss', 'train_val_gap',
        'gate_mean', 'gate_std', 'learning_rate'
    ])

    # Training loop
    logger.info("\n6. Starting training...")
    logger.info("="*80)

    best_val_auc = 0
    best_epoch = 0
    patience_counter = 0
    epoch_metrics = []

    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        logger.info("-"*80)

        # Train
        train_loss, train_auc, train_logloss = train_epoch(
            model, train_loader, optimizer, criterion, device, logger,
            grad_clip_norm=args.grad_clip_norm,
            scheduler=scheduler
        )

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        logger.info(f"\n  Train - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, LogLoss: {train_logloss:.4f}")

        # Validate
        val_loss, val_auc, val_logloss, gate_stats = evaluate(
            model, val_loader, criterion, device, logger, return_gates=True
        )

        train_val_gap = train_auc - val_auc

        logger.info(f"  Val   - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, LogLoss: {val_logloss:.4f}")
        logger.info(f"  Gap   - Train-Val AUC: {train_val_gap:+.4f}")
        logger.info(f"  Gate  - Mean: {gate_stats['mean']:.4f}, Std: {gate_stats['std']:.4f}, "
                   f"Range: [{gate_stats['min']:.4f}, {gate_stats['max']:.4f}]")
        logger.info(f"  LR    - {current_lr:.6e}")

        # Log to CSV
        csv_writer.writerow([
            epoch + 1, train_loss, train_auc, train_logloss,
            val_loss, val_auc, val_logloss, train_val_gap,
            gate_stats['mean'], gate_stats['std'], current_lr
        ])
        csv_file.flush()

        # Store epoch metrics
        epoch_metrics.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_auc': train_auc,
            'train_logloss': train_logloss,
            'val_loss': val_loss,
            'val_auc': val_auc,
            'val_logloss': val_logloss,
            'train_val_gap': train_val_gap,
            'gate_mean': gate_stats['mean'],
            'gate_std': gate_stats['std'],
            'lr': current_lr
        })

        # Save checkpoint every epoch (for learning curve analysis)
        checkpoint_path = checkpoint_dir / f'{model_name}_{phase_name}_epoch{epoch+1}.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_auc': train_auc,
            'val_auc': val_auc,
            'train_val_gap': train_val_gap,
            'gate_stats': gate_stats,
            'config': vars(args)
        }, checkpoint_path)

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            patience_counter = 0

            best_checkpoint_path = checkpoint_dir / f'{model_name}_{phase_name}_best.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_auc': train_auc,
                'val_auc': val_auc,
                'train_val_gap': train_val_gap,
                'gate_stats': gate_stats,
                'config': vars(args)
            }, best_checkpoint_path)

            logger.info(f"\n  ✓ New best model saved! Val AUC: {val_auc:.4f}")
        else:
            patience_counter += 1
            logger.info(f"\n  Patience: {patience_counter}/{args.early_stopping_patience}")

        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    csv_file.close()

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("PHASE 3 TRAINING COMPLETED")
    logger.info("="*80)
    logger.info(f"\nBest Performance:")
    logger.info(f"  Best Epoch: {best_epoch}")
    logger.info(f"  Best Val AUC: {best_val_auc:.4f}")

    # Get best epoch metrics
    best_metrics = epoch_metrics[best_epoch - 1]
    logger.info(f"  Train AUC at Best: {best_metrics['train_auc']:.4f}")
    logger.info(f"  Train-Val Gap at Best: {best_metrics['train_val_gap']:+.4f}")
    logger.info(f"  Gate Mean at Best: {best_metrics['gate_mean']:.4f}")

    logger.info(f"\nMetrics CSV saved to: {csv_path}")
    logger.info(f"Best checkpoint saved to: {checkpoint_dir / f'{model_name}_{phase_name}_best.pth'}")
    logger.info(f"Training log saved to: {log_file}")

    # Tier classification
    logger.info("\n" + "="*80)
    logger.info("TIER CLASSIFICATION")
    logger.info("="*80)

    tier1_val_auc = best_val_auc >= 0.578
    tier1_epoch = 3 <= best_epoch <= 6
    tier1_gap = 0.05 <= best_metrics['train_val_gap'] <= 0.12
    tier1_train_auc = best_metrics['train_auc'] <= 0.70

    logger.info(f"\nTier 1 Criteria (Green Light):")
    logger.info(f"  Best Val AUC >= 0.578:      {tier1_val_auc} ({best_val_auc:.4f})")
    logger.info(f"  Best Epoch in [3, 6]:       {tier1_epoch} ({best_epoch})")
    logger.info(f"  Train-Val Gap in [0.05, 0.12]: {tier1_gap} ({best_metrics['train_val_gap']:.4f})")
    logger.info(f"  Train AUC <= 0.70:          {tier1_train_auc} ({best_metrics['train_auc']:.4f})")

    tier2_val_auc = 0.570 <= best_val_auc < 0.578
    tier2_epoch = 2 <= best_epoch <= 7
    tier2_gap = best_metrics['train_val_gap'] < 0.15

    logger.info(f"\nTier 2 Criteria (Yellow Light):")
    logger.info(f"  Best Val AUC in [0.570, 0.578): {tier2_val_auc} ({best_val_auc:.4f})")
    logger.info(f"  Best Epoch in [2, 7]:        {tier2_epoch} ({best_epoch})")
    logger.info(f"  Train-Val Gap < 0.15:        {tier2_gap} ({best_metrics['train_val_gap']:.4f})")

    if all([tier1_val_auc, tier1_epoch, tier1_gap, tier1_train_auc]):
        tier = "TIER 1 - GREEN LIGHT"
        recommendation = "PROCEED TO PHASE 4 MULTI-SEED EVALUATION"
    elif all([tier2_val_auc, tier2_epoch, tier2_gap]):
        tier = "TIER 2 - YELLOW LIGHT"
        recommendation = "RUN ONE MORE DIAGNOSTIC ITERATION"
    else:
        tier = "TIER 3 - RED LIGHT"
        recommendation = "ESCALATE TO RESEARCH_ARCHITECT"

    logger.info(f"\n{'='*80}")
    logger.info(f"CLASSIFICATION: {tier}")
    logger.info(f"RECOMMENDATION: {recommendation}")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    main()
