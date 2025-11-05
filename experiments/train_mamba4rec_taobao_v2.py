"""
Train Mamba4Rec model on Taobao dataset - VERSION 2 WITH ENHANCED REGULARIZATION

Mamba4Rec v2: Addresses severe overfitting from v1 training
Target: Achieve publication-quality training dynamics with controlled generalization gap

V2 Improvements:
- Increased dropout (0.2 → 0.3)
- Increased weight decay (1e-5 → 5e-5)
- Label smoothing (0.05)
- Gradient clipping (max_norm=1.0)
- Warmup + cosine annealing LR schedule
- Extended training (20 epochs with patience=5)
- Enhanced monitoring and logging

Success Criteria:
- Train AUC at best epoch: 0.60-0.70 (must be <0.80)
- Val AUC at best epoch: ≥0.5730
- Train-Val gap at best epoch: ≤0.08
- No validation collapse pattern
"""

import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import logging
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, log_loss
from pathlib import Path
import pickle
import math

from models.mdaf.mamba4rec import Mamba4Rec
from utils.taobao_dataset import get_taobao_dataloader


class BCEWithLabelSmoothing(nn.Module):
    """
    Binary Cross-Entropy Loss with Label Smoothing

    Label smoothing prevents overconfident predictions by smoothing targets:
    - 0 → smoothing (e.g., 0.025)
    - 1 → 1 - smoothing (e.g., 0.975)

    This acts as a regularizer by preventing the model from becoming
    too confident about training examples.
    """

    def __init__(self, smoothing=0.05):
        """
        Args:
            smoothing: Label smoothing parameter (0 = no smoothing, 0.1 = 10% smoothing)
        """
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        """
        Args:
            pred: (batch_size,) - predicted probabilities (after sigmoid)
            target: (batch_size,) - true binary labels {0, 1}
        Returns:
            loss: scalar tensor
        """
        # Apply label smoothing: push labels away from 0 and 1
        target_smooth = target * (1 - self.smoothing) + self.smoothing * 0.5
        return self.bce(pred, target_smooth)


def get_warmup_cosine_scheduler(optimizer, warmup_epochs, max_epochs, min_lr_ratio=0.1):
    """
    Create learning rate scheduler with warmup + cosine annealing

    Phase 1 (warmup): Linear increase from 0 to base_lr over warmup_epochs
    Phase 2 (cosine): Cosine decay from base_lr to min_lr over remaining epochs

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs
        max_epochs: Total number of epochs
        min_lr_ratio: Minimum LR as ratio of base LR (e.g., 0.1 = decay to 10% of base)

    Returns:
        scheduler: LambdaLR scheduler
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup: 0 → 1 over warmup_epochs
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing after warmup
            progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def setup_logging(log_file):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def train_epoch(model, train_loader, criterion, optimizer, device, logger, grad_clip_norm=1.0):
    """
    Train for one epoch with gradient clipping

    Args:
        model: Mamba4Rec model
        train_loader: Training data loader
        criterion: Loss function (BCEWithLabelSmoothing)
        optimizer: Optimizer
        device: torch device
        logger: Logger instance
        grad_clip_norm: Maximum gradient norm for clipping

    Returns:
        avg_loss: Average training loss
        auc: Training AUC
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(train_loader, desc="Training", ncols=100)
    for target_items, target_categories, item_histories, category_histories, other_features, labels in pbar:
        # Move to device
        target_items = target_items.to(device)
        target_categories = target_categories.to(device)
        item_histories = item_histories.to(device)
        category_histories = category_histories.to(device)
        other_features = {k: v.to(device) for k, v in other_features.items()}
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        preds = model(target_items, target_categories, item_histories, category_histories, other_features)
        loss = criterion(preds, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        optimizer.step()

        # Collect metrics
        total_loss += loss.item()
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Calculate epoch metrics
    avg_loss = total_loss / len(train_loader)
    auc = roc_auc_score(all_labels, all_preds)

    return avg_loss, auc


def evaluate(model, val_loader, criterion, device, logger):
    """Evaluate on validation set"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", ncols=100)
        for target_items, target_categories, item_histories, category_histories, other_features, labels in pbar:
            # Move to device
            target_items = target_items.to(device)
            target_categories = target_categories.to(device)
            item_histories = item_histories.to(device)
            category_histories = category_histories.to(device)
            other_features = {k: v.to(device) for k, v in other_features.items()}
            labels = labels.to(device)

            # Forward pass
            preds = model(target_items, target_categories, item_histories, category_histories, other_features)
            loss = criterion(preds, labels)

            # Collect metrics
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Calculate metrics
    avg_loss = total_loss / len(val_loader)
    auc = roc_auc_score(all_labels, all_preds)
    logloss = log_loss(all_labels, all_preds)

    return avg_loss, auc, logloss


def main():
    parser = argparse.ArgumentParser(description='Train Mamba4Rec on Taobao - V2 with Enhanced Regularization')
    parser.add_argument('--data_dir', type=str, default='data/processed/taobao',
                        help='Data directory')

    # Model architecture (unchanged from v1)
    parser.add_argument('--item_embed_dim', type=int, default=64,
                        help='Item embedding dimension')
    parser.add_argument('--category_embed_dim', type=int, default=32,
                        help='Category embedding dimension')
    parser.add_argument('--static_embed_dim', type=int, default=16,
                        help='Static feature embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension for Mamba layers')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of Mamba layers')
    parser.add_argument('--d_state', type=int, default=16,
                        help='SSM state dimension')
    parser.add_argument('--d_conv', type=int, default=4,
                        help='Convolution kernel size')
    parser.add_argument('--expand', type=int, default=2,
                        help='Expansion factor for Mamba blocks')
    parser.add_argument('--mlp_hidden_dims', type=int, nargs='+', default=[256, 128, 64],
                        help='MLP hidden dimensions')

    # V2 Regularization improvements
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (increased from 0.2)')
    parser.add_argument('--weight_decay', type=float, default=5e-5,
                        help='Weight decay (increased from 1e-5)')
    parser.add_argument('--label_smoothing', type=float, default=0.05,
                        help='Label smoothing for loss (NEW in v2)')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0,
                        help='Gradient clipping max norm (NEW in v2)')

    # Training configuration
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs (increased from 5)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Base learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                        help='Warmup epochs (NEW in v2)')
    parser.add_argument('--min_lr_ratio', type=float, default=0.1,
                        help='Minimum LR as ratio of base LR')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help='Early stopping patience (increased from 3)')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0005,
                        help='Minimum improvement to reset patience')

    # Logging
    parser.add_argument('--log_file', type=str, default='results/mamba4rec_taobao_v2_training.log',
                        help='Log file path')
    parser.add_argument('--checkpoint_dir', type=str, default='results/checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--report_file', type=str, default='results/mamba4rec_v2_training_report.txt',
                        help='Training report file')

    args = parser.parse_args()

    # Create output directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_file).parent.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(args.log_file)

    # Log configuration
    logger.info("="*80)
    logger.info("Mamba4Rec V2 Training Configuration (Enhanced Regularization)")
    logger.info("="*80)
    logger.info("\nV2 IMPROVEMENTS:")
    logger.info("  - Dropout: 0.2 → 0.3")
    logger.info("  - Weight Decay: 1e-5 → 5e-5")
    logger.info("  - Label Smoothing: 0.05 (NEW)")
    logger.info("  - Gradient Clipping: max_norm=1.0 (NEW)")
    logger.info("  - LR Schedule: Warmup + Cosine Annealing (NEW)")
    logger.info("  - Early Stopping Patience: 3 → 5")
    logger.info("  - Max Epochs: 5 → 20")
    logger.info("\nSUCCESS CRITERIA:")
    logger.info("  - Train AUC at best epoch: 0.60-0.70 (< 0.80)")
    logger.info("  - Val AUC at best epoch: >= 0.5730")
    logger.info("  - Train-Val gap at best epoch: <= 0.08")
    logger.info("  - No validation collapse (>0.005 decline for 3 epochs)")
    logger.info("="*80)

    logger.info("\nFULL CONFIGURATION:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    logger.info("="*80)

    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    logger.info(f"\nUsing device: {device}")

    # Load data
    data_dir = Path(args.data_dir)
    logger.info("\nLoading datasets...")

    train_loader, metadata = get_taobao_dataloader(
        data_path=data_dir / 'train.parquet',
        metadata_path=data_dir / 'metadata.pkl',
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader, _ = get_taobao_dataloader(
        data_path=data_dir / 'val.parquet',
        metadata_path=data_dir / 'metadata.pkl',
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")

    # Create model with increased dropout
    logger.info("\nInitializing Mamba4Rec model (v2)...")
    model = Mamba4Rec(
        item_vocab_size=metadata['vocab_sizes']['item_id'],
        category_vocab_size=metadata['vocab_sizes']['category_id'],
        user_vocab_size=metadata['vocab_sizes']['user_id'],
        hour_vocab_size=metadata['vocab_sizes']['hour'],
        dayofweek_vocab_size=metadata['vocab_sizes']['dayofweek'],
        item_embed_dim=args.item_embed_dim,
        category_embed_dim=args.category_embed_dim,
        static_embed_dim=args.static_embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        mlp_hidden_dims=args.mlp_hidden_dims,
        dropout=args.dropout  # 0.3 instead of 0.2
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Loss and optimizer with enhanced regularization
    criterion = BCEWithLabelSmoothing(smoothing=args.label_smoothing)
    logger.info(f"\nUsing BCEWithLabelSmoothing (smoothing={args.label_smoothing})")

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay  # 5e-5 instead of 1e-5
    )

    # Learning rate scheduler with warmup + cosine annealing
    scheduler = get_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        min_lr_ratio=args.min_lr_ratio
    )
    logger.info(f"Using warmup ({args.warmup_epochs} epochs) + cosine annealing scheduler")

    # Training loop
    logger.info("\n" + "="*80)
    logger.info("Starting training...")
    logger.info("="*80)

    best_auc = 0
    best_epoch = 0
    best_train_auc = 0
    best_train_val_gap = 0
    patience_counter = 0

    # Track validation decline for collapse detection
    val_auc_history = []
    consecutive_declines = 0

    for epoch in range(args.epochs):
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs} (LR: {current_lr:.6f})")
        logger.info("-" * 80)

        # Train
        train_loss, train_auc = train_epoch(
            model, train_loader, criterion, optimizer, device, logger,
            grad_clip_norm=args.grad_clip_norm
        )

        # Validate
        val_loss, val_auc, val_logloss = evaluate(model, val_loader, criterion, device, logger)

        # Calculate train-val gap
        train_val_gap = abs(train_auc - val_auc)

        # Log results
        logger.info(f"\nEpoch {epoch + 1} Results:")
        logger.info(f"  Train - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}")
        logger.info(f"  Val   - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, LogLoss: {val_logloss:.4f}")
        logger.info(f"  Train-Val Gap: {train_val_gap:.4f}")

        # Warn if gap exceeds threshold
        if train_val_gap > 0.10:
            logger.warning(f"  WARNING: High train-val gap detected: {train_val_gap:.4f}")

        # Check for validation collapse pattern
        if len(val_auc_history) > 0:
            auc_change = val_auc - val_auc_history[-1]
            if auc_change < -0.005:
                consecutive_declines += 1
                if consecutive_declines >= 3:
                    logger.warning(f"  WARNING: Validation collapse detected (3+ consecutive declines)")
            else:
                consecutive_declines = 0

        val_auc_history.append(val_auc)

        # Save best model
        improvement = val_auc - best_auc
        if improvement > args.early_stopping_min_delta:
            best_auc = val_auc
            best_epoch = epoch + 1
            best_train_auc = train_auc
            best_train_val_gap = train_val_gap
            patience_counter = 0

            checkpoint_path = Path(args.checkpoint_dir) / 'mamba4rec_taobao_v2_best.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_auc': best_auc,
                'best_train_auc': best_train_auc,
                'train_val_gap': best_train_val_gap,
                'val_loss': val_loss,
                'val_logloss': val_logloss,
                'args': vars(args)
            }, checkpoint_path)

            logger.info(f"  >> New best model saved! Val AUC: {best_auc:.4f} (improvement: +{improvement:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  No significant improvement. Patience: {patience_counter}/{args.early_stopping_patience}")

        # Step scheduler
        scheduler.step()

        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    # Final results
    logger.info("\n" + "="*80)
    logger.info("Training Complete!")
    logger.info("="*80)
    logger.info(f"Best Validation AUC: {best_auc:.4f} (Epoch {best_epoch})")
    logger.info(f"Train AUC at best epoch: {best_train_auc:.4f}")
    logger.info(f"Train-Val Gap at best epoch: {best_train_val_gap:.4f}")
    logger.info(f"\nV1 Baseline: 0.5814 AUC")
    logger.info(f"BST Baseline: 0.5711 AUC")
    improvement_v1 = (best_auc - 0.5814) * 100
    improvement_bst = (best_auc - 0.5711) * 100
    logger.info(f"Change vs V1: {improvement_v1:+.2f}%p")
    logger.info(f"Improvement over BST: {improvement_bst:+.2f}%p")

    # Check success criteria
    logger.info("\n" + "="*80)
    logger.info("SUCCESS CRITERIA EVALUATION:")
    logger.info("="*80)

    train_auc_check = 0.60 <= best_train_auc <= 0.70
    val_auc_check = best_auc >= 0.5730
    gap_check = best_train_val_gap <= 0.08
    collapse_check = consecutive_declines < 3

    logger.info(f"1. Train AUC in [0.60, 0.70]: {best_train_auc:.4f} - {'PASS' if train_auc_check else 'FAIL'}")
    logger.info(f"2. Val AUC >= 0.5730: {best_auc:.4f} - {'PASS' if val_auc_check else 'FAIL'}")
    logger.info(f"3. Train-Val gap <= 0.08: {best_train_val_gap:.4f} - {'PASS' if gap_check else 'FAIL'}")
    logger.info(f"4. No validation collapse: {'PASS' if collapse_check else 'FAIL'}")

    all_pass = train_auc_check and val_auc_check and gap_check and collapse_check
    logger.info(f"\nOVERALL: {'PUBLICATION READY' if all_pass else 'NEEDS FURTHER TUNING'}")
    logger.info("="*80)

    # Generate training report
    report_path = Path(args.report_file)
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Mamba4Rec V2 Training Report\n")
        f.write("="*80 + "\n\n")

        f.write("TRAINING CONFIGURATION:\n")
        f.write("-" * 80 + "\n")
        for arg, value in vars(args).items():
            f.write(f"  {arg}: {value}\n")
        f.write("\n")

        f.write("V2 IMPROVEMENTS OVER V1:\n")
        f.write("-" * 80 + "\n")
        f.write("  - Dropout: 0.2 → 0.3\n")
        f.write("  - Weight Decay: 1e-5 → 5e-5\n")
        f.write("  - Label Smoothing: 0.05 (NEW)\n")
        f.write("  - Gradient Clipping: max_norm=1.0 (NEW)\n")
        f.write("  - LR Schedule: Warmup + Cosine Annealing (NEW)\n")
        f.write("  - Early Stopping Patience: 3 → 5\n\n")

        f.write("FINAL RESULTS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Best Epoch: {best_epoch}\n")
        f.write(f"  Best Val AUC: {best_auc:.4f}\n")
        f.write(f"  Train AUC at best epoch: {best_train_auc:.4f}\n")
        f.write(f"  Train-Val Gap: {best_train_val_gap:.4f}\n\n")

        f.write("COMPARISON WITH BASELINES:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  V1 Val AUC: 0.5814\n")
        f.write(f"  V2 Val AUC: {best_auc:.4f}\n")
        f.write(f"  Change: {improvement_v1:+.2f}%p\n\n")
        f.write(f"  BST Val AUC: 0.5711\n")
        f.write(f"  V2 Val AUC: {best_auc:.4f}\n")
        f.write(f"  Improvement: {improvement_bst:+.2f}%p\n\n")

        f.write("SUCCESS CRITERIA EVALUATION:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  1. Train AUC in [0.60, 0.70]: {best_train_auc:.4f} - {'PASS' if train_auc_check else 'FAIL'}\n")
        f.write(f"  2. Val AUC >= 0.5730: {best_auc:.4f} - {'PASS' if val_auc_check else 'FAIL'}\n")
        f.write(f"  3. Train-Val gap <= 0.08: {best_train_val_gap:.4f} - {'PASS' if gap_check else 'FAIL'}\n")
        f.write(f"  4. No validation collapse: {'PASS' if collapse_check else 'FAIL'}\n\n")
        f.write(f"  OVERALL: {'PUBLICATION READY' if all_pass else 'NEEDS FURTHER TUNING'}\n")
        f.write("="*80 + "\n")

    logger.info(f"\nTraining report saved to: {report_path}")


if __name__ == '__main__':
    main()
