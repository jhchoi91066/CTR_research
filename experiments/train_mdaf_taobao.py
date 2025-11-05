#!/usr/bin/env python3
"""
MDAF Training Script for Taobao Dataset

Usage:
    ./venv/bin/python experiments/train_mdaf_taobao.py --model mamba  # MDAF-Mamba
    ./venv/bin/python experiments/train_mdaf_taobao.py --model bst    # MDAF-BST

Features:
- Trains MDAF (Multi-branch Dual Attention Fusion) models on Taobao dataset
- Supports both Mamba4Rec and BST as sequential branches
- Implements early stopping, checkpointing, and gate value analysis
- Logs training progress and model performance
"""

import argparse
import json
import pickle
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np

from models.mdaf.mdaf_mamba import MDAF_Mamba
from models.mdaf.mdaf_bst import MDAF_BST
from utils.taobao_dataset import TaobaoDataset, collate_fn


# ==================== Configuration ====================

DEFAULT_CONFIG = {
    # Training
    'batch_size': 2048,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'max_epochs': 15,
    'early_stopping_patience': 5,
    'warmup_epochs': 2,

    # Model architecture
    'embedding_dim': 128,  # Target dimension for fusion
    'prediction_hidden_dims': [128, 64],
    'dropout': 0.2,

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
    'bst_embed_dim': 128,  # Must match embedding_dim
    'bst_max_seq_len': 50,
    'bst_num_transformer_layers': 2,
    'bst_num_heads': 4,
    'bst_d_ff': 256,
    'bst_dropout': 0.1,
}


# ==================== Setup Logging ====================

def setup_logging(model_name, output_dir):
    """Setup logging to file and console"""
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{model_name}_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


# ==================== Training Functions ====================

def train_epoch(model, dataloader, optimizer, criterion, device, logger, grad_clip_norm=1.0):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        all_preds.extend(predictions.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Log progress (reduced frequency for speed)
        if (batch_idx + 1) % 200 == 0:
            logger.info(f'   Batch [{batch_idx+1}/{len(dataloader)}] Loss: {loss.item():.4f}')

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_preds)

    return avg_loss, auc


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

    if return_gates:
        gate_stats = {
            'mean': np.mean(all_gates),
            'std': np.std(all_gates),
            'min': np.min(all_gates),
            'max': np.max(all_gates),
            'median': np.median(all_gates)
        }
        return avg_loss, auc, gate_stats

    return avg_loss, auc


# ==================== Main Training Loop ====================

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train MDAF models on Taobao dataset')
    parser.add_argument('--model', type=str, required=True, choices=['mamba', 'bst'],
                       help='Sequential branch type: mamba or bst')
    parser.add_argument('--epochs', type=int, default=15, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 regularization)')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing factor')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--warmup_epochs', type=int, default=2, help='Number of warmup epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='auto', help='Device: auto, cpu, cuda, or mps')
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

    # Setup logging
    logger = setup_logging(model_name, output_dir)
    logger.info("="*60)
    logger.info(f"Training {model_name.upper()}")
    logger.info("="*60)

    # Load metadata
    logger.info("\n1. Loading Taobao metadata...")
    with open(data_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    logger.info(f"   Vocabulary sizes:")
    logger.info(f"      Items: {metadata['vocab_sizes']['item_id']:,}")
    logger.info(f"      Categories: {metadata['vocab_sizes']['category_id']:,}")
    logger.info(f"      Users: {metadata['vocab_sizes']['user_id']:,}")
    logger.info(f"   Max sequence length: {metadata['max_seq_len']}")

    # Update config
    config = DEFAULT_CONFIG.copy()
    config.update({
        'item_vocab_size': metadata['vocab_sizes']['item_id'],
        'category_vocab_size': metadata['vocab_sizes']['category_id'],
        'user_vocab_size': metadata['vocab_sizes']['user_id'],
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'max_epochs': args.epochs,
        'dropout': args.dropout,
        'weight_decay': args.weight_decay,
        'label_smoothing': args.label_smoothing,
        'grad_clip_norm': args.grad_clip_norm,
        'warmup_epochs': args.warmup_epochs
    })

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,  # Use multiprocessing for faster data loading
        pin_memory=True,  # Faster data transfer to GPU
        persistent_workers=True  # Keep workers alive between epochs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,  # Less workers for validation
        pin_memory=True
    )

    logger.info(f"   Train batches: {len(train_loader)}")
    logger.info(f"   Val batches: {len(val_loader)}")

    # Create model
    logger.info(f"\n4. Creating {model_name.upper()} model...")

    if args.model == 'mamba':
        model = MDAF_Mamba(
            item_vocab_size=config['item_vocab_size'],
            category_vocab_size=config['category_vocab_size'],
            user_vocab_size=config['user_vocab_size'],
            dcnv3_embed_dim=config['dcnv3_embed_dim'],
            dcnv3_lcn_layers=config['dcnv3_lcn_layers'],
            dcnv3_ecn_layers=config['dcnv3_ecn_layers'],
            dcnv3_dropout=config['dcnv3_dropout'],
            item_embed_dim=config['mamba_item_embed_dim'],
            category_embed_dim=config['mamba_category_embed_dim'],
            static_embed_dim=config['mamba_static_embed_dim'],
            mamba_hidden_dim=config['mamba_hidden_dim'],
            mamba_num_layers=config['mamba_num_layers'],
            mamba_d_state=config['mamba_d_state'],
            mamba_d_conv=config['mamba_d_conv'],
            mamba_expand=config['mamba_expand'],
            embedding_dim=config['embedding_dim'],
            prediction_hidden_dims=config['prediction_hidden_dims'],
            dropout=config['dropout']
        )
    else:  # bst
        model = MDAF_BST(
            item_vocab_size=config['item_vocab_size'],
            category_vocab_size=config['category_vocab_size'],
            user_vocab_size=config['user_vocab_size'],
            dcnv3_embed_dim=config['dcnv3_embed_dim'],
            dcnv3_lcn_layers=config['dcnv3_lcn_layers'],
            dcnv3_ecn_layers=config['dcnv3_ecn_layers'],
            dcnv3_dropout=config['dcnv3_dropout'],
            bst_embed_dim=config['bst_embed_dim'],
            bst_max_seq_len=config['bst_max_seq_len'],
            bst_num_transformer_layers=config['bst_num_transformer_layers'],
            bst_num_heads=config['bst_num_heads'],
            bst_d_ff=config['bst_d_ff'],
            bst_dropout=config['bst_dropout'],
            embedding_dim=config['embedding_dim'],
            prediction_hidden_dims=config['prediction_hidden_dims'],
            dropout=config['dropout']
        )

    model = model.to(device)
    logger.info(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Log hyperparameters
    logger.info(f"\n   Hyperparameters:")
    logger.info(f"      Learning rate: {config['learning_rate']}")
    logger.info(f"      Dropout: {config['dropout']}")
    logger.info(f"      Weight decay: {config['weight_decay']}")
    logger.info(f"      Label smoothing: {config['label_smoothing']}")
    logger.info(f"      Gradient clip norm: {config['grad_clip_norm']}")
    logger.info(f"      Warmup epochs: {config['warmup_epochs']}")

    # Optimizer and criterion with label smoothing
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # BCELoss with label smoothing
    if config['label_smoothing'] > 0:
        # Custom BCE loss with label smoothing
        class BCEWithLabelSmoothing(nn.Module):
            def __init__(self, smoothing=0.0):
                super().__init__()
                self.smoothing = smoothing

            def forward(self, predictions, labels):
                # Apply label smoothing: y_smooth = y * (1 - smoothing) + 0.5 * smoothing
                labels_smooth = labels * (1.0 - self.smoothing) + 0.5 * self.smoothing
                return nn.functional.binary_cross_entropy(predictions, labels_smooth)

        criterion = BCEWithLabelSmoothing(smoothing=config['label_smoothing'])
    else:
        criterion = nn.BCELoss()

    # Training loop
    logger.info("\n5. Starting training...")
    best_val_auc = 0
    best_epoch = 0
    patience_counter = 0

    # Track metrics for all epochs
    metrics_history = []

    for epoch in range(config['max_epochs']):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{config['max_epochs']}")
        logger.info(f"{'='*60}")

        # Train
        train_loss, train_auc = train_epoch(
            model, train_loader, optimizer, criterion, device, logger,
            grad_clip_norm=config['grad_clip_norm']
        )
        logger.info(f"\nTrain - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}")

        # Validate
        val_loss, val_auc, gate_stats = evaluate(model, val_loader, criterion, device, logger, return_gates=True)
        logger.info(f"Val   - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")
        logger.info(f"Gate  - Mean: {gate_stats['mean']:.4f}, Std: {gate_stats['std']:.4f}, "
                   f"Range: [{gate_stats['min']:.4f}, {gate_stats['max']:.4f}]")

        # Track metrics
        train_val_gap = train_auc - val_auc
        logger.info(f"Gap   - Train-Val: {train_val_gap:+.4f}")

        metrics_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_auc': train_auc,
            'val_loss': val_loss,
            'val_auc': val_auc,
            'train_val_gap': train_val_gap,
            'gate_mean': gate_stats['mean'],
            'gate_std': gate_stats['std'],
            'gate_min': gate_stats['min'],
            'gate_max': gate_stats['max']
        })

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            patience_counter = 0

            checkpoint_path = checkpoint_dir / f'{model_name}_phase3_revised_best.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'train_auc': train_auc,
                'gate_stats': gate_stats,
                'config': config,
                'metrics_history': metrics_history
            }, checkpoint_path)

            logger.info(f"✓ New best model saved! Val AUC: {val_auc:.4f}")
        else:
            patience_counter += 1
            logger.info(f"Patience: {patience_counter}/{config['early_stopping_patience']}")

        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    # Final evaluation
    logger.info("\n" + "="*60)
    logger.info("Training completed!")
    logger.info(f"Best validation AUC: {best_val_auc:.4f} (Epoch {best_epoch})")
    logger.info("="*60)

    # Save metrics to CSV
    import csv
    metrics_csv_path = output_dir / 'phase3_revised_metrics.csv'
    with open(metrics_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics_history[0].keys())
        writer.writeheader()
        writer.writerows(metrics_history)
    logger.info(f"\nMetrics saved to: {metrics_csv_path}")

    return best_val_auc, best_epoch, metrics_history


if __name__ == '__main__':
    main()
