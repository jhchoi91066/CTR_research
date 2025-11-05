"""
Train Mamba4Rec model on Taobao dataset

Mamba4Rec: Selective State Space Model for Sequential CTR Prediction
Target: Beat BST baseline (0.5711 AUC) with more efficient architecture
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

from models.mdaf.mamba4rec import Mamba4Rec
from utils.taobao_dataset import get_taobao_dataloader


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


def train_epoch(model, train_loader, criterion, optimizer, device, logger):
    """Train for one epoch"""
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
    parser = argparse.ArgumentParser(description='Train Mamba4Rec on Taobao')
    parser.add_argument('--data_dir', type=str, default='data/processed/taobao',
                        help='Data directory')
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
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--log_file', type=str, default='results/mamba4rec_taobao_training.log',
                        help='Log file path')
    parser.add_argument('--checkpoint_dir', type=str, default='results/checkpoints',
                        help='Checkpoint directory')

    args = parser.parse_args()

    # Create output directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(args.log_file)

    # Log configuration
    logger.info("="*80)
    logger.info("Mamba4Rec Training Configuration")
    logger.info("="*80)
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info("="*80)

    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    logger.info(f"Using device: {device}")

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

    # Create model
    logger.info("\nInitializing Mamba4Rec model...")
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
        dropout=args.dropout
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Training loop
    logger.info("\n" + "="*80)
    logger.info("Starting training...")
    logger.info("="*80)

    best_auc = 0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        logger.info("-" * 80)

        # Train
        train_loss, train_auc = train_epoch(model, train_loader, criterion, optimizer, device, logger)

        # Validate
        val_loss, val_auc, val_logloss = evaluate(model, val_loader, criterion, device, logger)

        # Log results
        logger.info(f"\nEpoch {epoch + 1} Results:")
        logger.info(f"  Train - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}")
        logger.info(f"  Val   - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, LogLoss: {val_logloss:.4f}")

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch + 1
            patience_counter = 0

            checkpoint_path = Path(args.checkpoint_dir) / 'mamba4rec_taobao_best.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'val_loss': val_loss,
                'val_logloss': val_logloss,
                'args': vars(args)
            }, checkpoint_path)

            logger.info(f"  >> New best model saved! AUC: {best_auc:.4f}")
        else:
            patience_counter += 1
            logger.info(f"  No improvement. Patience: {patience_counter}/{args.early_stopping_patience}")

        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    # Final results
    logger.info("\n" + "="*80)
    logger.info("Training Complete!")
    logger.info("="*80)
    logger.info(f"Best Validation AUC: {best_auc:.4f} (Epoch {best_epoch})")
    logger.info(f"BST Baseline AUC: 0.5711")
    improvement = (best_auc - 0.5711) * 100
    logger.info(f"Improvement over BST: {improvement:+.2f}%p")
    logger.info("="*80)


if __name__ == '__main__':
    main()
