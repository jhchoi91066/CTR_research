"""
Train BST model on Taobao dataset
"""

import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, log_loss
from pathlib import Path

from models.baseline.bst_fixed import BST
from utils.taobao_dataset import get_taobao_dataloader


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(train_loader, desc="Training")
    for target_items, target_categories, item_histories, category_histories, other_features, labels in pbar:
        target_items = target_items.to(device)
        target_categories = target_categories.to(device)
        item_histories = item_histories.to(device)
        category_histories = category_histories.to(device)  # ⭐ 추가
        other_features = {k: v.to(device) for k, v in other_features.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        preds = model(target_items, target_categories, item_histories, category_histories, other_features)  # ⭐ 수정
        loss = criterion(preds, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(train_loader)
    auc = roc_auc_score(all_labels, all_preds)

    return avg_loss, auc


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for target_items, target_categories, item_histories, category_histories, other_features, labels in tqdm(val_loader, desc="Validation"):
            target_items = target_items.to(device)
            target_categories = target_categories.to(device)
            item_histories = item_histories.to(device)
            category_histories = category_histories.to(device)  # ⭐ 추가
            other_features = {k: v.to(device) for k, v in other_features.items()}
            labels = labels.to(device)

            preds = model(target_items, target_categories, item_histories, category_histories, other_features)  # ⭐ 수정
            loss = criterion(preds, labels)

            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    auc = roc_auc_score(all_labels, all_preds)
    logloss = log_loss(all_labels, all_preds)

    return avg_loss, auc, logloss


def main():
    parser = argparse.ArgumentParser(description='Train BST on Taobao')
    parser.add_argument('--data_dir', type=str, default='data/processed/taobao',
                        help='Data directory')
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--num_transformer_layers', type=int, default=1,
                        help='Number of transformer layers (논문 권장: 1)')
    parser.add_argument('--num_heads', type=int, default=2,
                        help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=256,
                        help='Feed-forward dimension')
    parser.add_argument('--dnn_hidden_units', type=int, nargs='+', default=[256, 128],
                        help='DNN hidden units')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')

    args = parser.parse_args()

    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # Load data
    data_dir = Path(args.data_dir)

    train_loader, metadata = get_taobao_dataloader(
        data_path=data_dir / 'train.parquet',
        metadata_path=data_dir / 'metadata.pkl',
        batch_size=args.batch_size,
        shuffle=True
    )

    val_loader, _ = get_taobao_dataloader(
        data_path=data_dir / 'val.parquet',
        metadata_path=data_dir / 'metadata.pkl',
        batch_size=args.batch_size,
        shuffle=False
    )

    # Create model (BST_fixed - 논문 정확 구현)
    model = BST(
        item_vocab_size=metadata['vocab_sizes']['item_id'],
        category_vocab_size=metadata['vocab_sizes']['category_id'],
        other_feature_dims={
            'user_id': metadata['vocab_sizes']['user_id'],
            'hour': metadata['vocab_sizes']['hour'],
            'dayofweek': metadata['vocab_sizes']['dayofweek']
        },
        embed_dim=args.embed_dim,
        max_seq_len=metadata['max_seq_len'],
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        dnn_hidden_units=args.dnn_hidden_units
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_auc = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 60)

        train_loss, train_auc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc, val_logloss = evaluate(model, val_loader, criterion, device)

        print(f"\n📊 Epoch {epoch + 1} Results:")
        print(f"   Train - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}")
        print(f"   Val   - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, LogLoss: {val_logloss:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            # Save best model
            Path('results/checkpoints').mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), 'results/checkpoints/bst_best.pth')
            print(f"   - Best Validation AUC: {best_auc:.4f}")

    print(f"\n🎉 최종 Best Validation AUC: {best_auc:.4f}")


if __name__ == '__main__':
    main()
