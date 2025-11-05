"""
Train DCN-V2 model on Taobao dataset (static features only)

주의: DCNv2는 순차 모델링 능력이 없으므로,
시퀀스 정보를 제외하고 정적 특징만 사용합니다.
이는 BST와의 공정한 비교를 위함입니다.
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

from models.baseline.dcnv2 import DCNV2
from utils.taobao_dataset import get_taobao_static_dataloader


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(train_loader, desc="Training")
    for num_features, cat_features, labels in pbar:
        num_features = num_features.to(device)
        cat_features = {k: v.to(device) for k, v in cat_features.items()}
        labels = labels.to(device).squeeze()

        optimizer.zero_grad()
        preds = model(num_features, cat_features)
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
        for num_features, cat_features, labels in tqdm(val_loader, desc="Validation"):
            num_features = num_features.to(device)
            cat_features = {k: v.to(device) for k, v in cat_features.items()}
            labels = labels.to(device).squeeze()

            preds = model(num_features, cat_features)
            loss = criterion(preds, labels)

            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    auc = roc_auc_score(all_labels, all_preds)
    logloss = log_loss(all_labels, all_preds)

    return avg_loss, auc, logloss


def main():
    parser = argparse.ArgumentParser(description='Train DCN-V2 on Taobao (static features)')
    parser.add_argument('--data_dir', type=str, default='data/processed/taobao',
                        help='Data directory')
    parser.add_argument('--embed_dim', type=int, default=16,
                        help='Embedding dimension')
    parser.add_argument('--cross_num_layers', type=int, default=3,
                        help='Number of cross layers')
    parser.add_argument('--dnn_hidden_units', type=int, nargs='+', default=[256, 128, 64],
                        help='DNN hidden units')
    parser.add_argument('--structure', type=str, default='parallel', choices=['parallel', 'stacked'],
                        help='Network structure')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=1024,
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
    print(f"\n{'='*60}")
    print(f"🎯 DCNv2 on Taobao (Static Features)")
    print(f"{'='*60}")
    print(f"⚠️  주의: 시퀀스 정보 제외, 정적 특징만 사용")
    print(f"   BST와의 공정한 비교를 위한 설정")
    print(f"{'='*60}\n")

    # Load data using static dataloader
    data_dir = Path(args.data_dir)

    train_loader, metadata = get_taobao_static_dataloader(
        data_path=data_dir / 'train.parquet',
        metadata_path=data_dir / 'metadata.pkl',
        batch_size=args.batch_size,
        shuffle=True
    )

    val_loader, _ = get_taobao_static_dataloader(
        data_path=data_dir / 'val.parquet',
        metadata_path=data_dir / 'metadata.pkl',
        batch_size=args.batch_size,
        shuffle=False
    )

    # Create model
    model = DCNV2(
        num_features=len(metadata['num_features']),  # 0
        cat_vocab_sizes=metadata['cat_vocab'],
        embed_dim=args.embed_dim,
        cross_num_layers=args.cross_num_layers,
        dnn_hidden_units=args.dnn_hidden_units,
        dropout=args.dropout,
        structure=args.structure
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Categorical features: {list(metadata['cat_vocab'].keys())}")
    print(f"Vocab sizes: {metadata['cat_vocab']}\n")

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
            torch.save(model.state_dict(), 'results/checkpoints/dcnv2_taobao_best.pth')
            print(f"   ✅ Best Validation AUC: {best_auc:.4f}")

    print(f"\n{'='*60}")
    print(f"🎉 최종 Best Validation AUC: {best_auc:.4f}")
    print(f"{'='*60}")
    print(f"\n💾 모델 저장: results/checkpoints/dcnv2_taobao_best.pth")


if __name__ == '__main__':
    main()
