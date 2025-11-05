"""
Train DCNv3 model on Criteo dataset

DCNv3 특징:
- Linear Cross Network (LCN): 선형 증가 (1차, 2차, 3차, ...)
- Exponential Cross Network (ECN): 지수 증가 (1차, 2차, 4차, 8차, ...)
- Self-Mask: LayerNorm 기반 노이즈 필터링
- Tri-BCE Loss: 적응형 가중치를 사용한 3개의 BCE 손실
"""

import sys
sys.path.append('.')

import torch
import torch.optim as optim
import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, log_loss
from pathlib import Path

from models.mdaf.dcnv3 import DCNV3, TriBCELoss
from utils.dataset import get_dataloader


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

        # DCNv3 forward with auxiliary outputs for Tri-BCE
        preds, lcn_preds, ecn_preds = model(num_features, cat_features, return_aux=True)

        # Tri-BCE loss
        loss = criterion(preds, lcn_preds, ecn_preds, labels)

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

            # Validation: return_aux=True for Tri-BCE
            preds, lcn_preds, ecn_preds = model(num_features, cat_features, return_aux=True)
            loss = criterion(preds, lcn_preds, ecn_preds, labels)

            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    auc = roc_auc_score(all_labels, all_preds)
    logloss = log_loss(all_labels, all_preds)

    return avg_loss, auc, logloss


def main():
    parser = argparse.ArgumentParser(description='Train DCNv3 on Criteo')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Data directory')
    parser.add_argument('--embed_dim', type=int, default=16,
                        help='Embedding dimension')
    parser.add_argument('--lcn_num_layers', type=int, default=3,
                        help='Number of Linear Cross Network layers')
    parser.add_argument('--ecn_num_layers', type=int, default=3,
                        help='Number of Exponential Cross Network layers')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate for regularization')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size (논문 권장: 4096)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (논문 권장: 0.001)')

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
    print(f"🚀 DCNv3 on Criteo (Feature Interaction)")
    print(f"{'='*60}")
    print(f"📖 논문 기반 구현:")
    print(f"   - Linear Cross Network (LCN): {args.lcn_num_layers} layers")
    print(f"   - Exponential Cross Network (ECN): {args.ecn_num_layers} layers")
    print(f"   - Self-Mask: LayerNorm 기반 노이즈 필터링")
    print(f"   - Tri-BCE Loss: 적응형 가중치")
    print(f"\n🔧 정규화 설정:")
    print(f"   - Dropout: {args.dropout}")
    print(f"   - Weight Decay: {args.weight_decay}")
    print(f"{'='*60}\n")

    # Load data
    data_dir = Path(args.data_dir)

    train_loader, metadata = get_dataloader(
        data_path=data_dir / 'train.parquet',
        metadata_path=data_dir / 'metadata.pkl',
        batch_size=args.batch_size,
        shuffle=True
    )

    val_loader, _ = get_dataloader(
        data_path=data_dir / 'val.parquet',
        metadata_path=data_dir / 'metadata.pkl',
        batch_size=args.batch_size,
        shuffle=False
    )

    # Create model
    model = DCNV3(
        num_features=len(metadata['num_features']),
        cat_vocab_sizes=metadata['cat_vocab'],
        embed_dim=args.embed_dim,
        lcn_num_layers=args.lcn_num_layers,
        ecn_num_layers=args.ecn_num_layers,
        dropout=args.dropout,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Numeric features: {len(metadata['num_features'])}")
    print(f"Categorical features: {list(metadata['cat_vocab'].keys())}")
    print(f"Total feature fields: {len(metadata['num_features']) + len(metadata['cat_vocab'])}\n")

    # Tri-BCE Loss and optimizer
    criterion = TriBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
            torch.save(model.state_dict(), 'results/checkpoints/dcnv3_criteo_best.pth')
            print(f"   ✅ New Best Validation AUC: {best_auc:.4f}")

    print(f"\n{'='*60}")
    print(f"🎉 Training Complete!")
    print(f"{'='*60}")
    print(f"Best Validation AUC: {best_auc:.4f}")
    print(f"{'='*60}")
    print(f"\n💾 모델 저장: results/checkpoints/dcnv3_criteo_best.pth")
    print(f"\n🎯 목표 대비 성능:")
    print(f"   - DCNv2 (Criteo): 0.7722")
    print(f"   - AutoInt (Criteo): 0.7802")
    print(f"   - DCNv3 (Criteo): {best_auc:.4f}")

    if best_auc > 0.7802:
        print(f"   ✅ SOTA 달성! (+{(best_auc - 0.7802)*100:.2f}%)")
    elif best_auc > 0.7722:
        print(f"   ✅ DCNv2 초과! (+{(best_auc - 0.7722)*100:.2f}%)")
    else:
        print(f"   ⚠️  목표 미달. 추가 튜닝 필요")


if __name__ == '__main__':
    main()
