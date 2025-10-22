"""
DeepFM 학습 스크립트

Usage:
    python experiments/train_deepfm.py --epochs 10 --batch_size 1024
"""

import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse
from tqdm import tqdm
import pickle

from models.baseline.deepfm import DeepFM
from utils.dataset import get_dataloader
from sklearn.metrics import roc_auc_score, log_loss


class Trainer:
    """DeepFM 학습 클래스"""

    def __init__(self, model, train_loader, val_loader, device, lr=0.001):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        print(f"\n✅ Trainer 초기화 완료")
        print(f"   - Device: {device}")
        print(f"   - Learning rate: {lr}")
        print(f"   - Train batches: {len(train_loader)}")
        print(f"   - Val batches: {len(val_loader)}")

    def train_epoch(self):
        """1 에폭 학습"""
        self.model.train()

        total_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc="Training")
        for numeric, categorical, labels in pbar:
            numeric = numeric.to(self.device)
            categorical = {k: v.to(self.device) for k, v in categorical.items()}
            labels = labels.to(self.device)

            outputs = self.model(numeric, categorical)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy().flatten())
            all_labels.extend(labels.detach().cpu().numpy().flatten())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(self.train_loader)
        auc = roc_auc_score(all_labels, all_preds)

        return avg_loss, auc

    def validate(self):
        """검증"""
        self.model.eval()

        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for numeric, categorical, labels in tqdm(self.val_loader, desc="Validating"):
                numeric = numeric.to(self.device)
                categorical = {k: v.to(self.device) for k, v in categorical.items()}
                labels = labels.to(self.device)

                outputs = self.model(numeric, categorical)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

        avg_loss = total_loss / len(self.val_loader)
        auc = roc_auc_score(all_labels, all_preds)
        logloss = log_loss(all_labels, all_preds)

        return avg_loss, auc, logloss

    def train(self, epochs, save_dir='results/checkpoints'):
        """전체 학습"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        best_auc = 0

        print(f"\n{'='*60}")
        print(f"🚀 학습 시작")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")
            print(f"-" * 60)

            train_loss, train_auc = self.train_epoch()
            val_loss, val_auc, val_logloss = self.validate()

            print(f"\n📊 Epoch {epoch} Results:")
            print(f"   Train - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}")
            print(f"   Val   - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, LogLoss: {val_logloss:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                checkpoint_path = save_dir / 'deepfm_best.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_auc': val_auc,
                    'val_logloss': val_logloss
                }, checkpoint_path)
                print(f"   ✅ Best 모델 저장: {checkpoint_path} (AUC: {val_auc:.4f})")

            print()

        print(f"{'='*60}")
        print(f"✅ 학습 완료!")
        print(f"   - Best Validation AUC: {best_auc:.4f}")
        print(f"{'='*60}\n")

        return best_auc


def main():
    parser = argparse.ArgumentParser(description='DeepFM 학습')
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--hidden_units', type=int, nargs='+', default=[256, 128, 64])
    parser.add_argument('--dropout', type=float, default=0.1)

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"✅ Device: {device}")

    with open(data_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    print(f"\n📥 데이터 로딩 중...")
    train_loader = get_dataloader(
        data_path=data_dir / 'train.parquet',
        metadata_path=data_dir / 'metadata.pkl',
        batch_size=args.batch_size,
        shuffle=True
    )

    val_loader = get_dataloader(
        data_path=data_dir / 'val.parquet',
        metadata_path=data_dir / 'metadata.pkl',
        batch_size=args.batch_size,
        shuffle=False
    )

    print(f"\n🏗️  모델 생성 중...")
    model = DeepFM(
        num_features=len(metadata['num_features']),
        cat_vocab_sizes=metadata['cat_vocab'],
        embed_dim=args.embed_dim,
        hidden_units=args.hidden_units,
        dropout=args.dropout
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr
    )

    best_auc = trainer.train(epochs=args.epochs)
    print(f"\n🎉 최종 Best Validation AUC: {best_auc:.4f}")


if __name__ == '__main__':
    main()
