"""
Criteo 데이터셋 로더
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
from pathlib import Path


class CriteoDataset(Dataset):
    """Criteo CTR 데이터셋"""

    def __init__(self, data_path, metadata_path):
        """
        Args:
            data_path: train/val/test.parquet 경로
            metadata_path: metadata.pkl 경로
        """
        self.data = pd.read_parquet(data_path)

        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        self.num_features = self.metadata['num_features']
        self.cat_features = self.metadata['cat_features']

        print(f"✅ 데이터셋 로딩 완료")
        print(f"   - 데이터 크기: {len(self.data):,} rows")
        print(f"   - 수치형 특징: {len(self.num_features)}개")
        print(f"   - 범주형 특징: {len(self.cat_features)}개")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            numeric_features: Tensor (num_features,)
            categorical_features: Dict of {feat_name: int}
            label: Tensor (1,)
        """
        row = self.data.iloc[idx]

        # 수치형 특징
        numeric_features = torch.tensor(
            row[self.num_features].values.astype('float32')
        )

        # 범주형 특징
        categorical_features = {
            feat: int(row[feat])
            for feat in self.cat_features
        }

        # 라벨
        label = torch.tensor([row['label']], dtype=torch.float32)

        return numeric_features, categorical_features, label


def collate_fn(batch):
    """
    배치 데이터 묶기

    Args:
        batch: List of (numeric_features, categorical_features, label)

    Returns:
        numeric_batch: Tensor (batch_size, num_features)
        categorical_batch: Dict of {feat_name: Tensor (batch_size,)}
        label_batch: Tensor (batch_size, 1)
    """
    numeric_list, cat_list, label_list = zip(*batch)

    # Numeric features: stack
    numeric_batch = torch.stack(numeric_list)  # (batch, num_features)

    # Categorical features: dict of tensors
    cat_feature_names = list(cat_list[0].keys())
    categorical_batch = {
        feat: torch.tensor([item[feat] for item in cat_list], dtype=torch.long)
        for feat in cat_feature_names
    }

    # Labels: stack
    label_batch = torch.stack(label_list)  # (batch, 1)

    return numeric_batch, categorical_batch, label_batch


def get_dataloader(data_path, metadata_path, batch_size=1024, shuffle=True, num_workers=0):
    """
    DataLoader 생성

    Args:
        data_path: train/val/test.parquet 경로
        metadata_path: metadata.pkl 경로
        batch_size: 배치 크기
        shuffle: 셔플 여부
        num_workers: 워커 수

    Returns:
        DataLoader
    """
    dataset = CriteoDataset(data_path, metadata_path)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True  # GPU 학습 시 성능 향상
    )

    return dataloader


if __name__ == '__main__':
    # 테스트
    print("="*60)
    print("🧪 데이터셋 로더 테스트")
    print("="*60)

    data_dir = Path('data/processed')

    # DataLoader 생성
    train_loader = get_dataloader(
        data_path=data_dir / 'train.parquet',
        metadata_path=data_dir / 'metadata.pkl',
        batch_size=128,
        shuffle=True
    )

    # 첫 번째 배치 확인
    numeric, categorical, labels = next(iter(train_loader))

    print(f"\n📦 배치 정보:")
    print(f"   - Numeric shape: {numeric.shape}")
    print(f"   - Categorical features: {len(categorical)}개")
    print(f"   - Labels shape: {labels.shape}")
    print(f"   - Labels range: [{labels.min():.0f}, {labels.max():.0f}]")
    print(f"   - CTR: {labels.mean():.4f}")

    # 범주형 특징 샘플
    print(f"\n📋 범주형 특징 샘플:")
    for feat_name in list(categorical.keys())[:3]:
        feat_tensor = categorical[feat_name]
        print(f"   - {feat_name}: shape={feat_tensor.shape}, range=[{feat_tensor.min()}, {feat_tensor.max()}]")

    print("\n" + "="*60)
    print("✅ 테스트 완료!")
    print("="*60)
