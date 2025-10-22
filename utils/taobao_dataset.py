"""
Taobao 데이터셋 로더
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
from pathlib import Path


class TaobaoDataset(Dataset):
    """Taobao UserBehavior CTR 데이터셋"""

    def __init__(self, data_path, metadata_path):
        """
        Args:
            data_path: train/val/test.parquet 경로
            metadata_path: metadata.pkl 경로
        """
        self.data = pd.read_parquet(data_path)

        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        print(f"✅ Taobao 데이터셋 로딩 완료")
        print(f"   - 데이터 크기: {len(self.data):,} rows")
        print(f"   - 사용자 수: {self.data['user_id_encoded'].nunique():,}")
        print(f"   - 상품 수: {self.data['item_id_encoded'].nunique():,}")
        print(f"   - Max sequence length: {self.metadata['max_seq_len']}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            target_item: int - 타겟 아이템 ID
            target_category: int - 타겟 카테고리 ID
            item_history: Tensor (max_seq_len,) - 아이템 히스토리
            category_history: Tensor (max_seq_len,) - 카테고리 히스토리 ⭐ 추가
            other_features: Dict - {feature_name: value}
            label: Tensor - 레이블
        """
        row = self.data.iloc[idx]

        # Target item & category
        target_item = int(row['item_id_encoded'])
        target_category = int(row['category_id_encoded'])

        # Item history sequence
        item_history = torch.tensor(row['item_history'], dtype=torch.long)
        category_history = torch.tensor(row['category_history'], dtype=torch.long)  # ⭐ 추가

        # Other features
        other_features = {
            'user_id': int(row['user_id_encoded']),
            'hour': int(row['hour']),
            'dayofweek': int(row['dayofweek'])
        }

        # Label
        label = torch.tensor(row['label'], dtype=torch.float32)

        return target_item, target_category, item_history, category_history, other_features, label


def collate_fn(batch):
    """
    배치 데이터 묶기

    Args:
        batch: List of (target_item, target_category, item_history, category_history, other_features, label)

    Returns:
        target_items: Tensor (batch_size,)
        target_categories: Tensor (batch_size,)
        item_histories: Tensor (batch_size, max_seq_len)
        category_histories: Tensor (batch_size, max_seq_len) ⭐ 추가
        other_features: Dict of {feat_name: Tensor (batch_size,)}
        labels: Tensor (batch_size,)
    """
    target_items, target_categories, item_histories, category_histories, other_features_list, labels = zip(*batch)

    # Stack target items & categories
    target_items = torch.tensor(target_items, dtype=torch.long)
    target_categories = torch.tensor(target_categories, dtype=torch.long)

    # Stack item histories
    item_histories = torch.stack(item_histories)
    category_histories = torch.stack(category_histories)  # ⭐ 추가

    # Combine other features
    other_features = {}
    for key in other_features_list[0].keys():
        other_features[key] = torch.tensor(
            [item[key] for item in other_features_list],
            dtype=torch.long
        )

    # Stack labels
    labels = torch.stack(labels)

    return target_items, target_categories, item_histories, category_histories, other_features, labels


def get_taobao_dataloader(data_path, metadata_path, batch_size=1024, shuffle=True, num_workers=0):
    """
    Taobao DataLoader 생성

    Args:
        data_path: train/val/test.parquet 경로
        metadata_path: metadata.pkl 경로
        batch_size: 배치 크기
        shuffle: 셔플 여부
        num_workers: 워커 수

    Returns:
        DataLoader, metadata
    """
    dataset = TaobaoDataset(data_path, metadata_path)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False  # MPS에서는 pin_memory 사용 안함
    )

    return dataloader, dataset.metadata


if __name__ == '__main__':
    # 테스트
    print("="*60)
    print("🧪 Taobao 데이터셋 로더 테스트")
    print("="*60)

    data_dir = Path('data/processed/taobao')

    # DataLoader 생성
    train_loader, metadata = get_taobao_dataloader(
        data_path=data_dir / 'train.parquet',
        metadata_path=data_dir / 'metadata.pkl',
        batch_size=128,
        shuffle=True
    )

    # 첫 번째 배치 확인
    target_items, target_categories, item_histories, category_histories, other_features, labels = next(iter(train_loader))

    print(f"\n📦 배치 정보:")
    print(f"   - Target items shape: {target_items.shape}")
    print(f"   - Target categories shape: {target_categories.shape}")
    print(f"   - Item histories shape: {item_histories.shape}")
    print(f"   - Category histories shape: {category_histories.shape}")  # ⭐ 추가
    print(f"   - Other features: {list(other_features.keys())}")
    print(f"   - Labels shape: {labels.shape}")
    print(f"   - Labels range: [{labels.min():.0f}, {labels.max():.0f}]")
    print(f"   - CTR: {labels.mean():.4f}")

    # 샘플 확인
    print(f"\n�� 샘플 데이터:")
    print(f"   - Target item: {target_items[0].item()}")
    print(f"   - Target category: {target_categories[0].item()}")
    print(f"   - Item history: {item_histories[0].tolist()}")
    print(f"   - Category history: {category_histories[0].tolist()}")  # ⭐ 추가
    print(f"   - User ID: {other_features['user_id'][0].item()}")
    print(f"   - Hour: {other_features['hour'][0].item()}")
    print(f"   - Label: {labels[0].item()}")

    print("\n" + "="*60)
    print("✅ 테스트 완료!")
    print("="*60)
