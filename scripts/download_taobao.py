"""
Taobao UserBehavior 데이터셋 다운로드 및 전처리

데이터셋: https://tianchi.aliyun.com/dataset/dataDetail?dataId=649
또는 Kaggle: https://www.kaggle.com/datasets/pavansanagapati/ad-displayclick-data-on-taobaocom

데이터 구조:
- user_id: 사용자 ID
- item_id: 상품 ID
- category_id: 카테고리 ID
- behavior_type: 행동 타입 (pv, buy, cart, fav)
- timestamp: 타임스탬프
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 경로 설정
data_dir = Path('data/raw/taobao')
output_dir = Path('data/processed/taobao')
output_dir.mkdir(parents=True, exist_ok=True)

print("="*60)
print("Taobao UserBehavior 데이터셋 전처리")
print("="*60)

# 샘플 데이터 생성 (실제 데이터 다운로드 전 테스트용)
print("\n📥 샘플 데이터 생성 중...")

# Taobao 형식의 샘플 데이터 생성
np.random.seed(42)

n_samples = 100000
n_users = 1000
n_items = 5000
n_categories = 100

# 사용자 행동 시퀀스 생성
data = {
    'user_id': np.random.randint(1, n_users+1, n_samples),
    'item_id': np.random.randint(1, n_items+1, n_samples),
    'category_id': np.random.randint(1, n_categories+1, n_samples),
    'behavior_type': np.random.choice(['pv', 'cart', 'fav', 'buy'], n_samples, p=[0.7, 0.1, 0.1, 0.1]),
    'timestamp': np.random.randint(1511539200, 1512057600, n_samples)  # 2017-11-25 ~ 2017-12-01
}

df = pd.DataFrame(data)
df = df.sort_values(['user_id', 'timestamp'])

print(f"   샘플 데이터: {len(df):,} rows")
print(f"   사용자 수: {df['user_id'].nunique():,}")
print(f"   상품 수: {df['item_id'].nunique():,}")

# 타겟 생성: buy 행동을 1로, 나머지를 0으로
df['label'] = (df['behavior_type'] == 'buy').astype(int)

print(f"\n📊 레이블 분포:")
print(df['label'].value_counts())
print(f"   CTR: {df['label'].mean():.4f}")

# 사용자별 시퀀스 길이 계산
user_seq_lengths = df.groupby('user_id').size()
print(f"\n📈 사용자별 행동 시퀀스 통계:")
print(f"   평균: {user_seq_lengths.mean():.1f}")
print(f"   중앙값: {user_seq_lengths.median():.1f}")
print(f"   최대: {user_seq_lengths.max()}")

# Feature Engineering
print(f"\n🔧 Feature Engineering...")

# LabelEncoder로 ID 인코딩
le_user = LabelEncoder()
le_item = LabelEncoder()
le_category = LabelEncoder()

df['user_id_encoded'] = le_user.fit_transform(df['user_id'])
df['item_id_encoded'] = le_item.fit_transform(df['item_id'])
df['category_id_encoded'] = le_category.fit_transform(df['category_id'])

# 시간 특성 추출
df['hour'] = pd.to_datetime(df['timestamp'], unit='s').dt.hour
df['dayofweek'] = pd.to_datetime(df['timestamp'], unit='s').dt.dayofweek

# 사용자 행동 시퀀스 특성 (간단 버전)
# 각 사용자의 최근 N개 아이템 ID를 시퀀스로 저장
MAX_SEQ_LEN = 10

user_item_sequences = {}
user_category_sequences = {}

for user_id in tqdm(df['user_id_encoded'].unique(), desc="시퀀스 생성"):
    user_data = df[df['user_id_encoded'] == user_id].sort_values('timestamp')
    item_seq = user_data['item_id_encoded'].tolist()
    cat_seq = user_data['category_id_encoded'].tolist()

    user_item_sequences[user_id] = item_seq
    user_category_sequences[user_id] = cat_seq

# 각 샘플에 대해 히스토리 시퀀스 추가
def get_history_sequence(row_idx, user_id, max_len=MAX_SEQ_LEN):
    """이전 행동 시퀀스 반환"""
    user_seq = user_item_sequences[user_id]
    user_idx = list(df[df['user_id_encoded'] == user_id].index).index(row_idx)

    # 현재 샘플 이전의 행동들
    hist_seq = user_seq[:user_idx][-max_len:]

    # Padding
    if len(hist_seq) < max_len:
        hist_seq = [0] * (max_len - len(hist_seq)) + hist_seq

    return hist_seq

print("   히스토리 시퀀스 생성 중...")
df['item_history'] = [
    get_history_sequence(idx, row['user_id_encoded'])
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="히스토리")
]

# Train/Val/Test 분할
print(f"\n✂️  데이터 분할 중...")
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

print(f"   Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
print(f"   Val:   {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
print(f"   Test:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")

# 메타데이터 저장
metadata = {
    'n_users': df['user_id_encoded'].nunique(),
    'n_items': df['item_id_encoded'].nunique(),
    'n_categories': df['category_id_encoded'].nunique(),
    'max_seq_len': MAX_SEQ_LEN,
    'vocab_sizes': {
        'user_id': df['user_id_encoded'].max() + 1,
        'item_id': df['item_id_encoded'].max() + 1,
        'category_id': df['category_id_encoded'].max() + 1,
        'hour': 24,
        'dayofweek': 7
    }
}

# 저장
print(f"\n💾 데이터 저장 중...")
train_df.to_parquet(output_dir / 'train.parquet', index=False)
val_df.to_parquet(output_dir / 'val.parquet', index=False)
test_df.to_parquet(output_dir / 'test.parquet', index=False)

with open(output_dir / 'metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print(f"   ✅ 저장 완료: {output_dir}")

print("\n" + "="*60)
print("✅ Taobao 데이터 전처리 완료!")
print("="*60)
print(f"\n📂 저장 경로: {output_dir}")
print(f"   - train.parquet: {len(train_df):,} rows")
print(f"   - val.parquet: {len(val_df):,} rows")
print(f"   - test.parquet: {len(test_df):,} rows")
print(f"   - metadata.pkl")
print(f"\n📊 메타데이터:")
for key, value in metadata.items():
    if key != 'vocab_sizes':
        print(f"   {key}: {value}")
    else:
        print(f"   vocab_sizes:")
        for k, v in value.items():
            print(f"      {k}: {v}")
