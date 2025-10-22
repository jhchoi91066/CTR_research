"""
Taobao UserBehavior 실제 데이터셋 다운로드 및 전처리

데이터셋 출처:
1. Kaggle: https://www.kaggle.com/datasets/pavansanagapati/ad-displayclick-data-on-taobaocom
2. Tianchi: https://tianchi.aliyun.com/dataset/dataDetail?dataId=649

데이터 구조:
- user_id: 사용자 ID
- item_id: 상품 ID
- category_id: 카테고리 ID
- behavior_type: 행동 타입 (pv, buy, cart, fav)
- timestamp: 타임스탬프

사용법:
1. Kaggle CLI로 다운로드 (권장):
   kaggle datasets download -d pavansanagapati/ad-displayclick-data-on-taobaocom

2. 또는 수동 다운로드 후 data/raw/taobao/ 에 저장
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# 경로 설정
data_dir = Path('data/raw/taobao')
output_dir = Path('data/processed/taobao')
output_dir.mkdir(parents=True, exist_ok=True)

print("="*60)
print("Taobao UserBehavior 실제 데이터셋 전처리")
print("="*60)

# 실제 데이터 파일 확인
raw_file_candidates = [
    data_dir / 'UserBehavior.csv',
    data_dir / 'taobao_data.csv',
    data_dir / 'user_behavior.csv',
]

raw_file = None
for candidate in raw_file_candidates:
    if candidate.exists():
        raw_file = candidate
        print(f"\n✅ 데이터 파일 발견: {raw_file}")
        break

if raw_file is None:
    print("\n⚠️  실제 데이터 파일을 찾을 수 없습니다.")
    print("\n다음 방법으로 데이터를 다운로드하세요:")
    print("\n1. Kaggle CLI 사용 (권장):")
    print("   kaggle datasets download -d pavansanagapati/ad-displayclick-data-on-taobaocom")
    print("   unzip ad-displayclick-data-on-taobaocom.zip -d data/raw/taobao/")
    print("\n2. 또는 https://www.kaggle.com/datasets/pavansanagapati/ad-displayclick-data-on-taobaocom")
    print("   에서 수동 다운로드 후 data/raw/taobao/ 에 저장")

    # 임시로 소규모 샘플 데이터 사용
    print("\n⚠️  임시로 샘플 데이터를 생성합니다...")
    print("   (실제 성능 검증을 위해서는 실제 데이터 사용 필요)")

    exec(open('scripts/download_taobao.py').read())
    exit(0)

# 실제 데이터 로드
print(f"\n📥 데이터 로딩 중... (이 과정은 시간이 걸릴 수 있습니다)")

# Taobao 데이터는 보통 컬럼명이 없거나 다를 수 있음
# 일반적인 형식: user_id, item_id, category_id, behavior_type, timestamp
try:
    df = pd.read_csv(raw_file, header=None, names=['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp'])
except:
    # 헤더가 있는 경우
    df = pd.read_csv(raw_file)

print(f"   원본 데이터: {len(df):,} rows")
print(f"   컬럼: {df.columns.tolist()}")

# 데이터 샘플링 (전체 데이터가 너무 큰 경우)
SAMPLE_SIZE = 1_000_000  # 100만개로 제한 (필요시 조정)
if len(df) > SAMPLE_SIZE:
    print(f"\n⚠️  데이터가 너무 큽니다 ({len(df):,} rows). {SAMPLE_SIZE:,} rows로 샘플링합니다.")
    df = df.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"   샘플링 후: {len(df):,} rows")

# 정렬
df = df.sort_values(['user_id', 'timestamp'])

print(f"\n📊 데이터 기본 통계:")
print(f"   사용자 수: {df['user_id'].nunique():,}")
print(f"   상품 수: {df['item_id'].nunique():,}")
print(f"   카테고리 수: {df['category_id'].nunique():,}")

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

# 최소 시퀀스 길이 필터링 (너무 짧은 시퀀스 제거)
MIN_SEQ_LEN = 3
user_counts = df.groupby('user_id').size()
valid_users = user_counts[user_counts >= MIN_SEQ_LEN].index
df = df[df['user_id'].isin(valid_users)]

print(f"\n🔧 최소 시퀀스 길이 {MIN_SEQ_LEN} 필터링 후: {len(df):,} rows")

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

# 사용자 행동 시퀀스 특성
# 각 사용자의 최근 N개 아이템 ID를 시퀀스로 저장
MAX_SEQ_LEN = 50  # 논문에서 사용한 길이

print(f"   시퀀스 최대 길이: {MAX_SEQ_LEN}")

# 메모리 효율적인 시퀀스 생성
print("   히스토리 시퀀스 생성 중...")
df = df.reset_index(drop=True)

item_histories = []
for user_id in tqdm(df['user_id_encoded'].unique(), desc="시퀀스 생성"):
    user_mask = df['user_id_encoded'] == user_id
    user_indices = df.index[user_mask].tolist()
    user_items = df.loc[user_mask, 'item_id_encoded'].tolist()

    for i, idx in enumerate(user_indices):
        # 현재 샘플 이전의 행동들
        hist_seq = user_items[:i][-MAX_SEQ_LEN:]

        # Padding
        if len(hist_seq) < MAX_SEQ_LEN:
            hist_seq = [0] * (MAX_SEQ_LEN - len(hist_seq)) + hist_seq

        item_histories.append(hist_seq)

df['item_history'] = item_histories

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
print("✅ Taobao 실제 데이터 전처리 완료!")
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
