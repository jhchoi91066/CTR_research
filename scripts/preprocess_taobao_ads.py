"""
Taobao Ad Click 데이터셋 전처리

데이터 구조:
- raw_sample.csv: user, time_stamp, adgroup_id, pid, nonclk, clk
- ad_feature.csv: adgroup_id, cate_id, campaign_id, customer, brand, price
- user_profile.csv: userid, cms_segid, cms_group_id, final_gender_code, age_level, pvalue_level, shopping_level, occupation, new_user_class_level

BST 모델을 위한 시퀀스 특성 생성:
- 사용자의 과거 클릭한 광고 시퀀스를 item_history로 사용
- adgroup_id를 item_id로, cate_id를 category_id로 사용
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
print("Taobao Ad Click 데이터셋 전처리")
print("="*60)

# 1. raw_sample.csv 로드 (메인 데이터)
print("\n📥 raw_sample.csv 로딩 중...")
df = pd.read_csv(data_dir / 'raw_sample.csv')
print(f"   원본 데이터: {len(df):,} rows")
print(f"   컬럼: {df.columns.tolist()}")

# 2. ad_feature.csv 로드 (광고 특성)
print("\n📥 ad_feature.csv 로딩 중...")
ad_features = pd.read_csv(data_dir / 'ad_feature.csv')
print(f"   광고 특성: {len(ad_features):,} rows")

# 3. user_profile.csv 로드 (사용자 프로필)
print("\n📥 user_profile.csv 로딩 중...")
user_profile = pd.read_csv(data_dir / 'user_profile.csv')
print(f"   사용자 프로필: {len(user_profile):,} rows")

# 4. 데이터 조인
print("\n🔗 데이터 조인 중...")
df = df.merge(ad_features, on='adgroup_id', how='left')
df = df.merge(user_profile, left_on='user', right_on='userid', how='left')

print(f"   조인 후: {len(df):,} rows")

# 5. 샘플링 (너무 크면)
SAMPLE_SIZE = 2_000_000  # 200만개로 제한
if len(df) > SAMPLE_SIZE:
    print(f"\n⚠️  데이터가 너무 큽니다 ({len(df):,} rows). {SAMPLE_SIZE:,} rows로 샘플링합니다.")
    df = df.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"   샘플링 후: {len(df):,} rows")

# 6. 타겟 변수 생성
df['label'] = df['clk']  # clk = 1이면 클릭, 0이면 비클릭

print(f"\n📊 레이블 분포:")
print(df['label'].value_counts())
print(f"   CTR: {df['label'].mean():.4f}")

# 7. 시간 순 정렬
df = df.sort_values(['user', 'time_stamp'])
print(f"\n📈 시간 순 정렬 완료")

# 8. 필수 컬럼 결측치 처리
print(f"\n🔧 결측치 처리 중...")
df['cate_id'] = df['cate_id'].fillna(0).astype(int)
df['brand'] = df['brand'].fillna(0).astype(int)
df['price'] = df['price'].fillna(df['price'].median())

# 사용자 프로필 결측치 처리
df['final_gender_code'] = df['final_gender_code'].fillna(0).astype(int)
df['age_level'] = df['age_level'].fillna(0).astype(int)
df['pvalue_level'] = df['pvalue_level'].fillna(0).astype(int)

# 9. Feature Engineering
print(f"\n🔧 Feature Engineering...")

# LabelEncoder로 ID 인코딩
le_user = LabelEncoder()
le_adgroup = LabelEncoder()
le_cate = LabelEncoder()

df['user_id_encoded'] = le_user.fit_transform(df['user'])
df['item_id_encoded'] = le_adgroup.fit_transform(df['adgroup_id'])  # adgroup_id를 item_id로
df['category_id_encoded'] = le_cate.fit_transform(df['cate_id'])

# 시간 특성 추출
df['hour'] = pd.to_datetime(df['time_stamp'], unit='s').dt.hour
df['dayofweek'] = pd.to_datetime(df['time_stamp'], unit='s').dt.dayofweek

print(f"   사용자 수: {df['user_id_encoded'].nunique():,}")
print(f"   광고그룹 수: {df['item_id_encoded'].nunique():,}")
print(f"   카테고리 수: {df['category_id_encoded'].nunique():,}")

# 10. 사용자 행동 시퀀스 생성
MAX_SEQ_LEN = 50  # 논문 권장 길이

print(f"\n📊 사용자 행동 시퀀스 생성 중... (MAX_SEQ_LEN={MAX_SEQ_LEN})")

# 최소 시퀀스 길이 필터링
user_counts = df.groupby('user_id_encoded').size()
valid_users = user_counts[user_counts >= 3].index
df = df[df['user_id_encoded'].isin(valid_users)]

print(f"   최소 시퀀스 길이 3 필터링 후: {len(df):,} rows")
print(f"   유효 사용자 수: {len(valid_users):,}")

# 메모리 효율적인 시퀀스 생성
df = df.reset_index(drop=True)

item_histories = []
category_histories = []  # ⭐ 추가: 카테고리 히스토리
print("   히스토리 시퀀스 생성 중...")

for user_id in tqdm(df['user_id_encoded'].unique(), desc="시퀀스 생성"):
    user_mask = df['user_id_encoded'] == user_id
    user_indices = df.index[user_mask].tolist()
    user_items = df.loc[user_mask, 'item_id_encoded'].tolist()
    user_categories = df.loc[user_mask, 'category_id_encoded'].tolist()  # ⭐ 추가

    for i, idx in enumerate(user_indices):
        # 현재 샘플 이전의 클릭한 아이템들만
        hist_seq = user_items[:i][-MAX_SEQ_LEN:]
        hist_cat_seq = user_categories[:i][-MAX_SEQ_LEN:]  # ⭐ 추가

        # Padding
        if len(hist_seq) < MAX_SEQ_LEN:
            hist_seq = [0] * (MAX_SEQ_LEN - len(hist_seq)) + hist_seq
            hist_cat_seq = [0] * (MAX_SEQ_LEN - len(hist_cat_seq)) + hist_cat_seq  # ⭐ 추가

        item_histories.append(hist_seq)
        category_histories.append(hist_cat_seq)  # ⭐ 추가

df['item_history'] = item_histories
df['category_history'] = category_histories  # ⭐ 추가

# 11. Train/Val/Test 분할
print(f"\n✂️  데이터 분할 중...")
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

print(f"   Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
print(f"   Val:   {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
print(f"   Test:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")

print(f"\n   Train CTR: {train_df['label'].mean():.4f}")
print(f"   Val CTR:   {val_df['label'].mean():.4f}")
print(f"   Test CTR:  {test_df['label'].mean():.4f}")

# 12. 메타데이터 저장
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

# 13. 저장
print(f"\n💾 데이터 저장 중...")

# 필요한 컬럼만 선택
save_columns = [
    'user_id_encoded', 'item_id_encoded', 'category_id_encoded',
    'hour', 'dayofweek', 'item_history', 'category_history', 'label'  # ⭐ category_history 추가
]

train_df[save_columns].to_parquet(output_dir / 'train.parquet', index=False)
val_df[save_columns].to_parquet(output_dir / 'val.parquet', index=False)
test_df[save_columns].to_parquet(output_dir / 'test.parquet', index=False)

with open(output_dir / 'metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print(f"   ✅ 저장 완료: {output_dir}")

print("\n" + "="*60)
print("✅ Taobao Ad Click 데이터 전처리 완료!")
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
