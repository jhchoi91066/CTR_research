"""
전처리된 데이터 검증 스크립트

Usage:
    python experiments/verify_preprocessing.py
"""

import pandas as pd
import pickle
from pathlib import Path


def verify_preprocessing(data_dir='data/processed'):
    """전처리된 데이터 검증"""
    data_dir = Path(data_dir)

    print("="*60)
    print("🔍 전처리 데이터 검증")
    print("="*60)

    # 1. 파일 존재 확인
    print("\n1️⃣ 파일 존재 확인:")
    required_files = ['train.parquet', 'val.parquet', 'test.parquet', 'metadata.pkl']
    for file in required_files:
        file_path = data_dir / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / 1024**2
            print(f"   ✅ {file}: {size_mb:.2f} MB")
        else:
            print(f"   ❌ {file}: 파일 없음")

    # 2. 메타데이터 로딩
    print("\n2️⃣ 메타데이터 확인:")
    with open(data_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    print(f"   - 수치형 특징: {len(metadata['num_features'])}개")
    print(f"   - 범주형 특징: {len(metadata['cat_features'])}개")
    print(f"   - 빈도 임계값: {metadata['freq_threshold']}")
    print(f"   - Vocabulary 크기 (상위 5개):")
    for col, size in list(metadata['cat_vocab'].items())[:5]:
        print(f"      {col}: {size}")

    # 3. 데이터 로딩 및 검증
    print("\n3️⃣ 데이터 로딩 및 기본 검증:")
    train = pd.read_parquet(data_dir / 'train.parquet')
    val = pd.read_parquet(data_dir / 'val.parquet')
    test = pd.read_parquet(data_dir / 'test.parquet')

    print(f"   - Train: {train.shape}")
    print(f"   - Val:   {val.shape}")
    print(f"   - Test:  {test.shape}")

    # 4. 결측치 확인
    print("\n4️⃣ 결측치 확인:")
    for name, df in [('Train', train), ('Val', val), ('Test', test)]:
        missing = df.isnull().sum().sum()
        print(f"   - {name}: {missing}개 (전체의 {missing/df.size*100:.4f}%)")

    # 5. 라벨 분포 확인
    print("\n5️⃣ 라벨 분포:")
    for name, df in [('Train', train), ('Val', val), ('Test', test)]:
        ctr = df['label'].mean()
        print(f"   - {name} CTR: {ctr:.4%}")

    # 6. 수치형 특징 범위 확인
    print("\n6️⃣ 수치형 특징 통계 (Train):")
    num_stats = train[metadata['num_features']].describe()
    print(num_stats)

    # 7. 범주형 특징 범위 확인
    print("\n7️⃣ 범주형 특징 고유값 개수 (Train):")
    cat_nunique = train[metadata['cat_features']].nunique()
    print(cat_nunique)

    print("\n" + "="*60)
    print("✅ 검증 완료!")
    print("="*60)

    return train, val, test, metadata


if __name__ == '__main__':
    verify_preprocessing()
