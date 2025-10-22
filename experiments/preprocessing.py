"""
Criteo 데이터셋 전처리 스크립트

전처리 단계:
1. 데이터 로딩 (청크 단위)
2. 결측치 처리 (수치형: median, 범주형: 'missing')
3. 수치형 특징 로그 변환
4. 범주형 특징 빈도 필터링 및 인코딩
5. Train/Val/Test 분할
6. 처리된 데이터 저장 (.parquet)

Usage:
    python experiments/preprocessing.py --sample_size 1000000
    python experiments/preprocessing.py --full  # 전체 데이터
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import pickle


class CriteoPreprocessor:
    """Criteo 데이터 전처리 클래스"""

    def __init__(self, data_path, output_dir, sample_size=None, freq_threshold=10):
        """
        Args:
            data_path: train.txt 경로
            output_dir: 전처리된 데이터 저장 경로
            sample_size: 샘플링 크기 (None이면 전체 데이터)
            freq_threshold: 범주형 특징 빈도 임계값
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_size = sample_size
        self.freq_threshold = freq_threshold

        # 컬럼명 정의
        self.num_features = [f'I{i}' for i in range(1, 14)]
        self.cat_features = [f'C{i}' for i in range(1, 27)]
        self.columns = ['label'] + self.num_features + self.cat_features

        # 전처리에 필요한 통계 저장
        self.num_medians = {}
        self.cat_encoders = {}
        self.cat_vocab = {}

        print(f"✅ Preprocessor 초기화 완료")
        print(f"   - 데이터 경로: {self.data_path}")
        print(f"   - 출력 경로: {self.output_dir}")
        print(f"   - 샘플 크기: {self.sample_size if self.sample_size else '전체'}")
        print(f"   - 빈도 임계값: {self.freq_threshold}")

    def load_data(self):
        """데이터 로딩"""
        print(f"\n📥 데이터 로딩 중...")

        df = pd.read_csv(
            self.data_path,
            sep='\t',
            header=None,
            names=self.columns,
            nrows=self.sample_size
        )

        print(f"✅ 데이터 로딩 완료: {df.shape}")
        print(f"   - 메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"   - CTR: {df['label'].mean():.4%}")

        return df

    def handle_missing_values(self, df):
        """결측치 처리"""
        print(f"\n🔧 결측치 처리 중...")

        # 수치형: median으로 대체
        for col in tqdm(self.num_features, desc="수치형 특징"):
            median_val = df[col].median()
            self.num_medians[col] = median_val
            df[col].fillna(median_val, inplace=True)

        # 범주형: 'missing' 문자열로 대체
        for col in tqdm(self.cat_features, desc="범주형 특징"):
            df[col].fillna('missing', inplace=True)

        print(f"✅ 결측치 처리 완료")
        print(f"   - 수치형 median 값: {len(self.num_medians)}개 저장")

        return df

    def transform_numeric_features(self, df):
        """수치형 특징 로그 변환"""
        print(f"\n🔧 수치형 특징 로그 변환 중...")

        for col in tqdm(self.num_features, desc="로그 변환"):
            # 음수 값 처리 후 로그 변환
            df[col] = np.log1p(df[col].clip(lower=0))

        print(f"✅ 로그 변환 완료")

        return df

    def encode_categorical_features(self, df):
        """범주형 특징 인코딩"""
        print(f"\n🔧 범주형 특징 인코딩 중...")

        for col in tqdm(self.cat_features, desc="범주형 인코딩"):
            # 빈도 계산
            value_counts = df[col].value_counts()

            # 빈도가 임계값 이상인 값만 유지, 나머지는 '<unk>'로 변환
            valid_values = value_counts[value_counts >= self.freq_threshold].index.tolist()
            df[col] = df[col].apply(lambda x: x if x in valid_values else '<unk>')

            # Label Encoding
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

            # 인코더 및 vocab 저장
            self.cat_encoders[col] = le
            self.cat_vocab[col] = len(le.classes_)

        print(f"✅ 범주형 인코딩 완료")
        print(f"   - Vocabulary 크기:")
        for col, vocab_size in list(self.cat_vocab.items())[:5]:
            print(f"      {col}: {vocab_size}")
        print(f"      ... (총 {len(self.cat_vocab)}개)")

        return df

    def split_data(self, df, test_size=0.2, val_size=0.1):
        """Train/Val/Test 분할"""
        print(f"\n🔧 데이터 분할 중...")

        # Train + Val vs Test
        train_val, test = train_test_split(
            df, test_size=test_size, random_state=42, stratify=df['label']
        )

        # Train vs Val
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val, test_size=val_ratio, random_state=42, stratify=train_val['label']
        )

        print(f"✅ 데이터 분할 완료")
        print(f"   - Train: {len(train):,} ({len(train)/len(df)*100:.1f}%)")
        print(f"   - Val:   {len(val):,} ({len(val)/len(df)*100:.1f}%)")
        print(f"   - Test:  {len(test):,} ({len(test)/len(df)*100:.1f}%)")

        return train, val, test

    def save_data(self, train, val, test):
        """전처리된 데이터 저장"""
        print(f"\n💾 데이터 저장 중...")

        # Parquet 형식으로 저장
        train.to_parquet(self.output_dir / 'train.parquet', index=False)
        val.to_parquet(self.output_dir / 'val.parquet', index=False)
        test.to_parquet(self.output_dir / 'test.parquet', index=False)

        # 전처리 메타데이터 저장
        metadata = {
            'num_medians': self.num_medians,
            'cat_encoders': self.cat_encoders,
            'cat_vocab': self.cat_vocab,
            'num_features': self.num_features,
            'cat_features': self.cat_features,
            'freq_threshold': self.freq_threshold
        }

        with open(self.output_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)

        print(f"✅ 데이터 저장 완료")
        print(f"   - 저장 위치: {self.output_dir}")
        print(f"   - 파일:")
        for file in ['train.parquet', 'val.parquet', 'test.parquet', 'metadata.pkl']:
            file_path = self.output_dir / file
            if file_path.exists():
                size_mb = file_path.stat().st_size / 1024**2
                print(f"      {file}: {size_mb:.2f} MB")

    def run(self):
        """전체 전처리 파이프라인 실행"""
        print("="*60)
        print("🚀 Criteo 데이터 전처리 시작")
        print("="*60)

        # 1. 데이터 로딩
        df = self.load_data()

        # 2. 결측치 처리
        df = self.handle_missing_values(df)

        # 3. 수치형 특징 변환
        df = self.transform_numeric_features(df)

        # 4. 범주형 특징 인코딩
        df = self.encode_categorical_features(df)

        # 5. 데이터 분할
        train, val, test = self.split_data(df)

        # 6. 데이터 저장
        self.save_data(train, val, test)

        print("\n" + "="*60)
        print("✅ 전처리 완료!")
        print("="*60)
        print(f"\n다음 단계:")
        print(f"  1. python experiments/verify_preprocessing.py  # 데이터 검증")
        print(f"  2. 베이스라인 모델 구현 시작")

        return train, val, test


def main():
    parser = argparse.ArgumentParser(description='Criteo 데이터 전처리')
    parser.add_argument('--data_path', type=str,
                        default='data/raw/dac/train.txt',
                        help='train.txt 경로')
    parser.add_argument('--output_dir', type=str,
                        default='data/processed',
                        help='출력 디렉토리')
    parser.add_argument('--sample_size', type=int, default=1_000_000,
                        help='샘플 크기 (전체 사용시 --full 옵션)')
    parser.add_argument('--full', action='store_true',
                        help='전체 데이터 사용')
    parser.add_argument('--freq_threshold', type=int, default=10,
                        help='범주형 특징 빈도 임계값')

    args = parser.parse_args()

    # 전체 데이터 사용 옵션
    sample_size = None if args.full else args.sample_size

    # 전처리 실행
    preprocessor = CriteoPreprocessor(
        data_path=args.data_path,
        output_dir=args.output_dir,
        sample_size=sample_size,
        freq_threshold=args.freq_threshold
    )

    preprocessor.run()


if __name__ == '__main__':
    main()
