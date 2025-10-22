# 데이터셋 다운로드 가이드

## 1. Criteo 데이터셋

### 옵션 A: Kaggle Criteo Display Advertising Challenge (추천)
**크기**: ~11GB (압축), ~45M 레코드
**URL**: https://www.kaggle.com/c/criteo-display-ad-challenge/data

**다운로드 방법**:

#### 1-1. Kaggle CLI 사용 (자동화)
```bash
# Kaggle API 설치
pip install kaggle

# Kaggle API 토큰 설정
# 1. https://www.kaggle.com/settings 접속
# 2. "Create New API Token" 클릭
# 3. kaggle.json 다운로드
# 4. 파일을 ~/.kaggle/kaggle.json으로 이동
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 데이터 다운로드
cd data/raw
kaggle competitions download -c criteo-display-ad-challenge
unzip criteo-display-ad-challenge.zip
```

#### 1-2. 수동 다운로드
1. Kaggle 계정으로 로그인
2. https://www.kaggle.com/c/criteo-display-ad-challenge/data 접속
3. "Download All" 버튼 클릭
4. `data/raw/` 디렉토리로 이동
5. 압축 해제

**파일 구조**:
```
data/raw/
├── train.txt          # 학습 데이터 (~45M rows)
├── test.txt           # 테스트 데이터 (라벨 없음)
└── sampleSubmission.csv
```

### 옵션 B: Criteo 1TB 데이터셋 (선택)
**크기**: ~1TB
**URL**: https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/

⚠️ **주의**: 매우 큰 용량, 초기 실험에는 Kaggle 버전 추천

---

## 2. Taobao 데이터셋

### Taobao User Behavior Dataset
**크기**: ~3.5GB (압축), ~100M 레코드
**URL**: https://tianchi.aliyun.com/dataset/649

**다운로드 방법**:

#### 2-1. 수동 다운로드 (필수)
1. Alibaba Cloud 계정 생성 (무료)
2. https://tianchi.aliyun.com/dataset/649 접속
3. "데이터 다운로드" 버튼 클릭
4. `UserBehavior.csv.zip` 다운로드
5. `data/raw/` 디렉토리로 이동
6. 압축 해제

```bash
cd data/raw
unzip UserBehavior.csv.zip
```

**파일 구조**:
```
data/raw/
└── UserBehavior.csv   # user_id, item_id, category_id, behavior_type, timestamp
```

**데이터 컬럼**:
- `user_id`: 사용자 ID
- `item_id`: 아이템 ID
- `category_id`: 카테고리 ID
- `behavior_type`: pv(페이지뷰), buy(구매), cart(장바구니), fav(즐겨찾기)
- `timestamp`: Unix timestamp

#### 2-2. 대안: Kaggle 버전
Tianchi 접속이 어려운 경우 Kaggle 미러 사용:
```bash
kaggle datasets download -d pavansanagapati/e-commerce-user-behavior-data
```

---

## 3. 데이터 샘플링 (빠른 실험용)

전체 데이터셋이 너무 큰 경우 샘플링:

```python
# utils/data_sampler.py
import pandas as pd

# Criteo 샘플링 (1M rows)
criteo_sample = pd.read_csv('data/raw/train.txt', sep='\t', nrows=1_000_000, header=None)
criteo_sample.to_csv('data/raw/criteo_sample_1m.csv', index=False)

# Taobao 샘플링 (2M rows)
taobao_sample = pd.read_csv('data/raw/UserBehavior.csv', nrows=2_000_000)
taobao_sample.to_csv('data/raw/taobao_sample_2m.csv', index=False)

print("샘플링 완료!")
```

실행:
```bash
python utils/data_sampler.py
```

---

## 4. 다운로드 확인

```bash
# 파일 크기 확인
ls -lh data/raw/

# 예상 출력:
# -rw-r--r--  11G  train.txt           (Criteo)
# -rw-r--r--  3.5G UserBehavior.csv    (Taobao)
```

---

## 다음 단계

데이터 다운로드 완료 후:
1. `notebooks/01_data_exploration.ipynb` 실행 (EDA)
2. `experiments/preprocessing.py` 실행 (전처리)
3. 전처리된 데이터 검증

---

## 문제 해결

### Kaggle API 오류
```bash
# 권한 오류 시
chmod 600 ~/.kaggle/kaggle.json

# 파이썬 경로 문제 시
python -m pip install --upgrade kaggle
```

### 메모리 부족
- 샘플링 사용 (1-2M rows)
- 청크 단위 읽기 (pandas.read_csv(..., chunksize=100000))

### 다운로드 속도 느림
- 학교/회사 네트워크 사용
- wget 또는 aria2c 사용 (멀티스레드 다운로드)
