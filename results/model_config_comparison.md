# 모델 학습 설정 비교 (Option 1 분석)

## 주요 발견: BST만 **다른 데이터셋** 사용 중! 🚨

| 구분 | AutoInt | DCNv2 | BST |
|------|---------|-------|-----|
| **데이터셋** | ⚠️ **Criteo** | ⚠️ **Criteo** | ✅ **Taobao** |
| **데이터 로더** | `utils/dataset.py` | `utils/dataset.py` | `utils/taobao_dataset.py` |
| **입력 형식** | num_features + cat_features | num_features + cat_features | target_item + history + features |

## 학습 하이퍼파라미터 비교

| 파라미터 | AutoInt | DCNv2 | BST |
|----------|---------|-------|-----|
| **Batch Size** | 1024 | 1024 | **512** ⚠️ |
| **Learning Rate** | 0.001 | 0.001 | 0.001 ✅ |
| **Epochs** | 5 | 5 | 5 ✅ |
| **Optimizer** | Adam | Adam | Adam ✅ |
| **Loss Function** | BCELoss | BCELoss | BCELoss ✅ |
| **Embedding Dim** | 16 | 16 | **64** ⚠️ |
| **Dropout** | 0.1 | 0.1 | 0.1 ✅ |

## 모델 아키텍처 비교

| 구성 요소 | AutoInt | DCNv2 | BST |
|-----------|---------|-------|-----|
| **특징 추출** | Self-attention (3 layers) | Cross Network (3 layers) | Transformer (1 layer) |
| **Attention Heads** | 2 | - | 2 |
| **DNN Hidden** | [256, 128, 64] | [256, 128, 64] | **[256, 128]** ⚠️ |
| **추가 구성** | Residual connections | Parallel/Stacked structure | Position encoding |

## ⚠️ 중대한 문제 발견

### 1. **데이터셋 불일치**
- AutoInt와 DCNv2: **Criteo 데이터셋** 사용
- BST: **Taobao 데이터셋** 사용

**결론**: **AutoInt와 DCNv2의 성능(0.78)을 BST와 직접 비교할 수 없습니다!**

다른 데이터셋이므로 AUC 0.78 vs 0.57을 직접 비교하는 것은 의미가 없습니다.

### 2. 하이퍼파라미터 차이
- BST의 batch_size가 절반 (512 vs 1024)
- BST의 embedding_dim이 4배 크다 (64 vs 16)
- BST의 DNN이 한 층 적다 ([256, 128] vs [256, 128, 64])

## 올바른 비교를 위한 방법

### Option A: Taobao에서 모든 모델 학습
```bash
# AutoInt를 Taobao 데이터로 재학습
python experiments/train_autoint_taobao.py

# DCNv2를 Taobao 데이터로 재학습  
python experiments/train_dcnv2_taobao.py

# 그리고 BST와 비교
```

### Option B: BST를 Criteo로 학습
```bash
# BST를 Criteo 데이터로 학습
python experiments/train_bst_criteo.py

# AutoInt, DCNv2와 비교
```

### Option C: 논문에서 보고된 Taobao 성능 확인
BST 논문(Alibaba 2019)에서 Taobao 데이터에 대한 다른 baseline 성능 확인

## 추가 확인 필요 사항

1. **AutoInt, DCNv2가 Taobao에서 실제로 0.78 AUC를 달성했는지?**
   - 현재 코드로는 Criteo에서만 학습되는 것으로 보임
   - 로그 파일 확인 필요

2. **BST 0.57이 정말 낮은 성능인지?**
   - Taobao 데이터셋의 난이도
   - 다른 baseline들의 Taobao에서의 실제 성능
   - 논문 reported numbers와 비교

## 다음 단계 권장사항

### 🔥 최우선: 데이터셋 확인
```bash
# AutoInt의 실제 학습 로그 확인
cat results/train_autoint.log | grep "dataset\|data"

# DCNv2의 실제 학습 로그 확인  
cat results/train_dcnv2.log | grep "dataset\|data"
```

이것이 확인되기 전까지는 **BST가 underperform한다고 결론 내릴 수 없습니다!**
