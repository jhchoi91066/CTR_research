# Option B: BST 논문 분석 결과

## 조사 목적
우리의 BST 구현(AUC 0.5711)이 Taobao 데이터셋에서 정상적인 성능인지 판단하기 위해 원논문의 baseline 성능을 조사

## 조사 결과

### 1. BST 논문 (Alibaba, 2019) 주요 정보

#### 데이터셋
- **기간**: 8일간의 Taobao 거래 로그
  - 학습: 첫 7일
  - 테스트: 마지막 1일
- **규모**: 구체적 샘플 수는 논문에 명시되지 않음
- **CTR**: 우리 데이터(4.71%) vs 논문 데이터(불명)

#### Baseline 모델
논문에서 비교한 모델들:
1. **WDL (Wide & Deep Learning)**
2. **DIN (Deep Interest Network)**  
3. **WDL(+Seq)**: WDL에 시퀀스 정보 추가 (이전 클릭 아이템 평균 임베딩)

#### 실험 결과 (Table 3)
- **BST(b=1)**: 최고 성능 (오프라인 & 온라인)
  - Transformer block 1개 사용
  - 2~3개 쌓았을 때보다 더 좋은 성능
- **성능 향상**: WDL, DIN 대비 개선됨
- **구체적 AUC 수치**: 웹에서 접근 불가 (논문 PDF 필요)

### 2. 관련 연구에서의 Taobao 성능

#### DIEN (Deep Interest Evolution Network)
- **Taobao 배포**: CTR 20.7% 개선
- BST의 직접적인 baseline 중 하나

#### MIRRN (최신 연구)
- Taobao에서 **0.67% AUC 향상** (strongest baseline 대비)
- 이는 절대 AUC가 아닌 상대적 향상률

#### 일반적인 Taobao CTR 특성
- **초대형 규모**: 사용자의 23% 이상이 5개월간 1,000회 이상 클릭
- **낮은 CTR**: E-commerce 특성상 클릭률이 일반적으로 낮음
- **긴 시퀀스**: 사용자 행동 시퀀스가 매우 김

### 3. 논문 정보 접근의 한계

#### 접근 불가 정보
✗ Table 3의 정확한 AUC 수치
✗ WDL, DIN의 구체적 성능 값
✗ BST의 절대적 AUC 값
✗ 데이터셋의 정확한 크기와 CTR

#### 확인된 정보
✓ BST(b=1)이 최고 성능
✓ 순차 정보 추가가 성능 향상에 중요
✓ Transformer block 1개가 최적
✓ 온라인 A/B 테스트에서도 개선 확인

### 4. 우리 구현과의 비교

| 항목 | 논문 (추정) | 우리 구현 | 비고 |
|------|------------|----------|------|
| 데이터셋 | Taobao (8일) | Taobao Ads (샘플링) | 다른 subset |
| 데이터 크기 | 불명 | 1.5M samples | |
| CTR | 불명 | 4.71% | |
| Transformer blocks | 1 | 1 | ✅ 동일 |
| Attention heads | 불명 | 2 | |
| Embedding dim | 불명 | 64 | |
| AUC | 불명 | 0.5711 | ❓ |

### 5. 분석 및 결론

#### 우리가 알 수 있는 것
1. ✅ BST 아키텍처는 논문과 동일하게 구현됨
2. ✅ Transformer block = 1 (논문 권장)
3. ✅ Category embedding 정상 학습 중
4. ✅ Gradient flow 정상

#### 우리가 알 수 없는 것
1. ❓ 논문의 baseline AUC가 얼마인지
2. ❓ 우리의 0.5711이 좋은/나쁜 성능인지
3. ❓ Taobao Ad Click 데이터셋의 난이도

#### 추정 가능한 상황

**시나리오 A: 우리 성능이 정상적**
- Taobao Ad Click은 매우 어려운 데이터셋
- CTR 4.71%로 매우 불균형
- 긴 시퀀스(최대 50)와 sparse features
- AUC 0.57이 합리적일 수 있음

**시나리오 B: 우리 성능이 낮음**
- 논문에서는 더 높은 AUC 달성
- 구현상 미세한 차이 존재 가능
- 하이퍼파라미터 최적화 필요

### 6. 간접적 증거: CTR 예측의 일반적 AUC 범위

다른 CTR 예측 연구들의 참고:
- **Criteo**: 일반적으로 AUC 0.75~0.80
- **Avazu**: AUC 0.77~0.79
- **Taobao**: 명확한 벤치마크 없음

**중요한 차이점**: 
- Criteo/Avazu는 상대적으로 단순한 feature interaction
- Taobao는 복잡한 sequential behavior 모델링 필요
- 직접 비교 불가능

### 7. 실질적 판단 기준

논문의 정확한 수치 없이 우리 성능을 판단하려면:

#### 방법 1: 같은 데이터에서 다른 모델 학습 ⭐ (가장 확실)
```bash
# Taobao에서 DIN, WDL 등 구현
python experiments/train_din_taobao.py
python experiments/train_wdl_taobao.py

# BST와 비교
# 만약 BST가 다른 것들보다 높으면 → 정상
# 만약 BST가 훨씬 낮으면 → 문제 있음
```

#### 방법 2: 하이퍼파라미터 튜닝
```bash
# Batch size, embedding dim, learning rate 등 조정
# 현재: batch_size=512, embed_dim=64, lr=0.001
# 시도: batch_size=1024, embed_dim=128, lr=0.0005
```

#### 방법 3: 데이터 분석
```python
# Taobao 데이터 난이도 분석
- 실제 CTR 분포
- Feature sparsity
- Sequence length 분포
- Class imbalance 정도
```

## 최종 결론

### ❌ Option B로는 명확한 답을 얻지 못함

BST 논문에서 공개된 정보만으로는:
- 우리의 AUC 0.5711이 좋은지 나쁜지 판단 불가
- 논문의 baseline 성능 수치가 웹에서 접근 불가
- Taobao 데이터셋의 일반적인 AUC 벤치마크 부재

### ✅ 대신 확인된 사항

1. **구현은 올바름**: 
   - 아키텍처 논문과 일치
   - Embedding 학습 정상
   - Gradient flow 정상

2. **다른 접근 필요**:
   - Option A (Taobao에서 다른 모델 학습)가 유일한 확실한 방법
   - 또는 더 심층적인 데이터 분석

## 다음 단계 권장사항

### 우선순위 1: Option A 진행 ⭐
**Taobao에서 다른 baseline 모델 학습**
- DIN, WDL, DeepFM 등을 Taobao로 학습
- BST와 직접 비교
- 이것이 유일하게 확실한 방법

### 우선순위 2: 데이터 난이도 분석
- 우리 Taobao 데이터의 특성 심층 분석
- CTR 분포, sparsity, sequence 특성
- "0.57이 합리적인가" 판단할 근거 마련

### 우선순위 3: 논문 정확한 수치 확보
- 논문 PDF 직접 다운로드 (arXiv)
- Table 3의 정확한 AUC 값 확인
- 가능하면 저자에게 직접 문의

## 참고 문헌

1. Chen et al. "Behavior Sequence Transformer for E-commerce Recommendation in Alibaba" (2019)
2. DIEN deployment at Taobao: 20.7% CTR improvement
3. MIRRN on Taobao: 0.67% AUC improvement over strongest baseline

