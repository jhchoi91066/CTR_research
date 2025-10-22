# Mamba-DCN with Adaptive Fusion (MDAF) 연구 계획서
## 간소화 버전 - 4개월 집중 실행 계획

---

## 📋 Executive Summary

**연구 목표**: 효율적 순차 모델링(Mamba)과 명시적 특징 상호작용(DCNv3)을 적응형 융합으로 결합한 CTR 예측 모델 개발

**핵심 차별점**: 
- BST → Mamba4Rec (선형 복잡도, 4-5배 빠른 추론)
- CIN → DCNv3 (지수적 교차, Criteo 1위)
- 단순 게이트 → 학습된 적응형 융합 (OptFusion 개념 적용)

**목표 학회**: WSDM 2025, CIKM 2025, 또는 RecSys 2025 Late-Breaking

**예상 기여도**: 
- Criteo에서 0.3-0.5% AUC 개선 (SOTA 대비)
- 추론 속도 2-3배 향상 (BST 대비)
- 실무 배포 가능한 효율성

---

## 🎯 1. 연구 범위 명확화 (Scope Definition)

### 1.1 핵심 연구 질문

**RQ1**: Mamba 기반 순차 모델링이 Transformer 기반(BST) 대비 CTR 예측 성능과 효율성을 동시에 개선할 수 있는가?

**RQ2**: DCNv3의 지수적 특징 교차가 xDeepFM의 CIN 대비 더 효과적인 상호작용 모델링을 제공하는가?

**RQ3**: 적응형 융합 메커니즘이 단순 연결 대비 두 구성 요소의 보완성을 더 잘 활용하는가?

### 1.2 간소화된 실험 범위

**데이터셋**: 2개로 집중
- Criteo (특징 상호작용 중심 평가)
- Taobao (순차 모델링 중심 평가)

**베이스라인**: 6개로 제한
1. DeepFM (기본 베이스라인)
2. xDeepFM (CIN 비교군)
3. DCNv2 (교차 네트워크 비교군)
4. BST (순차 모델 비교군)
5. AutoInt (어텐션 기반 비교군)
6. DCNv3 (최신 SOTA, 가능한 경우)

**평가 지표**: 핵심 4개
- AUC (주요 지표)
- LogLoss (보조 지표)
- 추론 지연 (밀리초)
- 모델 크기 (파라미터 수)

**절제 연구**: 4가지 핵심 실험
1. Full MDAF vs w/o Adaptive Fusion (단순 연결)
2. Full MDAF vs Mamba-only
3. Full MDAF vs DCNv3-only
4. Full MDAF vs BST-based variant (순차 구성 요소 비교)

---

## 🏗️ 2. MDAF 모델 아키텍처 (간소화)

### 2.1 전체 구조

```
Input Features
    ↓
Shared Embedding Layer
    ↓
    ├─→ DCNv3 Module ────→ h_dcn
    └─→ Mamba4Rec Module ─→ h_mamba
              ↓
    Adaptive Fusion Layer
         (2가지 변형)
              ↓
         Final MLP
              ↓
    Sigmoid → CTR Prediction
```

### 2.2 핵심 구성 요소

**A. Shared Embedding Layer**
- 모든 입력 특징을 d=32 차원으로 임베딩
- 사용자, 아이템, 문맥, 순차 특징 통합
- Parameter efficiency를 위한 공유 구조

**B. DCNv3 Module (명시적 특징 상호작용)**
```python
# 지수적 교차 레이어
class ExponentialCrossLayer(nn.Module):
    def __init__(self, input_dim):
        self.weight = nn.Parameter(torch.randn(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
    
    def forward(self, x0, x):
        # x0: 초기 입력, x: 현재 레이어 입력
        return x0 * (self.weight @ x) + self.bias + x
```
- 정적 특징(user, item, context) 입력
- 3-layer stacked cross network
- 출력: h_dcn ∈ R^d

**C. Mamba4Rec Module (효율적 순차 모델링)**
```python
# Simplified Mamba block
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16):
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.ssm = SelectiveSSM(d_model, d_state)
        self.out_proj = nn.Linear(d_model, d_model)
```
- 사용자 행동 시퀀스 입력
- 위치 임베딩 추가
- 2-layer Mamba blocks
- 출력: h_mamba ∈ R^d (마지막 타임스텝)

**D. Adaptive Fusion Layer (2가지 변형)**

**변형 1: Soft Fusion (OptFusion 스타일)**
```python
# 4가지 융합 작업을 학습된 가중치로 결합
α_add, α_prod, α_concat, α_att = softmax(learnable_weights)

h_fused = α_add * (h_dcn + h_mamba) + 
          α_prod * (h_dcn ⊙ h_mamba) +
          α_concat * Concat(h_dcn, h_mamba) +
          α_att * Attention(h_dcn, h_mamba)
```

**변형 2: Context-Aware Gating**
```python
# 입력 특징에 따라 동적으로 가중치 결정
context = Concat([h_dcn, h_mamba, user_profile])
gate = Sigmoid(W_gate @ context + b_gate)

h_fused = gate ⊙ h_dcn + (1 - gate) ⊙ h_mamba
```

**E. Prediction Head**
- 2-layer MLP (256 → 128 → 1)
- Dropout 0.1 적용
- Sigmoid 활성화 함수

### 2.3 모델 복잡도 분석

| 구성 요소 | 시간 복잡도 | 공간 복잡도 |
|---------|-----------|-----------|
| DCNv3 | O(Ld²) | O(Ld) |
| Mamba4Rec | O(LNd) | O(Nd) |
| Adaptive Fusion | O(d²) | O(d²) |
| **Total** | **O(Ld² + LNd)** | **O(Ld + Nd)** |

L: DCNv3 레이어 수, N: 시퀀스 길이, d: 임베딩 차원

**BST 대비**: O(N²d) → O(LNd), 시퀀스가 길수록 효율적

---

## 📅 3. 4개월 실행 계획 (주차별)

### 월 1: 기반 구축 (주 1-4)

**주 1: 환경 설정 & 문헌 연구**
- [ ] RecBole 설치 및 환경 구성
- [ ] DCNv3, Mamba4Rec, OptFusion 논문 정독
- [ ] GitHub 리포지토리 생성
- [ ] 연구 노트 시스템 설정 (Notion/Obsidian)

**주 2: 데이터 준비**
- [ ] Criteo 데이터셋 다운로드 (Kaggle 45M)
- [ ] Taobao 데이터셋 확보
- [ ] 데이터 EDA 및 통계 분석
- [ ] 전처리 파이프라인 설계

**주 3: 데이터 전처리 구현**
```python
# 주요 전처리 단계
1. Criteo:
   - 수치형: Log transformation + MinMax scaling
   - 범주형: Frequency encoding (threshold=10)
   - 시퀀스 생성: User ID 기준 그룹화
   
2. Taobao:
   - 세션 기반 시퀀스 구성 (max_len=50)
   - Timestamp 정렬
   - Item ID 인코딩
```
- [ ] 전처리 스크립트 작성
- [ ] Train/Val/Test 분할 (8:1:1)
- [ ] 처리된 데이터 저장 (.pkl 또는 .pt)

**주 4: 베이스라인 구현 시작**
- [ ] RecBole로 DeepFM 구현
- [ ] RecBole로 xDeepFM 구현
- [ ] 초기 학습 파이프라인 검증

**✅ 월 1 Deliverable**: 
- 처리된 데이터셋 (Criteo + Taobao)
- 2개 베이스라인 재현 (AUC 검증)
- 실험 프레임워크 기본 구조

---

### 월 2: 모델 개발 (주 5-8)

**주 5: 베이스라인 완성**
- [ ] DCNv2, AutoInt 구현
- [ ] BST 구현 (RecBole 또는 직접)
- [ ] 모든 베이스라인 Criteo에서 학습 완료

**주 6: MDAF 핵심 구성 요소 개발**
```python
# 구현 순서
Day 1-2: DCNv3 Module
  - ExponentialCrossLayer 클래스
  - 3-layer stacking
  - 단위 테스트 (입출력 shape 검증)

Day 3-4: Mamba4Rec Module  
  - SelectiveSSM 구현 (기존 코드 참고)
  - MambaBlock 클래스
  - 위치 임베딩 통합

Day 5-6: Adaptive Fusion
  - Soft Fusion 구현
  - Context-Aware Gating 구현
  - 융합 메커니즘 비교 테스트

Day 7: 통합 및 디버깅
  - 전체 MDAF 모델 클래스
  - Forward pass 검증
  - Gradient flow 확인
```

**주 7: 초기 학습 & 디버깅**
- [ ] Criteo 소규모 샘플로 빠른 학습
- [ ] Overfitting 테스트 (10K 샘플)
- [ ] 학습 안정성 확인
- [ ] 하이퍼파라미터 초기 탐색

**주 8: 첫 번째 전체 실험**
- [ ] Criteo 전체 데이터로 MDAF 학습
- [ ] 베이스라인과 초기 비교
- [ ] 문제점 분석 및 개선

**✅ 월 2 Deliverable**:
- 작동하는 MDAF 모델 (2가지 융합 변형)
- 6개 베이스라인 구현 완료
- Criteo 초기 결과 (AUC 기준)

---

### 월 3: 최적화 & 실험 (주 9-12)

**주 9: 하이퍼파라미터 튜닝**
```python
# 그리드 서치 범위
search_space = {
    'embedding_dim': [32, 64],
    'learning_rate': [1e-4, 5e-4, 1e-3],
    'dcn_layers': [2, 3],
    'mamba_layers': [2, 3],
    'seq_length': [20, 50],
    'dropout': [0.1, 0.2],
    'batch_size': [2048, 4096]
}

# 우선순위: embedding_dim, learning_rate 먼저
```
- [ ] Validation set으로 그리드 서치
- [ ] Top-3 설정 선정
- [ ] 교차 검증 (3-fold)

**주 10: Taobao 데이터셋 실험**
- [ ] 최적 하이퍼파라미터로 Taobao 학습
- [ ] 베이스라인 비교
- [ ] 순차 모델링 능력 중점 분석

**주 11: 절제 연구**
```python
# 4가지 변형 모델
models = {
    'MDAF-Full': full_model,
    'MDAF-Concat': replace_fusion_with_concat,
    'MDAF-MambaOnly': remove_dcnv3,
    'MDAF-DCNOnly': remove_mamba,
}

# 각 변형에 대해 3회 실행 (다른 seed)
for name, model in models.items():
    for seed in [42, 123, 2024]:
        results = train_and_evaluate(model, seed)
```
- [ ] 4가지 변형 모델 학습
- [ ] 구성 요소 기여도 분석
- [ ] 융합 메커니즘 효과 검증

**주 12: 효율성 분석**
- [ ] 추론 지연 측정 (1000 샘플 평균)
- [ ] 처리량 측정 (예측/초)
- [ ] 모델 크기 비교
- [ ] GPU 메모리 사용량

**✅ 월 3 Deliverable**:
- 2개 데이터셋 최종 결과
- 절제 연구 완료
- 효율성 메트릭
- 모든 실험 로그 및 체크포인트

---

### 월 4: 분석 & 논문 작성 (주 13-16)

**주 13: 결과 정리 & 시각화**
```python
# 주요 시각화
1. 성능 비교 막대 그래프
   - AUC by Model (Criteo & Taobao)
   - Error bars (95% CI)

2. 절제 연구 히트맵
   - Component contribution matrix

3. 효율성 트레이드오프
   - Scatter plot: AUC vs Latency
   - Pareto frontier

4. 융합 가중치 분석
   - Gate activation distribution
   - Contextual patterns

5. 학습 곡선
   - Training/Validation loss
   - AUC over epochs
```
- [ ] 모든 표와 그래프 생성
- [ ] 통계적 유의성 검정 (t-test)
- [ ] 주요 발견 사항 정리

**주 14: 논문 초안 작성**
```markdown
# 구조 (ACM 형식, ~8 페이지)

1. Abstract (200 words)
   - 문제, 방법, 주요 결과

2. Introduction (1.5 pages)
   - 동기 및 연구 공백
   - 3가지 핵심 기여

3. Related Work (1 page)
   - Feature Interaction Models
   - Sequential Recommendation
   - Fusion Mechanisms

4. Methodology (2 pages)
   - MDAF 아키텍처
   - DCNv3 & Mamba4Rec 구성 요소
   - Adaptive Fusion 메커니즘
   - 알고리즘 의사코드

5. Experiments (2.5 pages)
   - 데이터셋 & 설정
   - 베이스라인 비교 (표 1)
   - 절제 연구 (표 2)
   - 효율성 분석 (표 3)
   - 시각화 (그림 2-4)

6. Conclusion (0.5 page)
   - 요약, 한계, 향후 연구

7. References (1 page)
   - 30-40개 주요 논문
```
- [ ] 각 섹션 초안 완성
- [ ] 표와 그림 배치
- [ ] 수식 및 알고리즘 작성

**주 15: 검토 & 수정**
- [ ] 내부 리뷰 (동료/어드바이저)
- [ ] 논리 흐름 개선
- [ ] 명확성 향상 (애매한 표현 제거)
- [ ] 참고문헌 완성
- [ ] 보충 자료 준비

**주 16: 최종 마무리 & 제출**
- [ ] 맞춤법 & 문법 검사
- [ ] 형식 확인 (학회 템플릿)
- [ ] 익명화 (저자 정보 제거)
- [ ] 코드 정리 (GitHub 공개 준비)
- [ ] 제출 전 최종 검토
- [ ] 학회 제출 시스템 업로드

**✅ 월 4 Deliverable**:
- 완성된 논문 원고 (~8 페이지)
- 보충 자료 (추가 실험 결과)
- 공개 가능한 코드베이스
- 학회 제출 완료

---

## 🔬 4. 핵심 실험 설계 (상세)

### 4.1 주요 비교 실험

**실험 1: SOTA 베이스라인과 성능 비교**

| Model | Criteo AUC | Taobao AUC | Latency (ms) | Params (M) |
|-------|-----------|-----------|--------------|-----------|
| DeepFM | 0.8045 | 0.6823 | 1.2 | 8.5 |
| xDeepFM | 0.8078 | 0.6891 | 2.8 | 12.3 |
| DCNv2 | 0.8088 | 0.6905 | 1.8 | 9.2 |
| BST | 0.8062 | 0.6978 | 3.5 | 11.8 |
| AutoInt | 0.8071 | 0.6912 | 2.1 | 10.5 |
| DCNv3 | 0.8115* | 0.6932 | 2.0 | 10.1 |
| **MDAF (Ours)** | **0.8125±0.0003** | **0.7012±0.0005** | **1.9** | **11.4** |

*목표: Criteo에서 +0.10% AUC, Taobao에서 +0.34% AUC*

**실험 2: 절제 연구**

| Model Variant | Criteo AUC | Taobao AUC | 성능 하락 |
|--------------|-----------|-----------|---------|
| MDAF-Full | **0.8125** | **0.7012** | - |
| w/o Adaptive Fusion | 0.8108 | 0.6993 | -0.17%, -0.19% |
| Mamba-only | 0.8091 | 0.6998 | -0.34%, -0.14% |
| DCNv3-only | 0.8118 | 0.6945 | -0.07%, -0.67% |

**핵심 발견 목표**:
1. 적응형 융합이 단순 연결보다 효과적 (실험 2)
2. 두 구성 요소 모두 필수적 (실험 2)
3. Mamba가 순차 데이터(Taobao)에서 특히 효과적
4. DCNv3가 정적 특징(Criteo)에서 핵심적

### 4.2 통계적 검정

```python
# 5회 실행 결과로 paired t-test
from scipy.stats import ttest_rel

# MDAF vs Best Baseline
mdaf_scores = [0.8123, 0.8125, 0.8126, 0.8124, 0.8127]
baseline_scores = [0.8113, 0.8115, 0.8116, 0.8114, 0.8117]

t_stat, p_value = ttest_rel(mdaf_scores, baseline_scores)
print(f"p-value: {p_value:.6f}")  # 목표: p < 0.001
```

**보고 형식**: "MDAF는 DCNv3 대비 Criteo에서 0.10% AUC 개선을 보였으며, 이는 통계적으로 유의미함 (p < 0.001, paired t-test)."

---

## 💡 5. 위험 관리 및 백업 플랜

### 5.1 주요 위험 요소

**위험 1: 성능 개선이 미미함 (< 0.3% AUC)**

백업 플랜:
- 효율성 강조로 전환 (속도 2-3배 향상)
- 논문 포지셔닝: "효율성-성능 트레이드오프 개선"
- 실무 배포 관점 강조
- 추가 분석: 롱테일 아이템, 콜드 스타트

**위험 2: Mamba4Rec 구현 어려움**

백업 플랜:
- 기존 구현 코드 활용 (GitHub)
- LSTM 또는 GRU로 대체 (덜 혁신적이지만 안전)
- SASRec (Self-Attention) 사용 고려

**위험 3: 계산 자원 부족**

백업 플랜:
- Criteo 샘플링 (5-10M 레코드)
- 클라우드 크레딧 활용 (GCP/AWS)
- 학교 GPU 클러스터 신청
- 배치 크기 축소, gradient accumulation

**위험 4: 베이스라인 재현 실패**

백업 플랜:
- 공식 구현 코드 사용 (RecBole, 저자 GitHub)
- 논문 보고 결과 인용 (재현 불가 명시)
- 상대적 개선율로 보고

**위험 5: 논문 제출 마감 못 맞춤**

백업 옵션:
- Workshop 제출 (RecSys, KDD 워크샵)
- arXiv preprint 먼저 게시
- 다음 학회 사이클 (6개월 후)
- 저널 고려 (시간 여유 있음)

### 5.2 일정 버퍼

각 월 마지막 주를 버퍼로 활용:
- 주 4: 데이터 문제 해결
- 주 8: 구현 디버깅
- 주 12: 추가 실험
- 주 16: 논문 수정

---

## 📊 6. 예상 결과 및 기여도

### 6.1 정량적 결과 (목표)

**성능 개선**:
- Criteo: 0.8125 AUC (DCNv3 대비 +0.10%)
- Taobao: 0.7012 AUC (BST 대비 +0.34%)

**효율성 개선**:
- 추론 속도: BST 대비 1.8배 빠름
- 메모리: BST와 유사 (±5%)
- 학습 시간: DCNv3보다 느리지만 수용 가능

### 6.2 핵심 기여 (논문에 명시)

**기여 1**: Mamba 기반 SSM을 CTR 예측에 효과적으로 적용
- 첫 번째 Mamba-CTR 결합 연구 (또는 초기 연구 중 하나)
- 선형 복잡도로 장기 시퀀스 효율적 처리

**기여 2**: DCNv3와 Mamba의 상호보완성 검증
- 정적 vs 동적 신호의 효과적 결합
- 절제 연구로 각 구성 요소 중요성 입증

**기여 3**: 적응형 융합 메커니즘의 효과 실증
- 단순 연결 대비 성능 향상
- 컨텍스트 기반 동적 가중치의 중요성

**기여 4**: 실무 적용 가능성 제시
- 효율성-성능 트레이드오프 우수
- 대규모 시스템 배포 가능한 수준

### 6.3 목표 학회 및 전략

**1순위: CIKM 2025**
- 마감: 2025년 5월
- 채택률: ~25%
- 강점: 정보 검색 & 추천 특화
- 전략: 효율성 강조, 실무 적용성

**2순위: RecSys 2025 Late-Breaking**
- 마감: 2025년 7월
- 짧은 형식 (4 페이지)
- 강점: 최신 결과 빠른 공유
- 전략: 핵심 결과만 간결하게

**3순위: WSDM 2025**
- 마감: 2024년 8월
- 채택률: ~18%
- 강점: 웹 검색 & 데이터 마이닝
- 전략: 대규모 데이터 처리 강조

**백업: AAAI 2025 워크샵**
- 마감: 2024년 11월
- 낮은 기준, 빠른 피드백
- 전략: 예비 결과 공유, 커뮤니티 피드백

---

## ✅ 7. 주요 체크리스트

### 월별 핵심 마일스톤

**월 1 종료 시**:
- [ ] 데이터 전처리 완료
- [ ] 2개 베이스라인 재현
- [ ] 실험 프레임워크 작동

**월 2 종료 시**:
- [ ] MDAF 모델 구현 완료
- [ ] 6개 베이스라인 완성
- [ ] Criteo 초기 결과 확보

**월 3 종료 시**:
- [ ] 2개 데이터셋 최종 결과
- [ ] 절제 연구 완료
- [ ] 통계 검정 완료

**월 4 종료 시**:
- [ ] 논문 원고 완성
- [ ] 코드 정리 및 문서화
- [ ] 학회 제출 완료

### 일일 진행 추적

```markdown
# 주간 리뷰 템플릿 (매주 금요일)

## Week X Summary
- 달성한 것:
- 배운 것:
- 막힌 부분:
- 다음 주 계획:

## Key Metrics
- 코드 라인 수:
- 실험 완료 수:
- AUC 현황:
- 남은 작업:
```

---

## 🚀 8. 즉시 시작 가이드 (Day 1)

### 오늘 할 일 (4시간)

**1. 환경 설정 (1시간)**
```bash
# Python 환경
conda create -n mdaf python=3.9
conda activate mdaf

# 핵심 라이브러리
pip install torch==2.0.0
pip install recbole
pip install pandas numpy scikit-learn
pip install wandb  # 실험 추적

# 프로젝트 구조
mkdir -p mdaf_project/{data,models,experiments,results}
cd mdaf_project
git init
```

**2. 논문 다운로드 (30분)**
- [ ] DCNv3 (arXiv 2407.13349)
- [ ] Mamba4Rec (arXiv 2403.03900)
- [ ] OptFusion (arXiv 2411.15731)
- [ ] BST (DLP-KDD 2019)

**3. 데이터 다운로드 시작 (30분)**
```python
# Criteo Kaggle
# https://www.kaggle.com/c/criteo-display-ad-challenge

# Taobao
# https://tianchi.aliyun.com/dataset/dataDetail?dataId=649
```

**4. 프로젝트 계획 작성 (2시간)**
- [ ] 이 문서를 Notion/Markdown으로 정리
- [ ] Gantt 차트 생성 (주차별 태스크)
- [ ] GitHub 리포지토리 생성
- [ ] README 초안 작성

---

## 📚 9. 참고 자료

### 핵심 논문 (반드시 읽기)

1. **DCNv3** (2024): 지수적 교차 레이어
2. **Mamba4Rec** (2024): SSM 기반 순차 추천
3. **OptFusion** (2024): 자동화된 융합 학습
4. **BST** (2019): Transformer 순차 모델링
5. **xDeepFM** (2018): CIN 아키텍처

### 구현 참고 코드

- RecBole: https://github.com/RUCAIBox/RecBole
- Mamba: https://github.com/state-spaces/mamba
- DeepCTR-Torch: https://github.com/shenweichen/DeepCTR-Torch

### 유용한 도구

- Weights & Biases: 실험 추적
- Overleaf: 논문 작성 (LaTeX)
- Notion: 프로젝트 관리
- Papers with Code: 벤치마크 확인

---

## 💪 마무리: 성공을 위한 조언

1. **매일 조금씩**: 하루 4시간, 주 5일만 집중해도 충분
2. **막히면 건너뛰기**: 완벽주의 금지, 일단 돌아가게
3. **자주 저장**: Git commit 자주, 실험 로그 꼼꼼히
4. **도움 요청**: 막히면 Stack Overflow, GitHub Issues 활용
5. **건강 관리**: 번아웃 방지, 주말은 쉬기

**행운을 빕니다! 🎉**

---

*이 계획은 가이드라인입니다. 상황에 따라 유연하게 조정하세요.*