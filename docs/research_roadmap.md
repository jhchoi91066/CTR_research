# MDAF (Mamba-DCN with Adaptive Fusion) 연구 로드맵

본 로드맵은 총 4개월(16주) 집중 일정으로 구성되며, 각 단계별 목표, 사용할 기술 스택, 구체적인 실행 방안을 명시합니다.

---

## 🎯 핵심 연구 전략: 두 데이터셋 접근법

### 왜 Criteo와 Taobao 두 데이터셋을 함께 사용하는가?

MDAF는 **두 가지 핵심 능력**을 결합한 하이브리드 모델입니다:

1. **정적 특징 상호작용 모델링** (DCNv3)
2. **동적 순차 행동 모델링** (Mamba4Rec)

각 능력을 가장 엄격하게 검증하기 위해, 두 데이터셋을 전략적으로 활용합니다:

#### Criteo 데이터셋의 역할
- **특성**: 13개 수치형 + 26개 고차원 범주형 특징
- **핵심 과제**: 복잡한 특징 간 상호작용 학습
- **검증 대상**: **DCNv3 구성요소**의 특징 상호작용 능력
- **비교 베이스라인**: xDeepFM, DCNv2, AutoInt (특징 상호작용 모델)
- **목표**: "DCNv3가 Criteo의 복잡한 정적 특징들을 SOTA 수준으로 학습하는가?"

#### Taobao 데이터셋의 역할
- **특성**: 사용자 행동 시퀀스, 시간적 의존성
- **핵심 과제**: 사용자 관심사의 동적 변화 포착
- **검증 대상**: **Mamba4Rec 구성요소**의 순차 모델링 능력
- **비교 베이스라인**: BST, DIN (순차 추천 모델)
- **목표**: "Mamba4Rec이 사용자 행동 패턴을 BST보다 효율적으로 학습하는가?"

#### 전략적 의의
> MDAF가 **두 데이터셋 모두**에서 우수한 성능을 보이면, 이는 단순히 두 모델을 합친 것이 아니라 **각기 다른 데이터 특성에 적응하는 진정한 하이브리드 모델**임을 증명합니다.

마치 10종 경기 선수가 단거리와 투포환 모두에서 뛰어난 것처럼, MDAF는 정적/동적 신호 모두를 효과적으로 활용하는 종합적 우수성을 입증합니다.

---

## 월 1 (Week 1-4): 환경 설정 및 문헌 연구

### 목표
모든 실험을 일관된 환경에서 재현 가능하게 수행할 수 있는 견고한 기반을 마련하고, 핵심 논문 분석을 완료합니다.

### 주요 기술 스택
- **프로그래밍 언어**: Python 3.9+
- **딥러닝 프레임워크**: PyTorch 2.0+
- **핵심 라이브러리**:
  - `deepctr-torch` (베이스라인 모델)
  - `mamba-ssm` (Mamba 구현)
  - `pandas`, `numpy`, `scikit-learn`

### 실행 계획

#### Week 1: 개발 환경 설정
- Conda/venv를 사용한 격리된 Python 가상 환경 구성
- PyTorch 및 CUDA Toolkit 설치 (GPU 필수)
- 필수 라이브러리 설치:
  ```bash
  pip install deepctr-torch mamba-ssm torch torchvision pandas numpy scikit-learn
  ```
- GitHub 리포지토리 생성 및 초기 구조 설정
- 프로젝트 디렉토리 구조 생성:
  ```
  Research/
  ├── data/              # 데이터셋
  ├── models/            # 모델 구현
  ├── experiments/       # 실험 스크립트
  ├── results/           # 결과 저장
  ├── notebooks/         # 분석 노트북
  └── utils/             # 유틸리티 함수
  ```

#### Week 2-3: 핵심 논문 분석
- **DCNv3** 논문 정독 및 구현 분석
- **Mamba** 논문 정독 및 SSM 메커니즘 이해
- **Mamba4Rec** 논문 분석 (순차 추천 적용 방법)
- 각 논문의 핵심 수식 및 알고리즘 정리
- 기존 구현 코드 분석 (GitHub 리포지토리)

#### Week 4: 베이스라인 논문 조사
- DeepFM, xDeepFM, DCNv2, BST, AutoInt 논문 리뷰
- 각 모델의 강점/약점 분석
- MDAF의 차별점 명확화

---

## 월 2 (Week 5-8): 데이터셋 준비 및 베이스라인 구축

### 목표
Criteo와 Taobao 데이터셋을 준비하고, 모든 베이스라인 모델의 학습 파이프라인을 구축합니다.

### 실행 계획

#### Week 5: Criteo 데이터셋 준비
- **Criteo 1TB 데이터셋** 다운로드
- 초기 개발용 샘플링 (1-2일치 데이터)
- **전처리 파이프라인 구축**:
  1. 결측치 처리:
     - 수치형(13개): 평균/중앙값 대체
     - 범주형(26개): 'unknown' 카테고리 생성
  2. 이상치 처리: 로그 변환
  3. 범주형 인코딩: 빈도 기반 필터링 + Label Encoding
  4. Train/Val/Test 분할 (8:1:1)
- 전처리된 데이터를 Parquet 형식으로 저장

#### Week 6: Taobao 데이터셋 준비
- **Taobao User Behavior 데이터셋** 다운로드
- 세션화: `user_id`로 그룹화 + `timestamp` 정렬
- 시퀀스 생성: 각 사용자별 클릭 시퀀스 구축
- 아이템 ID 인덱싱 및 패딩
- Train/Val/Test 분할
- Parquet 형식으로 저장

#### Week 7: 베이스라인 모델 구축 - Criteo (특징 상호작용 검증용)
- `deepctr-torch`를 활용한 구현:
  - **DeepFM**: 1차/2차 특징 상호작용
  - **xDeepFM**: CIN 레이어 (DCNv3와 비교 대상)
  - **DCNv2**: Cross Network v2 (DCNv3와 비교 대상)
  - **AutoInt**: Multi-head Self-Attention
- 소규모 샘플 데이터로 학습 파이프라인 검증
- 평가 지표 계산: AUC, Logloss
- 실험 결과 로깅 시스템 구축

**목표**: DCNv3의 성능 비교를 위한 Criteo 베이스라인 확보

#### Week 8: 베이스라인 모델 구축 - Taobao (순차 모델링 검증용)
- 순차 모델 구현:
  - **BST**: Transformer 기반 순차 모델링 (Mamba4Rec과 비교 대상)
  - **DeepFM/AutoInt (Taobao 버전)**: 교차 검증용
- Taobao 데이터셋으로 학습 파이프라인 검증
- 두 데이터셋에 대한 통합 실험 프레임워크 완성

**목표**: Mamba4Rec의 성능 비교를 위한 Taobao 베이스라인 확보

**중요**:
- Criteo 베이스라인: xDeepFM, DCNv2, AutoInt 등 특징 상호작용 모델
- Taobao 베이스라인: BST + 최소 1-2개의 추가 모델 (공정한 비교를 위해)

---

## 월 3 (Week 9-12): MDAF 모델 구현 및 단위 테스트

### 목표
MDAF 모델을 완전히 구현하고, 각 구성 요소의 정상 동작을 확인합니다.

### 실행 계획

#### Week 9: DCNv3 Module 구현
- **DCNv3 레이어 구현**:
  - Low-rank Cross Network 구조
  - Mixture of Experts (MoE) 메커니즘
  - 효율적 행렬 연산 최적화
- 더미 텐서를 활용한 단위 테스트
- Output shape 및 gradient flow 검증

#### Week 10: Mamba4Rec Module 구현
- **Mamba4Rec 레이어 구현**:
  - Selective SSM 메커니즘
  - 순차 데이터 인코딩
  - Position encoding 통합
- 시퀀스 데이터로 단위 테스트
- 계산 효율성 검증 (O(N) 복잡도 확인)

#### Week 11: Adaptive Fusion Layer 구현
- **학습 가능한 융합 메커니즘 구현**:
  ```python
  # Gating mechanism
  gate = sigmoid(W_g @ [h_dcn; h_mamba] + b_g)
  h_fused = gate * h_dcn + (1 - gate) * h_mamba
  ```
- 게이트 값 시각화 도구 구현
- Ablation을 위한 대안 융합 방식 구현:
  - Simple Concatenation
  - Weighted Sum (고정 가중치)

#### Week 12: MDAF 통합 및 End-to-End 테스트
- 전체 MDAF 모델 클래스 구현
- `deepctr-torch`의 `BaseModel` 스타일로 통합
- 소규모 데이터로 Forward/Backward pass 검증
- 학습 안정성 테스트
- 메모리 사용량 및 속도 프로파일링

---

## 월 4 (Week 13-16): 실험 실행 및 결과 분석

### 목표
체계적인 실험을 통해 MDAF의 성능을 검증하고, 논문 작성을 위한 결과를 수집합니다.

### 실행 계획

#### Week 13: 하이퍼파라미터 튜닝
- 검증 데이터셋을 활용한 최적 하이퍼파라미터 탐색
- **주요 튜닝 대상**:
  - Embedding dimension: [16, 32, 64]
  - Learning rate: [1e-4, 5e-4, 1e-3]
  - Dropout rate: [0.1, 0.3, 0.5]
  - DCNv3 cross layers: [2, 3, 4]
  - Mamba hidden dimension: [64, 128, 256]
  - Batch size: [512, 1024, 2048]
- `Optuna`를 활용한 자동 탐색
- 베이스라인 모델들도 동일하게 튜닝

#### Week 14: 전체 데이터셋 학습 및 평가

**두 데이터셋 실험 전략**:

**실험 1: Criteo 데이터셋 (특징 상호작용 능력 검증)**
- 튜닝된 최적 하이퍼파라미터 사용
- **5개 랜덤 시드**로 다음 모델 학습:
  - 베이스라인: DeepFM, xDeepFM, DCNv2, AutoInt
  - DCNv3 (단독): MDAF의 구성요소 검증
  - **MDAF (Full)**: 최종 모델
- 검증 목표: "DCNv3와 MDAF가 Criteo의 복잡한 특징 상호작용을 SOTA 수준으로 학습하는가?"
- 기대: MDAF > DCNv3 (단독) > xDeepFM/DCNv2

**실험 2: Taobao 데이터셋 (순차 모델링 능력 검증)**
- 튜닝된 최적 하이퍼파라미터 사용
- **5개 랜덤 시드**로 다음 모델 학습:
  - 베이스라인: BST, DeepFM (Taobao), AutoInt (Taobao)
  - Mamba4Rec (단독): MDAF의 구성요소 검증
  - **MDAF (Full)**: 최종 모델
- 검증 목표: "Mamba4Rec과 MDAF가 사용자 행동 순서를 BST보다 효율적으로 학습하는가?"
- 기대: MDAF > Mamba4Rec (단독) > BST

**공통 평가**:
- 테스트셋 평가: **AUC**, **Logloss**, **추론 속도**
- 평균 및 표준편차 계산
- 학습 곡선 및 수렴 속도 비교
- 통계적 유의성 검정 (paired t-test, p < 0.001)

#### Week 15: Ablation Study 및 분석

**Ablation 실험 (두 데이터셋 모두에서 수행)**:

**변형 모델**:
  1. **MDAF-Full**: DCNv3 + Mamba4Rec + Adaptive Fusion (전체)
  2. **MDAF w/o Fusion**: DCNv3 + Mamba4Rec (단순 concatenation)
  3. **DCNv3 only**: DCNv3만 사용
  4. **Mamba4Rec only**: Mamba4Rec만 사용

**분석 목표**:
- **Criteo 결과 해석**:
  - DCNv3 only vs Mamba only → Criteo에서는 DCNv3가 더 중요함을 예상
  - MDAF-Full vs DCNv3 only → Mamba가 추가 기여하는가?
  - MDAF-Full vs w/o Fusion → Adaptive Fusion의 효과

- **Taobao 결과 해석**:
  - Mamba only vs DCNv3 only → Taobao에서는 Mamba가 더 중요함을 예상
  - MDAF-Full vs Mamba only → DCNv3가 추가 기여하는가?
  - MDAF-Full vs w/o Fusion → Adaptive Fusion의 효과

**정성적 분석**:
  - 게이트 값 분포 시각화 (데이터셋별)
  - Criteo vs Taobao에서의 융합 패턴 차이
  - DCNv3/Mamba의 상대적 기여도 분석

#### Week 16: 결과 정리 및 논문 초안 작성
- 모든 실험 결과를 표로 정리
- 통계적 유의성 검정 (paired t-test)
- 시각화 자료 생성:
  - 성능 비교 막대 그래프
  - 학습 곡선
  - 게이트 값 히트맵
  - Ablation 결과 비교
- 논문 초안 작성:
  - Abstract & Introduction
  - Related Work
  - Proposed Method (MDAF 아키텍처 상세 설명)
  - Experiments (설정, 결과, 분석)
  - Conclusion & Future Work
- 목표 학회 제출 준비 (WSDM 2025 / CIKM 2025 / RecSys 2025)

---

## 주요 체크포인트

### 월별 완료 기준
- **월 1 완료**: ✅ 환경 설정 완료 + 핵심 논문 3편 이상 분석
- **월 2 완료**: ✅ 두 데이터셋 전처리 완료 + 베이스라인 6종 학습 성공
- **월 3 완료**: ✅ MDAF 전체 구현 완료 + 단위 테스트 통과
- **월 4 완료**: ✅ 실험 완료 + 논문 초안 작성

### 리스크 관리
- **데이터 문제**: 샘플 데이터로 먼저 파이프라인 검증
- **구현 난이도**: 기존 오픈소스 코드 참조 및 모듈화
- **학습 시간**: GPU 리소스 확보, 효율적 배치 실험
- **성능 미달**: Ablation을 통한 문제 진단, 하이퍼파라미터 재탐색

---

## 참고 자료

### 핵심 논문
1. **DCNv3**: "DCN V3: Towards Effective Deep Cross Networks for CTR Prediction" (2023)
2. **Mamba**: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
3. **Mamba4Rec**: "Mamba4Rec: Towards Efficient Sequential Recommendation with SSM" (2024)

### 베이스라인 논문
4. **DeepFM**: "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction" (IJCAI 2017)
5. **xDeepFM**: "xDeepFM: Combining Explicit and Implicit Feature Interactions" (KDD 2018)
6. **DCNv2**: "DCN V2: Improved Deep & Cross Network" (WWW 2021)
7. **BST**: "Behavior Sequence Transformer for E-commerce Recommendation" (RecSys 2019)
8. **AutoInt**: "AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks" (CIKM 2019)

### 데이터셋
- **Criteo 1TB**: https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/
- **Taobao User Behavior**: https://tianchi.aliyun.com/dataset/649

### 코드 리포지토리
- **DeepCTR-Torch**: https://github.com/shenweichen/DeepCTR-Torch
- **Mamba**: https://github.com/state-spaces/mamba
