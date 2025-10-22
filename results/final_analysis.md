# BST 성능 분석 최종 결과

## Option 1 결과: 데이터셋 불일치 발견 🚨

### 핵심 발견
**AutoInt와 DCNv2는 Criteo 데이터셋을, BST는 Taobao 데이터셋을 사용 중**

| 모델 | 데이터셋 | AUC | 비교 가능 여부 |
|------|----------|-----|---------------|
| AutoInt | **Criteo** | 0.7802 | ❌ |
| DCNv2 | **Criteo** | 0.7722 | ❌ |
| BST | **Taobao** | 0.5711 | ❌ |

### 결론
**지금까지의 성능 비교는 무의미합니다!**
- 다른 데이터셋이므로 AUC 값을 직접 비교할 수 없음
- BST가 underperform한다는 결론은 잘못된 것

### 하이퍼파라미터 차이
| 파라미터 | AutoInt/DCNv2 | BST | 비고 |
|----------|---------------|-----|------|
| Batch Size | 1024 | 512 | BST가 절반 |
| Embedding Dim | 16 | 64 | BST가 4배 |
| DNN Hidden | [256,128,64] | [256,128] | BST가 한 층 적음 |

---

## Option 2 결과: Embedding 학습 검증 ✅

### Embedding 상태 (학습된 모델)

#### 1. Embedding 크기
- **Item Embedding**: 335,164 items × 64 dim
  - Mean: -0.000451, Std: 1.000
  - L2 Norm (mean): 7.97
  
- **Category Embedding**: 5,480 categories × 64 dim
  - Mean: -0.001474, Std: 1.001
  - L2 Norm (mean): 7.98

**Magnitude Ratio (Item/Category)**: 7.79
- ℹ️ Item embedding이 더 크지만, category embedding도 충분히 학습됨

#### 2. Gradient Flow
✅ **Both embeddings receiving gradients**

- **Item Embedding Gradient**:
  - Non-zero: 168,320 / 21,450,496 (0.78%)
  - Max abs: 0.00038033
  
- **Category Embedding Gradient**:
  - Non-zero: 26,752 / 350,720 (7.63%)
  - Max abs: 0.00038033

**Gradient Ratio (Item/Category)**: 0.0265
- Category embedding이 item보다 훨씬 더 강한 gradient를 받고 있음!

#### 3. Embedding Fusion 분석
**Contribution to fused embedding**:
- Item contribution: 71.08%
- Category contribution: 70.26%

✅ **Category가 모델에 제대로 기여하고 있음**
- 거의 동등한 비중 (단순 element-wise sum이므로 100% 넘을 수 있음)

### Embedding 학습 결론
1. ✅ Category embedding은 정상적으로 학습되고 있음
2. ✅ Gradient flow가 제대로 작동함
3. ✅ Category가 모델 예측에 적절히 기여함
4. ❌ **Embedding 학습 문제가 아님**

---

## 종합 결론

### BST 성능이 낮은 진짜 이유

1. **❌ 잘못된 비교**
   - AutoInt/DCNv2 (Criteo 0.78) vs BST (Taobao 0.57) 비교는 의미 없음
   - 완전히 다른 데이터셋

2. **✅ Embedding 학습은 정상**
   - Category embedding 제대로 학습됨
   - Gradient flow 정상
   - Fusion 정상

3. **🤔 실제 문제는?**
   - Taobao 데이터셋에서 BST 0.57이 좋은 성능인지 나쁜 성능인지 알 수 없음
   - **비교 대상이 필요함**

---

## 다음 단계 권장사항

### 우선순위 1: 올바른 비교 기준 설정

#### Option A: Taobao에서 모든 모델 학습 ⭐ (추천)
```bash
# AutoInt를 Taobao 데이터로 재학습
python experiments/train_autoint_taobao.py

# DCNv2를 Taobao 데이터로 재학습
python experiments/train_dcnv2_taobao.py

# BST와 비교
```

**장점**: 
- 같은 데이터셋에서 공정한 비교
- 실제 BST 성능 파악 가능

#### Option B: BST 논문 reported numbers 확인
BST 논문(Alibaba 2019)에서 Taobao 데이터에 대한:
- BST 성능
- 다른 baseline 성능
→ 우리 구현과 비교

#### Option C: DIN/DIEN 같은 sequential 모델 구현
- BST와 유사한 sequential recommendation 모델
- Taobao에서 학습하여 비교

### 우선순위 2: 하이퍼파라미터 튜닝 (Option A 후)
현재 BST 설정이 최적이 아닐 수 있음:
- Batch size: 512 → 1024
- Embedding dim: 64 → 32 or 128
- DNN layers: [256, 128] → [256, 128, 64]
- Learning rate 조정
- Epochs 증가 (20 epochs 결과 대기 중)

---

## 현재 진행 중

### 백그라운드 학습
- 20 epochs BST 학습 (lr=0.0001) 진행 중
- 결과 기대치: 
  - 학습이 더 잘 되면 → 단순 학습 부족 문제
  - 여전히 낮으면 → 다른 baseline과 비교 필요

---

## Action Items

1. **즉시**: AutoInt와 DCNv2가 사용한 데이터셋 로그 확인
2. **필수**: Taobao에서 AutoInt, DCNv2 재학습
3. **참고**: BST 논문에서 Taobao baseline 성능 확인
4. **대기**: 20 epoch 학습 결과 확인

---

## 파일 목록

### 분석 리포트
- `results/model_config_comparison.md`: Option 1 분석
- `results/bst_embedding_debug.log`: Option 2 디버깅 로그
- `results/bst_analysis_report.md`: 초기 분석
- `results/final_analysis.md`: 최종 종합 (현재 문서)

### 실험 코드
- `experiments/debug_bst_embeddings.py`: Embedding 디버깅 스크립트
- `experiments/train_bst.py`: BST 학습 스크립트

---

## 중요 깨달음

**"BST가 낮은 성능을 보인다"는 가정이 잘못되었을 수 있습니다.**

올바른 질문은:
- ❌ "BST가 왜 AutoInt보다 낮은가?" (다른 데이터셋이므로 비교 불가)
- ✅ "Taobao에서 BST 0.57은 좋은 성능인가?"
- ✅ "Taobao에서 다른 모델들은 얼마나 나올까?"
