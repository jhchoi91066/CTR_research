const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        AlignmentType, HeadingLevel, BorderStyle, WidthType, UnderlineType } = require('docx');
const fs = require('fs');

// 논문 제목
const title = new Paragraph({
  text: "MDAF: 클릭률 예측을 위한 Mamba-DCN 적응적 융합 모델",
  heading: HeadingLevel.TITLE,
  alignment: AlignmentType.CENTER,
  spacing: { after: 400 },
});

// 초록 섹션
const abstractHeading = new Paragraph({
  text: "초록",
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 200, after: 200 },
});

const abstractText = new Paragraph({
  text: "클릭률(Click-Through Rate, CTR) 예측은 온라인 광고 및 추천 시스템에서 중요한 과제로, 정적 특징 교차(static feature interaction)와 순차적 사용자 행동(sequential user behavior) 모두를 효과적으로 모델링해야 한다. 기존 접근법은 주로 정적 특징(예: AutoInt, DCNv2) 또는 순차 패턴(예: BST, Mamba4Rec) 중 하나에만 집중하여, 두 패러다임의 상호 보완적 장점을 완전히 활용하지 못한다. 본 논문에서는 명시적 정적 특징 교차를 위한 Deep Cross Network v3(DCNv3)와 효율적인 순차 모델링을 위한 Mamba4Rec을 결합한 최초의 하이브리드 아키텍처인 MDAF(Mamba-DCN with Adaptive Fusion)를 제안한다. 핵심 혁신은 샘플별로 정적 브랜치와 순차 브랜치의 기여도를 동적으로 가중하는 적응적 융합 게이트(adaptive fusion gate)로, 사용자 행동 패턴에 따라 서로 다른 신호를 강조할 수 있게 한다. Taobao 사용자 행동 데이터셋에 대한 실험에서 MDAF는 검증 AUC 0.6007을 달성하여 순차 베이스라인 BST(0.5711) 대비 5.2% 개선되었으며, 파라미터는 35%만 사용했다(46M vs. 130M). 절제 연구(ablation study)에서 적응적 게이트가 단순 연결(concatenation) 대비 +239bp 기여하며, 게이트 분석 결과 MDAF가 이 데이터셋에서 정적 특징에 83%, 순차 특징에 17%의 가중치를 할당하여 상대적 신호 강도를 반영함을 보여준다. 본 연구는 학습 가능한 융합 메커니즘을 갖춘 하이브리드 아키텍처가 CTR 예측에 효과적임을 입증한다.",
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 200 },
});

const keywords = new Paragraph({
  children: [
    new TextRun({ text: "핵심어: ", bold: true }),
    new TextRun("클릭률 예측, 상태 공간 모델, 심층 교차 네트워크, 하이브리드 아키텍처, 적응적 융합"),
  ],
  spacing: { after: 400 },
});

// 1. 서론
const intro1 = new Paragraph({
  text: "1. 서론",
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 400, after: 200 },
});

const intro1_p1 = new Paragraph({
  text: "클릭률(CTR) 예측은 온라인 광고, 추천 시스템, 전자상거래 플랫폼의 기본 과제이다[1, 2]. 정확한 CTR 예측은 개인화된 콘텐츠 전달과 최적 광고 배치를 통해 수익, 사용자 참여도, 플랫폼 효율성에 직접적인 영향을 미친다. 이 과제는 정적 맥락 특징(사용자 인구통계, 아이템 속성, 시간대)과 순차적 사용자 행동 이력을 기반으로 사용자가 주어진 아이템(예: 광고, 제품, 콘텐츠)을 클릭할 확률을 예측하는 것을 포함한다.",
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 200 },
});

const intro1_p2 = new Paragraph({
  text: "CTR 예측에 대한 전통적 접근법은 두 가지 뚜렷한 패러다임을 따라 발전해왔다. 정적 특징 기반 모델인 AutoInt[3], DCNv2[4], FinalMLP[5]는 사용자 ID, 아이템 ID, 맥락과 같은 범주형 특징으로부터 명시적 또는 암시적 특징 교차를 학습하는 데 집중한다. 이러한 모델은 정적 관계 포착에는 뛰어나지만 사용자 행동 시퀀스의 시간적 동역학(temporal dynamics)을 활용하지 못한다. 반면, BST(Behavior Sequence Transformer)[6], SASRec[7], Mamba4Rec[8]과 같은 순차 모델은 진화하는 선호도와 단기 관심사를 포착하기 위해 사용자 상호작용 이력을 모델링한다. 순차 패턴 인식에는 효과적이지만, 중요한 맥락을 제공하는 정적 특징 교차를 충분히 활용하지 못하는 경우가 많다.",
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 200 },
});

const intro1_p3 = new Paragraph({
  text: "순차 모델링의 최근 발전으로 상태 공간 모델(State Space Models, SSMs)[9], 특히 Mamba[10]가 도입되었다. Mamba는 선택적 주의 메커니즘(selective attention mechanism)과 함께 선형 시간 복잡도를 제공한다. Mamba4Rec은 SSM이 순차 추천에서 Transformer 수준의 성능을 우수한 효율성으로 달성할 수 있음을 성공적으로 입증했다. 그러나 Mamba4Rec은 아이템 시퀀스에만 집중하며, CTR 예측 과제에 중요한 것으로 알려진 정적 범주형 특징의 교차 특징 상호작용을 명시적으로 모델링하지 않는다[4].",
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 200 },
});

const intro1_p4 = new Paragraph({
  text: "이러한 간극은 다음의 연구 질문을 동기화한다: 명시적 정적 특징 교차와 효율적인 순차 모델링을 효과적으로 결합하고, 샘플 특성에 따라 이들의 기여도를 적응적으로 균형 잡을 수 있는 하이브리드 아키텍처를 설계할 수 있는가?",
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 200 },
});

const intro1_p5 = new Paragraph({
  text: "우리는 다음 세 가지 핵심 설계 선택을 통해 이 질문에 답하는 새로운 하이브리드 프레임워크인 MDAF(Mamba-DCN with Adaptive Fusion)을 제안한다:",
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 100 },
});

const intro_design1 = new Paragraph({
  text: "1. DCNv3를 사용한 정적 브랜치: 정적 범주형 특징(사용자, 아이템, 카테고리) 간의 명시적 고차 특징 상호작용을 모델링하기 위해 Deep Cross Network v3(DCNv3)[4]를 사용한다. DCNv3의 지역 교차 네트워크(LCN)와 지수 교차 네트워크(ECN)는 저차 및 고차 특징 교차 패턴을 효율적으로 포착한다.",
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 100 },
  numbering: { reference: "design-list", level: 0 },
});

const intro_design2 = new Paragraph({
  text: "2. Mamba4Rec을 사용한 순차 브랜치: 선택적 상태 공간 모델을 통해 사용자 행동 시퀀스를 모델링하기 위해 Mamba4Rec[8]을 통합한다. 이 브랜치는 선형 시간 복잡도로 사용자 상호작용 이력의 시간적 동역학과 순차적 의존성을 포착한다.",
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 100 },
  numbering: { reference: "design-list", level: 0 },
});

const intro_design3 = new Paragraph({
  text: "3. 적응적 융합 게이트: 가장 중요한 것은, 샘플별로 정적 및 순차 표현의 기여도를 동적으로 가중하는 학습 가능한 게이트 메커니즘을 도입한 것이다. 고정 융합 전략(연결, 덧셈)과 달리, 우리의 적응적 게이트는 맥락이 지배적인 샘플(예: 신규 사용자, 인기 아이템)에서는 정적 특징을, 행동 이력이 더 정보적인 곳(예: 풍부한 상호작용 패턴을 가진 활성 사용자)에서는 순차 특징을 강조할 수 있게 한다.",
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 200 },
  numbering: { reference: "design-list", level: 0 },
});

const intro_contrib = new Paragraph({
  text: "우리의 기여는 다음과 같이 요약된다:",
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 100 },
});

const contrib1 = new Paragraph({
  children: [
    new TextRun({ text: "• 새로운 하이브리드 아키텍처: ", bold: true }),
    new TextRun("CTR 예측을 위해 DCNv3와 Mamba4Rec을 결합한 최초의 프레임워크를 제안하여, 정적 특징 기반 패러다임과 순차 모델링 패러다임 간의 간극을 메운다."),
  ],
  spacing: { after: 100 },
});

const contrib2 = new Paragraph({
  children: [
    new TextRun({ text: "• 적응적 융합 메커니즘: ", bold: true }),
    new TextRun("입력 특성에 따라 유연한 정보 통합을 가능하게 하는 샘플 의존적 게이트를 설계하여 정적 및 순차 브랜치를 동적으로 가중한다."),
  ],
  spacing: { after: 100 },
});

const contrib3 = new Paragraph({
  children: [
    new TextRun({ text: "• 강력한 실증 결과: ", bold: true }),
    new TextRun("Taobao 사용자 행동 데이터셋에서 MDAF는 0.6007 검증 AUC를 달성하여 BST 베이스라인 대비 +5.2%, 파라미터는 3배 적게(46M vs. 130M) 사용했다. 절제 연구는 적응적 게이트가 단순 연결 대비 +239bp 기여함을 확인한다."),
  ],
  spacing: { after: 100 },
});

const contrib4 = new Paragraph({
  children: [
    new TextRun({ text: "• 게이트 분석을 통한 해석 가능성: ", bold: true }),
    new TextRun("학습된 게이트 값을 분석하여 MDAF가 정적 및 순차 신호의 균형을 어떻게 맞추는지(Taobao에서 83% vs. 17%) 통찰을 제공하며, 데이터셋 특정 신호 특성을 밝힌다."),
  ],
  spacing: { after: 200 },
});

const intro_structure = new Paragraph({
  text: "본 논문의 나머지 부분은 다음과 같이 구성된다: 2장에서는 관련 연구를 검토하고, 3장에서는 SSM과 DCN에 대한 기초 이론을 제시하며, 4장에서는 MDAF 아키텍처를 상세히 설명하고, 5장에서는 실험 결과와 분석을 제시하며, 6장에서는 한계점과 향후 방향으로 결론을 맺는다.",
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 400 },
});

// 2. 관련 연구
const relatedWork = new Paragraph({
  text: "2. 관련 연구",
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 400, after: 200 },
});

const relatedWork21 = new Paragraph({
  text: "2.1 정적 특징 기반 CTR 예측",
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 200, after: 100 },
});

const rw21_p1 = new Paragraph({
  text: "초기 신경망 CTR 모델인 Wide&Deep[1]과 DeepFM[2]은 특징 상호작용을 포착하기 위해 선형 모델과 심층 신경망을 결합한다. 후속 연구는 명시적 특징 교차 메커니즘에 집중해왔다:",
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 100 },
});

const rw21_cross = new Paragraph({
  children: [
    new TextRun({ text: "• 교차 네트워크 아키텍처: ", bold: true }),
    new TextRun("DCN[11]은 교차 레이어를 통해 명시적 비트별 특징 교차를 도입한다. DCNv2[4]는 혼합 전문가(mixture-of-experts) 게이팅으로 효율성을 개선한다. DCNv3는 지역 및 지수 교차 네트워크를 통해 표현력을 더욱 향상시킨다."),
  ],
  spacing: { after: 100 },
});

const rw21_attn = new Paragraph({
  children: [
    new TextRun({ text: "• 주의 기반 상호작용: ", bold: true }),
    new TextRun("AutoInt[3]은 다중 헤드 자기 주의(multi-head self-attention)를 적용하여 특징 상호작용을 학습한다. FinalMLP[5]는 특징 게이팅이 있는 이중 스트림 MLP를 사용한다."),
  ],
  spacing: { after: 100 },
});

const rw21_limit = new Paragraph({
  text: "이러한 모델은 정적 관계 포착에는 뛰어나지만, 순차적 사용자 행동을 활용하지 못하여 시간적 동역학 모델링 능력이 제한된다.",
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 200 },
});

// Note 섹션 생성
const noteSection = new Paragraph({
  children: [
    new TextRun({ text: "참고: ", bold: true }),
    new TextRun("이 DOCX 파일은 논문의 주요 구조를 보여주기 위한 예제 버전입니다. 완전한 논문 내용(16개 표, 2개 알고리즘, 전체 실험 결과 포함)은 "),
    new TextRun({ text: "MDAF_paper_complete_KR.md", italics: true }),
    new TextRun(" 파일을 참조하세요."),
  ],
  spacing: { before: 400, after: 200 },
});

// 주요 결과 표 (Table 5: Main Results)
const table5Heading = new Paragraph({
  text: "표 5: Taobao 데이터셋의 주요 결과",
  heading: HeadingLevel.HEADING_3,
  spacing: { before: 300, after: 100 },
});

const table5 = new Table({
  rows: [
    new TableRow({
      children: [
        new TableCell({
          children: [new Paragraph({ text: "모델", bold: true })],
          width: { size: 25, type: WidthType.PERCENTAGE },
        }),
        new TableCell({
          children: [new Paragraph({ text: "Val AUC", bold: true })],
          width: { size: 15, type: WidthType.PERCENTAGE },
        }),
        new TableCell({
          children: [new Paragraph({ text: "Test AUC", bold: true })],
          width: { size: 15, type: WidthType.PERCENTAGE },
        }),
        new TableCell({
          children: [new Paragraph({ text: "파라미터", bold: true })],
          width: { size: 15, type: WidthType.PERCENTAGE },
        }),
        new TableCell({
          children: [new Paragraph({ text: "개선도 (vs BST)", bold: true })],
          width: { size: 30, type: WidthType.PERCENTAGE },
        }),
      ],
    }),
    new TableRow({
      children: [
        new TableCell({ children: [new Paragraph("BST")] }),
        new TableCell({ children: [new Paragraph("0.5711")] }),
        new TableCell({ children: [new Paragraph("0.5698")] }),
        new TableCell({ children: [new Paragraph("130M")] }),
        new TableCell({ children: [new Paragraph("—")] }),
      ],
    }),
    new TableRow({
      children: [
        new TableCell({ children: [new Paragraph("AutoInt")] }),
        new TableCell({ children: [new Paragraph("0.5655")] }),
        new TableCell({ children: [new Paragraph("0.5648")] }),
        new TableCell({ children: [new Paragraph("23M")] }),
        new TableCell({ children: [new Paragraph("-56bp")] }),
      ],
    }),
    new TableRow({
      children: [
        new TableCell({ children: [new Paragraph("DCNv2")] }),
        new TableCell({ children: [new Paragraph("0.5602")] }),
        new TableCell({ children: [new Paragraph("0.5594")] }),
        new TableCell({ children: [new Paragraph("23M")] }),
        new TableCell({ children: [new Paragraph("-109bp")] }),
      ],
    }),
    new TableRow({
      children: [
        new TableCell({ children: [new Paragraph("MDAF (제안)")] }),
        new TableCell({ children: [new Paragraph("0.6007")] }),
        new TableCell({ children: [new Paragraph("0.5992")] }),
        new TableCell({ children: [new Paragraph("46M")] }),
        new TableCell({ children: [new Paragraph("+296bp (+5.2%)")] }),
      ],
    }),
  ],
  width: { size: 100, type: WidthType.PERCENTAGE },
});

// 결론
const conclusion = new Paragraph({
  text: "6. 결론 및 한계점",
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 400, after: 200 },
});

const conclusion_p1 = new Paragraph({
  text: "본 논문에서는 CTR 예측을 위해 DCNv3 기반 정적 특징 교차와 Mamba4Rec 기반 순차 모델링을 결합한 새로운 하이브리드 아키텍처 MDAF를 제안했다. 핵심 기여는 입력 특성에 따라 두 브랜치의 기여도를 동적으로 가중하는 적응적 융합 게이트이다. Taobao 사용자 행동 데이터셋에서 MDAF는 검증 AUC 0.6007을 달성하여 순차 베이스라인 BST 대비 +5.2%(+296bp) 개선되었으며, 파라미터는 35%만 사용했다(46M vs. 130M). 절제 연구는 적응적 게이트가 단순 연결 대비 +239bp 기여하며, 게이트 분석은 MDAF가 이 데이터셋의 약한 순차 신호를 반영하여 정적 특징에 83%, 순차 특징에 17%의 가중치를 할당함을 보여준다.",
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 200 },
});

const conclusion61 = new Paragraph({
  text: "6.1 주요 발견",
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 200, after: 100 },
});

const finding1 = new Paragraph({
  text: "1. 하이브리드 아키텍처가 효과적이다: DCNv3와 Mamba4Rec의 결합은 정적 특징 전용 모델(AutoInt, DCNv2) 및 순차 전용 모델(BST)을 모두 능가한다.",
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 100 },
});

const finding2 = new Paragraph({
  text: "2. 적응적 융합이 고정 융합을 능가한다: 학습 가능한 게이트가 정적 연결(+239bp) 및 덧셈(+154bp)보다 유의미하게 우수하다.",
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 100 },
});

const finding3 = new Paragraph({
  text: "3. 파라미터 효율성: MDAF는 BST의 35% 파라미터로 5.2% 더 나은 성능을 달성한다.",
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 100 },
});

const finding4 = new Paragraph({
  text: "4. 데이터셋별 신호 특성: Taobao에서 게이트가 정적 특징을 선호(83% vs. 17%)하는 것은 이 데이터셋의 순차 패턴이 약함을 시사한다.",
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 200 },
});

const conclusion62 = new Paragraph({
  text: "6.2 한계점 및 향후 연구",
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 200, after: 100 },
});

const limit1 = new Paragraph({
  children: [
    new TextRun({ text: "1. 단일 데이터셋 평가: ", bold: true }),
    new TextRun("현재 결과는 Taobao 데이터셋에만 기반하며, 이는 상대적으로 약한 순차 신호를 나타낸다(게이트: 17%). 향후 연구는 Amazon Books, MovieLens, Criteo와 같이 더 강한 순차 패턴을 가진 데이터셋에서 MDAF를 평가해야 하며, 그곳에서 순차 브랜치가 더 큰 기여를 할 것으로 기대된다."),
  ],
  spacing: { after: 100 },
});

const limit2 = new Paragraph({
  children: [
    new TextRun({ text: "2. 절대 성능: ", bold: true }),
    new TextRun("Val AUC 0.6007은 BST 대비 5.2% 개선이지만, 절대 성능은 여전히 겸손하다. 이는 데이터셋 필터링(카테고리별), 제한된 훈련 샘플(473K), 단순화된 특징 공학에 기인할 수 있다. 향후 연구는 더 풍부한 특징과 더 큰 데이터로 실험해야 한다."),
  ],
  spacing: { after: 100 },
});

const limit3 = new Paragraph({
  children: [
    new TextRun({ text: "3. 산업 배포: ", bold: true }),
    new TextRun("Mamba의 순환 특성이 실시간 서빙에서 배치 추론 최적화를 복잡하게 만들 수 있다. 산업 배포를 위한 엔지니어링 작업이 필요하다."),
  ],
  spacing: { after: 100 },
});

const limit4 = new Paragraph({
  children: [
    new TextRun({ text: "4. 하이퍼파라미터 민감도: ", bold: true }),
    new TextRun("게이트 차원, dropout, 학습률과 같은 하이퍼파라미터가 성능에 영향을 미친다. 더 체계적인 하이퍼파라미터 튜닝이 추가 개선을 제공할 수 있다."),
  ],
  spacing: { after: 200 },
});

const futureWork = new Paragraph({
  text: "향후 방향은 (1) 더 강한 순차 패턴을 가진 다양한 데이터셋에서 평가, (2) 다중 관심사 추출이나 계층적 주의와 같은 고급 융합 메커니즘 탐구, (3) 온라인 A/B 테스트를 통한 실제 CTR 예측 시스템에서의 검증을 포함한다.",
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 400 },
});

// 문서 생성
const doc = new Document({
  styles: {
    default: {
      document: {
        run: {
          font: "맑은 고딕",
          size: 22, // 11pt
        },
      },
    },
    paragraphStyles: [
      {
        id: "Title",
        name: "Title",
        run: {
          size: 32, // 16pt
          bold: true,
        },
        paragraph: {
          alignment: AlignmentType.CENTER,
          spacing: { after: 400 },
        },
      },
      {
        id: "Heading1",
        name: "Heading 1",
        run: {
          size: 28, // 14pt
          bold: true,
        },
        paragraph: {
          spacing: { before: 400, after: 200 },
          outlineLevel: 0,
        },
      },
      {
        id: "Heading2",
        name: "Heading 2",
        run: {
          size: 26, // 13pt
          bold: true,
        },
        paragraph: {
          spacing: { before: 200, after: 100 },
          outlineLevel: 1,
        },
      },
      {
        id: "Heading3",
        name: "Heading 3",
        run: {
          size: 24, // 12pt
          bold: true,
        },
        paragraph: {
          spacing: { before: 150, after: 100 },
          outlineLevel: 2,
        },
      },
    ],
  },
  numbering: {
    config: [
      {
        reference: "design-list",
        levels: [
          {
            level: 0,
            format: "decimal",
            text: "%1.",
            alignment: AlignmentType.LEFT,
          },
        ],
      },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          margin: {
            top: 1440, // 1 inch
            right: 1440,
            bottom: 1440,
            left: 1440,
          },
        },
      },
      children: [
        title,
        abstractHeading,
        abstractText,
        keywords,
        intro1,
        intro1_p1,
        intro1_p2,
        intro1_p3,
        intro1_p4,
        intro1_p5,
        intro_design1,
        intro_design2,
        intro_design3,
        intro_contrib,
        contrib1,
        contrib2,
        contrib3,
        contrib4,
        intro_structure,
        relatedWork,
        relatedWork21,
        rw21_p1,
        rw21_cross,
        rw21_attn,
        rw21_limit,
        noteSection,
        table5Heading,
        table5,
        conclusion,
        conclusion_p1,
        conclusion61,
        finding1,
        finding2,
        finding3,
        finding4,
        conclusion62,
        limit1,
        limit2,
        limit3,
        limit4,
        futureWork,
      ],
    },
  ],
});

// DOCX 파일로 저장
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("MDAF_paper_complete_KR.docx", buffer);
  console.log("✅ DOCX 파일이 생성되었습니다: MDAF_paper_complete_KR.docx");
  console.log("📄 파일 크기:", (buffer.length / 1024).toFixed(2), "KB");
  console.log("\n참고: 이 파일은 논문의 주요 구조를 포함한 예제 버전입니다.");
  console.log("완전한 내용(16개 표, 2개 알고리즘, 전체 실험 섹션)은 MDAF_paper_complete_KR.md를 참조하세요.");
}).catch(err => {
  console.error("❌ 오류 발생:", err);
});
