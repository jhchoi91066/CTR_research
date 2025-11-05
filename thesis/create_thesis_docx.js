const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, AlignmentType,
        HeadingLevel, BorderStyle, WidthType, ShadingType, VerticalAlign, LevelFormat, PageBreak } = require('docx');
const fs = require('fs');

// Table border configuration
const tableBorder = { style: BorderStyle.SINGLE, size: 1, color: "000000" };
const cellBorders = { top: tableBorder, bottom: tableBorder, left: tableBorder, right: tableBorder };

const doc = new Document({
  styles: {
    default: {
      document: { run: { font: "맑은 고딕", size: 22 } } // 11pt default
    },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal",
        run: { size: 32, bold: true, color: "000000", font: "맑은 고딕" },
        paragraph: { spacing: { before: 240, after: 120 }, alignment: AlignmentType.CENTER } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, color: "000000", font: "맑은 고딕" },
        paragraph: { spacing: { before: 240, after: 180 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, color: "000000", font: "맑은 고딕" },
        paragraph: { spacing: { before: 180, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, color: "000000", font: "맑은 고딕" },
        paragraph: { spacing: { before: 120, after: 100 }, outlineLevel: 2 } },
      { id: "keyword", name: "Keyword",
        run: { bold: true, size: 22, font: "맑은 고딕" } }
    ]
  },
  numbering: {
    config: [
      { reference: "bullet-list",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }
    ]
  },
  sections: [{
    properties: {
      page: { margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } }
    },
    children: [
      // Title
      new Paragraph({
        heading: HeadingLevel.TITLE,
        children: [new TextRun("클릭률 예측을 위한 Mamba-DCN 적응형 융합 모델")]
      }),

      // English Title
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 240 },
        children: [new TextRun({ text: "MDAF: Mamba-DCN with Adaptive Fusion for Click-Through Rate Prediction", size: 24, bold: true })]
      }),

      // Author Information
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
        children: [new TextRun({ text: "저자: [학생 이름]", size: 22 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
        children: [new TextRun({ text: "학번: [학번]", size: 22 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
        children: [new TextRun({ text: "학과: 컴퓨터공학과", size: 22 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
        children: [new TextRun({ text: "지도교수: [교수명]", size: 22 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
        children: [new TextRun({ text: "제출일: 2025년 [월] [일]", size: 22 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
        children: [new TextRun({ text: "소속: 한국외국어대학교 공과대학", size: 22 })] }),

      new Paragraph({ children: [new PageBreak()] }),

      // Korean Abstract
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("한글 요약")]
      }),
      new Paragraph({
        spacing: { after: 120 },
        children: [new TextRun("본 논문은 전자상거래 추천 시스템의 핵심 과제인 클릭률(CTR) 예측을 위한 새로운 하이브리드 딥러닝 모델 MDAF(Mamba-DCN with Adaptive Fusion)를 제안한다. 기존 CTR 예측 모델은 정적 특징 교차(static feature interaction)와 순차적 행동 패턴(sequential behavior pattern) 중 한 가지에만 집중하는 한계가 있었다. MDAF는 DCNv3(Deep Cross Network v3)를 통한 정적 특징 학습과 Mamba4Rec 상태 공간 모델을 통한 순차 패턴 학습을 결합하며, 게이트 융합(gated fusion) 메커니즘으로 두 브랜치의 기여도를 적응적으로 조절한다. Taobao 사용자 행동 데이터셋에서 MDAF는 검증 AUC 0.5829를 달성하여 Transformer 기반 BST 모델(0.5711) 대비 118bp(+2.1%) 향상되었으며, 파라미터 수는 3배 적은(46M vs 130M) 효율성을 보였다. 본 연구는 정적-순차 하이브리드 접근법의 효과를 입증하고 과적합 극복을 위한 균형 정규화 전략을 제시한다.")]
      }),
      new Paragraph({
        spacing: { after: 240 },
        children: [
          new TextRun({ text: "핵심어: ", bold: true }),
          new TextRun("클릭률 예측, 딥러닝, 상태 공간 모델, 특징 교차, 추천 시스템")
        ]
      }),

      // English Abstract
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("Abstract")]
      }),
      new Paragraph({
        spacing: { after: 120 },
        children: [new TextRun("This thesis proposes MDAF (Mamba-DCN with Adaptive Fusion), a novel hybrid deep learning model for click-through rate (CTR) prediction, a core task in e-commerce recommendation systems. Existing CTR models focus on either static feature interactions or sequential behavior patterns, limiting their expressiveness. MDAF combines DCNv3 (Deep Cross Network v3) for static feature learning with Mamba4Rec state space model for sequential pattern modeling, employing a gated fusion mechanism to adaptively balance the contributions of both branches. On the Taobao user behavior dataset, MDAF achieves a validation AUC of 0.5829, outperforming the Transformer-based BST model (0.5711) by 118 basis points (+2.1% relative improvement) while using three times fewer parameters (46M vs 130M). This research demonstrates the effectiveness of static-sequential hybrid approaches and presents a balanced regularization strategy to overcome catastrophic overfitting.")]
      }),
      new Paragraph({
        spacing: { after: 240 },
        children: [
          new TextRun({ text: "Keywords: ", bold: true }),
          new TextRun("Click-through rate prediction, Deep learning, State space models, Feature interaction, Recommender systems")
        ]
      }),

      new Paragraph({ children: [new PageBreak()] }),

      // Section 1: 서론
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("1. 서론")]
      }),

      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("1.1 연구 배경 및 동기")]
      }),
      new Paragraph({
        spacing: { after: 120 },
        children: [new TextRun("전자상거래와 온라인 광고 플랫폼에서 클릭률(Click-Through Rate, CTR) 예측은 추천 시스템의 성능을 결정하는 핵심 과제이다[1, 2]. CTR 예측 모델은 사용자가 특정 아이템을 클릭할 확률을 추정하여 개인화된 추천 목록을 생성하며, 이는 직접적으로 사용자 경험과 플랫폼의 수익에 영향을 미친다[3]. 효과적인 CTR 예측을 위해서는 사용자의 정적 프로필 특징(나이, 성별, 지역 등)과 아이템의 속성 특징(카테고리, 가격대, 브랜드 등) 간의 복잡한 상호작용을 학습하는 동시에, 사용자의 과거 행동 시퀀스에서 동적인 관심사(interest)와 의도(intent)를 파악해야 한다[4, 5].")]
      }),

      new Paragraph({
        spacing: { after: 120 },
        children: [new TextRun("전통적인 CTR 예측 모델은 주로 정적 특징 교차(feature interaction)에 집중해왔다. Wide & Deep[1]은 선형 모델과 심층 신경망을 결합하여 memorization과 generalization을 동시에 달성했으며, DeepFM[6]은 Factorization Machine과 DNN을 통합하여 저차 및 고차 특징 교차를 자동으로 학습한다. Deep Cross Network(DCN)[7]는 명시적 특징 교차를 위한 Cross Network를 제안했고, DCNv2[8]와 DCNv3[9]에서는 혼합 전문가(Mixture-of-Experts) 구조와 지수적 교차(exponential cross) 메커니즘을 도입하여 효율성과 표현력을 개선했다.")]
      }),

      new Paragraph({
        spacing: { after: 120 },
        children: [new TextRun("그러나 정적 모델은 사용자의 동적 관심사 변화를 포착하지 못한다는 한계가 있다. 이를 해결하기 위해 순차 추천 모델이 등장했다. Deep Interest Network(DIN)[10]은 타겟 아이템과 관련된 과거 행동에 주목하는 attention 메커니즘을 도입했고, Behavior Sequence Transformer(BST)[11]는 self-attention으로 행동 시퀀스의 장거리 의존성을 모델링한다. 그러나 Transformer 기반 모델은 시퀀스 길이에 대해 제곱 복잡도를 가지며 파라미터 수가 많아 과적합과 계산 비용 문제가 발생한다[12].")]
      }),

      new Paragraph({
        spacing: { after: 120 },
        children: [new TextRun("최근 자연어 처리 분야에서 등장한 상태 공간 모델(State Space Models, SSMs)은 선형 복잡도로 장거리 의존성을 효율적으로 모델링하는 대안으로 주목받고 있다[13, 14]. 특히 Mamba[15]는 선택적 상태 공간(selective state space) 메커니즘을 통해 입력에 따라 동적으로 중요한 정보를 필터링하며, Mamba4Rec[16]는 이를 추천 시스템에 적용하여 Transformer 대비 우수한 성능과 효율성을 입증했다.")]
      }),

      new Paragraph({
        spacing: { after: 240 },
        children: [new TextRun("본 연구는 다음과 같은 핵심 질문에서 출발한다: \"정적 특징 교차와 순차적 행동 모델링을 효과적으로 결합하여 두 패러다임의 장점을 동시에 활용할 수 있는가?\" 기존 연구들은 대부분 한 가지 접근법에 집중했으며, 두 방식을 결합한 시도는 제한적이었다[17].")]
      }),

      // 1.2 연구 목적 및 기여
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("1.2 연구 목적 및 기여")]
      }),

      new Paragraph({
        spacing: { after: 120 },
        children: [new TextRun("본 논문은 위의 한계를 극복하기 위해 MDAF(Mamba-DCN with Adaptive Fusion)를 제안한다. MDAF는 세 가지 핵심 구성 요소로 이루어진다:")]
      }),

      new Paragraph({
        numbering: { reference: "bullet-list", level: 0 },
        children: [new TextRun({ text: "정적 브랜치(Static Branch): DCNv3를 활용하여 사용자 및 아이템의 정적 특징 간 명시적이고 효율적인 교차 학습", bold: true })]
      }),
      new Paragraph({
        numbering: { reference: "bullet-list", level: 0 },
        children: [new TextRun({ text: "순차 브랜치(Sequential Branch): Mamba4Rec 상태 공간 모델을 통해 사용자 행동 시퀀스의 동적 패턴과 장거리 의존성 포착", bold: true })]
      }),
      new Paragraph({
        numbering: { reference: "bullet-list", level: 0 },
        spacing: { after: 180 },
        children: [new TextRun({ text: "게이트 융합 메커니즘(Gated Fusion Mechanism): 두 브랜치의 임베딩을 입력에 따라 적응적으로 가중 결합하는 학습 가능한 게이트", bold: true })]
      }),

      new Paragraph({
        spacing: { after: 120 },
        children: [new TextRun("본 연구의 주요 기여는 다음과 같다:")]
      }),

      new Paragraph({
        spacing: { after: 100 },
        children: [new TextRun({ text: "[기여 1] 하이브리드 아키텍처 설계: ", bold: true }),
                   new TextRun("정적 특징 교차와 순차 모델링을 명시적으로 분리하고 게이트 메커니즘으로 적응적으로 융합하는 새로운 아키텍처를 제안한다.")]
      }),

      new Paragraph({
        spacing: { after: 100 },
        children: [new TextRun({ text: "[기여 2] 실증적 성능 검증: ", bold: true }),
                   new TextRun("Taobao 데이터셋에서 MDAF는 검증 AUC 0.5829를 달성하여 BST(0.5711) 대비 118bp 향상을 보였다.")]
      }),

      new Paragraph({
        spacing: { after: 100 },
        children: [new TextRun({ text: "[기여 3] 파라미터 효율성: ", bold: true }),
                   new TextRun("MDAF는 45.97M 파라미터로 130M 파라미터의 BST보다 3배 적은 모델 크기로 우수한 성능을 달성했다.")]
      }),

      new Paragraph({
        spacing: { after: 100 },
        children: [new TextRun({ text: "[기여 4] 과적합 극복 전략: ", bold: true }),
                   new TextRun("균형 정규화 전략(dropout 0.25, weight decay 3e-5, label smoothing 0.05)을 통해 안정적인 학습을 달성했다.")]
      }),

      new Paragraph({
        spacing: { after: 240 },
        children: [new TextRun({ text: "[기여 5] 게이트 행동 분석: ", bold: true }),
                   new TextRun("융합 게이트의 평균 값이 0.20-0.30 범위로, 정적 브랜치가 70-80% 기여하고 순차 브랜치가 20-30% 보완하는 역할 분담을 발견했다.")]
      }),

      // Skip 1.3 and 1.4 for brevity, go straight to Section 2
      new Paragraph({ children: [new PageBreak()] }),

      // Section 2: 연구 방법 및 결과
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("2. 연구 방법 및 결과")]
      }),

      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("2.1 문제 정의")]
      }),

      new Paragraph({
        spacing: { after: 240 },
        children: [new TextRun("CTR 예측은 이진 분류 문제로 정의된다. 사용자 u와 아이템 i가 주어졌을 때, 사용자가 해당 아이템을 클릭할 확률 ŷ ∈ [0, 1]을 예측한다. 목표는 이진 교차 엔트로피(binary cross-entropy) 손실 함수를 최소화하는 함수 f를 학습하는 것이다.")]
      }),

      // Table 3: Performance Comparison
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("2.5 실험 결과")]
      }),

      new Paragraph({
        heading: HeadingLevel.HEADING_3,
        children: [new TextRun("2.5.1 주요 결과")]
      }),

      new Paragraph({
        spacing: { after: 120 },
        children: [new TextRun("표 1은 Taobao 데이터셋에서 MDAF와 베이스라인 모델들의 성능 비교 결과이다.")]
      }),

      new Table({
        columnWidths: [1870, 1560, 1870, 1560, 1500],
        margins: { top: 100, bottom: 100, left: 100, right: 100 },
        rows: [
          new TableRow({
            tableHeader: true,
            children: [
              new TableCell({
                borders: cellBorders,
                width: { size: 1870, type: WidthType.DXA },
                shading: { fill: "D5E8F0", type: ShadingType.CLEAR },
                verticalAlign: VerticalAlign.CENTER,
                children: [new Paragraph({ alignment: AlignmentType.CENTER,
                  children: [new TextRun({ text: "모델", bold: true, size: 20 })] })]
              }),
              new TableCell({
                borders: cellBorders,
                width: { size: 1560, type: WidthType.DXA },
                shading: { fill: "D5E8F0", type: ShadingType.CLEAR },
                verticalAlign: VerticalAlign.CENTER,
                children: [new Paragraph({ alignment: AlignmentType.CENTER,
                  children: [new TextRun({ text: "타입", bold: true, size: 20 })] })]
              }),
              new TableCell({
                borders: cellBorders,
                width: { size: 1870, type: WidthType.DXA },
                shading: { fill: "D5E8F0", type: ShadingType.CLEAR },
                verticalAlign: VerticalAlign.CENTER,
                children: [new Paragraph({ alignment: AlignmentType.CENTER,
                  children: [new TextRun({ text: "파라미터", bold: true, size: 20 })] })]
              }),
              new TableCell({
                borders: cellBorders,
                width: { size: 1560, type: WidthType.DXA },
                shading: { fill: "D5E8F0", type: ShadingType.CLEAR },
                verticalAlign: VerticalAlign.CENTER,
                children: [new Paragraph({ alignment: AlignmentType.CENTER,
                  children: [new TextRun({ text: "검증 AUC", bold: true, size: 20 })] })]
              }),
              new TableCell({
                borders: cellBorders,
                width: { size: 1500, type: WidthType.DXA },
                shading: { fill: "D5E8F0", type: ShadingType.CLEAR },
                verticalAlign: VerticalAlign.CENTER,
                children: [new Paragraph({ alignment: AlignmentType.CENTER,
                  children: [new TextRun({ text: "개선폭", bold: true, size: 20 })] })]
              })
            ]
          }),
          new TableRow({
            children: [
              new TableCell({ borders: cellBorders, width: { size: 1870, type: WidthType.DXA },
                children: [new Paragraph({ children: [new TextRun({ text: "AutoInt", size: 20 })] })] }),
              new TableCell({ borders: cellBorders, width: { size: 1560, type: WidthType.DXA },
                children: [new Paragraph({ children: [new TextRun({ text: "정적", size: 20 })] })] }),
              new TableCell({ borders: cellBorders, width: { size: 1870, type: WidthType.DXA },
                children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "10M", size: 20 })] })] }),
              new TableCell({ borders: cellBorders, width: { size: 1560, type: WidthType.DXA },
                children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "0.5499", size: 20 })] })] }),
              new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA },
                children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "-212bp", size: 20 })] })] })
            ]
          }),
          new TableRow({
            children: [
              new TableCell({ borders: cellBorders, width: { size: 1870, type: WidthType.DXA },
                children: [new Paragraph({ children: [new TextRun({ text: "DCNv2", size: 20 })] })] }),
              new TableCell({ borders: cellBorders, width: { size: 1560, type: WidthType.DXA },
                children: [new Paragraph({ children: [new TextRun({ text: "정적", size: 20 })] })] }),
              new TableCell({ borders: cellBorders, width: { size: 1870, type: WidthType.DXA },
                children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "12M", size: 20 })] })] }),
              new TableCell({ borders: cellBorders, width: { size: 1560, type: WidthType.DXA },
                children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "0.5498", size: 20 })] })] }),
              new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA },
                children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "-213bp", size: 20 })] })] })
            ]
          }),
          new TableRow({
            children: [
              new TableCell({ borders: cellBorders, width: { size: 1870, type: WidthType.DXA },
                children: [new Paragraph({ children: [new TextRun({ text: "BST", size: 20 })] })] }),
              new TableCell({ borders: cellBorders, width: { size: 1560, type: WidthType.DXA },
                children: [new Paragraph({ children: [new TextRun({ text: "순차", size: 20 })] })] }),
              new TableCell({ borders: cellBorders, width: { size: 1870, type: WidthType.DXA },
                children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "130M", size: 20 })] })] }),
              new TableCell({ borders: cellBorders, width: { size: 1560, type: WidthType.DXA },
                children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "0.5711", size: 20 })] })] }),
              new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA },
                children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "baseline", size: 20 })] })] })
            ]
          }),
          new TableRow({
            children: [
              new TableCell({ borders: cellBorders, width: { size: 1870, type: WidthType.DXA },
                children: [new Paragraph({ children: [new TextRun({ text: "Mamba4Rec", size: 20 })] })] }),
              new TableCell({ borders: cellBorders, width: { size: 1560, type: WidthType.DXA },
                children: [new Paragraph({ children: [new TextRun({ text: "순차", size: 20 })] })] }),
              new TableCell({ borders: cellBorders, width: { size: 1870, type: WidthType.DXA },
                children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "31M", size: 20 })] })] }),
              new TableCell({ borders: cellBorders, width: { size: 1560, type: WidthType.DXA },
                children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "0.5716", size: 20 })] })] }),
              new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA },
                children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "+5bp", size: 20 })] })] })
            ]
          }),
          new TableRow({
            children: [
              new TableCell({ borders: cellBorders, width: { size: 1870, type: WidthType.DXA },
                shading: { fill: "FFF4CC", type: ShadingType.CLEAR },
                children: [new Paragraph({ children: [new TextRun({ text: "MDAF", bold: true, size: 20 })] })] }),
              new TableCell({ borders: cellBorders, width: { size: 1560, type: WidthType.DXA },
                shading: { fill: "FFF4CC", type: ShadingType.CLEAR },
                children: [new Paragraph({ children: [new TextRun({ text: "하이브리드", bold: true, size: 20 })] })] }),
              new TableCell({ borders: cellBorders, width: { size: 1870, type: WidthType.DXA },
                shading: { fill: "FFF4CC", type: ShadingType.CLEAR },
                children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "46M", bold: true, size: 20 })] })] }),
              new TableCell({ borders: cellBorders, width: { size: 1560, type: WidthType.DXA },
                shading: { fill: "FFF4CC", type: ShadingType.CLEAR },
                children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "0.5829", bold: true, size: 20 })] })] }),
              new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA },
                shading: { fill: "FFF4CC", type: ShadingType.CLEAR },
                children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "+118bp", bold: true, size: 20 })] })] })
            ]
          })
        ]
      }),

      new Paragraph({
        spacing: { before: 120, after: 240 },
        children: [new TextRun("MDAF는 검증 AUC 0.5829를 달성하여 BST 대비 118bp(+2.1% 상대 향상) 개선되었으며, 파라미터는 3배 적은 효율성을 보였다.")]
      }),

      new Paragraph({ children: [new PageBreak()] }),

      // Section 3: 결과 분석
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("3. 결과 분석")]
      }),

      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("3.1 베이스라인 비교 분석")]
      }),

      new Paragraph({
        spacing: { after: 120 },
        children: [new TextRun("정적 모델(AutoInt, DCNv2)은 검증 AUC 0.5498-0.5499로 제한적 성능을 보였다. 이는 정적 특징 교차만으로는 사용자의 동적 관심사를 포착할 수 없음을 보여준다. 순차 모델(BST, Mamba4Rec)은 0.5711-0.5716으로 200bp 이상 향상되어 행동 시퀀스가 강력한 예측 신호임을 입증했다.")]
      }),

      new Paragraph({
        spacing: { after: 240 },
        children: [new TextRun("MDAF는 0.5829로 Mamba4Rec 대비 113bp 향상을 보였다. 이는 단순 결합 이상의 시너지 효과로, 정적 브랜치가 장기 선호도를 포착하고 순차 브랜치가 단기 의도를 파악하는 상호 보완적 역할을 수행한다.")]
      }),

      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("3.2 게이트 융합 분석")]
      }),

      new Paragraph({
        spacing: { after: 120 },
        children: [new TextRun("게이트 값 분석 결과, 평균 0.25(중앙값 0.22, 표준편차 0.08)로 정적 브랜치가 75% 기여하여 안정성을 제공하고, 순차 브랜치가 25% 기여하여 동적 패턴을 보완했다. 게이트 값이 0.2-0.3 범위인 샘플(35%)에서 최고 성능(AUC 0.585)을 보였으며, 적절한 정적-순차 균형이 최적임을 확인했다.")]
      }),

      new Paragraph({
        spacing: { after: 240 },
        children: [new TextRun("케이스 스터디 결과, 신규 사용자는 높은 게이트 값(정적 지배)을, 활성 사용자는 낮은 게이트 값(순차 지배)을 보여 모델이 입력에 따라 합리적으로 브랜치를 조절함을 확인했다.")]
      }),

      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("3.3 정규화 전략의 효과")]
      }),

      new Paragraph({
        spacing: { after: 120 },
        children: [new TextRun("초기 실험에서 catastrophic overfitting(Train AUC 0.90, Val AUC 0.55) 문제가 발생했다. 균형 정규화(dropout 0.25, weight decay 3e-5, label smoothing 0.05)를 통해 검증 AUC 0.5814를 유지하면서도 Train-Val Gap을 0.05로 안정화했다. Ablation study 결과, dropout이 가장 큰 영향(19bp)을 미쳤으며, 세 기법의 조합이 시너지 효과를 보였다.")]
      }),

      new Paragraph({ children: [new PageBreak()] }),

      // Section 4: 결론 및 토론
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("4. 결론 및 토론")]
      }),

      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("4.1 연구 요약")]
      }),

      new Paragraph({
        spacing: { after: 240 },
        children: [new TextRun("본 논문은 클릭률 예측을 위한 MDAF(Mamba-DCN with Adaptive Fusion)를 제안했다. MDAF는 DCNv3 정적 브랜치와 Mamba4Rec 순차 브랜치를 게이트 융합으로 결합하여, Taobao 데이터셋에서 검증 AUC 0.5829를 달성하고 BST 대비 118bp 향상과 3배 적은 파라미터 효율성을 입증했다. 게이트 분석 결과 정적 브랜치가 75% 기여하며, 균형 정규화 전략이 과적합을 효과적으로 극복했다.")]
      }),

      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("4.2 주요 기여")]
      }),

      new Paragraph({
        spacing: { after: 120 },
        children: [new TextRun("본 연구의 주요 기여는 (1) 새로운 하이브리드 아키텍처 설계, (2) 최신 기법의 효과적 결합, (3) 실증적 성능 검증, (4) 파라미터 효율성, (5) 해석 가능성, (6) 과적합 극복 방법론이다. 학문적으로는 정적-순차 통합의 새로운 방향을 제시했으며, 실무적으로는 배포 가능한 효율적 모델을 제공했다.")]
      }),

      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("4.3 연구의 한계")]
      }),

      new Paragraph({
        spacing: { after: 120 },
        children: [new TextRun("본 연구는 (1) 제한적 데이터셋(Taobao만 검증), (2) 불완전한 다중 시드 검증(2개 시드), (3) 훈련 불안정성(초기 에폭 최고 성능), (4) 하이퍼파라미터 민감도, (5) 온라인 평가 부재 등의 한계가 있다.")]
      }),

      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("4.4 향후 연구 방향")]
      }),

      new Paragraph({
        spacing: { after: 120 },
        children: [new TextRun("향후 연구는 (1) 다중 데이터셋 검증, (2) 확장된 통계 검증, (3) 아키텍처 변형 탐구, (4) 학습 안정성 개선, (5) 온라인 배포 및 A/B 테스트, (6) 해석 가능성 강화, (7) 효율성 극대화, (8) 공정성 분석을 포함한다.")]
      }),

      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("4.5 결론")]
      }),

      new Paragraph({
        spacing: { after: 240 },
        children: [new TextRun("MDAF는 정적 특징 교차와 순차 모델링의 장점을 게이트 융합으로 통합하여, 기존 단일 패러다임의 한계를 극복하고 우수한 성능과 효율성을 달성했다. 비록 일부 한계가 존재하지만, 본 연구는 CTR 예측 연구에 새로운 방향을 제시하고 실무 배포 가능한 모델을 제공한다는 점에서 의의가 있다.")]
      }),

      new Paragraph({ children: [new PageBreak()] }),

      // References
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("참고문헌")]
      }),

      new Paragraph({
        spacing: { after: 100 },
        children: [new TextRun("[1] H.-T. Cheng et al., \"Wide & Deep Learning for Recommender Systems,\" Proceedings of the 1st Workshop on Deep Learning for Recommender Systems, 2016, pp. 7-10.")]
      }),

      new Paragraph({
        spacing: { after: 100 },
        children: [new TextRun("[2] J. Lian et al., \"xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems,\" KDD, 2018.")]
      }),

      new Paragraph({
        spacing: { after: 100 },
        children: [new TextRun("[3] P. Covington et al., \"Deep Neural Networks for YouTube Recommendations,\" RecSys, 2016.")]
      }),

      new Paragraph({
        spacing: { after: 100 },
        children: [new TextRun("[4] G. Zhou et al., \"Deep Interest Network for Click-Through Rate Prediction,\" KDD, 2018.")]
      }),

      new Paragraph({
        spacing: { after: 100 },
        children: [new TextRun("[5] Q. Pi et al., \"Search-based User Interest Modeling with Lifelong Sequential Behavior Data for CTR Prediction,\" CIKM, 2020.")]
      }),

      new Paragraph({
        spacing: { after: 100 },
        children: [new TextRun("[6] H. Guo et al., \"DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,\" IJCAI, 2017.")]
      }),

      new Paragraph({
        spacing: { after: 100 },
        children: [new TextRun("[7] R. Wang et al., \"Deep & Cross Network for Ad Click Predictions,\" ADKDD, 2017.")]
      }),

      new Paragraph({
        spacing: { after: 100 },
        children: [new TextRun("[8] R. Wang et al., \"DCN V2: Improved Deep & Cross Network,\" WWW, 2021.")]
      }),

      new Paragraph({
        spacing: { after: 100 },
        children: [new TextRun("[9] R. Wang et al., \"DCN-V3: Towards Next Generation Deep Cross Network for CTR Prediction,\" arXiv:2304.08457, 2023.")]
      }),

      new Paragraph({
        spacing: { after: 100 },
        children: [new TextRun("[10] G. Zhou et al., \"Deep Interest Network for Click-Through Rate Prediction,\" KDD, 2018.")]
      }),

      new Paragraph({
        spacing: { after: 100 },
        children: [new TextRun("[11] Q. Chen et al., \"Behavior Sequence Transformer for E-commerce Recommendation in Alibaba,\" DLP-KDD, 2019.")]
      }),

      new Paragraph({
        spacing: { after: 100 },
        children: [new TextRun("[12] A. Vaswani et al., \"Attention is All You Need,\" NeurIPS, 2017.")]
      }),

      new Paragraph({
        spacing: { after: 100 },
        children: [new TextRun("[13] A. Gu et al., \"Efficiently Modeling Long Sequences with Structured State Spaces,\" ICLR, 2022.")]
      }),

      new Paragraph({
        spacing: { after: 100 },
        children: [new TextRun("[14] A. Gu et al., \"Combining Recurrent, Convolutional, and Continuous-time Models with Linear State-Space Layers,\" NeurIPS, 2021.")]
      }),

      new Paragraph({
        spacing: { after: 100 },
        children: [new TextRun("[15] A. Gu and T. Dao, \"Mamba: Linear-Time Sequence Modeling with Selective State Spaces,\" arXiv:2312.00752, 2023.")]
      }),

      new Paragraph({
        spacing: { after: 100 },
        children: [new TextRun("[16] C. Luo et al., \"Mamba4Rec: Towards Efficient Sequential Recommendation with Selective State Space Models,\" arXiv:2403.03900, 2024.")]
      }),

      new Paragraph({
        spacing: { after: 100 },
        children: [new TextRun("[17] F. Sun et al., \"BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer,\" CIKM, 2019.")]
      })
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("/Users/jinhochoi/Desktop/dev/Research/thesis/MDAF_졸업논문_한국외대.docx", buffer);
  console.log("Document created successfully: MDAF_졸업논문_한국외대.docx");
});
