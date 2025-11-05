"""
DCNv3: Towards Next Generation Deep Cross Network for CTR Prediction

논문: DCNv3: Towards Next Generation Deep Cross Network for CTR Prediction
ArXiv: https://arxiv.org/abs/2407.13349v6

핵심 구성요소:
1. LCN (Linear Cross Network): 선형적으로 증가하는 특징 교차 (1차, 2차, 3차, ...)
2. ECN (Exponential Cross Network): 지수적으로 증가하는 특징 교차 (1차, 2차, 4차, 8차, ...)
3. Self-Mask: LayerNorm을 사용한 노이즈 필터링
4. Tri-BCE Loss: 3개의 BCE 손실을 적응적 가중치로 결합
"""

import torch
import torch.nn as nn


class SelfMask(nn.Module):
    """
    Self-Mask operation for filtering noisy interactions

    논문 수식:
    Mask(c_l) = c_l ⊙ max(0, LN(c_l))

    where LayerNorm is:
    LN(c_l) = g ⊙ Norm(c_l) + b
    Norm(c_l) = (c_l - μ) / δ
    """

    def __init__(self, input_dim):
        super(SelfMask, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            masked_x: (batch_size, input_dim)
        """
        # LayerNorm으로 정규화
        ln_x = self.layer_norm(x)

        # max(0, LN(c_l))로 마스크 생성
        mask = torch.relu(ln_x)

        # Self-Mask 적용
        masked_x = x * mask

        return masked_x


class LinearCrossNetwork(nn.Module):
    """
    Linear Cross Network (LCN)

    선형적으로 증가하는 특징 교차 (1차, 2차, 3차, ...)

    논문 수식:
    c_l = W_l * x_l + b_l
    x_{l+1} = x_1 ⊙ [c_l || Mask(c_l)] + x_l

    where:
    - c_l ∈ R^{D/2}: cross vector
    - W_l ∈ R^{D/2 × D}: weight matrix
    - b_l ∈ R^{D/2}: bias
    - [c_l || Mask(c_l)]: concatenation (총 D 차원)
    """

    def __init__(self, input_dim, num_layers=3, dropout=0.0):
        super(LinearCrossNetwork, self).__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.cross_dim = input_dim // 2

        # Weight matrices and biases for each layer
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(self.cross_dim, input_dim) * 0.01)
            for _ in range(num_layers)
        ])

        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(self.cross_dim))
            for _ in range(num_layers)
        ])

        # Self-Mask layers
        self.masks = nn.ModuleList([
            SelfMask(self.cross_dim)
            for _ in range(num_layers)
        ])

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x0):
        """
        Args:
            x0: (batch_size, input_dim) - 초기 임베딩
        Returns:
            x_l: (batch_size, input_dim) - L차 교차 결과
        """
        x_l = x0

        for layer in range(self.num_layers):
            # Cross vector: c_l = W_l * x_l + b_l
            c_l = torch.matmul(x_l, self.weights[layer].T) + self.biases[layer]
            # c_l shape: (batch, D/2)

            # Masked cross vector: Mask(c_l)
            masked_c_l = self.masks[layer](c_l)
            # masked_c_l shape: (batch, D/2)

            # Concatenate: [c_l || Mask(c_l)]
            concat_c = torch.cat([c_l, masked_c_l], dim=1)
            # concat_c shape: (batch, D)

            # Feature interaction: x_{l+1} = x_1 ⊙ [c_l || Mask(c_l)] + x_l
            x_l = x0 * concat_c + x_l

            # Apply dropout
            x_l = self.dropout(x_l)

        return x_l


class ExponentialCrossNetwork(nn.Module):
    """
    Exponential Cross Network (ECN)

    지수적으로 증가하는 특징 교차 (1차, 2차, 4차, 8차, ...)

    논문 수식:
    c_l = W_l * x_{2^{l-1}} + b_l
    x_{2^l} = x_{2^{l-1}} ⊙ [c_l || Mask(c_l)] + x_{2^{l-1}}

    이는 layer l에서 2^l차 교차를 생성
    """

    def __init__(self, input_dim, num_layers=3, dropout=0.0):
        super(ExponentialCrossNetwork, self).__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.cross_dim = input_dim // 2

        # Weight matrices and biases for each layer
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(self.cross_dim, input_dim) * 0.01)
            for _ in range(num_layers)
        ])

        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(self.cross_dim))
            for _ in range(num_layers)
        ])

        # Self-Mask layers
        self.masks = nn.ModuleList([
            SelfMask(self.cross_dim)
            for _ in range(num_layers)
        ])

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x0):
        """
        Args:
            x0: (batch_size, input_dim) - 초기 임베딩
        Returns:
            x_l: (batch_size, input_dim) - 2^L차 교차 결과
        """
        x_prev = x0  # x_{2^{l-1}}

        for layer in range(self.num_layers):
            # Cross vector: c_l = W_l * x_{2^{l-1}} + b_l
            c_l = torch.matmul(x_prev, self.weights[layer].T) + self.biases[layer]
            # c_l shape: (batch, D/2)

            # Masked cross vector: Mask(c_l)
            masked_c_l = self.masks[layer](c_l)
            # masked_c_l shape: (batch, D/2)

            # Concatenate: [c_l || Mask(c_l)]
            concat_c = torch.cat([c_l, masked_c_l], dim=1)
            # concat_c shape: (batch, D)

            # Feature interaction: x_{2^l} = x_{2^{l-1}} ⊙ [c_l || Mask(c_l)] + x_{2^{l-1}}
            x_curr = x_prev * concat_c + x_prev

            # Apply dropout
            x_curr = self.dropout(x_curr)

            # Update for next iteration
            x_prev = x_curr

        return x_prev


class DCNV3(nn.Module):
    """
    DCNv3: Deep Cross Network V3

    구조:
    1. Embedding layer
    2. Linear Cross Network (LCN) - 선형 증가
    3. Exponential Cross Network (ECN) - 지수 증가
    4. Dual prediction heads (LCN & ECN)
    5. Mean fusion

    논문의 Tri-BCE loss는 학습 스크립트에서 구현
    """

    def __init__(
        self,
        num_features,
        cat_vocab_sizes,
        embed_dim=16,
        lcn_num_layers=3,
        ecn_num_layers=3,
        dropout=0.0,
    ):
        super(DCNV3, self).__init__()

        # 1. Embedding layers
        self.numeric_embeddings = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(num_features)
        ])

        self.cat_embeddings = nn.ModuleDict({
            feat: nn.Embedding(vocab_size, embed_dim)
            for feat, vocab_size in cat_vocab_sizes.items()
        })

        num_fields = num_features + len(cat_vocab_sizes)
        input_dim = num_fields * embed_dim

        # 2. Linear Cross Network (LCN)
        self.lcn = LinearCrossNetwork(
            input_dim=input_dim,
            num_layers=lcn_num_layers,
            dropout=dropout
        )

        # 3. Exponential Cross Network (ECN)
        self.ecn = ExponentialCrossNetwork(
            input_dim=input_dim,
            num_layers=ecn_num_layers,
            dropout=dropout
        )

        # 4. Prediction heads
        self.lcn_head = nn.Linear(input_dim, 1)
        self.ecn_head = nn.Linear(input_dim, 1)

    def forward(self, num_features, cat_features, return_aux=False, return_embedding=False):
        """
        Args:
            num_features: (batch_size, num_numeric_features)
            cat_features: dict of {feat_name: (batch_size,)}
            return_aux: Tri-BCE를 위해 보조 출력도 반환할지 여부
            return_embedding: MDAF 통합을 위해 중간 임베딩 반환 여부

        Returns:
            If return_embedding=True:
                embedding: (batch_size, input_dim) - LCN+ECN 평균 임베딩
            If return_aux=False:
                output: (batch_size,) - 최종 예측값
            If return_aux=True:
                (output, lcn_output, ecn_output) - Tri-BCE용
        """
        # Embeddings
        num_embeds = []
        for i, emb_layer in enumerate(self.numeric_embeddings):
            num_embeds.append(emb_layer(num_features[:, i:i+1]))

        cat_embeds = []
        for feat_name in sorted(cat_features.keys()):
            cat_embeds.append(self.cat_embeddings[feat_name](cat_features[feat_name]))

        # Concatenate all embeddings: (batch, num_fields * embed_dim)
        all_embeds = num_embeds + cat_embeds
        embed_concat = torch.cat(all_embeds, dim=1)

        # LCN and ECN forward
        lcn_out = self.lcn(embed_concat)  # (batch, input_dim)
        ecn_out = self.ecn(embed_concat)  # (batch, input_dim)

        # For MDAF: return intermediate embedding (mean of LCN and ECN)
        if return_embedding:
            # Return combined embedding before prediction heads
            return (lcn_out + ecn_out) / 2.0

        # Predictions from both networks
        lcn_logit = self.lcn_head(lcn_out)  # (batch, 1)
        ecn_logit = self.ecn_head(ecn_out)  # (batch, 1)

        lcn_pred = torch.sigmoid(lcn_logit.squeeze())  # (batch,)
        ecn_pred = torch.sigmoid(ecn_logit.squeeze())  # (batch,)

        # Mean fusion (논문 수식: ŷ = Mean(ŷ_D, ŷ_S))
        final_pred = (lcn_pred + ecn_pred) / 2.0

        if return_aux:
            # Tri-BCE loss를 위해 보조 출력 반환
            return final_pred, lcn_pred, ecn_pred
        else:
            return final_pred


class TriBCELoss(nn.Module):
    """
    Tri-BCE Loss Function

    논문 수식:
    L = -(1/N) Σ [y_i log(ŷ_i) + (1-y_i)log(1-ŷ_i)]
    L_D = BCE(y, ŷ_D)  # ECN loss
    L_S = BCE(y, ŷ_S)  # LCN loss

    Adaptive weights:
    w_D = max(0, L_D - L)
    w_S = max(0, L_S - L)

    Combined loss:
    L_Tri = L + w_D·L_D + w_S·L_S
    """

    def __init__(self):
        super(TriBCELoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, y_pred, lcn_pred, ecn_pred, y_true):
        """
        Args:
            y_pred: (batch,) - 최종 예측값 (LCN + ECN 평균)
            lcn_pred: (batch,) - LCN 예측값
            ecn_pred: (batch,) - ECN 예측값
            y_true: (batch,) - 실제 레이블

        Returns:
            loss: scalar - Tri-BCE loss
        """
        # Primary loss (최종 예측)
        loss_main = self.bce(y_pred, y_true)

        # Auxiliary losses
        loss_lcn = self.bce(lcn_pred, y_true)
        loss_ecn = self.bce(ecn_pred, y_true)

        # Adaptive weights
        w_lcn = max(0.0, (loss_lcn - loss_main).item())
        w_ecn = max(0.0, (loss_ecn - loss_main).item())

        # Combined Tri-BCE loss
        loss_tri = loss_main + w_lcn * loss_lcn + w_ecn * loss_ecn

        return loss_tri
