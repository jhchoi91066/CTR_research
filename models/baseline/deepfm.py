"""
DeepFM: Factorization Machine + Deep Neural Network

Reference:
    Guo et al. "DeepFM: A Factorization-Machine based Neural Network
    for CTR Prediction" (IJCAI 2017)

Architecture:
    Input → Embedding (shared) → [Linear + FM + DNN] → Sigmoid → Output

논문 수식:
    y_hat = sigmoid(y_FM + y_DNN)
    where:
        y_FM = <w, x> + Σ_i Σ_j <V_i, V_j> x_i x_j  (1st-order + 2nd-order)
        y_DNN = MLP(a^0) where a^0 = [e_1, e_2, ..., e_m]

핵심: FM과 DNN이 동일한 embedding V를 공유
"""

import torch
import torch.nn as nn


class FactorizationMachine(nn.Module):
    """
    Factorization Machine component

    논문 수식 (2nd-order interaction):
        Σ_i Σ_j <V_i, V_j> x_i x_j
        = 1/2 Σ_f=1^k ((Σ_i v_i,f x_i)^2 - Σ_i v_i,f^2 x_i^2)
    """

    def __init__(self):
        super(FactorizationMachine, self).__init__()

    def forward(self, embeddings):
        """
        Args:
            embeddings: (batch_size, num_fields, embed_dim)
                각 field의 embedding vector V_i

        Returns:
            fm_output: (batch_size, 1) - FM의 2차 상호작용 항
        """
        # 논문 수식 그대로 구현
        # square_of_sum: (Σ_i V_i)^2
        square_of_sum = torch.sum(embeddings, dim=1) ** 2  # (batch, embed_dim)

        # sum_of_square: Σ_i (V_i)^2
        sum_of_square = torch.sum(embeddings ** 2, dim=1)  # (batch, embed_dim)

        # FM 2차 상호작용: 0.5 * Σ_f ((Σ_i v_i,f)^2 - Σ_i v_i,f^2)
        fm_output = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)

        return fm_output  # (batch, 1)


class DeepNeuralNetwork(nn.Module):
    """
    Deep Neural Network component

    논문 수식:
        a^(l+1) = σ(W^l a^l + b^l)
        where a^0 = [e_1, e_2, ..., e_m] (concatenated embeddings)
        y_DNN = σ(W^|H| a^|H| + b^|H|)
    """

    def __init__(self, input_dim, hidden_units=[256, 128, 64], dropout=0.1):
        """
        Args:
            input_dim: Input dimension (num_fields * embed_dim)
            hidden_units: List of hidden layer sizes (논문의 H)
            dropout: Dropout rate
        """
        super(DeepNeuralNetwork, self).__init__()

        layers = []
        in_dim = input_dim

        # 논문의 Hidden layers
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())  # 논문의 activation function σ
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.dnn = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, 1)  # 논문의 마지막 레이어

    def forward(self, embeddings):
        """
        Args:
            embeddings: (batch_size, num_fields, embed_dim)
                논문의 a^0 = [e_1, e_2, ..., e_m]

        Returns:
            dnn_output: (batch_size, 1) - y_DNN
        """
        batch_size = embeddings.size(0)

        # Flatten embeddings: a^0
        x = embeddings.view(batch_size, -1)  # (batch, num_fields * embed_dim)

        # DNN forward: a^1, a^2, ..., a^|H|
        x = self.dnn(x)

        # Output: y_DNN
        dnn_output = self.output_layer(x)  # (batch, 1)

        return dnn_output


class DeepFM(nn.Module):
    """
    DeepFM Model (논문 정확한 구현)

    논문 수식:
        y_hat = sigmoid(y_FM + y_DNN)
        where:
            y_FM = <w, x> + Σ_i Σ_j <V_i, V_j> x_i x_j
            y_DNN = MLP([e_1, e_2, ..., e_m])

    핵심:
        - FM과 DNN이 동일한 embedding layer V를 공유
        - 1st-order는 별도의 weight w
        - 2nd-order는 embedding interaction
    """

    def __init__(self,
                 num_features,
                 cat_vocab_sizes,
                 embed_dim=16,
                 hidden_units=[256, 128, 64],
                 dropout=0.1):
        """
        Args:
            num_features: Number of numeric features
            cat_vocab_sizes: Dict of {feature_name: vocab_size}
            embed_dim: Embedding dimension (논문의 k)
            hidden_units: DNN hidden layer sizes
            dropout: Dropout rate
        """
        super(DeepFM, self).__init__()

        self.num_features = num_features
        self.cat_vocab_sizes = cat_vocab_sizes
        self.embed_dim = embed_dim

        # ========== Shared Embeddings (핵심!) ==========
        # 논문: "FM and DNN share the same feature embedding"

        # Numeric features → embedding (treating as one field)
        self.numeric_embed = nn.Linear(num_features, embed_dim)

        # Categorical features → embeddings
        self.cat_embeddings = nn.ModuleDict({
            feat: nn.Embedding(vocab_size, embed_dim)
            for feat, vocab_size in cat_vocab_sizes.items()
        })

        # ========== 1st-order weights (Linear part) ==========
        # 논문: <w, x>
        # 각 field마다 scalar weight
        num_fields = 1 + len(cat_vocab_sizes)  # numeric + categorical
        self.first_order_weights = nn.Embedding(num_fields, 1)
        self.first_order_bias = nn.Parameter(torch.zeros((1,)))

        # ========== FM component (2nd-order) ==========
        self.fm = FactorizationMachine()

        # ========== DNN component ==========
        dnn_input_dim = num_fields * embed_dim
        self.dnn = DeepNeuralNetwork(dnn_input_dim, hidden_units, dropout)

        # Weight initialization (논문 참고)
        self._initialize_weights()

        print(f"✅ DeepFM 초기화 완료 (논문 정확 구현)")
        print(f"   - 수치형 특징: {num_features}개")
        print(f"   - 범주형 특징: {len(cat_vocab_sizes)}개")
        print(f"   - Total fields: {num_fields}")
        print(f"   - Embedding 차원: {embed_dim}")
        print(f"   - DNN 구조: {hidden_units}")

    def _initialize_weights(self):
        """Weight initialization"""
        # Embedding uniform initialization
        nn.init.xavier_uniform_(self.first_order_weights.weight)

    def forward(self, numeric_features, categorical_features):
        """
        논문 수식 그대로 구현:
            y_hat = sigmoid(y_FM + y_DNN)
            where y_FM = y_linear + y_interaction

        Args:
            numeric_features: (batch_size, num_features)
            categorical_features: Dict of {feat_name: (batch_size,)}

        Returns:
            output: (batch_size, 1) - CTR 예측값
        """
        batch_size = numeric_features.size(0)

        # ========== Shared Embeddings (FM & DNN 공유) ==========
        embeddings_list = []
        field_indices = []

        # Field 0: Numeric features
        numeric_embed = self.numeric_embed(numeric_features).unsqueeze(1)  # (batch, 1, embed_dim)
        embeddings_list.append(numeric_embed)
        field_indices.append(torch.zeros(batch_size, dtype=torch.long, device=numeric_features.device))

        # Field 1~m: Categorical features
        for field_idx, feat_name in enumerate(sorted(self.cat_embeddings.keys()), start=1):
            feat_idx = categorical_features[feat_name]  # (batch,)
            cat_embed = self.cat_embeddings[feat_name](feat_idx).unsqueeze(1)  # (batch, 1, embed_dim)
            embeddings_list.append(cat_embed)
            field_indices.append(torch.full((batch_size,), field_idx, dtype=torch.long, device=numeric_features.device))

        # Concatenate all embeddings
        embeddings = torch.cat(embeddings_list, dim=1)  # (batch, num_fields, embed_dim)
        field_indices = torch.stack(field_indices, dim=1)  # (batch, num_fields)

        # ========== 1st-order: y_linear = <w, x> + b ==========
        # 각 field의 weight를 가져와서 sum
        first_order = self.first_order_weights(field_indices)  # (batch, num_fields, 1)
        first_order = torch.sum(first_order, dim=1) + self.first_order_bias  # (batch, 1)

        # ========== 2nd-order: y_interaction = Σ_i Σ_j <V_i, V_j> x_i x_j ==========
        second_order = self.fm(embeddings)  # (batch, 1)

        # ========== y_FM = y_linear + y_interaction ==========
        y_fm = first_order + second_order  # (batch, 1)

        # ========== y_DNN = MLP(embeddings) ==========
        y_dnn = self.dnn(embeddings)  # (batch, 1)

        # ========== 최종 출력: y_hat = sigmoid(y_FM + y_DNN) ==========
        logit = y_fm + y_dnn  # (batch, 1)
        output = torch.sigmoid(logit)

        return output

    def predict(self, numeric_features, categorical_features):
        """Inference mode"""
        self.eval()
        with torch.no_grad():
            output = self.forward(numeric_features, categorical_features)
        return output


if __name__ == '__main__':
    # 테스트
    print("="*60)
    print("🧪 DeepFM 단위 테스트")
    print("="*60)

    # 더미 데이터
    batch_size = 32
    num_features = 13
    cat_vocab_sizes = {
        f'C{i}': 100 + i * 10 for i in range(1, 27)
    }

    # 모델 생성
    model = DeepFM(
        num_features=num_features,
        cat_vocab_sizes=cat_vocab_sizes,
        embed_dim=16,
        hidden_units=[256, 128, 64],
        dropout=0.1
    )

    # 더미 입력
    numeric_input = torch.randn(batch_size, num_features)
    categorical_input = {
        f'C{i}': torch.randint(0, 100 + i * 10, (batch_size,))
        for i in range(1, 27)
    }

    # Forward pass
    print(f"\n📥 입력:")
    print(f"   - Numeric: {numeric_input.shape}")
    print(f"   - Categorical: {len(categorical_input)} features")

    output = model(numeric_input, categorical_input)

    print(f"\n📤 출력:")
    print(f"   - Shape: {output.shape}")
    print(f"   - Range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"   - Mean: {output.mean().item():.4f}")

    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n📊 모델 정보:")
    print(f"   - 총 파라미터: {total_params:,}")
    print(f"   - 학습 가능: {trainable_params:,}")

    print("\n" + "="*60)
    print("✅ 테스트 완료!")
    print("="*60)
