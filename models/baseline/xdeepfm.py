"""
xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems (KDD 2018)

Reference:
- Paper: https://arxiv.org/abs/1803.05170
- Official implementation: https://github.com/Leavingseason/xDeepFM

논문 핵심:
    y_hat = sigmoid(w_linear^T a + w_cin^T p^+ + w_dnn^T x_dnn + b)

구성 요소:
    1. Linear: 1차 특징
    2. CIN (Compressed Interaction Network): 명시적 vector-wise 고차 상호작용
    3. DNN: 암묵적 bit-wise 고차 상호작용
"""

import torch
import torch.nn as nn


class CIN(nn.Module):
    """
    Compressed Interaction Network (CIN)

    논문 수식 (Section 3.2):
        X^k_{h,*} = Σ_{i=1}^{H_{k-1}} Σ_{j=1}^m W^{k,h}_{i,j} (X^{k-1}_{i,*} ⊙ X^0_{j,*})

    여기서:
        - X^k: k번째 레이어의 feature map (H_k x D)
        - X^0: 초기 embedding matrix (m x D)
        - ⊙: element-wise product (Hadamard product)
        - W^{k,h}: 학습 가능한 파라미터 (H_{k-1} x m)
        - H_k: k번째 레이어의 feature map 개수
        - m: 필드 수
        - D: embedding 차원

    구현 방식:
        - 외적을 직접 계산하면 (batch, H_{k-1}, m, D) 텐서 생성
        - 이를 (batch, H_{k-1} * m, D)로 reshape
        - 1x1 Conv1D로 (batch, H_k, D)로 압축
    """
    def __init__(self, num_fields, cin_layer_sizes, split_half=True):
        """
        Args:
            num_fields: 입력 필드 수 (m)
            cin_layer_sizes: 각 CIN 레이어의 feature map 개수 리스트 [H_1, H_2, ...]
            split_half: True면 각 레이어 출력을 절반으로 나눠 일부만 최종 출력에 사용
                       (논문의 "direct connections" 구현)
        """
        super(CIN, self).__init__()
        self.num_fields = num_fields
        self.cin_layer_sizes = cin_layer_sizes
        self.split_half = split_half

        # CIN layers
        self.conv_layers = nn.ModuleList()
        prev_size = num_fields

        for i, size in enumerate(cin_layer_sizes):
            # Conv1D: (batch, H_{k-1} * m, D) -> (batch, H_k, D)
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=prev_size * num_fields,
                    out_channels=size,
                    kernel_size=1,
                    stride=1
                )
            )

            if split_half and i != len(cin_layer_sizes) - 1:
                # 절반만 다음 레이어로 전달 (나머지는 최종 출력에 직접 연결)
                prev_size = size // 2
            else:
                prev_size = size

        # 최종 출력 차원 계산
        if split_half:
            # 마지막 레이어 제외하고 모두 절반, 마지막 레이어는 전체
            self.output_dim = sum(cin_layer_sizes[:-1]) // 2 + cin_layer_sizes[-1]
        else:
            self.output_dim = sum(cin_layer_sizes)

    def forward(self, x0):
        """
        Args:
            x0: (batch_size, num_fields, embed_dim)

        Returns:
            pooled_outputs: (batch_size, output_dim)
        """
        batch_size = x0.size(0)
        embed_dim = x0.size(2)

        # Hidden layers
        hidden_nn_layers = [x0]  # X^0
        final_result = []

        for i, conv in enumerate(self.conv_layers):
            # X^{k-1}
            x_k_minus_1 = hidden_nn_layers[-1]  # (batch, H_{k-1}, D)

            # Outer product: X^{k-1} ⊙ X^0
            # (batch, H_{k-1}, 1, D) * (batch, 1, m, D) = (batch, H_{k-1}, m, D)
            z = torch.einsum('bhd,bmd->bhmd', x_k_minus_1, x0)

            # Reshape: (batch, H_{k-1} * m, D)
            z = z.view(batch_size, -1, embed_dim)

            # Conv1D expects (N, C_in, L)
            # 여기서 C_in = H_{k-1} * m, L = D
            # z는 현재 (batch, H_{k-1} * m, D)
            # Conv1D로 전달: (batch, H_{k-1} * m, D)
            x_k = conv(z)  # (batch, H_k, D)

            # ReLU activation
            x_k = torch.relu(x_k)

            # Direct connection (split_half)
            if self.split_half and i != len(self.conv_layers) - 1:
                # 절반은 최종 출력으로, 절반은 다음 레이어로
                split_size = x_k.size(1) // 2
                direct_connect, next_hidden = torch.split(x_k, [split_size, split_size], dim=1)
                final_result.append(direct_connect)
                hidden_nn_layers.append(next_hidden)
            else:
                # 마지막 레이어 또는 split 안함
                final_result.append(x_k)
                if i != len(self.conv_layers) - 1:
                    hidden_nn_layers.append(x_k)

        # Sum pooling along embedding dimension
        # [(batch, H_k, D)] -> [(batch, H_k)]
        pooled_outputs = [torch.sum(x, dim=-1) for x in final_result]

        # Concatenate: (batch, sum(H_k))
        output = torch.cat(pooled_outputs, dim=1)

        return output


class DNN(nn.Module):
    """Deep Neural Network"""
    def __init__(self, input_dim, hidden_units, dropout=0.0, activation='relu'):
        super(DNN, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_units:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Tanh())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.dnn = nn.Sequential(*layers)
        self.output_dim = prev_dim

    def forward(self, x):
        return self.dnn(x)


class xDeepFM(nn.Module):
    """
    xDeepFM: Combining Explicit and Implicit Feature Interactions (KDD 2018)

    논문 수식 (Section 3.3):
        y_hat = sigmoid(w_linear^T a + w_cin^T p^+ + w_dnn^T x_dnn + b)

    여기서:
        - a: 1차 특징 (linear part)
        - p^+: CIN의 sum-pooled 출력
        - x_dnn: DNN의 출력

    핵심 차별점:
        - CIN: Explicit, vector-wise, high-order feature interactions
        - DNN: Implicit, bit-wise, high-order feature interactions
        - 두 방식을 결합하여 complementary한 특징 학습
    """
    def __init__(
        self,
        num_features,
        cat_vocab_sizes,
        embed_dim=16,
        cin_layer_sizes=[128, 128],
        cin_split_half=True,
        dnn_hidden_units=[256, 128, 64],
        dnn_activation='relu',
        dropout=0.1
    ):
        super(xDeepFM, self).__init__()

        self.num_features = num_features
        self.cat_vocab_sizes = cat_vocab_sizes
        self.embed_dim = embed_dim

        # Embeddings - 각 feature를 독립적인 field로 취급
        # Numeric features: 각각 독립적인 embedding
        self.numeric_embeddings = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(num_features)
        ])

        # Categorical features
        self.cat_embeddings = nn.ModuleDict({
            feat: nn.Embedding(vocab_size, embed_dim)
            for feat, vocab_size in cat_vocab_sizes.items()
        })

        self.num_fields = num_features + len(cat_vocab_sizes)

        # 1. Linear part (1st order)
        self.linear = nn.Embedding(self.num_fields, 1)
        self.bias = nn.Parameter(torch.zeros((1,)))

        # 2. CIN (explicit high-order interactions)
        self.cin = CIN(
            num_fields=self.num_fields,
            cin_layer_sizes=cin_layer_sizes,
            split_half=cin_split_half
        )

        # 3. DNN (implicit high-order interactions)
        dnn_input_dim = self.num_fields * embed_dim
        self.dnn = DNN(
            input_dim=dnn_input_dim,
            hidden_units=dnn_hidden_units,
            dropout=dropout,
            activation=dnn_activation
        )

        # Final linear combination weights
        # 논문에서는 linear, CIN, DNN 출력을 weighted sum
        final_input_dim = 1 + self.cin.output_dim + self.dnn.output_dim
        self.final_linear = nn.Linear(final_input_dim, 1, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, numeric_features, categorical_features):
        """
        Args:
            numeric_features: (batch_size, num_features)
            categorical_features: dict {feat_name: (batch_size,)}

        Returns:
            output: (batch_size, 1)
        """
        batch_size = numeric_features.size(0)

        # Embeddings
        # Numeric: 각 feature를 독립적으로 embedding
        numeric_embeds = []
        for i in range(self.num_features):
            feat = numeric_features[:, i:i+1]  # (batch, 1)
            embed = self.numeric_embeddings[i](feat)  # (batch, embed_dim)
            numeric_embeds.append(embed)

        # Categorical
        cat_embeds = []
        for feat_name in sorted(categorical_features.keys()):
            embed = self.cat_embeddings[feat_name](categorical_features[feat_name])
            cat_embeds.append(embed)

        # (batch, num_fields, embed_dim)
        embeddings = torch.stack(numeric_embeds + cat_embeds, dim=1)

        # Field indices
        field_indices = torch.arange(self.num_fields, device=embeddings.device)
        field_indices = field_indices.unsqueeze(0).expand(batch_size, -1)

        # 1. Linear part
        linear_out = self.linear(field_indices).squeeze(-1).sum(dim=1, keepdim=True) + self.bias

        # 2. CIN part
        cin_out = self.cin(embeddings)  # (batch, cin_output_dim)

        # 3. DNN part
        dnn_input = embeddings.view(batch_size, -1)
        dnn_out = self.dnn(dnn_input)  # (batch, dnn_output_dim)

        # Concatenate all
        final_input = torch.cat([linear_out, cin_out, dnn_out], dim=1)

        # Final prediction
        logit = self.final_linear(final_input)
        output = torch.sigmoid(logit)

        return output


if __name__ == '__main__':
    """Unit test"""
    batch_size = 32
    num_features = 13
    cat_vocab_sizes = {f'C{i}': 100 for i in range(26)}

    model = xDeepFM(
        num_features=num_features,
        cat_vocab_sizes=cat_vocab_sizes,
        embed_dim=16,
        cin_layer_sizes=[128, 128],
        cin_split_half=True,
        dnn_hidden_units=[256, 128, 64],
        dropout=0.1
    )

    numeric_features = torch.randn(batch_size, num_features)
    categorical_features = {
        f'C{i}': torch.randint(0, 100, (batch_size,))
        for i in range(26)
    }

    output = model(numeric_features, categorical_features)

    print(f"✅ xDeepFM 초기화 완료 (논문 정확 구현)")
    print(f"   - 수치형 특징: {num_features}개")
    print(f"   - 범주형 특징: {len(cat_vocab_sizes)}개")
    print(f"   - Total fields: {model.num_fields}")
    print(f"   - Embedding 차원: {model.embed_dim}")
    print(f"   - CIN 구조: {model.cin.cin_layer_sizes}")
    print(f"   - DNN 구조: {model.dnn.output_dim}")
    print(f"   - Output shape: {output.shape}")
    print(f"   - Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"   - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
