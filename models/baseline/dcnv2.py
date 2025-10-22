"""
DCN-V2: Improved Deep & Cross Network (DCN-V2, 2020)
논문: DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems

본 구현은 DCNv2 논문의 Matrix form (더 간단한 버전)을 사용합니다.
"""

import torch
import torch.nn as nn


class CrossNetworkV2(nn.Module):
    """
    Cross Network V2 - Matrix Form

    논문 수식:
    x_{l+1} = x_0 ⊙ (W_l · x_l + b_l) + x_l

    여기서:
    - W_l ∈ R^{d×d}: weight matrix
    - b_l ∈ R^d: bias
    - ⊙: element-wise product
    """

    def __init__(self, input_dim, num_layers=3):
        super(CrossNetworkV2, self).__init__()
        self.num_layers = num_layers

        # Weight matrices and biases for each layer
        self.weight = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, input_dim) * 0.01)
            for _ in range(num_layers)
        ])

        self.bias = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim))
            for _ in range(num_layers)
        ])

    def forward(self, x0):
        """
        Args:
            x0: (batch_size, input_dim)
        Returns:
            x_l: (batch_size, input_dim)
        """
        x_l = x0

        for layer in range(self.num_layers):
            # W_l · x_l: (batch, d)
            x_w = torch.matmul(x_l, self.weight[layer].T)

            # x_0 ⊙ (W_l · x_l + b_l) + x_l
            x_l = x0 * (x_w + self.bias[layer]) + x_l

        return x_l


class DCNV2(nn.Module):
    """
    DCN-V2: Improved Deep & Cross Network

    구조:
    1. Embedding layer
    2. Cross Network V2 (parallel with DNN)
    3. Deep Network
    4. Final prediction layer
    """

    def __init__(
        self,
        num_features,
        cat_vocab_sizes,
        embed_dim=16,
        cross_num_layers=3,
        dnn_hidden_units=[256, 128, 64],
        dropout=0.1,
        structure='parallel'  # 'parallel' or 'stacked'
    ):
        super(DCNV2, self).__init__()

        self.structure = structure

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

        # 2. Cross Network V2
        self.cross_net = CrossNetworkV2(
            input_dim=input_dim,
            num_layers=cross_num_layers
        )

        # 3. Deep Network
        dnn_layers = []
        prev_dim = input_dim

        for hidden_dim in dnn_hidden_units:
            dnn_layers.append(nn.Linear(prev_dim, hidden_dim))
            dnn_layers.append(nn.BatchNorm1d(hidden_dim))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.dnn = nn.Sequential(*dnn_layers)

        # 4. Final layer
        if structure == 'parallel':
            # Parallel: concat cross output + dnn output
            final_dim = input_dim + prev_dim
        else:
            # Stacked: cross -> dnn
            final_dim = prev_dim

        self.final_layer = nn.Linear(final_dim, 1)

    def forward(self, num_features, cat_features):
        """
        Args:
            num_features: (batch_size, num_numeric_features)
            cat_features: dict of {feat_name: (batch_size,)}
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

        if self.structure == 'parallel':
            # Parallel structure
            cross_out = self.cross_net(embed_concat)
            dnn_out = self.dnn(embed_concat)
            final_input = torch.cat([cross_out, dnn_out], dim=1)
        else:
            # Stacked structure: cross -> dnn
            cross_out = self.cross_net(embed_concat)
            dnn_out = self.dnn(cross_out)
            final_input = dnn_out

        output = self.final_layer(final_input)
        return torch.sigmoid(output.squeeze())
