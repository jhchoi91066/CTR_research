"""
AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks (CIKM 2019)
논문: https://arxiv.org/abs/1810.11921

핵심 아이디어:
- Multi-head Self-Attention을 사용한 고차원 feature interaction 자동 학습
- Transformer의 Self-Attention 메커니즘 활용
"""

import torch
import torch.nn as nn
import math


class InteractingLayer(nn.Module):
    """
    Multi-Head Self-Attention Layer for Feature Interaction

    논문 수식:
    Attention(Q, K, V) = softmax(QK^T / √d_k)V
    MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
    """

    def __init__(self, embed_dim, num_heads=2, use_residual=True):
        super(InteractingLayer, self).__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_residual = use_residual

        # Multi-head attention parameters
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)

        # Output projection
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)

        # Residual connection
        if use_residual:
            self.W_Res = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_fields, embed_dim)
        Returns:
            output: (batch_size, num_fields, embed_dim)
        """
        batch_size, num_fields, embed_dim = x.size()

        # Linear projections: (batch, num_fields, embed_dim) -> (batch, num_fields, num_heads, head_dim)
        Q = self.W_Q(x).view(batch_size, num_fields, self.num_heads, self.head_dim)
        K = self.W_K(x).view(batch_size, num_fields, self.num_heads, self.head_dim)
        V = self.W_V(x).view(batch_size, num_fields, self.num_heads, self.head_dim)

        # Transpose for attention: (batch, num_heads, num_fields, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled dot-product attention
        # (batch, num_heads, num_fields, num_fields)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply attention to values: (batch, num_heads, num_fields, head_dim)
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads: (batch, num_fields, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, num_fields, embed_dim)

        # Output projection
        output = self.W_O(attn_output)

        # Residual connection
        if self.use_residual:
            output = output + self.W_Res(x)

        return output


class AutoInt(nn.Module):
    """
    AutoInt: Automatic Feature Interaction Learning

    구조:
    1. Embedding layer
    2. Multiple Interacting layers (Self-Attention)
    3. Output layer
    """

    def __init__(
        self,
        num_features,
        cat_vocab_sizes,
        embed_dim=16,
        num_layers=3,
        num_heads=2,
        use_residual=True,
        use_dnn=False,
        dnn_hidden_units=[256, 128, 64],
        dropout=0.1
    ):
        super(AutoInt, self).__init__()

        self.use_dnn = use_dnn

        # 1. Embedding layers
        self.numeric_embeddings = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(num_features)
        ])

        self.cat_embeddings = nn.ModuleDict({
            feat: nn.Embedding(vocab_size, embed_dim)
            for feat, vocab_size in cat_vocab_sizes.items()
        })

        self.num_fields = num_features + len(cat_vocab_sizes)

        # 2. Interacting layers (Self-Attention)
        self.interacting_layers = nn.ModuleList([
            InteractingLayer(embed_dim, num_heads, use_residual)
            for _ in range(num_layers)
        ])

        # 3. Output layer
        if use_dnn:
            # AutoInt + DNN
            dnn_layers = []
            prev_dim = self.num_fields * embed_dim

            for hidden_dim in dnn_hidden_units:
                dnn_layers.append(nn.Linear(prev_dim, hidden_dim))
                dnn_layers.append(nn.BatchNorm1d(hidden_dim))
                dnn_layers.append(nn.ReLU())
                dnn_layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim

            self.dnn = nn.Sequential(*dnn_layers)

            # Final layer combines attention output + DNN output
            final_dim = self.num_fields * embed_dim + prev_dim
            self.final_layer = nn.Linear(final_dim, 1)
        else:
            # AutoInt only (논문의 기본 구조)
            self.final_layer = nn.Linear(self.num_fields * embed_dim, 1)

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

        # Stack all embeddings: (batch, num_fields, embed_dim)
        all_embeds = num_embeds + cat_embeds
        embed_stack = torch.stack(all_embeds, dim=1)

        # Apply interacting layers (Self-Attention)
        attn_output = embed_stack
        for layer in self.interacting_layers:
            attn_output = layer(attn_output)

        # Flatten: (batch, num_fields * embed_dim)
        attn_flat = attn_output.view(attn_output.size(0), -1)

        if self.use_dnn:
            # Combine attention output with DNN
            dnn_output = self.dnn(attn_flat)
            final_input = torch.cat([attn_flat, dnn_output], dim=1)
        else:
            final_input = attn_flat

        # Final prediction
        output = self.final_layer(final_input)
        return torch.sigmoid(output.squeeze())
