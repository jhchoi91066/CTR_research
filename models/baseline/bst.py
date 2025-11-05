"""
BST: Behavior Sequence Transformer (Alibaba, 2019)
논문: Behavior Sequence Transformer for E-commerce Recommendation in Alibaba

핵심 아이디어:
- Transformer의 Self-Attention을 사용한 사용자 행동 시퀀스 모델링
- 위치 인코딩으로 시간 순서 정보 포함
- Target item attention으로 관련성 높은 과거 행동에 집중
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer
    논문에서는 sinusoidal positional encoding 사용
    """

    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()

        # (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, seq_len):
        """
        Args:
            seq_len: sequence length
        Returns:
            pe: (seq_len, d_model)
        """
        return self.pe[:seq_len, :]


class TransformerLayer(nn.Module):
    """
    Single Transformer Encoder Layer
    Multi-Head Self-Attention + Feed Forward
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) - True for positions to mask
        Returns:
            output: (batch, seq_len, d_model)
        """
        # Self-Attention with residual connection
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed Forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x


class BST(nn.Module):
    """
    Behavior Sequence Transformer

    구조:
    1. Item Embedding + Positional Encoding
    2. Transformer Encoder Layers
    3. Target Attention (현재 아이템과의 관련성)
    4. DNN + Prediction
    """

    def __init__(
        self,
        item_vocab_size,
        category_vocab_size,
        other_feature_dims,  # {feature_name: vocab_size}
        embed_dim=64,
        max_seq_len=50,
        num_transformer_layers=2,
        num_heads=2,
        d_ff=256,
        dropout=0.1,
        dnn_hidden_units=[256, 128]
    ):
        super(BST, self).__init__()

        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # 1. Embeddings
        self.item_embedding = nn.Embedding(item_vocab_size, embed_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(category_vocab_size, embed_dim, padding_idx=0)

        # Other feature embeddings (user_id, hour, dayofweek 등)
        self.other_embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size, embed_dim)
            for name, vocab_size in other_feature_dims.items()
        })

        # Positional Encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len)

        # 2. Transformer Encoder
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, d_ff, dropout)
            for _ in range(num_transformer_layers)
        ])

        # 3. Target Attention (타겟 아이템과 시퀀스 간 attention)
        self.target_attention = nn.Linear(embed_dim, 1)

        # 4. DNN
        # Input: target item embed + category + other features + pooled sequence
        dnn_input_dim = embed_dim * (2 + len(other_feature_dims) + 1)  # item + cat + others + seq

        dnn_layers = []
        prev_dim = dnn_input_dim

        for hidden_dim in dnn_hidden_units:
            dnn_layers.append(nn.Linear(prev_dim, hidden_dim))
            dnn_layers.append(nn.BatchNorm1d(hidden_dim))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.dnn = nn.Sequential(*dnn_layers)

        # 5. Output layer
        self.output_layer = nn.Linear(prev_dim, 1)

    def forward(self, target_item, target_category, item_history, other_features, return_embedding=False):
        """
        Args:
            target_item: (batch_size,) - 타겟 아이템 ID
            target_category: (batch_size,) - 타겟 카테고리 ID
            item_history: (batch_size, seq_len) - 아이템 시퀀스 (0 is padding)
            other_features: dict of {name: (batch_size,)}
            return_embedding: if True, return intermediate embedding for MDAF
        Returns:
            If return_embedding=True:
                seq_pooled: (batch_size, embed_dim) - attention-pooled sequence embedding
            Else:
                output: (batch_size,) - 예측 확률
        """
        batch_size = target_item.size(0)
        seq_len = item_history.size(1)

        # 1. Target item embedding
        target_item_embed = self.item_embedding(target_item)  # (batch, embed_dim)
        target_cat_embed = self.category_embedding(target_category)  # (batch, embed_dim)

        # 2. Sequence embedding + positional encoding
        seq_embed = self.item_embedding(item_history)  # (batch, seq_len, embed_dim)

        # Add positional encoding
        pos_enc = self.pos_encoding(seq_len).unsqueeze(0)  # (1, seq_len, embed_dim)
        seq_embed = seq_embed + pos_enc

        # 3. Create padding mask (0인 위치는 True)
        padding_mask = (item_history == 0)  # (batch, seq_len)

        # 4. Transformer encoding
        for transformer in self.transformer_layers:
            seq_embed = transformer(seq_embed, padding_mask)

        # 5. Target Attention - 타겟 아이템과 시퀀스의 관련성
        # 각 시퀀스 아이템과 타겟 아이템의 similarity 계산
        target_expanded = target_item_embed.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, embed_dim)
        similarity = torch.sum(seq_embed * target_expanded, dim=-1)  # (batch, seq_len)

        # Mask out padding positions
        similarity = similarity.masked_fill(padding_mask, -1e9)

        # Attention weights
        attn_weights = torch.softmax(similarity, dim=-1)  # (batch, seq_len)

        # Weighted sum of sequence embeddings
        seq_pooled = torch.sum(seq_embed * attn_weights.unsqueeze(-1), dim=1)  # (batch, embed_dim)

        # For MDAF: return attention-pooled sequence embedding
        if return_embedding:
            return seq_pooled

        # 6. Other feature embeddings
        other_embeds = []
        for name in sorted(other_features.keys()):
            if name in self.other_embeddings:
                other_embeds.append(self.other_embeddings[name](other_features[name]))

        # 7. Concatenate all features
        all_features = [target_item_embed, target_cat_embed, seq_pooled] + other_embeds
        concat_features = torch.cat(all_features, dim=1)  # (batch, total_embed_dim)

        # 8. DNN
        dnn_output = self.dnn(concat_features)

        # 9. Prediction
        output = self.output_layer(dnn_output)
        return torch.sigmoid(output.squeeze())
