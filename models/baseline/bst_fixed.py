"""
BST: Behavior Sequence Transformer (Alibaba, 2019) - 논문 정확 구현
논문: Behavior Sequence Transformer for E-commerce Recommendation in Alibaba

논문 기반 정확한 구현:
1. 타겟 아이템을 시퀀스 맨 앞에 추가
2. Transformer로 시퀀스 인코딩
3. 타겟 아이템 위치의 출력만 사용 (논문 핵심!)
4. 1개 Transformer 블록 사용 (논문에서 b=1이 최적)
"""

import torch
import torch.nn as nn
import math


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
    Behavior Sequence Transformer - 논문 정확 구현

    핵심 구조 (논문):
    1. Embedding: Item + Category + Other features
    2. Sequence: [Target Item] + [History Items] (타겟을 맨 앞에)
    3. Transformer: 1개 블록 (b=1)
    4. Output: 타겟 위치(index 0)의 Transformer 출력만 사용
    5. MLP: Other features + Target output → Prediction
    """

    def __init__(
        self,
        item_vocab_size,
        category_vocab_size,
        other_feature_dims,  # {feature_name: vocab_size}
        embed_dim=64,
        max_seq_len=50,
        num_heads=2,
        d_ff=256,
        dropout=0.1,
        dnn_hidden_units=[256, 128, 64]
    ):
        super(BST, self).__init__()

        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # 1. Embeddings
        self.item_embedding = nn.Embedding(item_vocab_size, embed_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(category_vocab_size, embed_dim, padding_idx=0)

        # Other feature embeddings
        self.other_embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size, embed_dim)
            for name, vocab_size in other_feature_dims.items()
        })

        # Position embedding (learnable, 논문에서는 시간 차이 기반이지만 여기서는 간단히 learnable로)
        self.position_embedding = nn.Embedding(max_seq_len + 1, embed_dim)  # +1 for target

        # 2. Transformer Encoder - 논문에서는 1개 블록이 최적
        self.transformer = TransformerLayer(embed_dim, num_heads, d_ff, dropout)

        # 3. MLP (논문: 3 hidden layers)
        # Input: target transformer output + other features
        dnn_input_dim = embed_dim + embed_dim * len(other_feature_dims)

        dnn_layers = []
        prev_dim = dnn_input_dim

        for hidden_dim in dnn_hidden_units:
            dnn_layers.append(nn.Linear(prev_dim, hidden_dim))
            dnn_layers.append(nn.BatchNorm1d(hidden_dim))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.dnn = nn.Sequential(*dnn_layers)

        # 4. Output layer
        self.output_layer = nn.Linear(prev_dim, 1)

    def forward(self, target_item, target_category, item_history, category_history, other_features):
        """
        Args:
            target_item: (batch_size,) - 타겟 아이템 ID
            target_category: (batch_size,) - 타겟 카테고리 ID
            item_history: (batch_size, seq_len) - 아이템 시퀀스 (0 is padding)
            category_history: (batch_size, seq_len) - 카테고리 시퀀스 (0 is padding) ⭐ 추가
            other_features: dict of {name: (batch_size,)}
        Returns:
            output: (batch_size,) - 예측 확률
        """
        batch_size = target_item.size(0)
        hist_seq_len = item_history.size(1)

        # 1. 시퀀스 구성: [Target Item] + [History Items]
        # Target item을 맨 앞에 추가 (논문의 핵심!)
        target_item_unsqueezed = target_item.unsqueeze(1)  # (batch, 1)
        target_category_unsqueezed = target_category.unsqueeze(1)  # (batch, 1) ⭐ 추가

        full_item_sequence = torch.cat([target_item_unsqueezed, item_history], dim=1)  # (batch, seq_len+1)
        full_category_sequence = torch.cat([target_category_unsqueezed, category_history], dim=1)  # (batch, seq_len+1) ⭐ 추가

        # 2. Item + Category embedding (논문의 핵심: 두 임베딩을 더함!)
        item_embed = self.item_embedding(full_item_sequence)  # (batch, seq_len+1, embed_dim)
        category_embed = self.category_embedding(full_category_sequence)  # (batch, seq_len+1, embed_dim) ⭐ 추가
        seq_embed = item_embed + category_embed  # (batch, seq_len+1, embed_dim) ⭐ Element-wise sum

        # 3. Positional embedding
        seq_len = full_item_sequence.size(1)
        positions = torch.arange(seq_len, device=full_item_sequence.device).unsqueeze(0).expand(batch_size, -1)
        pos_embed = self.position_embedding(positions)  # (batch, seq_len+1, embed_dim)

        # Combine item + category + position embeddings
        seq_embed = seq_embed + pos_embed

        # 4. Padding mask (0인 위치는 True)
        # 주의: target item(index 0)은 절대 mask되면 안됨
        padding_mask = (full_item_sequence == 0)  # (batch, seq_len+1)
        padding_mask[:, 0] = False  # Target position은 항상 valid

        # 5. Transformer encoding (1개 블록, 논문 권장)
        seq_encoded = self.transformer(seq_embed, padding_mask)  # (batch, seq_len+1, embed_dim)

        # 6. 타겟 아이템 출력 추출 (논문의 핵심!)
        target_output = seq_encoded[:, 0, :]  # (batch, embed_dim) - 첫 번째 위치

        # 7. Other feature embeddings
        other_embeds = []
        for name in sorted(other_features.keys()):
            if name in self.other_embeddings:
                other_embeds.append(self.other_embeddings[name](other_features[name]))

        # 8. Concatenate: target output + other features
        if other_embeds:
            concat_features = torch.cat([target_output] + other_embeds, dim=1)
        else:
            concat_features = target_output

        # 9. MLP prediction
        dnn_output = self.dnn(concat_features)
        output = self.output_layer(dnn_output)

        return torch.sigmoid(output.squeeze())
