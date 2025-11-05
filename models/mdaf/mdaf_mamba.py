"""
MDAF-Mamba: Multi-branch Dual Attention Fusion with Mamba4Rec

Hybrid CTR prediction model that combines:
- Static branch: DCNv3 for cross-feature interactions
- Sequential branch: Mamba4Rec for behavioral sequence modeling
- Gated fusion: Adaptive weighting of both branches

Architecture:
    Input Features
    ├── Static Features → DCNv3 → static_embedding (128-dim)
    ├── Sequential Features → Mamba4Rec → sequential_embedding (128-dim)
    └── Gated Fusion → MLP Prediction Head → click_probability
"""

import torch
import torch.nn as nn
from models.mdaf.dcnv3 import DCNV3
from models.mdaf.mamba4rec import Mamba4Rec
from models.mdaf.mdaf_components import GatedFusion, PredictionHead


class MDAF_Mamba(nn.Module):
    """
    MDAF with Mamba4Rec as sequential branch.

    This model integrates static feature modeling (DCNv3) with sequential
    behavioral modeling (Mamba4Rec) through an adaptive gated fusion mechanism.
    """

    def __init__(
        self,
        # Vocabulary sizes
        item_vocab_size,
        category_vocab_size,
        user_vocab_size,
        hour_vocab_size=24,
        dayofweek_vocab_size=7,
        # DCNv3 configuration
        dcnv3_embed_dim=16,
        dcnv3_lcn_layers=3,
        dcnv3_ecn_layers=3,
        dcnv3_dropout=0.0,
        # Mamba4Rec configuration
        item_embed_dim=64,
        category_embed_dim=32,
        static_embed_dim=16,
        mamba_hidden_dim=128,
        mamba_num_layers=2,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        # MDAF configuration
        embedding_dim=128,  # Output dimension for both branches
        prediction_hidden_dims=[128, 64],
        dropout=0.2
    ):
        """
        Args:
            item_vocab_size: Number of unique items
            category_vocab_size: Number of unique categories
            user_vocab_size: Number of unique users
            hour_vocab_size: Number of hours (24)
            dayofweek_vocab_size: Number of weekdays (7)
            dcnv3_embed_dim: DCNv3 embedding dimension per feature
            dcnv3_lcn_layers: Number of Linear Cross Network layers
            dcnv3_ecn_layers: Number of Exponential Cross Network layers
            dcnv3_dropout: DCNv3 dropout rate
            item_embed_dim: Item embedding dimension for Mamba4Rec
            category_embed_dim: Category embedding dimension for Mamba4Rec
            static_embed_dim: Static feature embedding dimension
            mamba_hidden_dim: Hidden dimension for Mamba layers
            mamba_num_layers: Number of Mamba layers
            mamba_d_state: SSM state dimension
            mamba_d_conv: Convolution kernel size
            mamba_expand: Expansion factor for Mamba blocks
            embedding_dim: Target embedding dimension for fusion (must be 128)
            prediction_hidden_dims: Hidden dimensions for prediction MLP
            dropout: Dropout rate for fusion and prediction
        """
        super(MDAF_Mamba, self).__init__()

        # Critical: Both branches must output the same embedding dimension
        assert mamba_hidden_dim == embedding_dim, \
            f"Mamba hidden_dim ({mamba_hidden_dim}) must match embedding_dim ({embedding_dim})"

        self.embedding_dim = embedding_dim

        # === Static Branch: DCNv3 ===
        # For Taobao: 5 static categorical features
        # (target_item, target_category, user_id, hour, dayofweek)
        num_static_features = 0  # No numeric features
        cat_vocab_sizes = {
            'target_item': item_vocab_size,
            'target_category': category_vocab_size,
            'user_id': user_vocab_size,
            'hour': hour_vocab_size,
            'dayofweek': dayofweek_vocab_size
        }

        self.dcnv3 = DCNV3(
            num_features=num_static_features,
            cat_vocab_sizes=cat_vocab_sizes,
            embed_dim=dcnv3_embed_dim,
            lcn_num_layers=dcnv3_lcn_layers,
            ecn_num_layers=dcnv3_ecn_layers,
            dropout=dcnv3_dropout
        )

        # Calculate DCNv3 output dimension
        # DCNv3 outputs: (LCN + ECN) / 2, where each is (num_fields * embed_dim)
        num_fields = num_static_features + len(cat_vocab_sizes)
        dcnv3_output_dim = num_fields * dcnv3_embed_dim

        # Project DCNv3 output to target embedding dimension
        self.dcnv3_projection = nn.Sequential(
            nn.Linear(dcnv3_output_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # === Sequential Branch: Mamba4Rec ===
        self.mamba4rec = Mamba4Rec(
            item_vocab_size=item_vocab_size,
            category_vocab_size=category_vocab_size,
            user_vocab_size=user_vocab_size,
            hour_vocab_size=hour_vocab_size,
            dayofweek_vocab_size=dayofweek_vocab_size,
            item_embed_dim=item_embed_dim,
            category_embed_dim=category_embed_dim,
            static_embed_dim=static_embed_dim,
            hidden_dim=mamba_hidden_dim,  # Must equal embedding_dim
            num_layers=mamba_num_layers,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            dropout=dropout
        )

        # === Gated Fusion ===
        self.fusion = GatedFusion(embedding_dim=embedding_dim, dropout=dropout)

        # === Prediction Head ===
        self.prediction_head = PredictionHead(
            input_dim=embedding_dim,
            hidden_dims=prediction_hidden_dims,
            dropout=dropout
        )

    def forward(
        self,
        target_item,
        target_category,
        item_history,
        category_history,
        other_features,
        return_gate=False
    ):
        """
        Args:
            target_item: (batch_size,) - target item ID
            target_category: (batch_size,) - target category ID
            item_history: (batch_size, seq_len) - item sequence
            category_history: (batch_size, seq_len) - category sequence
            other_features: dict with keys 'user_id', 'hour', 'dayofweek'
            return_gate: if True, also return gate values for analysis

        Returns:
            If return_gate=False:
                logits: (batch_size,) - predicted click probability
            If return_gate=True:
                (logits, gate): logits and gate values for analysis
        """
        # === Static Branch: DCNv3 ===
        # Prepare DCNv3 inputs
        batch_size = target_item.size(0)
        num_features = torch.empty(batch_size, 0, dtype=torch.float32, device=target_item.device)

        cat_features = {
            'target_item': target_item,
            'target_category': target_category,
            'user_id': other_features['user_id'],
            'hour': other_features['hour'],
            'dayofweek': other_features['dayofweek']
        }

        # Get DCNv3 embedding
        dcnv3_emb = self.dcnv3(num_features, cat_features, return_embedding=True)  # (batch, dcnv3_output_dim)

        # Project to target dimension
        static_emb = self.dcnv3_projection(dcnv3_emb)  # (batch, embedding_dim)

        # === Sequential Branch: Mamba4Rec ===
        sequential_emb = self.mamba4rec(
            target_item=target_item,
            target_category=target_category,
            item_history=item_history,
            category_history=category_history,
            other_features=other_features,
            return_embedding=True
        )  # (batch, embedding_dim)

        # === Gated Fusion ===
        fusion_emb, gate = self.fusion(static_emb, sequential_emb)  # (batch, embedding_dim), (batch, 1)

        # === Prediction ===
        logits = self.prediction_head(fusion_emb).squeeze(-1)  # (batch,)

        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(logits)

        if return_gate:
            return probabilities, gate
        return probabilities


if __name__ == '__main__':
    """Test MDAF_Mamba with dummy Taobao data"""
    print("="*60)
    print("Testing MDAF_Mamba Model")
    print("="*60)

    # Dummy vocabulary sizes (from Taobao)
    item_vocab_size = 335164
    category_vocab_size = 5480
    user_vocab_size = 577482

    # Create model
    model = MDAF_Mamba(
        item_vocab_size=item_vocab_size,
        category_vocab_size=category_vocab_size,
        user_vocab_size=user_vocab_size,
        dcnv3_embed_dim=16,
        item_embed_dim=64,
        category_embed_dim=32,
        mamba_hidden_dim=128,
        embedding_dim=128
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy batch
    batch_size = 4
    seq_len = 50

    target_item = torch.randint(1, item_vocab_size, (batch_size,))
    target_category = torch.randint(1, category_vocab_size, (batch_size,))
    item_history = torch.randint(0, item_vocab_size, (batch_size, seq_len))
    category_history = torch.randint(0, category_vocab_size, (batch_size, seq_len))
    other_features = {
        'user_id': torch.randint(0, user_vocab_size, (batch_size,)),
        'hour': torch.randint(0, 24, (batch_size,)),
        'dayofweek': torch.randint(0, 7, (batch_size,))
    }

    # Forward pass
    print("\n1. Testing forward pass...")
    output, gate = model(
        target_item=target_item,
        target_category=target_category,
        item_history=item_history,
        category_history=category_history,
        other_features=other_features,
        return_gate=True
    )

    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"   Output sample: {output[:3]}")
    print(f"   Gate shape: {gate.shape}")
    print(f"   Gate values: {gate.squeeze().tolist()}")
    print(f"   Gate range: [{gate.min():.4f}, {gate.max():.4f}]")

    # Test backward pass
    print("\n2. Testing backward pass...")
    labels = torch.randint(0, 2, (batch_size,)).float()
    loss = torch.nn.functional.binary_cross_entropy(output, labels)
    loss.backward()

    print(f"   Loss: {loss.item():.4f}")
    print(f"   Gradients exist: {any(p.grad is not None for p in model.parameters())}")

    print("\n" + "="*60)
    print("✅ MDAF_Mamba test passed!")
    print("="*60)
