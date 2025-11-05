"""
MDAF Shared Components

This module contains the core fusion and prediction components used by both
MDAF-Mamba and MDAF-BST models to ensure identical fusion mechanisms.
"""

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """
    Gated Fusion Mechanism for MDAF

    Adaptively learns to weight static (DCNv3) and sequential (Mamba4Rec/BST)
    branch embeddings based on their relevance for each sample.

    Architecture:
        concat([static_emb, sequential_emb]) → MLP → Sigmoid → gate
        fusion = gate * static_emb + (1 - gate) * sequential_emb

    Gate interpretation:
        - gate ≈ 1.0: Model relies on static features (DCNv3)
        - gate ≈ 0.0: Model relies on sequential features (Mamba4Rec/BST)
        - gate ≈ 0.5: Model balances both branches
    """

    def __init__(self, embedding_dim=128, dropout=0.2):
        """
        Args:
            embedding_dim: Dimension of both static and sequential embeddings
            dropout: Dropout rate for regularization
        """
        super(GatedFusion, self).__init__()

        self.embedding_dim = embedding_dim

        # Gate network: learns adaptive weighting
        self.gate_network = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()  # Gate values in [0, 1]
        )

    def forward(self, static_emb, sequential_emb):
        """
        Args:
            static_emb: (batch_size, embedding_dim) from DCNv3
            sequential_emb: (batch_size, embedding_dim) from Mamba4Rec/BST

        Returns:
            fusion_emb: (batch_size, embedding_dim) fused representation
            gate: (batch_size, 1) learned gate values (for analysis)
        """
        # Input validation
        assert static_emb.shape == sequential_emb.shape, \
            f"Embedding shape mismatch: static={static_emb.shape}, sequential={sequential_emb.shape}"
        assert static_emb.shape[-1] == self.embedding_dim, \
            f"Expected embedding_dim={self.embedding_dim}, got {static_emb.shape[-1]}"

        # Concatenate embeddings
        concat_emb = torch.cat([static_emb, sequential_emb], dim=-1)  # (batch, 256)

        # Learn adaptive gate
        gate = self.gate_network(concat_emb)  # (batch, 1)

        # Gated fusion: weighted combination
        # fusion = gate * static + (1 - gate) * sequential
        fusion_emb = gate * static_emb + (1 - gate) * sequential_emb  # (batch, embedding_dim)

        return fusion_emb, gate


class PredictionHead(nn.Module):
    """
    MLP Prediction Head for MDAF

    Transforms the fused embedding into click probability prediction.
    Uses a multi-layer perceptron with ReLU activations and dropout.
    """

    def __init__(self, input_dim=128, hidden_dims=[128, 64], dropout=0.2):
        """
        Args:
            input_dim: Input dimension (from fusion)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate for regularization
        """
        super(PredictionHead, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Final prediction layer (no activation - use BCEWithLogitsLoss or apply sigmoid)
        layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, fusion_emb):
        """
        Args:
            fusion_emb: (batch_size, input_dim)

        Returns:
            logits: (batch_size, 1) - raw scores (apply sigmoid for probability)
        """
        return self.mlp(fusion_emb)


if __name__ == '__main__':
    """Test the components with dummy data"""
    print("="*60)
    print("Testing MDAF Components")
    print("="*60)

    batch_size = 4
    embedding_dim = 128

    # Create dummy embeddings
    static_emb = torch.randn(batch_size, embedding_dim)
    sequential_emb = torch.randn(batch_size, embedding_dim)

    # Test GatedFusion
    print("\n1. Testing GatedFusion...")
    fusion = GatedFusion(embedding_dim=embedding_dim)
    print(f"   GatedFusion parameters: {sum(p.numel() for p in fusion.parameters()):,}")

    fusion_emb, gate = fusion(static_emb, sequential_emb)

    print(f"   Static embedding shape: {static_emb.shape}")
    print(f"   Sequential embedding shape: {sequential_emb.shape}")
    print(f"   Fusion embedding shape: {fusion_emb.shape}")
    print(f"   Gate shape: {gate.shape}")
    print(f"   Gate values: {gate.squeeze().tolist()}")
    print(f"   Gate range: [{gate.min():.4f}, {gate.max():.4f}]")

    # Verify gate values are in [0, 1]
    assert (gate >= 0).all() and (gate <= 1).all(), "Gate values must be in [0, 1]"
    print("   ✓ Gate values are valid")

    # Test PredictionHead
    print("\n2. Testing PredictionHead...")
    pred_head = PredictionHead(input_dim=embedding_dim, hidden_dims=[128, 64])
    print(f"   PredictionHead parameters: {sum(p.numel() for p in pred_head.parameters()):,}")

    logits = pred_head(fusion_emb)
    probabilities = torch.sigmoid(logits)

    print(f"   Fusion embedding shape: {fusion_emb.shape}")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Probabilities shape: {probabilities.shape}")
    print(f"   Probability range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")

    # Test backward pass
    print("\n3. Testing gradient flow...")
    loss = torch.nn.functional.binary_cross_entropy(
        probabilities.squeeze(),
        torch.randint(0, 2, (batch_size,)).float()
    )
    loss.backward()

    # Check gradients exist
    fusion_grad_exists = any(p.grad is not None for p in fusion.parameters())
    pred_grad_exists = any(p.grad is not None for p in pred_head.parameters())

    print(f"   Fusion gradient exists: {fusion_grad_exists}")
    print(f"   Prediction head gradient exists: {pred_grad_exists}")
    print(f"   Loss value: {loss.item():.4f}")

    assert fusion_grad_exists and pred_grad_exists, "Gradients must flow through all components"
    print("   ✓ Gradient flow verified")

    print("\n" + "="*60)
    print("✅ All component tests passed!")
    print("="*60)
