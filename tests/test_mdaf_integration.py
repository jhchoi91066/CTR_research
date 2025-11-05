#!/usr/bin/env python3
"""
Integration test for MDAF models with real Taobao data

This script tests:
1. Data loading from Taobao dataset
2. Forward pass through MDAF-Mamba
3. Forward pass through MDAF-BST
4. Backward pass and gradient computation
5. Gate value analysis
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
import pickle

from models.mdaf.mdaf_mamba import MDAF_Mamba
from models.mdaf.mdaf_bst import MDAF_BST
from utils.taobao_dataset import TaobaoDataset, collate_fn


def test_mdaf_integration():
    """Test MDAF models with real Taobao data"""
    print("="*60)
    print("MDAF Integration Test")
    print("="*60)

    # Paths
    data_dir = Path('data/processed/taobao')

    # Load metadata
    print("\n1. Loading metadata...")
    with open(data_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    print(f"   Items: {metadata['vocab_sizes']['item_id']:,}")
    print(f"   Categories: {metadata['vocab_sizes']['category_id']:,}")
    print(f"   Users: {metadata['vocab_sizes']['user_id']:,}")

    # Create dataloader
    print("\n2. Creating dataloader...")
    val_dataset = TaobaoDataset(data_dir / 'val.parquet', data_dir / 'metadata.pkl')
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Get one batch
    print("\n3. Loading one batch...")
    batch = next(iter(val_loader))
    target_items, target_categories, item_histories, category_histories, other_features, labels = batch

    print(f"   Batch size: {target_items.size(0)}")
    print(f"   Sequence length: {item_histories.size(1)}")
    print(f"   Label distribution: {labels.float().mean():.4f}")

    # Device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n4. Device: {device}")

    # Move batch to device
    target_items = target_items.to(device)
    target_categories = target_categories.to(device)
    item_histories = item_histories.to(device)
    category_histories = category_histories.to(device)
    other_features = {k: v.to(device) for k, v in other_features.items()}
    labels = labels.to(device)

    # Test MDAF-Mamba
    print("\n" + "="*60)
    print("Testing MDAF-Mamba")
    print("="*60)

    mdaf_mamba = MDAF_Mamba(
        item_vocab_size=metadata['vocab_sizes']['item_id'],
        category_vocab_size=metadata['vocab_sizes']['category_id'],
        user_vocab_size=metadata['vocab_sizes']['user_id'],
        dcnv3_embed_dim=16,
        item_embed_dim=64,
        category_embed_dim=32,
        mamba_hidden_dim=128,
        embedding_dim=128
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in mdaf_mamba.parameters()):,}")

    # Forward pass
    print("\n1. Forward pass...")
    preds_mamba, gates_mamba = mdaf_mamba(
        target_item=target_items,
        target_category=target_categories,
        item_history=item_histories,
        category_history=category_histories,
        other_features=other_features,
        return_gate=True
    )

    print(f"   Predictions shape: {preds_mamba.shape}")
    print(f"   Predictions range: [{preds_mamba.min():.4f}, {preds_mamba.max():.4f}]")
    print(f"   Gates shape: {gates_mamba.shape}")
    print(f"   Gates mean: {gates_mamba.mean():.4f}")
    print(f"   Gates std: {gates_mamba.std():.4f}")

    # Backward pass
    print("\n2. Backward pass...")
    loss_mamba = torch.nn.functional.binary_cross_entropy(preds_mamba, labels)
    loss_mamba.backward()

    print(f"   Loss: {loss_mamba.item():.4f}")
    print(f"   Gradients exist: {any(p.grad is not None for p in mdaf_mamba.parameters())}")

    # Test MDAF-BST
    print("\n" + "="*60)
    print("Testing MDAF-BST")
    print("="*60)

    mdaf_bst = MDAF_BST(
        item_vocab_size=metadata['vocab_sizes']['item_id'],
        category_vocab_size=metadata['vocab_sizes']['category_id'],
        user_vocab_size=metadata['vocab_sizes']['user_id'],
        dcnv3_embed_dim=16,
        bst_embed_dim=128,
        embedding_dim=128
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in mdaf_bst.parameters()):,}")

    # Forward pass
    print("\n1. Forward pass...")
    preds_bst, gates_bst = mdaf_bst(
        target_item=target_items,
        target_category=target_categories,
        item_history=item_histories,
        category_history=category_histories,
        other_features=other_features,
        return_gate=True
    )

    print(f"   Predictions shape: {preds_bst.shape}")
    print(f"   Predictions range: [{preds_bst.min():.4f}, {preds_bst.max():.4f}]")
    print(f"   Gates shape: {gates_bst.shape}")
    print(f"   Gates mean: {gates_bst.mean():.4f}")
    print(f"   Gates std: {gates_bst.std():.4f}")

    # Backward pass
    print("\n2. Backward pass...")
    loss_bst = torch.nn.functional.binary_cross_entropy(preds_bst, labels)
    loss_bst.backward()

    print(f"   Loss: {loss_bst.item():.4f}")
    print(f"   Gradients exist: {any(p.grad is not None for p in mdaf_bst.parameters())}")

    # Summary
    print("\n" + "="*60)
    print("Integration Test Summary")
    print("="*60)
    print(f"MDAF-Mamba Loss: {loss_mamba.item():.4f}")
    print(f"MDAF-BST Loss: {loss_bst.item():.4f}")
    print(f"\nMDAF-Mamba Gate Statistics:")
    print(f"   Mean: {gates_mamba.mean():.4f}")
    print(f"   Std: {gates_mamba.std():.4f}")
    print(f"   Range: [{gates_mamba.min():.4f}, {gates_mamba.max():.4f}]")
    print(f"\nMDAF-BST Gate Statistics:")
    print(f"   Mean: {gates_bst.mean():.4f}")
    print(f"   Std: {gates_bst.std():.4f}")
    print(f"   Range: [{gates_bst.min():.4f}, {gates_bst.max():.4f}]")

    print("\n" + "="*60)
    print("✅ All integration tests passed!")
    print("="*60)


if __name__ == '__main__':
    test_mdaf_integration()
