"""
BST Embedding 학습 디버깅 스크립트
Category embedding이 실제로 학습되고 있는지 검증
"""

import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.baseline.bst_fixed import BST
from utils.taobao_dataset import get_taobao_dataloader
import pickle

def load_trained_model():
    """학습된 모델 로드"""
    # 메타데이터 로드
    with open('data/processed/taobao/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    # 모델 초기화
    model = BST(
        item_vocab_size=metadata['vocab_sizes']['item_id'],
        category_vocab_size=metadata['vocab_sizes']['category_id'],
        other_feature_dims={
            'user_id': metadata['vocab_sizes']['user_id'],
            'hour': metadata['vocab_sizes']['hour'],
            'dayofweek': metadata['vocab_sizes']['dayofweek']
        },
        embed_dim=64,
        max_seq_len=metadata['max_seq_len'],
        num_heads=2,
        d_ff=256,
        dropout=0.1,
        dnn_hidden_units=[256, 128]
    )

    # 학습된 모델 로드 (있다면)
    checkpoint_path = Path('results/checkpoints/bst_best.pth')
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        print(f"✅ Loaded trained model from {checkpoint_path}")
        return model, metadata, True
    else:
        print("⚠️  No trained model found, using randomly initialized model")
        return model, metadata, False

def analyze_embeddings(model, is_trained):
    """Embedding 분석"""
    print("\n" + "="*60)
    print("EMBEDDING ANALYSIS")
    print("="*60)

    # Item embedding
    item_emb = model.item_embedding.weight.data
    print(f"\n📊 Item Embedding:")
    print(f"   Shape: {item_emb.shape}")
    print(f"   Mean: {item_emb.mean().item():.6f}")
    print(f"   Std: {item_emb.std().item():.6f}")
    print(f"   Min: {item_emb.min().item():.6f}")
    print(f"   Max: {item_emb.max().item():.6f}")
    print(f"   L2 Norm (mean): {torch.norm(item_emb, dim=1).mean().item():.6f}")

    # Category embedding
    cat_emb = model.category_embedding.weight.data
    print(f"\n📊 Category Embedding:")
    print(f"   Shape: {cat_emb.shape}")
    print(f"   Mean: {cat_emb.mean().item():.6f}")
    print(f"   Std: {cat_emb.std().item():.6f}")
    print(f"   Min: {cat_emb.min().item():.6f}")
    print(f"   Max: {cat_emb.max().item():.6f}")
    print(f"   L2 Norm (mean): {torch.norm(cat_emb, dim=1).mean().item():.6f}")

    # Embedding 비교
    print(f"\n🔍 Item vs Category Embedding:")
    item_norm = torch.norm(item_emb).item()
    cat_norm = torch.norm(cat_emb).item()
    print(f"   Item total norm: {item_norm:.4f}")
    print(f"   Category total norm: {cat_norm:.4f}")
    print(f"   Magnitude ratio (item/category): {item_norm / cat_norm:.4f}")

    if is_trained:
        print(f"\n   💡 Interpretation:")
        if cat_norm < item_norm * 0.1:
            print(f"   ⚠️  Category embedding magnitude is very small!")
            print(f"   → May not be contributing much to the model")
        elif cat_norm > item_norm * 0.9:
            print(f"   ✅ Category and item embeddings have similar magnitudes")
        else:
            print(f"   ℹ️  Category embedding is smaller but present")

def check_gradient_flow(model, metadata):
    """Gradient flow 확인"""
    print("\n" + "="*60)
    print("GRADIENT FLOW CHECK")
    print("="*60)

    model.train()
    criterion = torch.nn.BCELoss()

    # 데이터 로드
    val_loader, _ = get_taobao_dataloader(
        data_path='data/processed/taobao/val.parquet',
        metadata_path='data/processed/taobao/metadata.pkl',
        batch_size=256,
        shuffle=False
    )

    # 한 배치로 forward + backward
    batch = next(iter(val_loader))
    target_items, target_categories, item_histories, category_histories, other_features, labels = batch

    # Forward
    outputs = model(target_items, target_categories, item_histories, category_histories, other_features)
    loss = criterion(outputs, labels)

    # Backward
    loss.backward()

    # Gradient 확인
    print(f"\n🔍 Gradients:")

    item_grad = model.item_embedding.weight.grad
    if item_grad is not None:
        print(f"\n   Item Embedding Grad:")
        print(f"      Mean: {item_grad.mean().item():.8f}")
        print(f"      Std: {item_grad.std().item():.8f}")
        print(f"      Max abs: {item_grad.abs().max().item():.8f}")
        print(f"      Non-zero: {(item_grad != 0).sum().item()} / {item_grad.numel()}")
    else:
        print(f"   Item Embedding Grad: None ❌")

    cat_grad = model.category_embedding.weight.grad
    if cat_grad is not None:
        print(f"\n   Category Embedding Grad:")
        print(f"      Mean: {cat_grad.mean().item():.8f}")
        print(f"      Std: {cat_grad.std().item():.8f}")
        print(f"      Max abs: {cat_grad.abs().max().item():.8f}")
        print(f"      Non-zero: {(cat_grad != 0).sum().item()} / {cat_grad.numel()}")
    else:
        print(f"   Category Embedding Grad: None ❌")

    if item_grad is not None and cat_grad is not None:
        item_grad_norm = item_grad.abs().mean().item()
        cat_grad_norm = cat_grad.abs().mean().item()
        print(f"\n   Gradient ratio (item/category): {item_grad_norm / cat_grad_norm:.4f}")
        
        if cat_grad_norm < item_grad_norm * 0.01:
            print(f"   ⚠️  Category gradient is very small - may not be learning effectively!")
        else:
            print(f"   ✅ Both embeddings receiving gradients")

def test_embedding_fusion(model, metadata):
    """Embedding fusion 테스트"""
    print("\n" + "="*60)
    print("EMBEDDING FUSION TEST")
    print("="*60)

    model.eval()

    # 데이터 로드
    val_loader, _ = get_taobao_dataloader(
        data_path='data/processed/taobao/val.parquet',
        metadata_path='data/processed/taobao/metadata.pkl',
        batch_size=256,
        shuffle=False
    )

    batch = next(iter(val_loader))
    target_items, target_categories, item_histories, category_histories, other_features, labels = batch

    with torch.no_grad():
        # Item embedding만
        item_emb = model.item_embedding(torch.cat([target_items.unsqueeze(1), item_histories], dim=1))

        # Category embedding만
        cat_emb = model.category_embedding(torch.cat([target_categories.unsqueeze(1), category_histories], dim=1))

        # Fusion (element-wise sum)
        fused_emb = item_emb + cat_emb

        print(f"\n📊 Sample from batch (first sequence):")
        print(f"   Item embedding norm: {torch.norm(item_emb[0], dim=1).mean().item():.6f}")
        print(f"   Category embedding norm: {torch.norm(cat_emb[0], dim=1).mean().item():.6f}")
        print(f"   Fused embedding norm: {torch.norm(fused_emb[0], dim=1).mean().item():.6f}")

        # Category의 기여도 확인
        item_contribution = torch.norm(item_emb) / torch.norm(fused_emb)
        cat_contribution = torch.norm(cat_emb) / torch.norm(fused_emb)

        print(f"\n🔍 Contribution to fused embedding:")
        print(f"   Item contribution: {item_contribution.item():.4f} ({item_contribution.item()*100:.2f}%)")
        print(f"   Category contribution: {cat_contribution.item():.4f} ({cat_contribution.item()*100:.2f}%)")
        
        if cat_contribution.item() < 0.1:
            print(f"\n   ⚠️  Category contributes very little to the fused embedding!")
            print(f"   → This may explain low performance")
        else:
            print(f"\n   ✅ Category is contributing to the model")

def main():
    print("\n" + "="*60)
    print("BST EMBEDDING DEBUGGING")
    print("="*60)

    # 모델과 데이터 로드
    model, metadata, is_trained = load_trained_model()

    # 1. Embedding 분석
    analyze_embeddings(model, is_trained)

    # 2. Gradient flow 확인
    check_gradient_flow(model, metadata)

    # 3. Embedding fusion 테스트
    test_embedding_fusion(model, metadata)

    print("\n" + "="*60)
    print("DEBUGGING COMPLETE")
    print("="*60)

if __name__ == '__main__':
    main()
