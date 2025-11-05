#!/bin/bash
# Quick test script for MDAF implementation
# This script runs all validation tests to ensure MDAF models are working correctly

echo "============================================================"
echo "MDAF Quick Validation Suite"
echo "============================================================"

# Set working directory
cd /Users/jinhochoi/Desktop/dev/Research

# Activate virtual environment
source venv/bin/activate

echo ""
echo "1. Testing GatedFusion and PredictionHead components..."
python -m models.mdaf.mdaf_components
if [ $? -ne 0 ]; then
    echo "❌ Component test failed!"
    exit 1
fi

echo ""
echo "2. Testing MDAF-Mamba model..."
python -m models.mdaf.mdaf_mamba
if [ $? -ne 0 ]; then
    echo "❌ MDAF-Mamba test failed!"
    exit 1
fi

echo ""
echo "3. Testing MDAF-BST model..."
python -m models.mdaf.mdaf_bst
if [ $? -ne 0 ]; then
    echo "❌ MDAF-BST test failed!"
    exit 1
fi

echo ""
echo "4. Running integration test with real Taobao data..."
python tests/test_mdaf_integration.py
if [ $? -ne 0 ]; then
    echo "❌ Integration test failed!"
    exit 1
fi

echo ""
echo "============================================================"
echo "✅ All MDAF validation tests passed!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Train MDAF-Mamba: ./venv/bin/python experiments/train_mdaf_taobao.py --model mamba"
echo "  2. Train MDAF-BST:   ./venv/bin/python experiments/train_mdaf_taobao.py --model bst"
echo ""
