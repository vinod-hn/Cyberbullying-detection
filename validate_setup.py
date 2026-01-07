#!/usr/bin/env python3
"""
Validation Script for Cyberbullying Detection Project Setup

This script validates that:
1. All required dependencies are importable
2. Preprocessing modules work correctly
3. Basic functionality is operational

Run this after installing dependencies to ensure everything is set up correctly.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Constants
NOTEBOOK_FILENAME = "Cyberbullying_Detection_Colab_Training.ipynb"


def setup_project_path(subdir=None):
    """Setup Python path for project imports"""
    if subdir:
        sys.path.insert(0, str(PROJECT_ROOT / subdir))
    return PROJECT_ROOT

def test_imports():
    """Test that all critical imports work"""
    print("=" * 60)
    print("Testing Critical Imports")
    print("=" * 60)
    
    tests = [
        ("numpy", "import numpy as np"),
        ("pandas", "import pandas as pd"),
        ("sklearn", "import sklearn"),
        ("torch", "import torch"),
    ]
    
    optional_tests = [
        ("transformers", "import transformers"),
        ("accelerate", "import accelerate"),
        ("emoji", "import emoji"),
        ("scipy", "import scipy"),
        ("matplotlib", "import matplotlib"),
        ("seaborn", "import seaborn"),
    ]
    
    passed = 0
    failed = 0
    
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"✅ {name:20s} - OK")
            passed += 1
        except ImportError as e:
            print(f"❌ {name:20s} - FAILED: {e}")
            failed += 1
    
    print("\nOptional dependencies:")
    for name, import_stmt in optional_tests:
        try:
            exec(import_stmt)
            print(f"✅ {name:20s} - OK")
        except ImportError:
            print(f"⚠️  {name:20s} - Not installed (optional)")
    
    print(f"\nCore dependencies: {passed}/{len(tests)} passed")
    return failed == 0

def test_preprocessing_modules():
    """Test preprocessing modules"""
    print("\n" + "=" * 60)
    print("Testing Preprocessing Modules")
    print("=" * 60)
    
    setup_project_path("01_preprocessing")
    
    tests = [
        ("TextNormalizer", "from text_normalizer import TextNormalizer"),
        ("EmojiHandler", "from emoji_handler import EmojiHandler"),
        ("CodeMixProcessor", "from code_mix_processor import CodeMixProcessor"),
        ("Transliterator", "from transliterator import Transliterator"),
    ]
    
    passed = 0
    failed = 0
    
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"✅ {name:20s} - OK")
            passed += 1
        except Exception as e:
            print(f"❌ {name:20s} - FAILED: {e}")
            failed += 1
    
    print(f"\nPreprocessing modules: {passed}/{len(tests)} passed")
    return failed == 0

def test_basic_functionality():
    """Test basic preprocessing functionality"""
    print("\n" + "=" * 60)
    print("Testing Basic Functionality")
    print("=" * 60)
    
    setup_project_path("01_preprocessing")
    
    try:
        from text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        test_text = "Hello   World!!!  @mention  #hashtag"
        result = normalizer.normalize(test_text)
        print(f"✅ TextNormalizer.normalize() works")
        print(f"   Input:  '{test_text}'")
        print(f"   Output: '{result}'")
        return True
    except Exception as e:
        print(f"❌ TextNormalizer.normalize() failed: {e}")
        return False

def check_environment():
    """Check Python environment"""
    print("\n" + "=" * 60)
    print("Environment Information")
    print("=" * 60)
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch not installed")
    
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("Transformers not installed")

def main():
    """Run all validation tests"""
    print("\n🚀 Cyberbullying Detection - Setup Validation")
    print("=" * 60)
    
    check_environment()
    
    results = []
    results.append(test_imports())
    results.append(test_preprocessing_modules())
    results.append(test_basic_functionality())
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if all(results):
        print("✅ All tests passed! Setup is ready.")
        print("\n📚 Next steps:")
        print(f"   1. Open {NOTEBOOK_FILENAME} in Colab")
        print("   2. Enable GPU runtime")
        print("   3. Follow the training guide in COLAB_TRAINING.md")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure you ran: pip install -r requirements.txt")
        print("   2. Or use: pip install -e .")
        print("   3. Check COLAB_TRAINING.md for more details")
        return 1

if __name__ == "__main__":
    sys.exit(main())
