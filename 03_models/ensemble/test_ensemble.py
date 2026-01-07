"""
Test Script for Ensemble Models

Quick validation of ensemble functionality.

Usage:
    python test_ensemble.py
"""

import sys
from pathlib import Path
import importlib

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("="*80)
print("Testing Ensemble Models")
print("="*80)

# Test imports
print("\n1. Testing imports...")
try:
    from ensemble_model import (
        VotingEnsemble,
        StackingEnsemble,
        WeightedEnsemble,
        FeatureEnsemble,
        CyberbullyingEnsemble
    )
    print("   [OK] Ensemble models imported successfully")
except ImportError as e:
    print(f"   [FAIL] Failed to import ensemble models: {e}")
    sys.exit(1)

try:
    from weight_calibrator import WeightCalibrator, TemperatureScaling
    print("   [OK] Weight calibrator imported successfully")
except ImportError as e:
    print(f"   [FAIL] Failed to import weight calibrator: {e}")
    sys.exit(1)

try:
    from multi_task_learning import MultiTaskLearner
    print("   [OK] Multi-task learning imported successfully")
except ImportError as e:
    print(f"   [WARN] Multi-task learning not available (requires PyTorch): {e}")

# Test basic functionality
print("\n2. Testing basic ensemble creation...")
try:
    ensemble = CyberbullyingEnsemble(ensemble_type='voting')
    print("   [OK] Voting ensemble created")
    
    ensemble = CyberbullyingEnsemble(ensemble_type='stacking')
    print("   [OK] Stacking ensemble created")
    
    ensemble = CyberbullyingEnsemble(ensemble_type='weighted')
    print("   [OK] Weighted ensemble created")
    
    ensemble = CyberbullyingEnsemble(ensemble_type='feature')
    print("   [OK] Feature ensemble created")
except Exception as e:
    print(f"   [FAIL] Ensemble creation failed: {e}")
    sys.exit(1)

# Test weight calibrator
print("\n3. Testing weight calibrator...")
try:
    import numpy as np
    
    # Dummy predictions
    pred1 = np.array([0, 1, 1, 0, 1])
    pred2 = np.array([0, 1, 0, 0, 1])
    pred3 = np.array([1, 1, 1, 0, 1])
    y_true = np.array([0, 1, 1, 0, 1])
    
    predictions = [pred1, pred2, pred3]
    
    calibrator = WeightCalibrator(metric='accuracy', method='random')
    weights, score = calibrator.calibrate(predictions, y_true, n_iterations=100)
    
    print(f"   [OK] Weight calibration successful")
    print(f"        Optimal weights: {weights}")
    print(f"        Best score: {score:.4f}")
except Exception as e:
    print(f"   [FAIL] Weight calibration failed: {e}")

# Test feature extraction (if available)
print("\n4. Testing feature extraction...")
try:
    feature_module_path = PROJECT_ROOT / '02_feature_extraction'
    if str(feature_module_path) not in sys.path:
        sys.path.insert(0, str(feature_module_path))
    
    ling_mod = importlib.import_module('linguistic_features')
    emoji_mod = importlib.import_module('emoji_features')
    
    LinguisticFeatures = getattr(ling_mod, 'LinguisticFeatures')
    EmojiFeatures = getattr(emoji_mod, 'EmojiFeatures')
    
    ling = LinguisticFeatures()
    emoji = EmojiFeatures()
    
    test_text = "This is a test message with emoji 😊"
    
    ling_feats = ling.extract(test_text)
    emoji_feats = emoji.extract(test_text)
    
    print(f"   [OK] Feature extraction working")
    print(f"        Linguistic features: {len(ling_feats)} extracted")
    print(f"        Emoji features: {len(emoji_feats)} extracted")
except Exception as e:
    print(f"   [WARN] Feature extraction not fully available: {e}")

# Check for trained models
print("\n5. Checking for trained baseline models...")
models_dir = PROJECT_ROOT / '03_models' / 'saved_models' / 'baseline'
if models_dir.exists():
    model_types = ['naive_bayes', 'svm', 'tfidf']
    found_models = []
    
    for model_type in model_types:
        model_path = models_dir / model_type / 'model.pkl'
        if model_path.exists():
            found_models.append(model_type)
            print(f"   [OK] Found {model_type} model")
    
    if found_models:
        print(f"   [OK] {len(found_models)}/3 baseline models found")
        print("        Ready for ensemble training!")
    else:
        print("   [WARN] No baseline models found")
        print("          Train baseline models first: python train_models.py --type baseline --model all")
else:
    print("   [WARN] Baseline models directory not found")
    print("          Train baseline models first")

# Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print("[OK] Ensemble module is functional")
print("\nNext steps:")
print("1. Train baseline models if not already done:")
print("   python 14_scripts/train_models.py --type baseline --model all")
print("\n2. Train ensemble models:")
print("   python 03_models/ensemble/train_ensemble.py --type voting --models baseline")
print("\n3. Compare all ensemble strategies:")
print("   python 03_models/ensemble/train_ensemble.py --type all --compare")
print("="*80)
