"""
Train Ensemble Models for Cyberbullying Detection

Combines baseline and/or context-aware models using various ensemble strategies.

Usage:
    python train_ensemble.py --type voting --models baseline
    python train_ensemble.py --type stacking --models all
    python train_ensemble.py --type weighted --models baseline --tune

Author: Cyberbullying Detection Project Team
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ensemble_model import (
    CyberbullyingEnsemble,
    create_ensemble_from_baseline
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(data_dir: str = None) -> Dict[str, tuple]:
    """Load train, validation, and test data."""
    if data_dir is None:
        data_dir = PROJECT_ROOT / '00_data' / 'processed'
    else:
        data_dir = Path(data_dir)
    
    data = {}
    
    # Load training data
    train_path = data_dir / 'train_data.csv'
    if train_path.exists():
        train_df = pd.read_csv(train_path)
        data['train'] = (
            train_df['message'].fillna('').tolist(),
            train_df['label'].fillna('neutral').str.lower().tolist()
        )
        logger.info(f"Loaded {len(data['train'][0])} training samples")
    
    # Load validation data
    val_path = data_dir / 'val_data.csv'
    if val_path.exists():
        val_df = pd.read_csv(val_path)
        data['val'] = (
            val_df['message'].fillna('').tolist(),
            val_df['label'].fillna('neutral').str.lower().tolist()
        )
        logger.info(f"Loaded {len(data['val'][0])} validation samples")
    
    # Load test data
    test_path = data_dir / 'test_data.csv'
    if test_path.exists():
        test_df = pd.read_csv(test_path)
        data['test'] = (
            test_df['message'].fillna('').tolist(),
            test_df['label'].fillna('neutral').str.lower().tolist()
        )
        logger.info(f"Loaded {len(data['test'][0])} test samples")
    
    return data


def load_baseline_models(models_dir: Path) -> List:
    """Load trained baseline models."""
    import pickle
    
    models = []
    model_types = ['naive_bayes', 'svm', 'tfidf']
    
    for model_type in model_types:
        model_path = models_dir / model_type / 'model.pkl'
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                models.append(model)
                logger.info(f"Loaded {model_type} model")
            except Exception as e:
                logger.warning(f"Failed to load {model_type}: {e}")
    
    return models


def train_voting_ensemble(args, data: Dict) -> CyberbullyingEnsemble:
    """Train voting ensemble."""
    logger.info("="*80)
    logger.info("Training Voting Ensemble")
    logger.info("="*80)
    
    config = {
        'voting': 'soft',  # or 'hard'
    }
    
    ensemble = CyberbullyingEnsemble(ensemble_type='voting', config=config)
    
    # Load models
    if args.models in ['baseline', 'all']:
        models_dir = PROJECT_ROOT / '03_models' / 'saved_models' / 'baseline'
        models = load_baseline_models(models_dir)
        for model in models:
            ensemble.add_model(model)
    
    # Build ensemble
    ensemble.build(X_val=data['val'][0], y_val=data['val'][1])
    
    # Evaluate
    results = ensemble.evaluate(data['test'][0], data['test'][1])
    
    # Save
    save_dir = PROJECT_ROOT / '03_models' / 'saved_models' / 'ensemble' / 'voting'
    ensemble.save(str(save_dir))
    
    return ensemble


def train_stacking_ensemble(args, data: Dict) -> CyberbullyingEnsemble:
    """Train stacking ensemble."""
    logger.info("="*80)
    logger.info("Training Stacking Ensemble")
    logger.info("="*80)
    
    config = {
        'use_features': True,  # Use original features in meta-model
    }
    
    ensemble = CyberbullyingEnsemble(ensemble_type='stacking', config=config)
    
    # Load models
    if args.models in ['baseline', 'all']:
        models_dir = PROJECT_ROOT / '03_models' / 'saved_models' / 'baseline'
        models = load_baseline_models(models_dir)
        for model in models:
            ensemble.add_model(model)
    
    # Build ensemble (requires fitting meta-model)
    ensemble.build(X_val=data['val'][0], y_val=data['val'][1])
    
    # Evaluate
    results = ensemble.evaluate(data['test'][0], data['test'][1])
    
    # Save
    save_dir = PROJECT_ROOT / '03_models' / 'saved_models' / 'ensemble' / 'stacking'
    ensemble.save(str(save_dir))
    
    return ensemble


def train_weighted_ensemble(args, data: Dict) -> CyberbullyingEnsemble:
    """Train weighted ensemble with optimized weights."""
    logger.info("="*80)
    logger.info("Training Weighted Ensemble")
    logger.info("="*80)
    
    ensemble = CyberbullyingEnsemble(ensemble_type='weighted')
    
    # Load models
    if args.models in ['baseline', 'all']:
        models_dir = PROJECT_ROOT / '03_models' / 'saved_models' / 'baseline'
        models = load_baseline_models(models_dir)
        for model in models:
            ensemble.add_model(model)
    
    # Build ensemble (optimizes weights on validation data)
    logger.info("Optimizing model weights...")
    ensemble.build(X_val=data['val'][0], y_val=data['val'][1])
    
    # Evaluate
    results = ensemble.evaluate(data['test'][0], data['test'][1])
    
    # Save
    save_dir = PROJECT_ROOT / '03_models' / 'saved_models' / 'ensemble' / 'weighted'
    ensemble.save(str(save_dir))
    
    return ensemble


def train_feature_ensemble(args, data: Dict) -> CyberbullyingEnsemble:
    """Train feature-level ensemble."""
    logger.info("="*80)
    logger.info("Training Feature-Level Ensemble")
    logger.info("="*80)
    
    config = {
        'feature_types': ['linguistic', 'emoji', 'contextual']
    }
    
    ensemble = CyberbullyingEnsemble(ensemble_type='feature', config=config)
    
    # Build ensemble (trains classifier on combined features)
    logger.info("Extracting and combining features...")
    ensemble.build(X_val=data['train'][0], y_val=data['train'][1])
    
    # Evaluate
    results = ensemble.evaluate(data['test'][0], data['test'][1])
    
    # Save
    save_dir = PROJECT_ROOT / '03_models' / 'saved_models' / 'ensemble' / 'feature'
    ensemble.save(str(save_dir))
    
    return ensemble


def compare_ensembles(ensembles: Dict[str, CyberbullyingEnsemble], data: Dict):
    """Compare different ensemble strategies."""
    logger.info("\n" + "="*80)
    logger.info("Ensemble Comparison")
    logger.info("="*80)
    
    results = {}
    for name, ensemble in ensembles.items():
        results[name] = ensemble.evaluate(data['test'][0], data['test'][1])
    
    # Print comparison table
    logger.info("\nComparison Table:")
    logger.info("-"*80)
    logger.info(f"{'Ensemble':<20} {'Accuracy':<12} {'F1 (Macro)':<12} {'F1 (Weighted)':<12}")
    logger.info("-"*80)
    
    for name, metrics in results.items():
        logger.info(f"{name:<20} {metrics['accuracy']:<12.4f} {metrics['f1_macro']:<12.4f} {metrics['f1_weighted']:<12.4f}")
    
    logger.info("-"*80)
    
    # Find best ensemble
    best_ensemble = max(results.items(), key=lambda x: x[1]['f1_weighted'])
    logger.info(f"\nBest Ensemble: {best_ensemble[0]}")
    logger.info(f"F1 (Weighted): {best_ensemble[1]['f1_weighted']:.4f}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train ensemble models for cyberbullying detection',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--type',
        choices=['voting', 'stacking', 'weighted', 'feature', 'all'],
        default='voting',
        help='Type of ensemble to train'
    )
    
    parser.add_argument(
        '--models',
        choices=['baseline', 'context', 'all'],
        default='baseline',
        help='Which models to include in ensemble'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare all ensemble strategies'
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info("Loading data...")
    data = load_data()
    
    if not data:
        logger.error("No data found. Please check data directory.")
        return
    
    # Train ensemble(s)
    ensembles = {}
    
    if args.type == 'all' or args.compare:
        # Train all ensemble types
        ensembles['voting'] = train_voting_ensemble(args, data)
        ensembles['stacking'] = train_stacking_ensemble(args, data)
        ensembles['weighted'] = train_weighted_ensemble(args, data)
        ensembles['feature'] = train_feature_ensemble(args, data)
        
        # Compare
        compare_ensembles(ensembles, data)
    
    else:
        # Train specific ensemble type
        if args.type == 'voting':
            ensembles['voting'] = train_voting_ensemble(args, data)
        elif args.type == 'stacking':
            ensembles['stacking'] = train_stacking_ensemble(args, data)
        elif args.type == 'weighted':
            ensembles['weighted'] = train_weighted_ensemble(args, data)
        elif args.type == 'feature':
            ensembles['feature'] = train_feature_ensemble(args, data)
    
    logger.info("\n[SUCCESS] Ensemble training complete!")


if __name__ == "__main__":
    main()
