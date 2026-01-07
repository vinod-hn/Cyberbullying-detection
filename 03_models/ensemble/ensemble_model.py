"""
Ensemble Model for Cyberbullying Detection

Combines multiple models using different ensemble strategies:
- Voting (soft/hard)
- Stacking with meta-learner
- Weighted averaging
- Feature-level ensemble

Integrates baseline and context-aware models with advanced feature extraction.

Author: Cyberbullying Detection Project Team
"""

import os
import sys
import json
import pickle
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import Counter
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Core dependencies
try:
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    BaseEstimator = object
    ClassifierMixin = object

# Feature extraction modules
# Use importlib to avoid static import resolution warnings when paths are dynamic.
import importlib
FEATURES_AVAILABLE = False
LinguisticFeatures = None
BehavioralFeatures = None
EmojiFeatures = None
ContextualFeatures = None
try:
    feature_module_path = PROJECT_ROOT / '02_feature_extraction'
    if str(feature_module_path) not in sys.path:
        sys.path.insert(0, str(feature_module_path))
    
    _ling_mod = importlib.import_module('linguistic_features')
    _emoji_mod = importlib.import_module('emoji_features')
    
    LinguisticFeatures = getattr(_ling_mod, 'LinguisticFeatures', None)
    EmojiFeatures = getattr(_emoji_mod, 'EmojiFeatures', None)
    
    try:
        _beh_mod = importlib.import_module('behavioral_features')
        BehavioralFeatures = getattr(_beh_mod, 'BehavioralFeatures', None)
    except Exception:
        BehavioralFeatures = None
    
    try:
        _ctx_mod = importlib.import_module('contextual_features')
        ContextualFeatures = getattr(_ctx_mod, 'ContextualFeatures', None)
    except Exception:
        ContextualFeatures = None
    
    FEATURES_AVAILABLE = all([LinguisticFeatures is not None, EmojiFeatures is not None])
except Exception as e:
    FEATURES_AVAILABLE = False

# PyTorch for deep models
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Ensemble Strategies
# =============================================================================

class VotingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Voting ensemble combining multiple models.
    
    Supports:
    - Hard voting (majority vote)
    - Soft voting (average probabilities)
    - Weighted voting (model confidence weighting)
    """
    
    def __init__(self, models: List[Any], weights: Optional[List[float]] = None,
                 voting: str = 'soft', label_encoder: Optional[LabelEncoder] = None):
        """
        Initialize VotingEnsemble.
        
        Args:
            models: List of trained model instances
            weights: Optional weights for each model
            voting: 'hard' or 'soft' voting
            label_encoder: Label encoder for consistent label handling
        """
        self.models = models
        self.weights = weights if weights else [1.0] * len(models)
        self.voting = voting
        self.label_encoder = label_encoder
        self.classes_ = None
        
        # Normalize weights
        total_weight = float(sum(self.weights))
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            # Fallback to uniform weights
            n = max(len(self.weights), 1)
            self.weights = [1.0 / n] * n
        else:
            self.weights = [max(0.0, float(w)) / total_weight for w in self.weights]
    
    def predict_proba(self, X: List[str]) -> np.ndarray:
        """
        Predict class probabilities using voting.
        
        Args:
            X: List of text samples
            
        Returns:
            Array of shape (n_samples, n_classes) with probabilities
        """
        all_probas = []
        
        for i, model in enumerate(self.models):
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                elif hasattr(model, 'decision_function'):
                    # Convert decision function to probabilities
                    scores = model.decision_function(X)
                    if scores.ndim == 1:
                        scores = np.column_stack([-scores, scores])
                    proba = self._softmax(scores)
                else:
                    # Hard predictions converted to one-hot
                    preds = model.predict(X)
                    proba = self._to_proba(preds)
                
                all_probas.append(proba * self.weights[i])
            except Exception as e:
                logger.warning(f"Model {i} prediction failed: {e}")
                continue
        
        if not all_probas:
            raise ValueError("No models produced valid predictions")
        
        # Average weighted probabilities
        avg_proba = np.sum(all_probas, axis=0)
        return avg_proba
    
    def predict(self, X: List[str]) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: List of text samples
            
        Returns:
            Array of predicted labels
        """
        if self.voting == 'hard':
            # Collect predictions from all models
            all_preds = []
            for model in self.models:
                try:
                    preds = model.predict(X)
                    all_preds.append(preds)
                except Exception as e:
                    logger.warning(f"Model prediction failed: {e}")
                    continue
            
            if not all_preds:
                raise ValueError("No models produced valid predictions for hard voting")
            
            # Majority vote
            predictions = []
            for i in range(len(X)):
                votes = [preds[i] for preds in all_preds]
                predictions.append(Counter(votes).most_common(1)[0][0])
            
            return np.array(predictions)
        else:
            # Soft voting - use probabilities
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]
    
    def fit(self, X: List[str], y: np.ndarray):
        """Fit is not needed - models are pre-trained."""
        if self.label_encoder:
            self.classes_ = self.label_encoder.classes_
        else:
            self.classes_ = np.unique(y)
        return self
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax function for converting scores to probabilities."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _to_proba(self, preds: np.ndarray) -> np.ndarray:
        """Convert hard predictions to probability matrix."""
        n_samples = len(preds)
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))
        
        for i, pred in enumerate(preds):
            class_idx = np.where(self.classes_ == pred)[0][0]
            proba[i, class_idx] = 1.0
        
        return proba


class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Stacking ensemble with meta-learner.
    
    Uses predictions from base models as features for a meta-classifier.
    """
    
    def __init__(self, base_models: List[Any], meta_model: Optional[Any] = None,
                 use_features: bool = True, label_encoder: Optional[LabelEncoder] = None):
        """
        Initialize StackingEnsemble.
        
        Args:
            base_models: List of base model instances
            meta_model: Meta-classifier (default: Logistic Regression)
            use_features: Whether to include original features in meta-model
            label_encoder: Label encoder for consistent labels
        """
        self.base_models = base_models
        self.meta_model = meta_model or LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42
        )
        self.use_features = use_features
        self.label_encoder = label_encoder
        self.classes_ = None
        self.feature_extractor = None
    
    def _get_base_predictions(self, X: List[str], predict_proba: bool = True) -> np.ndarray:
        """Get predictions from all base models."""
        base_preds = []
        
        for model in self.base_models:
            try:
                if predict_proba and hasattr(model, 'predict_proba'):
                    preds = model.predict_proba(X)
                else:
                    preds = model.predict(X)
                    if preds.ndim == 1:
                        preds = preds.reshape(-1, 1)
                
                base_preds.append(preds)
            except Exception as e:
                logger.warning(f"Base model prediction failed: {e}")
                continue
        
        if not base_preds:
            raise ValueError("No base models produced valid predictions")
        return np.hstack(base_preds)
    
    def fit(self, X: List[str], y: np.ndarray):
        """
        Fit the stacking ensemble.
        
        Args:
            X: Training texts
            y: Training labels
        """
        # Get base model predictions
        base_preds = self._get_base_predictions(X, predict_proba=True)
        
        # Optionally add original features
        if self.use_features and FEATURES_AVAILABLE:
            try:
                self.feature_extractor = LinguisticFeatures()
                orig_features = []
                for text in X:
                    feats = self.feature_extractor.extract(text)
                    orig_features.append(list(feats.values()))
                orig_features = np.array(orig_features)
                
                meta_features = np.hstack([base_preds, orig_features])
            except Exception as e:
                logger.warning(f"Feature extraction failed: {e}")
                meta_features = base_preds
        else:
            meta_features = base_preds
        
        # Fit meta-model
        self.meta_model.fit(meta_features, y)
        
        if self.label_encoder:
            self.classes_ = self.label_encoder.classes_
        else:
            self.classes_ = np.unique(y)
        
        return self
    
    def predict(self, X: List[str]) -> np.ndarray:
        """Predict class labels."""
        base_preds = self._get_base_predictions(X, predict_proba=True)
        
        if self.use_features and self.feature_extractor:
            try:
                orig_features = []
                for text in X:
                    feats = self.feature_extractor.extract(text)
                    orig_features.append(list(feats.values()))
                orig_features = np.array(orig_features)
                
                meta_features = np.hstack([base_preds, orig_features])
            except:
                meta_features = base_preds
        else:
            meta_features = base_preds
        
        return self.meta_model.predict(meta_features)
    
    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Predict class probabilities."""
        base_preds = self._get_base_predictions(X, predict_proba=True)
        
        if self.use_features and self.feature_extractor:
            try:
                orig_features = []
                for text in X:
                    feats = self.feature_extractor.extract(text)
                    orig_features.append(list(feats.values()))
                orig_features = np.array(orig_features)
                
                meta_features = np.hstack([base_preds, orig_features])
            except:
                meta_features = base_preds
        else:
            meta_features = base_preds
        
        if hasattr(self.meta_model, 'predict_proba'):
            return self.meta_model.predict_proba(meta_features)
        else:
            # Convert predictions to probabilities
            preds = self.meta_model.predict(meta_features)
            n_samples = len(preds)
            n_classes = len(self.classes_)
            proba = np.zeros((n_samples, n_classes))
            for i, pred in enumerate(preds):
                class_idx = np.where(self.classes_ == pred)[0][0]
                proba[i, class_idx] = 1.0
            return proba


class WeightedEnsemble(BaseEstimator, ClassifierMixin):
    """
    Weighted ensemble with optimized weights.
    
    Learns optimal weights for each model based on validation performance.
    """
    
    def __init__(self, models: List[Any], label_encoder: Optional[LabelEncoder] = None):
        """
        Initialize WeightedEnsemble.
        
        Args:
            models: List of trained models
            label_encoder: Label encoder
        """
        self.models = models
        self.label_encoder = label_encoder
        self.weights = None
        self.classes_ = None
    
    def fit(self, X: List[str], y: np.ndarray):
        """
        Learn optimal weights based on validation performance.
        
        Args:
            X: Validation texts
            y: Validation labels
        """
        # Get predictions from each model
        all_probas = []
        for model in self.models:
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    all_probas.append(proba)
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")
                all_probas.append(None)
        
        # Optimize weights using grid search
        best_weights = None
        best_score = 0.0
        
        # Try different weight combinations
        from itertools import product
        weight_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        # Determine a valid probability shape
        valid_shape = None
        for p in all_probas:
            if p is not None:
                valid_shape = p.shape
                break
        if valid_shape is None:
            raise ValueError("No models produced probability outputs for weighting")
        
        classes = self.label_encoder.classes_ if (self.label_encoder is not None and hasattr(self.label_encoder, 'classes_')) else np.unique(y)
        
        for weights in product(weight_options, repeat=len(self.models)):
            total_w = float(sum(weights))
            if not np.isfinite(total_w) or total_w <= 0.0:
                continue
            
            # Normalize weights safely
            norm_weights = np.array([max(0.0, float(w)) for w in weights], dtype=float) / total_w
            
            # Compute weighted average using first valid shape
            weighted_proba = np.zeros(valid_shape, dtype=float)
            for i, proba in enumerate(all_probas):
                if proba is not None:
                    weighted_proba += proba * norm_weights[i]
            
            # Get predictions and score
            pred_indices = np.argmax(weighted_proba, axis=1)
            preds = np.array([classes[idx] for idx in pred_indices])
            
            score = accuracy_score(y, preds)
            
            if score > best_score:
                best_score = score
                best_weights = norm_weights
        
        self.weights = best_weights if best_weights is not None else np.ones(len(self.models)) / len(self.models)
        
        if self.label_encoder:
            self.classes_ = self.label_encoder.classes_
        else:
            self.classes_ = np.unique(y)
        
        logger.info(f"Optimized weights: {self.weights}")
        logger.info(f"Best validation score: {best_score:.4f}")
        
        return self
    
    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Predict class probabilities."""
        if self.weights is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        weighted_proba = None
        
        for i, model in enumerate(self.models):
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    
                    if weighted_proba is None:
                        weighted_proba = proba * self.weights[i]
                    else:
                        weighted_proba += proba * self.weights[i]
            except Exception as e:
                logger.warning(f"Model {i} prediction failed: {e}")
                continue
        
        if weighted_proba is None:
            raise ValueError("No models produced probability outputs at prediction time")
        return weighted_proba
    
    def predict(self, X: List[str]) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# =============================================================================
# Feature-Level Ensemble
# =============================================================================

class FeatureEnsemble(BaseEstimator, ClassifierMixin):
    """
    Ensemble combining multiple feature extractors with a single classifier.
    
    Extracts linguistic, behavioral, emoji, and contextual features.
    """
    
    def __init__(self, classifier: Optional[Any] = None, 
                 feature_types: Optional[List[str]] = None):
        """
        Initialize FeatureEnsemble.
        
        Args:
            classifier: Classifier to use (default: Random Forest)
            feature_types: List of feature types to use
        """
        self.classifier = classifier or RandomForestClassifier(
            n_estimators=100, class_weight='balanced', random_state=42
        )
        self.feature_types = feature_types or ['linguistic', 'emoji', 'contextual']
        self.feature_extractors = {}
        self.classes_ = None
        self._feature_schema: List[Tuple[str, List[str]]] = []  # (feat_type, ordered_keys)
        
        # Initialize feature extractors
        if FEATURES_AVAILABLE:
            if 'linguistic' in self.feature_types:
                self.feature_extractors['linguistic'] = LinguisticFeatures()
            if 'behavioral' in self.feature_types:
                self.feature_extractors['behavioral'] = BehavioralFeatures()
            if 'emoji' in self.feature_types:
                self.feature_extractors['emoji'] = EmojiFeatures()
            if 'contextual' in self.feature_types:
                self.feature_extractors['contextual'] = ContextualFeatures()
    
    def _build_feature_schema(self, texts: List[str]):
        """Build a stable schema of feature keys across texts for consistent vectors."""
        schema: List[Tuple[str, List[str]]] = []
        for feat_type, extractor in self.feature_extractors.items():
            try:
                key_set = set()
                for text in texts:
                    feats = extractor.extract(text)
                    key_set.update(feats.keys())
                ordered_keys = sorted(list(key_set))
                schema.append((feat_type, ordered_keys))
            except Exception as e:
                logger.warning(f"{feat_type} schema build failed: {e}")
                continue
        self._feature_schema = schema
    
    def _extract_features(self, texts: List[str]) -> np.ndarray:
        """Extract all features from texts using the built schema."""
        if not self._feature_schema:
            # Build schema lazily if missing
            self._build_feature_schema(texts)
        all_features = []
        for text in texts:
            text_features = []
            for feat_type, keys in self._feature_schema:
                extractor = self.feature_extractors.get(feat_type)
                if extractor is None:
                    # Fill zeros if extractor not available
                    text_features.extend([0.0] * len(keys))
                    continue
                try:
                    feats = extractor.extract(text)
                    # Follow stable key order and fill missing with 0.0
                    text_features.extend([float(feats.get(k, 0.0)) for k in keys])
                except Exception as e:
                    logger.warning(f"{feat_type} extraction failed: {e}")
                    text_features.extend([0.0] * len(keys))
                    continue
            all_features.append(text_features)
        return np.array(all_features, dtype=float)
    
    def fit(self, X: List[str], y: np.ndarray):
        """Fit the feature ensemble."""
        if not self.feature_extractors:
            raise ValueError("No feature extractors available")
        
        # Extract features
        # Build schema on the training data for consistency
        self._build_feature_schema(X)
        features = self._extract_features(X)
        
        # Fit classifier
        self.classifier.fit(features, y)
        self.classes_ = np.unique(y)
        
        return self
    
    def predict(self, X: List[str]) -> np.ndarray:
        """Predict class labels."""
        features = self._extract_features(X)
        return self.classifier.predict(features)
    
    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Predict class probabilities."""
        features = self._extract_features(X)
        if hasattr(self.classifier, 'predict_proba'):
            return self.classifier.predict_proba(features)
        else:
            preds = self.classifier.predict(features)
            n_samples = len(preds)
            n_classes = len(self.classes_)
            proba = np.zeros((n_samples, n_classes))
            for i, pred in enumerate(preds):
                class_idx = np.where(self.classes_ == pred)[0][0]
                proba[i, class_idx] = 1.0
            return proba


# =============================================================================
# Main Ensemble Model
# =============================================================================

class CyberbullyingEnsemble:
    """
    Main ensemble model for cyberbullying detection.
    
    Combines multiple strategies and models for robust predictions.
    """
    
    def __init__(self, ensemble_type: str = 'voting', config: Optional[Dict] = None):
        """
        Initialize ensemble.
        
        Args:
            ensemble_type: Type of ensemble ('voting', 'stacking', 'weighted', 'feature')
            config: Configuration dictionary
        """
        self.ensemble_type = ensemble_type
        self.config = config or {}
        self.models = []
        self.ensemble = None
        self.label_encoder = LabelEncoder()
        
        logger.info(f"Initialized {ensemble_type} ensemble")
    
    def load_models(self, model_paths: List[str]):
        """
        Load pre-trained models from paths.
        
        Args:
            model_paths: List of paths to saved models
        """
        for path in model_paths:
            try:
                model_path = Path(path)
                if model_path.exists():
                    # Try loading as pickle
                    with open(model_path / 'model.pkl', 'rb') as f:
                        model = pickle.load(f)
                    self.models.append(model)
                    logger.info(f"Loaded model from {path}")
            except Exception as e:
                logger.warning(f"Failed to load model from {path}: {e}")
        
        logger.info(f"Loaded {len(self.models)} models")
    
    def add_model(self, model: Any):
        """Add a trained model to the ensemble."""
        self.models.append(model)
    
    def build(self, X_val: Optional[List[str]] = None, y_val: Optional[np.ndarray] = None):
        """
        Build the ensemble from loaded models.
        
        Args:
            X_val: Validation texts for weight optimization
            y_val: Validation labels
        """
        if not self.models:
            raise ValueError("No models loaded. Use load_models() or add_model() first.")
        
        if self.ensemble_type == 'voting':
            self.ensemble = VotingEnsemble(
                models=self.models,
                voting=self.config.get('voting', 'soft'),
                label_encoder=self.label_encoder
            )
            if X_val is not None and y_val is not None:
                self.ensemble.fit(X_val, y_val)
        
        elif self.ensemble_type == 'stacking':
            self.ensemble = StackingEnsemble(
                base_models=self.models,
                use_features=self.config.get('use_features', True),
                label_encoder=self.label_encoder
            )
            if X_val is not None and y_val is not None:
                self.ensemble.fit(X_val, y_val)
        
        elif self.ensemble_type == 'weighted':
            self.ensemble = WeightedEnsemble(
                models=self.models,
                label_encoder=self.label_encoder
            )
            if X_val is not None and y_val is not None:
                self.ensemble.fit(X_val, y_val)
        
        elif self.ensemble_type == 'feature':
            self.ensemble = FeatureEnsemble(
                feature_types=self.config.get('feature_types', ['linguistic', 'emoji', 'contextual'])
            )
            if X_val is not None and y_val is not None:
                self.ensemble.fit(X_val, y_val)
        
        else:
            raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")
        
        logger.info(f"Built {self.ensemble_type} ensemble")
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict labels for texts."""
        if self.ensemble is None:
            raise ValueError("Ensemble not built. Call build() first.")
        
        return self.ensemble.predict(texts)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict probabilities for texts."""
        if self.ensemble is None:
            raise ValueError("Ensemble not built. Call build() first.")
        
        return self.ensemble.predict_proba(texts)
    
    def evaluate(self, X_test: List[str], y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate ensemble on test data.
        
        Args:
            X_test: Test texts
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X_test)
        
        results = {
            'accuracy': accuracy_score(y_test, predictions),
            'f1_macro': f1_score(y_test, predictions, average='macro'),
            'f1_weighted': f1_score(y_test, predictions, average='weighted')
        }
        
        logger.info(f"Ensemble Results:")
        logger.info(f"  Accuracy: {results['accuracy']:.4f}")
        logger.info(f"  F1 (macro): {results['f1_macro']:.4f}")
        logger.info(f"  F1 (weighted): {results['f1_weighted']:.4f}")
        
        return results
    
    def save(self, save_path: str):
        """Save ensemble model."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save ensemble
        with open(save_path / 'ensemble.pkl', 'wb') as f:
            pickle.dump(self.ensemble, f)
        
        # Save config
        config = {
            'ensemble_type': self.ensemble_type,
            'n_models': len(self.models),
            'config': self.config
        }
        with open(save_path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Ensemble saved to {save_path}")
    
    @classmethod
    def load(cls, load_path: str) -> 'CyberbullyingEnsemble':
        """Load ensemble model."""
        load_path = Path(load_path)
        
        # Load config
        with open(load_path / 'config.json', 'r') as f:
            config = json.load(f)
        
        # Create instance
        instance = cls(
            ensemble_type=config['ensemble_type'],
            config=config.get('config', {})
        )
        
        # Load ensemble
        with open(load_path / 'ensemble.pkl', 'rb') as f:
            instance.ensemble = pickle.load(f)
        
        logger.info(f"Ensemble loaded from {load_path}")
        return instance


# =============================================================================
# Utility Functions
# =============================================================================

def create_ensemble_from_baseline(model_dir: str, ensemble_type: str = 'voting') -> CyberbullyingEnsemble:
    """
    Create ensemble from trained baseline models.
    
    Args:
        model_dir: Directory containing baseline models
        ensemble_type: Type of ensemble to create
        
    Returns:
        CyberbullyingEnsemble instance
    """
    model_dir = Path(model_dir)
    model_paths = [
        model_dir / 'naive_bayes',
        model_dir / 'svm',
        model_dir / 'tfidf'
    ]
    
    ensemble = CyberbullyingEnsemble(ensemble_type=ensemble_type)
    ensemble.load_models([str(p) for p in model_paths if p.exists()])
    
    return ensemble


if __name__ == "__main__":
    print("Ensemble Model Module")
    print("="*60)
    print("Available ensemble strategies:")
    print("  - VotingEnsemble: Combine model predictions via voting")
    print("  - StackingEnsemble: Use predictions as features for meta-learner")
    print("  - WeightedEnsemble: Optimize model weights")
    print("  - FeatureEnsemble: Combine multiple feature types")
    print("\nUsage:")
    print("  from ensemble_model import CyberbullyingEnsemble")
    print("  ensemble = CyberbullyingEnsemble(ensemble_type='voting')")
    print("  ensemble.load_models(model_paths)")
    print("  ensemble.build(X_val, y_val)")
    print("  predictions = ensemble.predict(X_test)")
