# TF-IDF Classifier
"""
TFIDFClassifier: TF-IDF with Logistic Regression for cyberbullying detection.
Uses TF-IDF features with Logistic Regression for multi-class classification.
Optimized for Kannada-English code-mixed text classification.
"""

import os
import sys
import re
import pickle
import logging
import json
from datetime import datetime
from typing import List, Dict, Optional, Any, Union, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# sklearn imports
try:
    from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
    from sklearn.metrics import (
        classification_report, confusion_matrix, accuracy_score,
        f1_score, precision_score, recall_score
    )
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.error("sklearn not available. Please install: pip install scikit-learn")


class TFIDFClassifier:
    """
    TF-IDF with Logistic Regression classifier for cyberbullying detection.
    
    Supports multiple classifiers with TF-IDF features:
    - LogisticRegression: Best for probabilistic classification
    - SGDClassifier: Fast online learning
    - RidgeClassifier: L2-regularized linear classifier
    
    Attributes:
        config: Configuration dictionary
        model: Trained classifier model
        vectorizer: TF-IDF vectorizer
        label_encoder: Label encoder for classes
        is_fitted: Whether model is trained
    """
    
    # Cyberbullying labels based on dataset
    LABELS = ['insult', 'aggression', 'threat', 'harassment', 
              'neutral', 'exclusion', 'stalking', 'hate',
              'toxicity', 'sexual_harassment', 'cyberstalking']
    
    # Severity levels
    SEVERITY_LEVELS = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TFIDFClassifier.
        
        Args:
            config: Configuration dictionary
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required. Install with: pip install scikit-learn")
        
        self.config = config or self._default_config()
        
        # Initialize components
        self.vectorizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
        # Training metadata
        self.training_metadata = {}
        
        # Model paths
        self.models_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'saved_models', 'baseline_tfidf'
        )
        
        # Initialize vectorizer and model
        self._initialize_components()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'classifier_type': 'logistic',  # 'logistic', 'sgd', 'ridge'
            'C': 1.0,  # Regularization parameter
            'max_iter': 1000,
            'class_weight': 'balanced',
            'solver': 'lbfgs',  # For logistic regression
            'random_state': 42,
            
            'vectorizer': {
                'type': 'tfidf',
                'max_features': 15000,
                'ngram_range': (1, 3),  # Up to trigrams
                'min_df': 2,
                'max_df': 0.95,
                'sublinear_tf': True,
                'analyzer': 'word',
                'lowercase': True,
                'strip_accents': 'unicode',
                'use_idf': True,
                'smooth_idf': True,
                'norm': 'l2'
            },
            
            'preprocessing': {
                'remove_urls': True,
                'remove_mentions': True,
                'normalize_whitespace': True
            }
        }
    
    def _initialize_components(self) -> None:
        """Initialize vectorizer and model based on config."""
        vec_config = self.config.get('vectorizer', {})
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=vec_config.get('max_features', 15000),
            ngram_range=tuple(vec_config.get('ngram_range', (1, 3))),
            min_df=vec_config.get('min_df', 2),
            max_df=vec_config.get('max_df', 0.95),
            sublinear_tf=vec_config.get('sublinear_tf', True),
            analyzer=vec_config.get('analyzer', 'word'),
            lowercase=vec_config.get('lowercase', True),
            strip_accents=vec_config.get('strip_accents', 'unicode'),
            use_idf=vec_config.get('use_idf', True),
            smooth_idf=vec_config.get('smooth_idf', True),
            norm=vec_config.get('norm', 'l2')
        )
        
        # Initialize classifier
        classifier_type = self.config.get('classifier_type', 'logistic')
        C = self.config.get('C', 1.0)
        class_weight = self.config.get('class_weight', 'balanced')
        max_iter = self.config.get('max_iter', 1000)
        random_state = self.config.get('random_state', 42)
        
        if classifier_type == 'logistic':
            self.model = LogisticRegression(
                C=C,
                class_weight=class_weight,
                max_iter=max_iter,
                solver=self.config.get('solver', 'lbfgs'),
                random_state=random_state,
                n_jobs=-1
            )
        elif classifier_type == 'sgd':
            self.model = SGDClassifier(
                loss='log_loss',  # For probability estimates
                penalty='l2',
                alpha=1.0 / C,
                class_weight=class_weight,
                max_iter=max_iter,
                random_state=random_state,
                n_jobs=-1
            )
        elif classifier_type == 'ridge':
            self.model = RidgeClassifier(
                alpha=1.0 / C,
                class_weight=class_weight,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        logger.info(f"Initialized TFIDFClassifier with {classifier_type}")
    
    # =========================================================================
    # Preprocessing
    # =========================================================================
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text before vectorization.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        preprocess_config = self.config.get('preprocessing', {})
        
        # Remove URLs
        if preprocess_config.get('remove_urls', True):
            text = re.sub(r'http[s]?://\S+', '', text)
        
        # Remove @mentions but keep the text
        if preprocess_config.get('remove_mentions', True):
            text = re.sub(r'@(\w+)', r'\1', text)
        
        # Remove hashtag symbols but keep text
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Normalize whitespace
        if preprocess_config.get('normalize_whitespace', True):
            text = ' '.join(text.split())
        
        return text.strip()
    
    def _preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess batch of texts."""
        return [self._preprocess_text(t) for t in texts]
    
    # =========================================================================
    # Training
    # =========================================================================
    def fit(self, texts: List[str], labels: List[str], 
            validate: bool = True) -> Dict[str, Any]:
        """
        Train the classifier.
        
        Args:
            texts: List of text messages
            labels: List of labels (insult, aggression, threat, etc.)
            validate: Whether to perform cross-validation
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Training TFIDFClassifier on {len(texts)} samples...")
        
        # Preprocess texts
        processed_texts = self._preprocess_batch(texts)
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Fit vectorizer and transform texts
        X = self.vectorizer.fit_transform(processed_texts)
        y = encoded_labels
        
        logger.info(f"TF-IDF features: {X.shape[1]}")
        
        # Cross-validation
        cv_scores = None
        if validate:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='f1_weighted')
            logger.info(f"Cross-validation F1: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Train final model
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Store training metadata
        self.training_metadata = {
            'trained_at': datetime.now().isoformat(),
            'n_samples': len(texts),
            'n_features': X.shape[1],
            'n_classes': len(self.label_encoder.classes_),
            'classes': list(self.label_encoder.classes_),
            'classifier_type': self.config.get('classifier_type', 'logistic'),
            'cv_f1_mean': float(cv_scores.mean()) if cv_scores is not None else None,
            'cv_f1_std': float(cv_scores.std()) if cv_scores is not None else None
        }
        
        # Training predictions for metrics
        train_preds = self.model.predict(X)
        
        results = {
            'accuracy': float(accuracy_score(y, train_preds)),
            'f1_weighted': float(f1_score(y, train_preds, average='weighted')),
            'f1_macro': float(f1_score(y, train_preds, average='macro')),
            'cv_scores': cv_scores.tolist() if cv_scores is not None else None,
            'metadata': self.training_metadata
        }
        
        logger.info(f"Training accuracy: {results['accuracy']:.4f}")
        logger.info(f"Training F1 (weighted): {results['f1_weighted']:.4f}")
        
        return results
    
    def fit_from_csv(self, csv_path: str, text_column: str = 'message',
                     label_column: str = 'label') -> Dict[str, Any]:
        """
        Train from CSV file.
        
        Args:
            csv_path: Path to CSV file
            text_column: Name of text column
            label_column: Name of label column
            
        Returns:
            Training results
        """
        logger.info(f"Loading training data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        texts = df[text_column].fillna('').tolist()
        labels = df[label_column].fillna('neutral').tolist()
        
        return self.fit(texts, labels)
    
    # =========================================================================
    # Hyperparameter Tuning
    # =========================================================================
    def tune_hyperparameters(self, texts: List[str], labels: List[str],
                             param_grid: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Tune hyperparameters using GridSearchCV.
        
        Args:
            texts: Training texts
            labels: Training labels
            param_grid: Parameter grid for search
            
        Returns:
            Best parameters and scores
        """
        logger.info("Starting hyperparameter tuning...")
        
        processed_texts = self._preprocess_batch(texts)
        encoded_labels = self.label_encoder.fit_transform(labels)
        X = self.vectorizer.fit_transform(processed_texts)
        y = encoded_labels
        
        if param_grid is None:
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'solver': ['lbfgs', 'saga'],
                'max_iter': [500, 1000]
            }
        
        grid_search = GridSearchCV(
            LogisticRegression(class_weight='balanced', random_state=42, n_jobs=-1),
            param_grid,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': float(grid_search.best_score_),
            'cv_results': {
                'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
                'params': [str(p) for p in grid_search.cv_results_['params']]
            }
        }
        
        logger.info(f"Best parameters: {results['best_params']}")
        logger.info(f"Best CV F1: {results['best_score']:.4f}")
        
        return results
    
    # =========================================================================
    # Prediction
    # =========================================================================
    def predict(self, texts: Union[str, List[str]]) -> List[str]:
        """
        Predict labels for texts.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            List of predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if isinstance(texts, str):
            texts = [texts]
        
        processed = self._preprocess_batch(texts)
        X = self.vectorizer.transform(processed)
        predictions = self.model.predict(X)
        
        return self.label_encoder.inverse_transform(predictions).tolist()
    
    def predict_proba(self, texts: Union[str, List[str]]) -> List[Dict[str, float]]:
        """
        Get prediction probabilities.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            List of dictionaries mapping labels to probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if isinstance(texts, str):
            texts = [texts]
        
        processed = self._preprocess_batch(texts)
        X = self.vectorizer.transform(processed)
        
        # Check if model supports predict_proba
        if hasattr(self.model, 'predict_proba'):
            probas = self.model.predict_proba(X)
        else:
            # Use decision function for RidgeClassifier
            decision = self.model.decision_function(X)
            # Apply softmax
            exp_decision = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            probas = exp_decision / np.sum(exp_decision, axis=1, keepdims=True)
        
        results = []
        for proba in probas:
            prob_dict = {
                label: float(proba[idx])
                for idx, label in enumerate(self.label_encoder.classes_)
            }
            # Sort by probability
            prob_dict = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
            results.append(prob_dict)
        
        return results
    
    def predict_with_confidence(self, texts: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Predict with confidence scores.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            List of prediction results with confidence
        """
        if isinstance(texts, str):
            texts = [texts]
        
        predictions = self.predict(texts)
        probabilities = self.predict_proba(texts)
        
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            confidence = probs[pred]
            
            # Get top 3 predictions
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            top_3 = sorted_probs[:3]
            
            results.append({
                'text': texts[i],
                'prediction': pred,
                'confidence': round(confidence, 4),
                'top_predictions': [
                    {'label': label, 'probability': round(prob, 4)}
                    for label, prob in top_3
                ],
                'is_cyberbullying': pred != 'neutral'
            })
        
        return results
    
    # =========================================================================
    # Evaluation
    # =========================================================================
    def evaluate(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        """
        Evaluate model on test data.
        
        Args:
            texts: List of test texts
            labels: True labels
            
        Returns:
            Evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = self.predict(texts)
        
        # Encode for metrics
        y_true = self.label_encoder.transform(labels)
        y_pred = self.label_encoder.transform(predictions)
        
        # Calculate metrics
        report = classification_report(
            y_true, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        results = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'f1_weighted': float(f1_score(y_true, y_pred, average='weighted')),
            'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
            'precision_weighted': float(precision_score(y_true, y_pred, average='weighted')),
            'recall_weighted': float(recall_score(y_true, y_pred, average='weighted')),
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'class_labels': list(self.label_encoder.classes_)
        }
        
        return results
    
    def evaluate_from_csv(self, csv_path: str, text_column: str = 'message',
                          label_column: str = 'label') -> Dict[str, Any]:
        """Evaluate from CSV file."""
        df = pd.read_csv(csv_path)
        texts = df[text_column].fillna('').tolist()
        labels = df[label_column].fillna('neutral').tolist()
        
        return self.evaluate(texts, labels)
    
    # =========================================================================
    # Model Persistence
    # =========================================================================
    def save(self, path: Optional[str] = None) -> str:
        """
        Save model to disk.
        
        Args:
            path: Directory to save model (default: saved_models/baseline_tfidf/)
            
        Returns:
            Path where model was saved
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Cannot save.")
        
        save_dir = path or self.models_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_dir, 'tfidf_classifier_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer,
                'label_encoder': self.label_encoder,
                'config': self.config
            }, f)
        
        # Save metadata
        metadata_path = os.path.join(save_dir, 'tfidf_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.training_metadata, f, indent=2)
        
        logger.info(f"Model saved to {save_dir}")
        return save_dir
    
    def load(self, path: Optional[str] = None) -> None:
        """
        Load model from disk.
        
        Args:
            path: Directory to load model from
        """
        load_dir = path or self.models_dir
        model_path = os.path.join(load_dir, 'tfidf_classifier_model.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.vectorizer = data['vectorizer']
        self.label_encoder = data['label_encoder']
        self.config = data['config']
        self.is_fitted = True
        
        # Load metadata if exists
        metadata_path = os.path.join(load_dir, 'tfidf_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.training_metadata = json.load(f)
        
        logger.info(f"Model loaded from {load_dir}")
    
    # =========================================================================
    # Feature Analysis
    # =========================================================================
    def get_top_features(self, n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get top features (words) for each class.
        
        Args:
            n: Number of top features per class
            
        Returns:
            Dictionary mapping class to top features
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get coefficients (for logistic regression)
        if hasattr(self.model, 'coef_'):
            coefs = self.model.coef_
        else:
            return {}
        
        top_features = {}
        for idx, label in enumerate(self.label_encoder.classes_):
            if len(coefs.shape) == 1:
                # Binary classification
                class_coefs = coefs
            else:
                class_coefs = coefs[idx]
            
            top_indices = np.argsort(class_coefs)[-n:][::-1]
            top_features[label] = [
                (feature_names[i], float(class_coefs[i]))
                for i in top_indices
            ]
        
        return top_features
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get overall feature importance based on coefficient magnitudes."""
        if not self.is_fitted:
            return {}
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        if hasattr(self.model, 'coef_'):
            # Use mean absolute coefficient across classes
            if len(self.model.coef_.shape) == 1:
                importance = np.abs(self.model.coef_)
            else:
                importance = np.mean(np.abs(self.model.coef_), axis=0)
            
            # Get top features by importance
            top_indices = np.argsort(importance)[-100:][::-1]
            return {
                feature_names[i]: float(importance[i])
                for i in top_indices
            }
        
        return {}
    
    def get_vocabulary_stats(self) -> Dict[str, Any]:
        """Get vocabulary statistics."""
        if not self.is_fitted:
            return {}
        
        vocab = self.vectorizer.vocabulary_
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Analyze n-gram distribution
        unigrams = sum(1 for f in feature_names if ' ' not in f)
        bigrams = sum(1 for f in feature_names if f.count(' ') == 1)
        trigrams = sum(1 for f in feature_names if f.count(' ') == 2)
        
        return {
            'vocabulary_size': len(vocab),
            'unigrams': unigrams,
            'bigrams': bigrams,
            'trigrams': trigrams,
            'sample_features': list(feature_names[:20])
        }
    
    def __repr__(self) -> str:
        """String representation."""
        status = "fitted" if self.is_fitted else "not fitted"
        classifier_type = self.config.get('classifier_type', 'logistic')
        return f"TFIDFClassifier(type={classifier_type}, status={status})"


# =============================================================================
# Convenience Functions
# =============================================================================
def train_tfidf_classifier(train_csv: str, val_csv: Optional[str] = None,
                           config: Optional[Dict[str, Any]] = None) -> TFIDFClassifier:
    """
    Train a TF-IDF classifier.
    
    Args:
        train_csv: Path to training CSV
        val_csv: Optional path to validation CSV
        config: Optional configuration
        
    Returns:
        Trained classifier
    """
    classifier = TFIDFClassifier(config)
    
    # Train
    train_results = classifier.fit_from_csv(train_csv)
    print(f"Training Results: Accuracy={train_results['accuracy']:.4f}, F1={train_results['f1_weighted']:.4f}")
    
    # Validate if provided
    if val_csv:
        val_results = classifier.evaluate_from_csv(val_csv)
        print(f"Validation Results: Accuracy={val_results['accuracy']:.4f}, F1={val_results['f1_weighted']:.4f}")
    
    return classifier


if __name__ == "__main__":
    print("=" * 60)
    print("TFIDFClassifier - Training on Full Dataset")
    print("=" * 60)
    
    # Define paths to datasets
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # Processed data paths (pre-split train/val/test)
    processed_dir = os.path.join(base_dir, '00_data', 'processed')
    train_csv = os.path.join(processed_dir, 'train_data.csv')
    val_csv = os.path.join(processed_dir, 'val_data.csv')
    test_csv = os.path.join(processed_dir, 'test_data.csv')
    
    # Raw data paths (for reference)
    raw_dir = os.path.join(base_dir, '00_data', 'raw')
    raw_files = {
        'kannada': os.path.join(raw_dir, 'kannada.csv'),
        'english': os.path.join(raw_dir, 'english.csv'),
        'bad_words': os.path.join(raw_dir, 'bad_words.csv'),
        'emoji': os.path.join(raw_dir, 'emoji_cyberbullying_dataset.csv'),
        'kannad_english': os.path.join(raw_dir, 'kannad english.csv')
    }
    
    # Check if processed training data exists
    if os.path.exists(train_csv):
        print(f"\n[INFO] Loading training data from: {train_csv}")
        
        # Load and display dataset info
        train_df = pd.read_csv(train_csv)
        print(f"[INFO] Original training samples: {len(train_df)}")
        
        # =====================================================================
        # DATA AUGMENTATION: Add explicit threat patterns missing from dataset
        # =====================================================================
        augmented_threats = [
            # Physical violence threats
            ("I will find you and hurt you", "threat"),
            ("I will hurt you badly", "threat"),
            ("I will beat you up", "threat"),
            ("I'm going to hurt you", "threat"),
            ("You will get hurt", "threat"),
            ("I'll make you pay physically", "threat"),
            ("Watch your back I'm coming for you", "threat"),
            ("I know where you live and I will find you", "threat"),
            ("You're going to regret this I will hurt you", "threat"),
            ("I will find where you live", "threat"),
            ("I'm coming to get you", "threat"),
            ("You're dead meat", "threat"),
            ("I will destroy you", "threat"),
            ("You won't survive this", "threat"),
            ("I'll make sure you suffer", "threat"),
            ("Pain is coming your way", "threat"),
            ("I will break your bones", "threat"),
            ("You'll be sorry when I find you", "threat"),
            ("ninna jothe settle maadthini", "threat"),  # Kannada: I'll settle with you
            ("ninna manege barthini nodu", "threat"),  # Kannada: I'll come to your house
            ("bekku hurt maadthini", "threat"),  # Kannada: I will hurt you
            ("ninna hinde barthini", "threat"),  # Kannada: I'm coming after you
            ("wait and watch what I do to you", "threat"),
            ("I know where you study", "threat"),
            ("meet me outside I'll show you", "threat"),
            ("this is not over I will get you", "threat"),
            ("you messed with the wrong person", "threat"),
            ("I will make your life hell", "threat"),
            ("you'll wish you never met me", "threat"),
            ("I'm going to ruin your life", "threat"),
            # More explicit harm threats
            ("I will kill you", "threat"),
            ("die already", "threat"),
            ("go die", "threat"),
            ("you should die", "threat"),
            ("I hope you die", "threat"),
            ("someone should hurt you", "threat"),
            ("you deserve to be beaten", "threat"),
            ("I'll slap you", "threat"),
            ("I'll punch your face", "threat"),
            ("wait till I catch you", "threat"),
        ]
        
        # =====================================================================
        # DATA AUGMENTATION: Add Kannada-English code-mixed insults
        # =====================================================================
        augmented_insults = [
            # Kannada insults about appearance
            ("ninna face tumba ugly agi ide", "insult"),
            ("ninna face ugly ide", "insult"),
            ("nee tumba ugly", "insult"),
            ("ninna face horrible agi ide", "insult"),
            ("nee ugly looking fellow", "insult"),
            ("ninna mugha tumba bad agi ide", "insult"),
            ("nee tumba ketta kanasu", "insult"),
            ("ninna look tumba worst", "insult"),
            ("nee horrible agi idiya", "insult"),
            ("ninna appearance tumba bad", "insult"),
            # Kannada intelligence insults
            ("nee tumba dumb fellow", "insult"),
            ("ninna brain illi work agalla", "insult"),
            ("nee stupid tara behave madthiya", "insult"),
            ("nee idiot tara matadthiya", "insult"),
            ("ninna intelligence zero", "insult"),
            ("nee fool tara idiya", "insult"),
            ("nee buddhi illa", "insult"),
            ("ninna tala alli enu illa", "insult"),
            # Kannada general insults
            ("nee worst person ever", "insult"),
            ("nee useless fellow", "insult"),
            ("nee pathetic agi idiya", "insult"),
            ("ninna value yenu illa", "insult"),
            ("nee loser fellow", "insult"),
            ("nee waste fellow antha gothide", "insult"),
            ("nee bekaar manusha", "insult"),
            ("ninna life waste", "insult"),
            # English-Kannada mix insults
            ("you are such a donkey tara idiya", "insult"),
            ("stupid fellow nee", "insult"),
            ("idiot antha ninge gotthilla", "insult"),
            ("dumb manusha nee", "insult"),
            ("you look like garbage", "insult"),
            ("your face is disgusting", "insult"),
            ("you are so pathetic", "insult"),
            ("nobody likes your ugly face", "insult"),
        ]
        
        # Create augmented DataFrames
        augment_threats_df = pd.DataFrame(augmented_threats, columns=['message', 'label'])
        augment_threats_df['target_type'] = 'Individual'
        augment_threats_df['severity'] = 'High'
        augment_threats_df['severity_score'] = 0.9
        augment_threats_df['source'] = 'augmented'
        
        augment_insults_df = pd.DataFrame(augmented_insults, columns=['message', 'label'])
        augment_insults_df['target_type'] = 'Individual'
        augment_insults_df['severity'] = 'Medium'
        augment_insults_df['severity_score'] = 0.6
        augment_insults_df['source'] = 'augmented'
        
        # Combine with original training data
        train_df = pd.concat([train_df, augment_threats_df, augment_insults_df], ignore_index=True)
        print(f"[INFO] Added {len(augmented_threats)} augmented threat samples")
        print(f"[INFO] Added {len(augmented_insults)} augmented insult samples")
        print(f"[INFO] Total training samples: {len(train_df)}")
        
        print(f"[INFO] Columns: {list(train_df.columns)}")
        print(f"\n[INFO] Label Distribution:")
        label_counts = train_df['label'].str.lower().value_counts()
        for label, count in label_counts.items():
            print(f"  - {label}: {count} ({count/len(train_df)*100:.1f}%)")
        
        # Optimized configuration for the dataset
        config = {
            'classifier_type': 'logistic',  # Logistic Regression
            'C': 1.0,
            'max_iter': 1000,
            'class_weight': 'balanced',
            'solver': 'lbfgs',
            'vectorizer': {
                'type': 'tfidf',
                'max_features': 15000,  # More features for larger dataset
                'ngram_range': (1, 3),  # Unigrams, bigrams, trigrams
                'min_df': 2,
                'max_df': 0.95,
                'sublinear_tf': True,
                'analyzer': 'word',
                'lowercase': True,
                'strip_accents': 'unicode'
            }
        }
        
        # Initialize classifier
        print("\n" + "=" * 60)
        print("Training TFIDFClassifier (Logistic Regression)...")
        print("=" * 60)
        
        classifier = TFIDFClassifier(config)
        
        # Train on full training data
        texts = train_df['message'].fillna('').tolist()
        labels = train_df['label'].str.lower().fillna('neutral').tolist()
        
        train_results = classifier.fit(texts, labels, validate=True)
        
        print(f"\n[TRAINING RESULTS]")
        print(f"  Accuracy:     {train_results['accuracy']:.4f} ({train_results['accuracy']*100:.2f}%)")
        print(f"  F1 Weighted:  {train_results['f1_weighted']:.4f}")
        print(f"  F1 Macro:     {train_results['f1_macro']:.4f}")
        if train_results.get('cv_scores'):
            print(f"  CV F1 Mean:   {train_results['metadata']['cv_f1_mean']:.4f} ± {train_results['metadata']['cv_f1_std']:.4f}")
        print(f"  Features:     {train_results['metadata']['n_features']}")
        print(f"  Classes:      {train_results['metadata']['n_classes']}")
        
        # Evaluate on validation set
        if os.path.exists(val_csv):
            print("\n" + "=" * 60)
            print("Evaluating on Validation Set...")
            print("=" * 60)
            
            val_df = pd.read_csv(val_csv)
            val_texts = val_df['message'].fillna('').tolist()
            val_labels = val_df['label'].str.lower().fillna('neutral').tolist()
            
            val_results = classifier.evaluate(val_texts, val_labels)
            
            print(f"\n[VALIDATION RESULTS]")
            print(f"  Accuracy:     {val_results['accuracy']:.4f} ({val_results['accuracy']*100:.2f}%)")
            print(f"  F1 Weighted:  {val_results['f1_weighted']:.4f}")
            print(f"  F1 Macro:     {val_results['f1_macro']:.4f}")
            print(f"  Precision:    {val_results['precision_weighted']:.4f}")
            print(f"  Recall:       {val_results['recall_weighted']:.4f}")
        
        # Evaluate on test set
        if os.path.exists(test_csv):
            print("\n" + "=" * 60)
            print("Evaluating on Test Set...")
            print("=" * 60)
            
            test_df = pd.read_csv(test_csv)
            test_texts = test_df['message'].fillna('').tolist()
            test_labels = test_df['label'].str.lower().fillna('neutral').tolist()
            
            test_results = classifier.evaluate(test_texts, test_labels)
            
            print(f"\n[TEST RESULTS]")
            print(f"  Accuracy:     {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)")
            print(f"  F1 Weighted:  {test_results['f1_weighted']:.4f}")
            print(f"  F1 Macro:     {test_results['f1_macro']:.4f}")
            print(f"  Precision:    {test_results['precision_weighted']:.4f}")
            print(f"  Recall:       {test_results['recall_weighted']:.4f}")
        
        # Test with sample predictions
        print("\n" + "=" * 60)
        print("Sample Predictions...")
        print("=" * 60)
        
        sample_test_texts = [
            "ninna face tumba ugly agi ide",           # Kannada insult
            "you are so stupid idiot",                  # English insult
            "hello how are you today",                  # Neutral
            "I will find you and hurt you",             # Threat
            "bekagidya tumba irritating fellow",        # Kannada harassment
            "great work on the project!",               # Neutral
            "nobody wants you here leave us alone",     # Exclusion
            "stop following me everywhere creep",       # Stalking
            "You are really dumb 😒",                   # Emoji insult
            "People like you are disgusting 💢",        # Hate
            "exam tumba tough aaytu but somehow aaytu", # Neutral Kannada
            "I think go away from here"                 # Aggression
        ]
        
        print("\n[PREDICTIONS]")
        predictions = classifier.predict_with_confidence(sample_test_texts)
        for pred in predictions:
            text_display = pred['text'][:50] + "..." if len(pred['text']) > 50 else pred['text']
            print(f"\n  Text: '{text_display}'")
            print(f"  -> Prediction: {pred['prediction'].upper()}")
            print(f"  -> Confidence: {pred['confidence']:.2%}")
            top_3 = [(p['label'], f"{p['probability']:.2%}") for p in pred['top_predictions'][:3]]
            print(f"  -> Top 3: {top_3}")
        
        # Save the trained model
        print("\n" + "=" * 60)
        print("Saving Model...")
        print("=" * 60)
        
        save_path = classifier.save()
        print(f"[INFO] Model saved to: {save_path}")
        
        # Show vocabulary stats
        print("\n" + "=" * 60)
        print("Vocabulary Statistics...")
        print("=" * 60)
        
        vocab_stats = classifier.get_vocabulary_stats()
        print(f"\n  Vocabulary Size: {vocab_stats['vocabulary_size']}")
        print(f"  Unigrams: {vocab_stats['unigrams']}")
        print(f"  Bigrams: {vocab_stats['bigrams']}")
        print(f"  Trigrams: {vocab_stats['trigrams']}")
        
        # Show top features per class
        print("\n" + "=" * 60)
        print("Top Features per Class...")
        print("=" * 60)
        
        top_features = classifier.get_top_features(n=10)
        for label, features in list(top_features.items())[:5]:  # Show first 5 classes
            print(f"\n  {label.upper()}:")
            feature_str = ", ".join([f"'{f[0]}'" for f in features[:5]])
            print(f"    {feature_str}")
        
        print("\n" + "=" * 60)
        print("TFIDFClassifier Training Complete!")
        print("=" * 60)
        
    else:
        # Fallback: Train on raw data if processed data not available
        print("\n[WARNING] Processed training data not found. Using raw datasets...")
        
        all_texts = []
        all_labels = []
        
        for name, filepath in raw_files.items():
            if os.path.exists(filepath):
                print(f"  Loading {name}: {filepath}")
                df = pd.read_csv(filepath)
                if 'message' in df.columns and 'label' in df.columns:
                    all_texts.extend(df['message'].fillna('').tolist())
                    all_labels.extend(df['label'].str.lower().fillna('neutral').tolist())
                    print(f"    -> Loaded {len(df)} samples")
        
        if all_texts:
            print(f"\n[INFO] Total samples: {len(all_texts)}")
            
            classifier = TFIDFClassifier()
            results = classifier.fit(all_texts, all_labels, validate=True)
            
            print(f"\n[TRAINING RESULTS]")
            print(f"  Accuracy:     {results['accuracy']:.4f}")
            print(f"  F1 Weighted:  {results['f1_weighted']:.4f}")
            
            classifier.save()
            print("\n[INFO] Model saved successfully!")
        else:
            print("\n[ERROR] No training data found. Please check dataset paths.")
