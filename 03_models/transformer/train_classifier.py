"""
Train Cyberbullying Classifier

This script trains a high-accuracy text classifier using TF-IDF + Ensemble models.
Works without PyTorch - uses scikit-learn for reliable Windows compatibility.

Achieves high accuracy through:
- TF-IDF with n-grams for text representation
- Ensemble of multiple classifiers (SVM, Random Forest, Logistic Regression)
- Class weighting for imbalanced data
- Cross-validation for robust evaluation

Author: Cyberbullying Detection Project Team
"""

import sys
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / '00_data' / 'processed'
MODELS_PATH = PROJECT_ROOT / '03_models' / 'saved_models'

# Add preprocessing to path
sys.path.insert(0, str(PROJECT_ROOT / '01_preprocessing'))
try:
    from text_normalizer import TextNormalizer  # type: ignore[import-not-found]
    NORMALIZER_AVAILABLE = True
except ImportError:
    NORMALIZER_AVAILABLE = False
    TextNormalizer = None


class CyberbullyingClassifier:
    """
    High-accuracy text classifier for cyberbullying detection.
    
    Uses TF-IDF vectorization with an ensemble of classifiers.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize classifier with configuration."""
        self.config = config or self._default_config()
        
        # Components
        self.vectorizer = None
        self.classifier = None
        self.label_encoder = LabelEncoder()
        self.preprocessor = None
        
        # Initialize preprocessor if available
        if NORMALIZER_AVAILABLE:
            self.preprocessor = TextNormalizer()
        
        # Metrics
        self.training_metrics = {}
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'max_features': 15000,
            'ngram_range': (1, 3),
            'min_df': 2,
            'max_df': 0.95,
            'use_ensemble': True,
            'class_weight': 'balanced'
        }
    
    def _preprocess(self, texts: List[str]) -> List[str]:
        """Preprocess texts."""
        if self.preprocessor:
            return [self.preprocessor.normalize(t) for t in tqdm(texts, desc="Preprocessing")]
        return texts
    
    def _build_vectorizer(self) -> TfidfVectorizer:
        """Build TF-IDF vectorizer."""
        return TfidfVectorizer(
            max_features=self.config['max_features'],
            ngram_range=self.config['ngram_range'],
            min_df=self.config['min_df'],
            max_df=self.config['max_df'],
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            lowercase=True
        )
    
    def _build_ensemble(self) -> VotingClassifier:
        """Build ensemble of classifiers."""
        # Individual classifiers
        lr = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight=self.config['class_weight'],
            solver='lbfgs',
            multi_class='multinomial',
            random_state=42,
            n_jobs=-1
        )
        
        svm = CalibratedClassifierCV(
            LinearSVC(
                C=0.5,
                class_weight=self.config['class_weight'],
                max_iter=2000,
                random_state=42
            ),
            cv=3,
            n_jobs=-1
        )
        
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            class_weight=self.config['class_weight'],
            random_state=42,
            n_jobs=-1
        )
        
        nb = MultinomialNB(alpha=0.1)
        
        # Voting ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('lr', lr),
                ('svm', svm),
                ('rf', rf),
                ('nb', nb)
            ],
            voting='soft',
            n_jobs=-1
        )
        
        return ensemble
    
    def fit(self, X_train: List[str], y_train: np.ndarray,
            X_val: List[str] = None, y_val: np.ndarray = None) -> 'CyberbullyingClassifier':
        """
        Train the classifier.
        
        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Optional validation texts
            y_val: Optional validation labels
            
        Returns:
            Self
        """
        logger.info("=" * 60)
        logger.info("Training Cyberbullying Classifier")
        logger.info("=" * 60)
        
        # Preprocess texts
        logger.info("\n1. Preprocessing texts...")
        X_train_processed = self._preprocess(X_train)
        
        # Encode labels
        logger.info("\n2. Encoding labels...")
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        logger.info(f"   Classes: {list(self.label_encoder.classes_)}")
        
        # Build and fit vectorizer
        logger.info("\n3. Building TF-IDF vectorizer...")
        self.vectorizer = self._build_vectorizer()
        X_train_tfidf = self.vectorizer.fit_transform(X_train_processed)
        logger.info(f"   Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        logger.info(f"   Feature matrix: {X_train_tfidf.shape}")
        
        # Build classifier
        logger.info("\n4. Building classifier...")
        if self.config['use_ensemble']:
            self.classifier = self._build_ensemble()
            logger.info("   Using ensemble: LR + SVM + RF + NB")
        else:
            self.classifier = LogisticRegression(
                C=1.0, max_iter=1000,
                class_weight=self.config['class_weight'],
                random_state=42, n_jobs=-1
            )
            logger.info("   Using Logistic Regression")
        
        # Train
        logger.info("\n5. Training classifier...")
        self.classifier.fit(X_train_tfidf, y_train_encoded)
        logger.info("   Training complete!")
        
        # Evaluate on training data
        train_preds = self.classifier.predict(X_train_tfidf)
        train_acc = accuracy_score(y_train_encoded, train_preds)
        train_f1 = f1_score(y_train_encoded, train_preds, average='macro')
        
        self.training_metrics['train_accuracy'] = train_acc
        self.training_metrics['train_f1'] = train_f1
        
        logger.info(f"\n   Training Accuracy: {train_acc:.4f}")
        logger.info(f"   Training F1 (macro): {train_f1:.4f}")
        
        # Validate if data provided
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            self.training_metrics['val_accuracy'] = val_metrics['accuracy']
            self.training_metrics['val_f1'] = val_metrics['f1_macro']
        
        return self
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict labels for texts."""
        processed = self._preprocess(texts)
        X_tfidf = self.vectorizer.transform(processed)
        pred_encoded = self.classifier.predict(X_tfidf)
        return self.label_encoder.inverse_transform(pred_encoded)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict probabilities for texts."""
        processed = self._preprocess(texts)
        X_tfidf = self.vectorizer.transform(processed)
        return self.classifier.predict_proba(X_tfidf)
    
    def evaluate(self, X_test: List[str], y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate classifier on test data.
        
        Args:
            X_test: Test texts
            y_test: Test labels
            
        Returns:
            Dictionary of metrics
        """
        logger.info("\nEvaluating classifier...")
        
        # Preprocess and predict
        processed = self._preprocess(X_test)
        X_tfidf = self.vectorizer.transform(processed)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        predictions = self.classifier.predict(X_tfidf)
        
        # Compute metrics
        accuracy = accuracy_score(y_test_encoded, predictions)
        f1_macro = f1_score(y_test_encoded, predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test_encoded, predictions, average='weighted', zero_division=0)
        precision = precision_score(y_test_encoded, predictions, average='macro', zero_division=0)
        recall = recall_score(y_test_encoded, predictions, average='macro', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall
        }
        
        # Classification report
        report = classification_report(
            y_test_encoded, predictions,
            target_names=self.label_encoder.classes_,
            zero_division=0
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Accuracy:    {accuracy:.4f}")
        logger.info(f"F1 Macro:    {f1_macro:.4f}")
        logger.info(f"F1 Weighted: {f1_weighted:.4f}")
        logger.info(f"Precision:   {precision:.4f}")
        logger.info(f"Recall:      {recall:.4f}")
        logger.info("\nClassification Report:")
        logger.info("\n" + report)
        
        return metrics
    
    def save(self, save_path: str = None):
        """Save model to disk."""
        if save_path is None:
            save_path = MODELS_PATH / 'tfidf_ensemble'
        else:
            save_path = Path(save_path)
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save components
        with open(save_path / 'vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open(save_path / 'classifier.pkl', 'wb') as f:
            pickle.dump(self.classifier, f)
        
        with open(save_path / 'label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        with open(save_path / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        with open(save_path / 'metrics.json', 'w') as f:
            json.dump(self.training_metrics, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def load(cls, load_path: str) -> 'CyberbullyingClassifier':
        """Load model from disk."""
        load_path = Path(load_path)
        
        with open(load_path / 'config.json', 'r') as f:
            config = json.load(f)
        
        instance = cls(config)
        
        with open(load_path / 'vectorizer.pkl', 'rb') as f:
            instance.vectorizer = pickle.load(f)
        
        with open(load_path / 'classifier.pkl', 'rb') as f:
            instance.classifier = pickle.load(f)
        
        with open(load_path / 'label_encoder.pkl', 'rb') as f:
            instance.label_encoder = pickle.load(f)
        
        logger.info(f"Model loaded from {load_path}")
        return instance


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test datasets."""
    train_df = pd.read_csv(DATA_PATH / 'train_data.csv')
    val_df = pd.read_csv(DATA_PATH / 'val_data.csv')
    test_df = pd.read_csv(DATA_PATH / 'test_data.csv')
    
    logger.info(f"Loaded data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    return train_df, val_df, test_df


def train_model() -> CyberbullyingClassifier:
    """Train the cyberbullying classifier."""
    logger.info("=" * 60)
    logger.info("CYBERBULLYING CLASSIFIER TRAINING")
    logger.info("=" * 60)
    
    # Load data
    logger.info("\nLoading datasets...")
    train_df, val_df, test_df = load_data()
    
    # Extract texts and labels
    X_train = train_df['message'].fillna('').tolist()
    y_train = train_df['label'].values
    
    X_val = val_df['message'].fillna('').tolist()
    y_val = val_df['label'].values
    
    X_test = test_df['message'].fillna('').tolist()
    y_test = test_df['label'].values
    
    # Initialize classifier
    classifier = CyberbullyingClassifier()
    
    # Train
    classifier.fit(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    logger.info("\n" + "=" * 60)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 60)
    test_metrics = classifier.evaluate(X_test, y_test)
    
    # Save model
    classifier.save()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1 Macro: {test_metrics['f1_macro']:.4f}")
    logger.info(f"Model saved to: 03_models/saved_models/tfidf_ensemble/")
    
    return classifier


def demo_predictions(classifier: CyberbullyingClassifier = None):
    """Demonstrate predictions on sample texts."""
    sample_texts = [
        "nee tumba stupid tara idiya",  # Kannada insult
        "You are such a loser",  # English insult
        "Did you finish the assignment?",  # Neutral
        "I will make you pay for this",  # Threat
        "Nobody wants you here, leave",  # Exclusion
        "Good morning everyone!",  # Neutral
        "Shut up you idiot",  # Aggression
        "yavdru ivattu notes idre send maadu pls",  # Neutral Kannada
    ]
    
    if classifier is None:
        model_path = MODELS_PATH / 'tfidf_ensemble'
        if model_path.exists():
            classifier = CyberbullyingClassifier.load(str(model_path))
        else:
            logger.error("No trained model found. Please run training first.")
            return
    
    logger.info("\n" + "=" * 60)
    logger.info("SAMPLE PREDICTIONS")
    logger.info("=" * 60)
    
    predictions = classifier.predict(sample_texts)
    probas = classifier.predict_proba(sample_texts)
    
    for text, pred, proba in zip(sample_texts, predictions, probas):
        confidence = proba.max() * 100
        logger.info(f"\nText: '{text}'")
        logger.info(f"  Prediction: {pred} (confidence: {confidence:.1f}%)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Cyberbullying Classifier')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--demo', action='store_true', help='Run demo predictions')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate saved model')
    
    args = parser.parse_args()
    
    if args.train:
        classifier = train_model()
        demo_predictions(classifier)
    elif args.demo:
        demo_predictions()
    elif args.evaluate:
        model_path = MODELS_PATH / 'tfidf_ensemble'
        if model_path.exists():
            classifier = CyberbullyingClassifier.load(str(model_path))
            _, _, test_df = load_data()
            classifier.evaluate(
                test_df['message'].fillna('').tolist(),
                test_df['label'].values
            )
        else:
            logger.error("No trained model found.")
    else:
        print("\nCyberbullying Classifier Training Script")
        print("=" * 60)
        print("\nUsage:")
        print("  Train model: python train_classifier.py --train")
        print("  Run demo:    python train_classifier.py --demo")
        print("  Evaluate:    python train_classifier.py --evaluate")
