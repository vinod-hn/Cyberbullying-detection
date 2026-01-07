# Train Baseline
"""
Train Baseline Models for Cyberbullying Detection

This script trains and evaluates all baseline classifiers:
- Naive Bayes (MultinomialNB, ComplementNB)
- SVM (LinearSVC with calibration)
- TF-IDF + Logistic Regression

Supports:
- Training on processed data
- Cross-validation
- Hyperparameter tuning
- Model comparison
- Model persistence
- Results visualization

Usage:
    python train_baseline.py --model all --tune
    python train_baseline.py --model naive_bayes
    python train_baseline.py --model svm --tune
    python train_baseline.py --model tfidf

Author: Cyberbullying Detection Project Team
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

# sklearn imports
try:
    from sklearn.metrics import (
        classification_report, confusion_matrix, accuracy_score,
        f1_score, precision_score, recall_score
    )
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError as e:
    SKLEARN_AVAILABLE = False
    print(f"sklearn not available: {e}")
    print("Install with: pip install scikit-learn pandas numpy")

# Plotting imports (optional)
try:
    import matplotlib  # type: ignore
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None

# Import baseline classifiers
from naive_bayes_classifier import NaiveBayesClassifier
from svm_classifier import SVMClassifier
from tfidf_classifier import TFIDFClassifier


# =============================================================================
# Logging Configuration
# =============================================================================
def setup_logging(log_dir: str = None) -> logging.Logger:
    """Setup logging configuration."""
    if log_dir is None:
        log_dir = os.path.join(PROJECT_ROOT, '17_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'baseline_training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


# =============================================================================
# Data Loading
# =============================================================================
class DataLoader:
    """Load and prepare data for training."""
    
    def __init__(self, data_dir: str = None):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory containing processed data
        """
        if data_dir is None:
            self.data_dir = os.path.join(PROJECT_ROOT, '00_data', 'processed')
        else:
            self.data_dir = data_dir
        
        self.train_path = os.path.join(self.data_dir, 'train_data.csv')
        self.val_path = os.path.join(self.data_dir, 'val_data.csv')
        self.test_path = os.path.join(self.data_dir, 'test_data.csv')
    
    def load_data(self, text_column: str = 'message', 
                  label_column: str = 'label') -> Dict[str, Tuple[List[str], List[str]]]:
        """
        Load train, validation, and test data.
        
        Args:
            text_column: Name of text column
            label_column: Name of label column
            
        Returns:
            Dictionary with train, val, test data tuples
        """
        data = {}
        
        # Load training data
        if os.path.exists(self.train_path):
            train_df = pd.read_csv(self.train_path)
            data['train'] = (
                train_df[text_column].fillna('').tolist(),
                train_df[label_column].fillna('neutral').tolist()
            )
            print(f"Loaded {len(data['train'][0])} training samples")
        else:
            raise FileNotFoundError(f"Training data not found: {self.train_path}")
        
        # Load validation data
        if os.path.exists(self.val_path):
            val_df = pd.read_csv(self.val_path)
            data['val'] = (
                val_df[text_column].fillna('').tolist(),
                val_df[label_column].fillna('neutral').tolist()
            )
            print(f"Loaded {len(data['val'][0])} validation samples")
        else:
            print("Validation data not found. Will use train-test split.")
            data['val'] = None
        
        # Load test data
        if os.path.exists(self.test_path):
            test_df = pd.read_csv(self.test_path)
            data['test'] = (
                test_df[text_column].fillna('').tolist(),
                test_df[label_column].fillna('neutral').tolist()
            )
            print(f"Loaded {len(data['test'][0])} test samples")
        else:
            print("Test data not found.")
            data['test'] = None
        
        return data
    
    def get_label_distribution(self, labels: List[str]) -> Dict[str, int]:
        """Get label distribution."""
        from collections import Counter
        return dict(Counter(labels))


# =============================================================================
# Model Trainer
# =============================================================================
class BaselineTrainer:
    """
    Trainer class for baseline models.
    
    Handles training, evaluation, and comparison of all baseline classifiers.
    """
    
    # Model configurations
    MODEL_CONFIGS = {
        'naive_bayes': {
            'nb_type': 'complement',  # ComplementNB for imbalanced data
            'alpha': 0.1,
            'vectorizer': {
                'type': 'tfidf',
                'max_features': 15000,
                'ngram_range': (1, 3),
                'min_df': 2,
                'max_df': 0.95,
                'sublinear_tf': True
            }
        },
        'svm': {
            'svm_type': 'linear',
            'C': 1.0,
            'class_weight': 'balanced',
            'calibrate': True,
            'vectorizer': {
                'type': 'tfidf',
                'max_features': 15000,
                'ngram_range': (1, 3),
                'min_df': 2,
                'max_df': 0.95,
                'sublinear_tf': True
            }
        },
        'tfidf': {
            'classifier_type': 'logistic',
            'C': 1.0,
            'class_weight': 'balanced',
            'max_iter': 1000,
            'vectorizer': {
                'max_features': 15000,
                'ngram_range': (1, 3),
                'min_df': 2,
                'max_df': 0.95,
                'sublinear_tf': True
            }
        }
    }
    
    # Hyperparameter grids for tuning
    PARAM_GRIDS = {
        'naive_bayes': {
            'alpha': [0.01, 0.1, 0.5, 1.0],
            'fit_prior': [True, False]
        },
        'svm': {
            'C': [0.1, 0.5, 1.0, 5.0, 10.0],
            'max_iter': [5000, 10000]
        },
        'tfidf': {
            'C': [0.1, 0.5, 1.0, 5.0, 10.0],
            'solver': ['lbfgs', 'saga'],
            'max_iter': [500, 1000]
        }
    }
    
    def __init__(self, save_dir: str = None, logger: logging.Logger = None):
        """
        Initialize trainer.
        
        Args:
            save_dir: Directory to save models and results
            logger: Logger instance
        """
        if save_dir is None:
            self.save_dir = os.path.join(PROJECT_ROOT, '03_models', 'saved_models', 'baseline_tfidf')
        else:
            self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.results_dir = os.path.join(PROJECT_ROOT, '04_evaluation', 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.logger = logger or logging.getLogger(__name__)
        self.trained_models = {}
        self.results = {}
    
    def _get_classifier(self, model_type: str, config: Dict[str, Any] = None):
        """Get classifier instance by type."""
        config = config or self.MODEL_CONFIGS.get(model_type, {})
        
        if model_type == 'naive_bayes':
            return NaiveBayesClassifier(config)
        elif model_type == 'svm':
            return SVMClassifier(config)
        elif model_type == 'tfidf':
            return TFIDFClassifier(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train_model(self, model_type: str, 
                    train_texts: List[str], 
                    train_labels: List[str],
                    val_texts: List[str] = None,
                    val_labels: List[str] = None,
                    tune: bool = False) -> Dict[str, Any]:
        """
        Train a single model.
        
        Args:
            model_type: Type of model ('naive_bayes', 'svm', 'tfidf')
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            tune: Whether to perform hyperparameter tuning
            
        Returns:
            Training results
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Training {model_type.upper()} Classifier")
        self.logger.info(f"{'='*60}")
        
        # Get classifier
        classifier = self._get_classifier(model_type)
        
        # Hyperparameter tuning if requested
        if tune and hasattr(classifier, 'tune_hyperparameters'):
            self.logger.info("Performing hyperparameter tuning...")
            param_grid = self.PARAM_GRIDS.get(model_type, {})
            tune_results = classifier.tune_hyperparameters(
                train_texts, train_labels, param_grid
            )
            self.logger.info(f"Best params: {tune_results['best_params']}")
            self.logger.info(f"Best CV score: {tune_results['best_score']:.4f}")
        
        # Train model
        train_results = classifier.fit(train_texts, train_labels, validate=True)
        
        self.logger.info(f"Training Accuracy: {train_results['accuracy']:.4f}")
        self.logger.info(f"Training F1 (weighted): {train_results['f1_weighted']:.4f}")
        self.logger.info(f"Training F1 (macro): {train_results['f1_macro']:.4f}")
        
        # Validate if data provided
        val_results = None
        if val_texts is not None and val_labels is not None:
            self.logger.info("\nValidation Results:")
            val_results = classifier.evaluate(val_texts, val_labels)
            self.logger.info(f"Validation Accuracy: {val_results['accuracy']:.4f}")
            self.logger.info(f"Validation F1 (weighted): {val_results['f1_weighted']:.4f}")
            self.logger.info(f"Validation F1 (macro): {val_results['f1_macro']:.4f}")
        
        # Save model
        model_save_dir = os.path.join(self.save_dir, model_type)
        classifier.save(model_save_dir)
        self.logger.info(f"Model saved to: {model_save_dir}")
        
        # Store results
        self.trained_models[model_type] = classifier
        self.results[model_type] = {
            'train': train_results,
            'validation': val_results,
            'model_path': model_save_dir
        }
        
        return self.results[model_type]
    
    def train_all_models(self, train_texts: List[str], train_labels: List[str],
                         val_texts: List[str] = None, val_labels: List[str] = None,
                         tune: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Train all baseline models.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            tune: Whether to tune hyperparameters
            
        Returns:
            Results for all models
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("TRAINING ALL BASELINE MODELS")
        self.logger.info("="*70)
        
        model_types = ['naive_bayes', 'svm', 'tfidf']
        
        for model_type in model_types:
            try:
                self.train_model(
                    model_type, train_texts, train_labels,
                    val_texts, val_labels, tune
                )
            except Exception as e:
                self.logger.error(f"Error training {model_type}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        return self.results
    
    def evaluate_on_test(self, test_texts: List[str], 
                         test_labels: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all trained models on test data.
        
        Args:
            test_texts: Test texts
            test_labels: Test labels
            
        Returns:
            Test results for all models
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("EVALUATING ON TEST SET")
        self.logger.info("="*70)
        
        test_results = {}
        
        for model_type, classifier in self.trained_models.items():
            self.logger.info(f"\n{model_type.upper()} Test Results:")
            try:
                results = classifier.evaluate(test_texts, test_labels)
                test_results[model_type] = results
                
                self.logger.info(f"  Accuracy: {results['accuracy']:.4f}")
                self.logger.info(f"  F1 (weighted): {results['f1_weighted']:.4f}")
                self.logger.info(f"  F1 (macro): {results['f1_macro']:.4f}")
                self.logger.info(f"  Precision: {results['precision_weighted']:.4f}")
                self.logger.info(f"  Recall: {results['recall_weighted']:.4f}")
                
                # Update results
                self.results[model_type]['test'] = results
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_type}: {e}")
        
        return test_results
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all trained models.
        
        Returns:
            DataFrame with model comparison
        """
        comparison = []
        
        for model_type, result in self.results.items():
            row = {'Model': model_type.upper()}
            
            # Training metrics
            if 'train' in result and result['train']:
                row['Train_Acc'] = result['train']['accuracy']
                row['Train_F1'] = result['train']['f1_weighted']
            
            # Validation metrics
            if 'validation' in result and result['validation']:
                row['Val_Acc'] = result['validation']['accuracy']
                row['Val_F1'] = result['validation']['f1_weighted']
            
            # Test metrics
            if 'test' in result and result['test']:
                row['Test_Acc'] = result['test']['accuracy']
                row['Test_F1'] = result['test']['f1_weighted']
                row['Test_Precision'] = result['test']['precision_weighted']
                row['Test_Recall'] = result['test']['recall_weighted']
            
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        
        # Print comparison table
        self.logger.info("\n" + "="*70)
        self.logger.info("MODEL COMPARISON")
        self.logger.info("="*70)
        print("\n" + df.to_string(index=False))
        
        return df
    
    def plot_confusion_matrices(self, test_texts: List[str], 
                                test_labels: List[str],
                                save_path: str = None):
        """
        Plot confusion matrices for all models.
        
        Args:
            test_texts: Test texts
            test_labels: Test labels
            save_path: Path to save the plot
        """
        if not PLOTTING_AVAILABLE:
            self.logger.warning("Plotting libraries not available. Skipping confusion matrices.")
            return
        
        n_models = len(self.trained_models)
        if n_models == 0:
            self.logger.warning("No trained models available for plotting.")
            return
        
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        for ax, (model_type, classifier) in zip(axes, self.trained_models.items()):
            predictions = classifier.predict(test_texts)
            
            # Get unique labels
            all_labels = list(set(test_labels) | set(predictions))
            
            cm = confusion_matrix(test_labels, predictions, labels=all_labels)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=all_labels, yticklabels=all_labels)
            ax.set_title(f'{model_type.upper()}\nConfusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.results_dir, 'confusion_matrices.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"Confusion matrices saved to: {save_path}")
        plt.close()
    
    def plot_metrics_comparison(self, save_path: str = None):
        """
        Plot metrics comparison bar chart.
        
        Args:
            save_path: Path to save the plot
        """
        if not PLOTTING_AVAILABLE:
            self.logger.warning("Plotting libraries not available. Skipping metrics comparison.")
            return
        
        comparison_df = self.compare_models()
        
        if comparison_df.empty:
            self.logger.warning("No results to plot.")
            return
        
        # Select metric columns
        metric_cols = [col for col in comparison_df.columns if col != 'Model']
        
        if not metric_cols:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(comparison_df))
        width = 0.15
        
        for i, col in enumerate(metric_cols):
            if col in comparison_df.columns:
                values = comparison_df[col].fillna(0).values
                ax.bar(x + i*width, values, width, label=col)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Baseline Model Comparison')
        ax.set_xticks(x + width * (len(metric_cols)-1) / 2)
        ax.set_xticklabels(comparison_df['Model'])
        ax.legend(loc='lower right', bbox_to_anchor=(1.15, 0))
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.results_dir, 'metrics_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"Metrics comparison saved to: {save_path}")
        plt.close()
    
    def save_results(self, filename: str = None):
        """Save training results to JSON."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(self.results_dir, f'baseline_results_{timestamp}.json')
        
        # Prepare serializable results
        serializable_results = {}
        for model_type, result in self.results.items():
            serializable_results[model_type] = {}
            for split, metrics in result.items():
                if isinstance(metrics, dict):
                    # Convert numpy arrays to lists
                    clean_metrics = {}
                    for key, value in metrics.items():
                        if isinstance(value, np.ndarray):
                            clean_metrics[key] = value.tolist()
                        elif isinstance(value, (np.int64, np.float64)):
                            clean_metrics[key] = float(value)
                        else:
                            clean_metrics[key] = value
                    serializable_results[model_type][split] = clean_metrics
                else:
                    serializable_results[model_type][split] = metrics
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to: {filename}")
        return filename
    
    def print_classification_reports(self):
        """Print detailed classification reports for all models."""
        for model_type, result in self.results.items():
            if 'test' in result and result['test']:
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"{model_type.upper()} - Classification Report (Test Set)")
                self.logger.info(f"{'='*60}")
                
                report = result['test'].get('classification_report', {})
                if report:
                    report_df = pd.DataFrame(report).transpose()
                    print(report_df.to_string())


# =============================================================================
# Main Function
# =============================================================================
def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train baseline classifiers')
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'naive_bayes', 'svm', 'tfidf'],
                        help='Model type to train')
    parser.add_argument('--tune', action='store_true',
                        help='Perform hyperparameter tuning')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing processed data')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save models')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("\n" + "="*70)
    logger.info("CYBERBULLYING DETECTION - BASELINE MODEL TRAINING")
    logger.info("="*70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Hyperparameter tuning: {args.tune}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    # Check dependencies
    if not SKLEARN_AVAILABLE:
        logger.error("Required packages not installed. Exiting.")
        return
    
    # Load data
    logger.info("\nLoading data...")
    data_loader = DataLoader(args.data_dir)
    
    try:
        data = data_loader.load_data()
    except FileNotFoundError as e:
        logger.error(f"Data not found: {e}")
        return
    
    train_texts, train_labels = data['train']
    val_texts, val_labels = data.get('val') or (None, None)
    test_texts, test_labels = data.get('test') or (None, None)
    
    # Print label distribution
    logger.info("\nLabel Distribution (Training):")
    label_dist = data_loader.get_label_distribution(train_labels)
    for label, count in sorted(label_dist.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {label}: {count} ({100*count/len(train_labels):.1f}%)")
    
    # Initialize trainer
    trainer = BaselineTrainer(save_dir=args.save_dir, logger=logger)
    
    # Train models
    if args.model == 'all':
        trainer.train_all_models(
            train_texts, train_labels,
            val_texts, val_labels,
            tune=args.tune
        )
    else:
        trainer.train_model(
            args.model,
            train_texts, train_labels,
            val_texts, val_labels,
            tune=args.tune
        )
    
    # Evaluate on test set if available
    if test_texts is not None and test_labels is not None:
        trainer.evaluate_on_test(test_texts, test_labels)
    
    # Compare models
    comparison_df = trainer.compare_models()
    
    # Print classification reports
    trainer.print_classification_reports()
    
    # Generate plots
    if not args.no_plots:
        try:
            if test_texts is not None:
                trainer.plot_confusion_matrices(test_texts, test_labels)
            trainer.plot_metrics_comparison()
        except Exception as e:
            logger.warning(f"Could not generate plots: {e}")
    
    # Save results
    results_file = trainer.save_results()
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"Models trained: {list(trainer.trained_models.keys())}")
    logger.info(f"Results saved to: {results_file}")
    
    # Best model recommendation
    if 'test' in trainer.results.get('tfidf', {}):
        best_model = max(
            trainer.results.items(),
            key=lambda x: x[1].get('test', {}).get('f1_weighted', 0) if x[1].get('test') else 0
        )
        logger.info(f"\nBest performing model: {best_model[0].upper()}")
        if best_model[1].get('test'):
            logger.info(f"  Test F1 (weighted): {best_model[1]['test']['f1_weighted']:.4f}")


# =============================================================================
# Quick Training Functions
# =============================================================================
def train_quick(model_type: str = 'all') -> Dict[str, Any]:
    """
    Quick training function for programmatic use.
    
    Args:
        model_type: 'all', 'naive_bayes', 'svm', or 'tfidf'
        
    Returns:
        Training results
    """
    logger = setup_logging()
    data_loader = DataLoader()
    data = data_loader.load_data()
    
    trainer = BaselineTrainer(logger=logger)
    
    train_texts, train_labels = data['train']
    val_texts, val_labels = data.get('val') or (None, None)
    test_texts, test_labels = data.get('test') or (None, None)
    
    if model_type == 'all':
        trainer.train_all_models(train_texts, train_labels, val_texts, val_labels)
    else:
        trainer.train_model(model_type, train_texts, train_labels, val_texts, val_labels)
    
    if test_texts:
        trainer.evaluate_on_test(test_texts, test_labels)
    
    return trainer.results


def demo():
    """
    Demo function to test baseline classifiers.
    """
    print("\n" + "="*60)
    print("BASELINE CLASSIFIER DEMO")
    print("="*60)
    
    # Sample texts for testing
    test_samples = [
        "nee tumba thotha maga tara idiya maga",  # Insult (Kannada)
        "You are so stupid and useless",  # Insult (English)
        "mundina sari nodu yen agutte anta",  # Threat (Kannada)
        "Wait and see what happens to you",  # Threat (English)
        "yavdru ivattu notes idre send maadu pls",  # Neutral (Kannada)
        "Did anyone finish the assignment?",  # Neutral (English)
        "I will hurt you if you come near me",  # Threat
        "You don't belong with us, go away",  # Exclusion
    ]
    
    # Load a trained model if available
    saved_model_path = os.path.join(PROJECT_ROOT, '03_models', 'saved_models', 'baseline_tfidf', 'tfidf')
    
    if os.path.exists(saved_model_path):
        print("\nLoading trained TF-IDF classifier...")
        classifier = TFIDFClassifier()
        classifier.load(saved_model_path)
        
        print("\nPredictions:")
        print("-" * 60)
        
        for text in test_samples:
            result = classifier.predict_with_confidence([text])[0]
            print(f"Text: {text[:50]}...")
            print(f"  → Prediction: {result['prediction']} (confidence: {result['confidence']:.2%})")
            print(f"  → Is Cyberbullying: {result['is_cyberbullying']}")
            print()
    else:
        print("\nNo trained model found. Please run training first:")
        print("  python train_baseline.py --model all")


if __name__ == "__main__":
    main()
