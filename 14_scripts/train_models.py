"""
Unified Training Script for Cyberbullying Detection Models
Handles both baseline and context-aware models with class imbalance handling.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / '17_logs' / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def train_baseline_models(args):
    """Train baseline machine learning models."""
    logger.info("="*80)
    logger.info("TRAINING BASELINE MODELS")
    logger.info("="*80)
    
    try:
        # Import baseline training module
        sys.path.insert(0, str(project_root / '03_models' / 'baseline'))
        from train_baseline import BaselineTrainer, DataLoader
        
        # Load data
        data_loader = DataLoader(str(project_root / '00_data' / 'processed'))
        data = data_loader.load_data()
        
        # Initialize trainer
        save_dir = str(project_root / '03_models' / 'saved_models' / 'baseline')
        trainer = BaselineTrainer(save_dir=save_dir, logger=logger)
        
        # Train models based on args
        models_to_train = []
        if args.model == 'all':
            models_to_train = ['naive_bayes', 'svm', 'tfidf']
        elif args.model == 'nb':
            models_to_train = ['naive_bayes']
        elif args.model == 'svm':
            models_to_train = ['svm']
        elif args.model == 'tfidf':
            models_to_train = ['tfidf']
        
        for model_type in models_to_train:
            logger.info("\n" + "="*80)
            logger.info(f"Training {model_type.upper()} Model")
            logger.info("="*80)
            
            trainer.train_model(
                model_type=model_type,
                train_texts=data['train'][0],
                train_labels=data['train'][1],
                val_texts=data.get('val', (None, None))[0] if data.get('val') else None,
                val_labels=data.get('val', (None, None))[1] if data.get('val') else None,
                tune=args.tune
            )
        
        logger.info("\nBaseline model training complete!")
        
    except ImportError as e:
        logger.error(f"Failed to import baseline modules: {e}")
        logger.error("Make sure scikit-learn is installed: pip install scikit-learn")
    except Exception as e:
        logger.error(f"Error during baseline training: {e}")
        raise


def train_context_models(args):
    """Train deep learning context-aware models."""
    logger.info("="*80)
    logger.info("TRAINING CONTEXT-AWARE MODELS")
    logger.info("="*80)
    
    try:
        # Check PyTorch availability
        try:
            import torch
            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        except ImportError:
            logger.error("PyTorch is not installed!")
            logger.error("Install with: pip install torch")
            return
        
        # Import context-aware training module
        sys.path.insert(0, str(project_root / '03_models' / 'context_aware'))
        from train_context_model import ContextModelTrainer
        
        # Model configurations
        config = {
            'vocab_size': 10000,
            'embedding_dim': 300 if args.embedding == 'glove' else 128,
            'hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.3,
            'learning_rate': 0.001,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'patience': 5,
            'use_class_weights': True,  # Handle class imbalance
            'gradient_clip': 1.0
        }
        
        # Initialize trainer
        trainer = ContextModelTrainer(
            train_path=str(project_root / '00_data' / 'processed' / 'train_data.csv'),
            val_path=str(project_root / '00_data' / 'processed' / 'val_data.csv'),
            test_path=str(project_root / '00_data' / 'processed' / 'test_data.csv'),
            results_dir=str(project_root / '04_evaluation' / 'results' / 'context_aware'),
            config=config
        )
        
        # Load data
        logger.info("Loading and preparing data...")
        data = trainer.load_data()
        
        # Build vocabulary
        logger.info("Building vocabulary...")
        trainer.build_vocabulary(data['train'][0])
        
        # Train models based on args
        if args.model == 'all' or args.model == 'lstm':
            logger.info("\n" + "="*80)
            logger.info("Training LSTM Model")
            logger.info("="*80)
            trainer.train_model('lstm', data)
        
        if args.model == 'all' or args.model == 'bilstm':
            logger.info("\n" + "="*80)
            logger.info("Training BiLSTM Model")
            logger.info("="*80)
            trainer.train_model('bilstm', data)
        
        if args.model == 'all' or args.model == 'transformer':
            logger.info("\n" + "="*80)
            logger.info("Training Transformer Model")
            logger.info("="*80)
            trainer.train_model('transformer', data)
        
        logger.info("\n✓ Context-aware model training complete!")
        
    except ImportError as e:
        logger.error(f"Failed to import context-aware modules: {e}")
        logger.error("Make sure PyTorch is installed: pip install torch")
    except Exception as e:
        logger.error(f"Error during context-aware training: {e}")
        raise


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train cyberbullying detection models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all baseline models
  python train_models.py --type baseline --model all
  
  # Train specific baseline model with hyperparameter tuning
  python train_models.py --type baseline --model svm --tune
  
  # Train all context-aware models
  python train_models.py --type context --model all --epochs 20
  
  # Train specific context-aware model
  python train_models.py --type context --model lstm --batch-size 64
  
  # Train both baseline and context-aware models
  python train_models.py --type both --model all
        """
    )
    
    parser.add_argument(
        '--type',
        choices=['baseline', 'context', 'both'],
        default='baseline',
        help='Type of models to train (default: baseline)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='all',
        help='Specific model to train. For baseline: nb, svm, tfidf, all. '
             'For context: lstm, bilstm, transformer, all (default: all)'
    )
    
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Enable hyperparameter tuning for baseline models'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs for context-aware models (default: 10)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for context-aware models (default: 32)'
    )
    
    parser.add_argument(
        '--embedding',
        choices=['random', 'glove'],
        default='random',
        help='Embedding type for context-aware models (default: random)'
    )
    
    args = parser.parse_args()
    
    # Print configuration
    logger.info("="*80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Model Type: {args.type}")
    logger.info(f"Model: {args.model}")
    if args.type in ['baseline', 'both']:
        logger.info(f"Hyperparameter Tuning: {args.tune}")
    if args.type in ['context', 'both']:
        logger.info(f"Epochs: {args.epochs}")
        logger.info(f"Batch Size: {args.batch_size}")
        logger.info(f"Embedding: {args.embedding}")
    logger.info("="*80 + "\n")
    
    # Start training
    start_time = datetime.now()
    
    try:
        if args.type == 'baseline':
            train_baseline_models(args)
        elif args.type == 'context':
            train_context_models(args)
        elif args.type == 'both':
            train_baseline_models(args)
            train_context_models(args)
        
        # Print summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING SUMMARY")
        logger.info("="*80)
        logger.info(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total Duration: {duration}")
        logger.info("="*80)
        
        logger.info("\n[SUCCESS] All training tasks completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("\n\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"\n\n[FAILED] Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
