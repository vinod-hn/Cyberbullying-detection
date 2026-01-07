# Train Context Model
"""
Training Script for Context-Aware Cyberbullying Detection Models

Trains context-aware models:
- LSTM with Attention
- Transformer-based models
- Hierarchical models

Supports:
- Data loading and preprocessing
- Training with validation
- Model evaluation
- Hyperparameter tuning
- Model comparison
- Results visualization

Usage:
    python train_context_model.py --model lstm --epochs 50
    python train_context_model.py --model transformer --batch-size 32
    python train_context_model.py --model all

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
from collections import Counter

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. This script requires PyTorch.")
    print("Install with: pip install torch")
    sys.exit(1)

# Import context-aware models
try:
    from lstm_context_model import LSTMContextModel, save_lstm_model, load_lstm_model
    from conversation_transformer import ConversationTransformer, save_transformer_model
except ImportError as e:
    print(f"Error importing models: {e}")
    print("Make sure you're running from the correct directory.")
    sys.exit(1)


# =============================================================================
# Logging Configuration
# =============================================================================
def setup_logging(log_dir: str = None) -> logging.Logger:
    """Setup logging configuration."""
    if log_dir is None:
        log_dir = os.path.join(PROJECT_ROOT, '17_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'context_model_training_{timestamp}.log')
    
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
# Dataset Classes
# =============================================================================
class CyberbullyingDataset(Dataset):
    """
    Dataset for cyberbullying detection.
    
    Args:
        texts: List of text messages
        labels: List of labels
        vocab: Vocabulary dictionary
        max_len: Maximum sequence length
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[str],
        vocab: Dict[str, int],
        label_map: Dict[str, int],
        max_len: int = 100
    ):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.label_map = label_map
        self.max_len = max_len
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text to IDs."""
        tokens = text.lower().split()
        ids = [self.vocab.get(token, self.vocab.get('<UNK>', 1)) for token in tokens]
        return ids
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        token_ids = self.tokenize(text)
        
        # Pad or truncate
        if len(token_ids) > self.max_len:
            token_ids = token_ids[:self.max_len]
        else:
            token_ids = token_ids + [0] * (self.max_len - len(token_ids))
        
        # Create attention mask
        attention_mask = [1 if id != 0 else 0 for id in token_ids]
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float),
            'labels': torch.tensor(self.label_map[label], dtype=torch.long),
            'length': torch.tensor(sum(attention_mask), dtype=torch.long)
        }


# =============================================================================
# Vocabulary Builder
# =============================================================================
class VocabularyBuilder:
    """Build vocabulary from texts."""
    
    def __init__(self, min_freq: int = 2, max_vocab_size: int = 10000):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.vocab = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.word_counts = Counter()
    
    def build_vocab(self, texts: List[str]) -> Dict[str, int]:
        """Build vocabulary from texts."""
        # Count words
        for text in texts:
            tokens = text.lower().split()
            self.word_counts.update(tokens)
        
        # Add frequent words to vocab
        for word, count in self.word_counts.most_common(self.max_vocab_size):
            if count >= self.min_freq and word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        
        return self.vocab
    
    def save_vocab(self, path: str):
        """Save vocabulary to file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, indent=2, ensure_ascii=False)
    
    def load_vocab(self, path: str) -> Dict[str, int]:
        """Load vocabulary from file."""
        with open(path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        return self.vocab


# =============================================================================
# Data Loader
# =============================================================================
class ContextDataLoader:
    """Load and prepare data for context-aware models."""
    
    def __init__(self, data_dir: str = None, logger: logging.Logger = None):
        if data_dir is None:
            self.data_dir = os.path.join(PROJECT_ROOT, '00_data', 'processed')
        else:
            self.data_dir = data_dir
        
        self.logger = logger or logging.getLogger(__name__)
        
        self.train_path = os.path.join(self.data_dir, 'train_data.csv')
        self.val_path = os.path.join(self.data_dir, 'val_data.csv')
        self.test_path = os.path.join(self.data_dir, 'test_data.csv')
        
        self.vocab = None
        self.label_map = None
    
    def load_data(self) -> Dict[str, Tuple[List[str], List[str]]]:
        """Load all data splits."""
        data = {}
        
        # Load training data
        if os.path.exists(self.train_path):
            train_df = pd.read_csv(self.train_path)
            # Normalize labels to lowercase
            data['train'] = (
                train_df['message'].fillna('').astype(str).tolist(),
                train_df['label'].fillna('neutral').astype(str).str.lower().tolist()
            )
            self.logger.info(f"Loaded {len(data['train'][0])} training samples")
            
            # Show label distribution
            label_counts = Counter(data['train'][1])
            self.logger.info("Training label distribution:")
            for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
                self.logger.info(f"  {label}: {count} ({100*count/len(data['train'][1]):.1f}%)")
        else:
            raise FileNotFoundError(f"Training data not found: {self.train_path}")
        
        # Load validation data
        if os.path.exists(self.val_path):
            val_df = pd.read_csv(self.val_path)
            data['val'] = (
                val_df['message'].fillna('').astype(str).tolist(),
                val_df['label'].fillna('neutral').astype(str).str.lower().tolist()
            )
            self.logger.info(f"Loaded {len(data['val'][0])} validation samples")
        
        # Load test data
        if os.path.exists(self.test_path):
            test_df = pd.read_csv(self.test_path)
            data['test'] = (
                test_df['message'].fillna('').astype(str).tolist(),
                test_df['label'].fillna('neutral').astype(str).str.lower().tolist()
            )
            self.logger.info(f"Loaded {len(data['test'][0])} test samples")
        
        return data
    
    def build_vocabulary(self, texts: List[str], vocab_path: str = None) -> Dict[str, int]:
        """Build or load vocabulary."""
        if vocab_path and os.path.exists(vocab_path):
            vocab_builder = VocabularyBuilder()
            self.vocab = vocab_builder.load_vocab(vocab_path)
            self.logger.info(f"Loaded vocabulary: {len(self.vocab)} words")
        else:
            vocab_builder = VocabularyBuilder(min_freq=2, max_vocab_size=10000)
            self.vocab = vocab_builder.build_vocab(texts)
            self.logger.info(f"Built vocabulary: {len(self.vocab)} words")
            
            if vocab_path:
                vocab_builder.save_vocab(vocab_path)
                self.logger.info(f"Saved vocabulary to: {vocab_path}")
        
        return self.vocab
    
    def build_label_map(self, labels: List[str]) -> Dict[str, int]:
        """Build label to index mapping."""
        unique_labels = sorted(set(labels))
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.logger.info(f"Labels: {list(self.label_map.keys())}")
        return self.label_map
    
    def create_dataloaders(
        self,
        data: Dict[str, Tuple[List[str], List[str]]],
        batch_size: int = 32,
        max_len: int = 100
    ) -> Dict[str, DataLoader]:
        """Create PyTorch dataloaders."""
        dataloaders = {}
        
        for split, (texts, labels) in data.items():
            dataset = CyberbullyingDataset(
                texts, labels, self.vocab, self.label_map, max_len
            )
            
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=0  # Windows compatibility
            )
        
        return dataloaders


# =============================================================================
# Model Trainer
# =============================================================================
class ContextModelTrainer:
    """Trainer for context-aware models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        logger: logging.Logger = None
    ):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.logger = logger or logging.getLogger(__name__)
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in dataloader:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            loss = criterion(outputs['logits'], labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            preds = outputs['logits'].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total
        }
    
    def evaluate(
        self,
        dataloader: DataLoader,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs['logits'], labels)
                
                total_loss += loss.item()
                preds = outputs['logits'].argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute F1 score
        try:
            from sklearn.metrics import f1_score  # type: ignore
            f1 = f1_score(all_labels, all_preds, average='weighted')
        except ImportError:
            f1 = 0.0
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total,
            'f1_weighted': f1
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        learning_rate: float = 0.001,
        save_path: str = None
    ) -> Dict[str, List[float]]:
        """Train model for multiple epochs."""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        self.logger.info(f"\nStarting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, optimizer, criterion)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            
            # Validate
            val_metrics = self.evaluate(val_loader, criterion)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            # Log
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                # Save best model
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    self.logger.info(f"Saved best model to: {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        return self.history


# =============================================================================
# Main Function
# =============================================================================
def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train context-aware models')
    parser.add_argument('--model', type=str, default='lstm',
                        choices=['lstm', 'transformer', 'all'],
                        help='Model type to train')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--max-len', type=int, default=100,
                        help='Maximum sequence length')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Data directory')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Save directory')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    
    logger.info("\n" + "="*70)
    logger.info("CONTEXT-AWARE MODEL TRAINING")
    logger.info("="*70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Device: {args.device}")
    
    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available. Using CPU.")
        args.device = 'cpu'
    
    # Load data
    logger.info("\nLoading data...")
    data_loader = ContextDataLoader(args.data_dir, logger)
    data = data_loader.load_data()
    
    # Build vocabulary and label map
    vocab_path = os.path.join(PROJECT_ROOT, '03_models', 'saved_models', 'vocab.json')
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    
    vocab = data_loader.build_vocabulary(data['train'][0], vocab_path)
    label_map = data_loader.build_label_map(data['train'][1])
    
    # Create dataloaders
    logger.info("\nCreating dataloaders...")
    dataloaders = data_loader.create_dataloaders(
        data, batch_size=args.batch_size, max_len=args.max_len
    )
    
    # Model parameters
    vocab_size = len(vocab)
    num_classes = len(label_map)
    
    # Save directory
    if args.save_dir is None:
        save_dir = os.path.join(PROJECT_ROOT, '03_models', 'saved_models', 'context_models')
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # Train models
    results = {}
    
    if args.model in ['lstm', 'all']:
        logger.info("\n" + "="*70)
        logger.info("Training LSTM Context Model")
        logger.info("="*70)
        
        model = LSTMContextModel(
            vocab_size=vocab_size,
            embedding_dim=128,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            dropout=0.3
        )
        
        trainer = ContextModelTrainer(model, args.device, logger)
        
        save_path = os.path.join(save_dir, 'lstm_context_model.pth')
        history = trainer.train(
            dataloaders['train'],
            dataloaders['val'],
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            save_path=save_path
        )
        
        # Test evaluation
        if 'test' in dataloaders:
            test_metrics = trainer.evaluate(
                dataloaders['test'],
                nn.CrossEntropyLoss()
            )
            logger.info(f"\nLSTM Test Results:")
            logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"  F1 (weighted): {test_metrics['f1_weighted']:.4f}")
            
            results['lstm'] = {
                'history': history,
                'test_metrics': test_metrics
            }
    
    if args.model in ['transformer', 'all']:
        logger.info("\n" + "="*70)
        logger.info("Training Transformer Model")
        logger.info("="*70)
        
        model = ConversationTransformer(
            vocab_size=vocab_size,
            d_model=args.hidden_dim,
            num_heads=8,
            num_encoder_layers=4,
            d_ff=args.hidden_dim * 4,
            num_classes=num_classes,
            max_seq_len=args.max_len,
            dropout=0.1
        )
        
        trainer = ContextModelTrainer(model, args.device, logger)
        
        save_path = os.path.join(save_dir, 'transformer_model.pth')
        history = trainer.train(
            dataloaders['train'],
            dataloaders['val'],
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            save_path=save_path
        )
        
        # Test evaluation
        if 'test' in dataloaders:
            test_metrics = trainer.evaluate(
                dataloaders['test'],
                nn.CrossEntropyLoss()
            )
            logger.info(f"\nTransformer Test Results:")
            logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"  F1 (weighted): {test_metrics['f1_weighted']:.4f}")
            
            results['transformer'] = {
                'history': history,
                'test_metrics': test_metrics
            }
    
    # Save results
    results_path = os.path.join(save_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        # Convert to serializable format
        serializable_results = {}
        for model_name, model_results in results.items():
            serializable_results[model_name] = {
                'history': {k: [float(v) for v in vals] for k, vals in model_results['history'].items()},
                'test_metrics': {k: float(v) for k, v in model_results['test_metrics'].items()}
            }
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_path}")
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("\n" + "="*60)
        print("ERROR: PyTorch Not Installed")
        print("="*60)
        print("\nThis script requires PyTorch for training context-aware models.")
        print("\nInstall PyTorch:")
        print("  pip install torch")
        print("\nFor CPU-only (smaller):")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("="*60)
    else:
        main()

