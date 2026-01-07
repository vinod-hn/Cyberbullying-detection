"""
Multi-Task Learning for Cyberbullying Detection

Trains models to simultaneously predict:
- Cyberbullying category (classification)
- Severity score (regression)
- Target type (classification)
- Intent detection (classification)

Uses shared representations with task-specific heads.

Author: Cyberbullying Detection Project Team
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple

# Optional scikit-learn imports with safe fallbacks to avoid unresolved-import warnings
try:
    from sklearn.base import BaseEstimator
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    class BaseEstimator:  # Minimal fallback
        pass
    
    class LabelEncoder:  # Lightweight fallback encoder
        def __init__(self):
            self.classes_ = None
            self._mapping = {}
            self._inverse = []
        def fit(self, y):
            classes = np.unique(y)
            self.classes_ = classes
            self._mapping = {c: i for i, c in enumerate(classes)}
            self._inverse = list(classes)
            return self
        def fit_transform(self, y):
            self.fit(y)
            return np.array([self._mapping[c] for c in y], dtype=np.int64)
        def inverse_transform(self, y_idx):
            return np.array([self._inverse[int(i)] for i in y_idx])

logger = logging.getLogger(__name__)

# Optional PyTorch for neural multi-task learning
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


# =============================================================================
# Multi-Task Models
# =============================================================================

if TORCH_AVAILABLE:
    class MultiTaskCyberbullyingModel(nn.Module):
        """
        Neural multi-task learning model.
        
        Architecture:
        - Shared encoder (LSTM/Transformer)
        - Task-specific heads for each prediction task
        - Attention mechanism for importance weighting
        """
        
        def __init__(self, vocab_size: int, embedding_dim: int = 128,
                     hidden_dim: int = 256, n_classes: int = 11,
                     n_severity: int = 5, n_targets: int = 5,
                     dropout: float = 0.3):
            """
            Initialize multi-task model.
            
            Args:
                vocab_size: Vocabulary size
                embedding_dim: Embedding dimension
                hidden_dim: Hidden dimension for LSTM
                n_classes: Number of cyberbullying classes
                n_severity: Number of severity levels
                n_targets: Number of target types
                dropout: Dropout probability
            """
            super().__init__()
            
            # Shared layers
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.lstm = nn.LSTM(
                embedding_dim, hidden_dim,
                num_layers=2, bidirectional=True,
                batch_first=True, dropout=dropout
            )
            self.dropout = nn.Dropout(dropout)
            
            # Task-specific heads
            lstm_output_dim = hidden_dim * 2  # Bidirectional
            
            # Classification head (main task)
            self.class_head = nn.Sequential(
                nn.Linear(lstm_output_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes)
            )
            
            # Severity regression head
            self.severity_head = nn.Sequential(
                nn.Linear(lstm_output_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
            
            # Target type head
            self.target_head = nn.Sequential(
                nn.Linear(lstm_output_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, n_targets)
            )
            
            # Task attention weights
            self.task_attention = nn.Parameter(torch.ones(3))
        
        def forward(self, x, lengths: Optional[List[int]] = None) -> Dict[str, Any]:
            """
            Forward pass.
            
            Args:
                x: Input token IDs (batch_size, seq_len)
                lengths: Sequence lengths for packing
                
            Returns:
                Dictionary with predictions for each task
            """
            # Embedding
            embedded = self.embedding(x)
            embedded = self.dropout(embedded)
            
            # LSTM encoding
            if lengths is not None:
                packed = nn.utils.rnn.pack_padded_sequence(
                    embedded, lengths, batch_first=True, enforce_sorted=False
                )
                lstm_out, (hidden, cell) = self.lstm(packed)
                lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
            else:
                lstm_out, (hidden, cell) = self.lstm(embedded)
            
            # Use last hidden state
            # Concatenate forward and backward
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
            
            # Task-specific predictions
            class_logits = self.class_head(last_hidden)
            severity_score = self.severity_head(last_hidden).squeeze(-1)
            target_logits = self.target_head(last_hidden)
            
            return {
                'class_logits': class_logits,
                'severity_score': severity_score,
                'target_logits': target_logits
            }
        
        def get_task_weights(self) -> Any:
            """Get normalized task attention weights."""
            return F.softmax(self.task_attention, dim=0)


class MultiTaskLearner(BaseEstimator):
    """
    Multi-task learner for cyberbullying detection.
    
    Trains a single model for multiple related tasks:
    - Main task: Cyberbullying classification
    - Auxiliary tasks: Severity, target type, etc.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize multi-task learner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        self.model = None
        self.label_encoders = {}
        self.vocab = None
        
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = None
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'vocab_size': 10000,
            'embedding_dim': 128,
            'hidden_dim': 256,
            'dropout': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 10,
            'task_weights': {
                'classification': 1.0,
                'severity': 0.5,
                'target': 0.3
            }
        }
    
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts."""
        from collections import Counter
        
        word_freq = Counter()
        for text in texts:
            words = text.lower().split()
            word_freq.update(words)
        
        # Get most common words
        vocab_size = self.config['vocab_size']
        most_common = word_freq.most_common(vocab_size - 2)
        
        # Build vocab dict
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        for i, (word, _) in enumerate(most_common):
            self.vocab[word] = i + 2
        
        logger.info(f"Built vocabulary with {len(self.vocab)} words")
    
    def _tokenize(self, texts: List[str]) -> Tuple[np.ndarray, List[int]]:
        """Tokenize texts to token IDs."""
        token_ids = []
        lengths = []
        
        max_len = 100
        
        for text in texts:
            words = text.lower().split()
            ids = [self.vocab.get(w, 1) for w in words[:max_len]]
            token_ids.append(ids)
            lengths.append(len(ids))
        
        # Pad sequences
        padded = np.zeros((len(token_ids), max_len), dtype=np.int64)
        for i, ids in enumerate(token_ids):
            padded[i, :len(ids)] = ids
        
        return padded, lengths
    
    def fit(self, X: List[str], y: Dict[str, np.ndarray]) -> 'MultiTaskLearner':
        """
        Fit multi-task model.
        
        Args:
            X: Training texts
            y: Dictionary with labels for each task:
                - 'class': Main classification labels
                - 'severity': Severity scores (optional)
                - 'target': Target type labels (optional)
                
        Returns:
            Self
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for multi-task learning")
        
        # Build vocabulary
        if self.vocab is None:
            self.build_vocab(X)
        
        # Encode labels
        if 'class' not in y:
            raise ValueError("Main 'class' labels required")
        
        self.label_encoders['class'] = LabelEncoder()
        y_class = self.label_encoders['class'].fit_transform(y['class'])
        
        n_classes = len(self.label_encoders['class'].classes_)
        
        # Handle optional tasks
        y_severity = y.get('severity', np.zeros(len(X)))
        
        if 'target' in y:
            self.label_encoders['target'] = LabelEncoder()
            y_target = self.label_encoders['target'].fit_transform(y['target'])
            n_targets = len(self.label_encoders['target'].classes_)
        else:
            y_target = np.zeros(len(X), dtype=np.int64)
            n_targets = 5
        
        # Build model
        self.model = MultiTaskCyberbullyingModel(
            vocab_size=self.config['vocab_size'],
            embedding_dim=self.config['embedding_dim'],
            hidden_dim=self.config['hidden_dim'],
            n_classes=n_classes,
            n_severity=5,
            n_targets=n_targets,
            dropout=self.config['dropout']
        ).to(self.device)
        
        # Tokenize
        X_tokens, lengths = self._tokenize(X)
        
        # Training setup
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate']
        )
        
        class_criterion = nn.CrossEntropyLoss()
        severity_criterion = nn.MSELoss()
        target_criterion = nn.CrossEntropyLoss()
        
        # Training loop
        batch_size = self.config['batch_size']
        n_batches = len(X) // batch_size
        
        logger.info(f"Training multi-task model for {self.config['epochs']} epochs...")
        
        for epoch in range(self.config['epochs']):
            self.model.train()
            epoch_loss = 0.0
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                # Get batch
                batch_x = torch.tensor(X_tokens[start_idx:end_idx], device=self.device)
                batch_lengths = lengths[start_idx:end_idx]
                batch_y_class = torch.tensor(y_class[start_idx:end_idx], device=self.device)
                batch_y_severity = torch.tensor(y_severity[start_idx:end_idx], dtype=torch.float32, device=self.device)
                batch_y_target = torch.tensor(y_target[start_idx:end_idx], device=self.device)
                
                # Forward pass
                outputs = self.model(batch_x, batch_lengths)
                
                # Compute losses
                loss_class = class_criterion(outputs['class_logits'], batch_y_class)
                loss_severity = severity_criterion(outputs['severity_score'], batch_y_severity)
                loss_target = target_criterion(outputs['target_logits'], batch_y_target)
                
                # Weighted total loss
                task_weights = self.config['task_weights']
                total_loss = (
                    task_weights['classification'] * loss_class +
                    task_weights['severity'] * loss_severity +
                    task_weights['target'] * loss_target
                )
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += total_loss.item()
            
            avg_loss = epoch_loss / n_batches
            logger.info(f"Epoch {epoch+1}/{self.config['epochs']}, Loss: {avg_loss:.4f}")
        
        logger.info("Multi-task training complete!")
        return self
    
    def predict(self, X: List[str]) -> Dict[str, np.ndarray]:
        """
        Predict all tasks.
        
        Args:
            X: Input texts
            
        Returns:
            Dictionary with predictions for each task
        """
        if self.model is None:
            raise ValueError("Model not fitted")
        
        self.model.eval()
        
        # Tokenize
        X_tokens, lengths = self._tokenize(X)
        
        all_preds = {
            'class': [],
            'severity': [],
            'target': []
        }
        
        batch_size = self.config['batch_size']
        n_batches = (len(X) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(X))
                
                batch_x = torch.tensor(X_tokens[start_idx:end_idx], device=self.device)
                batch_lengths = lengths[start_idx:end_idx]
                
                # Forward pass
                outputs = self.model(batch_x, batch_lengths)
                
                # Get predictions
                class_preds = torch.argmax(outputs['class_logits'], dim=1).cpu().numpy()
                severity_preds = outputs['severity_score'].cpu().numpy()
                target_preds = torch.argmax(outputs['target_logits'], dim=1).cpu().numpy()
                
                all_preds['class'].extend(class_preds)
                all_preds['severity'].extend(severity_preds)
                all_preds['target'].extend(target_preds)
        
        # Decode labels
        all_preds['class'] = self.label_encoders['class'].inverse_transform(all_preds['class'])
        if 'target' in self.label_encoders:
            all_preds['target'] = self.label_encoders['target'].inverse_transform(all_preds['target'])
        
        return all_preds
    
    def predict_class(self, X: List[str]) -> np.ndarray:
        """Predict only main classification task."""
        predictions = self.predict(X)
        return predictions['class']


# =============================================================================
# Utility Functions
# =============================================================================

def combine_predictions(predictions: List[Dict[str, np.ndarray]], 
                       weights: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
    """
    Combine predictions from multiple multi-task models.
    
    Args:
        predictions: List of prediction dictionaries
        weights: Optional weights for each model
        
    Returns:
        Combined predictions
    """
    if weights is None:
        weights = [1.0] * len(predictions)
    
    # Normalize weights
    weights = np.array(weights) / sum(weights)
    
    # Combine predictions
    combined = {}
    
    for task in predictions[0].keys():
        if task == 'severity':
            # Average for regression
            combined[task] = np.zeros_like(predictions[0][task])
            for i, preds in enumerate(predictions):
                combined[task] += preds[task] * weights[i]
        else:
            # Voting for classification
            from collections import Counter
            combined[task] = []
            for sample_idx in range(len(predictions[0][task])):
                votes = [preds[task][sample_idx] for preds in predictions]
                combined[task].append(Counter(votes).most_common(1)[0][0])
            combined[task] = np.array(combined[task])
    
    return combined


if __name__ == "__main__":
    print("Multi-Task Learning Module")
    print("="*60)
    print("Supports simultaneous prediction of:")
    print("  - Cyberbullying classification (main task)")
    print("  - Severity score (auxiliary)")
    print("  - Target type (auxiliary)")
    print("\nRequires PyTorch for neural multi-task learning.")
    print("\nUsage:")
    print("  from multi_task_learning import MultiTaskLearner")
    print("  learner = MultiTaskLearner()")
    print("  learner.fit(X_train, {'class': y_class, 'severity': y_severity})")
    print("  predictions = learner.predict(X_test)")

