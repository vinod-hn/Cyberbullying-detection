"""
IndicBERT Classifier for Cyberbullying Detection

A comprehensive IndicBERT-based classifier optimized for Indian languages that:
- Uses pre-trained ai4bharat/indic-bert for code-mixed text (Kannada-English)
- Integrates with preprocessing pipeline
- Supports multi-class classification with 11 cyberbullying categories
- Includes severity prediction as auxiliary task
- Provides confidence scores and explanations

Author: Cyberbullying Detection Project Team
"""

import os
import sys
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import importlib

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project root setup
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Optional imports with graceful fallbacks
TORCH_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False

# Type stubs for when libraries are not available
torch = None
nn = None
F = None
Dataset = None
DataLoader = None
AdamW = None
AutoModel = None
AutoTokenizer = None
get_linear_schedule_with_warmup = None

try:
    import torch as _torch  # type: ignore[import-not-found]
    import torch.nn as _nn  # type: ignore[import-not-found]
    import torch.nn.functional as _F  # type: ignore[import-not-found]
    from torch.utils.data import Dataset as _Dataset, DataLoader as _DataLoader  # type: ignore[import-not-found]
    from torch.optim import AdamW as _AdamW  # type: ignore[import-not-found]
    
    # Assign to module-level variables
    torch = _torch
    nn = _nn
    F = _F
    Dataset = _Dataset
    DataLoader = _DataLoader
    AdamW = _AdamW
    TORCH_AVAILABLE = True
except ImportError:
    class _DummyDataset:
        """Dummy Dataset class when PyTorch is not available."""
        pass
    Dataset = _DummyDataset
    logger.warning("PyTorch not available. Install with: pip install torch")

try:
    from transformers import (
        AutoModel as _AutoModel, 
        AutoTokenizer as _AutoTokenizer,
        AlbertModel, AlbertTokenizer,
        get_linear_schedule_with_warmup as _get_linear_schedule_with_warmup
    )
    AutoModel = _AutoModel
    AutoTokenizer = _AutoTokenizer
    get_linear_schedule_with_warmup = _get_linear_schedule_with_warmup
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers not available. Install with: pip install transformers")

try:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        classification_report, confusion_matrix
    )
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = lambda x, **kwargs: x

# Import preprocessing modules dynamically
preprocessing_path = PROJECT_ROOT / '01_preprocessing'
if str(preprocessing_path) not in sys.path:
    sys.path.insert(0, str(preprocessing_path))

TextNormalizer = None
EmojiHandler = None
try:
    text_norm_mod = importlib.import_module('text_normalizer')
    TextNormalizer = getattr(text_norm_mod, 'TextNormalizer', None)
    emoji_mod = importlib.import_module('emoji_handler')
    EmojiHandler = getattr(emoji_mod, 'EmojiHandler', None)
except Exception as e:
    logger.warning(f"Preprocessing modules not fully available: {e}")

# Import feature extraction modules dynamically
feature_path = PROJECT_ROOT / '02_feature_extraction'
if str(feature_path) not in sys.path:
    sys.path.insert(0, str(feature_path))

LinguisticFeatures = None
EmojiFeatures = None
try:
    ling_mod = importlib.import_module('linguistic_features')
    LinguisticFeatures = getattr(ling_mod, 'LinguisticFeatures', None)
    emoji_feat_mod = importlib.import_module('emoji_features')
    EmojiFeatures = getattr(emoji_feat_mod, 'EmojiFeatures', None)
except Exception as e:
    logger.warning(f"Feature extraction modules not fully available: {e}")


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    # Model settings - IndicBERT is based on ALBERT architecture
    'model_name': 'ai4bharat/indic-bert',  # Best for Indian languages
    'max_length': 128,
    'hidden_dropout': 0.3,
    'classifier_dropout': 0.2,
    'num_labels': 11,
    
    # Training settings
    'batch_size': 16,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'epochs': 10,
    'warmup_ratio': 0.1,
    'max_grad_norm': 1.0,
    'early_stopping_patience': 3,
    
    # Data settings
    'train_split': 0.8,
    'val_split': 0.1,
    'test_split': 0.1,
    
    # Feature integration
    'use_preprocessing': True,
    'use_linguistic_features': True,
    'linguistic_feature_dim': 32,
    
    # Class labels
    'labels': [
        'neutral', 'insult', 'harassment', 'threat', 'exclusion',
        'aggression', 'toxicity', 'stalking', 'sexual_harassment',
        'hate', 'cyberstalking'
    ]
}


# =============================================================================
# Dataset Classes
# =============================================================================

_DatasetBase = Dataset if TORCH_AVAILABLE and Dataset is not None else object


class IndicCyberbullyingDataset(_DatasetBase):
    """
    PyTorch Dataset for cyberbullying detection with IndicBERT.
    
    Handles tokenization, preprocessing, and feature extraction.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer: Any = None,
        max_length: int = 128,
        preprocessor: Optional[Any] = None,
        extract_features: bool = False,
        linguistic_extractor: Optional[Any] = None
    ):
        """
        Initialize dataset.
        
        Args:
            texts: List of text messages
            labels: Optional list of label indices
            tokenizer: IndicBERT tokenizer
            max_length: Maximum sequence length
            preprocessor: Optional TextNormalizer instance
            extract_features: Whether to extract linguistic features
            linguistic_extractor: Optional LinguisticFeatures instance
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required. Install with: pip install torch")
        
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocessor = preprocessor
        self.extract_features = extract_features
        self.linguistic_extractor = linguistic_extractor
        
        # Pre-process all texts
        self.processed_texts = []
        for text in texts:
            if self.preprocessor:
                text = self.preprocessor.normalize(text)
            self.processed_texts.append(text)
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.processed_texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        # Add token type ids if available (IndicBERT/ALBERT uses them)
        if 'token_type_ids' in encoding:
            item['token_type_ids'] = encoding['token_type_ids'].flatten()
        
        # Add labels if available
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Extract linguistic features if enabled
        if self.extract_features and self.linguistic_extractor:
            try:
                feats = self.linguistic_extractor.extract(text)
                feat_values = list(feats.values())[:32]
                while len(feat_values) < 32:
                    feat_values.append(0.0)
                item['linguistic_features'] = torch.tensor(feat_values, dtype=torch.float)
            except Exception:
                item['linguistic_features'] = torch.zeros(32, dtype=torch.float)
        
        return item


# =============================================================================
# Model Architecture
# =============================================================================

_ModuleBase = nn.Module if TORCH_AVAILABLE and nn is not None else object


class IndicBertCyberbullyingClassifier(_ModuleBase):
    """
    IndicBERT-based classifier for cyberbullying detection.
    
    Architecture:
    - Pre-trained IndicBERT (ALBERT-based) encoder
    - Optional linguistic feature integration
    - Multi-head classification with dropout
    - Auxiliary severity prediction head
    """
    
    def __init__(
        self,
        model_name: str = 'ai4bharat/indic-bert',
        num_labels: int = 11,
        hidden_dropout: float = 0.3,
        classifier_dropout: float = 0.2,
        use_linguistic_features: bool = False,
        linguistic_feature_dim: int = 32
    ):
        """
        Initialize IndicBERT classifier.
        
        Args:
            model_name: Pre-trained model name
            num_labels: Number of classification labels
            hidden_dropout: Dropout for hidden layers
            classifier_dropout: Dropout for classifier
            use_linguistic_features: Whether to use linguistic features
            linguistic_feature_dim: Dimension of linguistic features
        """
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "PyTorch and Transformers are required. "
                "Install with: pip install torch transformers"
            )
        
        super().__init__()
        
        self.num_labels = num_labels
        self.use_linguistic_features = use_linguistic_features
        self.linguistic_feature_dim = linguistic_feature_dim
        
        # Load pre-trained IndicBERT (ALBERT architecture)
        self.indicbert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.indicbert.config.hidden_size
        
        # Dropout layers
        self.hidden_dropout = nn.Dropout(hidden_dropout)
        self.classifier_dropout = nn.Dropout(classifier_dropout)
        
        # Feature dimension calculation
        classifier_input_dim = self.hidden_size
        if use_linguistic_features:
            # Linguistic feature projection
            self.linguistic_projection = nn.Sequential(
                nn.Linear(linguistic_feature_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32)
            )
            classifier_input_dim += 32
        
        # Main classification head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(128, num_labels)
        )
        
        # Auxiliary severity prediction head
        self.severity_head = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Attention pooling layer
        self.attention_weights = nn.Linear(self.hidden_size, 1)
    
    def attention_pool(self, hidden_states: Any, attention_mask: Any) -> Any:
        """
        Apply attention pooling over sequence.
        
        Args:
            hidden_states: IndicBERT hidden states (batch, seq_len, hidden)
            attention_mask: Attention mask (batch, seq_len)
            
        Returns:
            Pooled representation (batch, hidden)
        """
        # Compute attention scores
        attention_scores = self.attention_weights(hidden_states).squeeze(-1)
        
        # Mask padding tokens
        attention_scores = attention_scores.masked_fill(
            attention_mask == 0, float('-inf')
        )
        
        # Softmax over sequence
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Weighted sum
        pooled = torch.bmm(attention_probs.unsqueeze(1), hidden_states).squeeze(1)
        
        return pooled
    
    def forward(
        self,
        input_ids: Any,
        attention_mask: Any,
        token_type_ids: Optional[Any] = None,
        linguistic_features: Optional[Any] = None,
        labels: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            token_type_ids: Optional token type IDs
            linguistic_features: Optional linguistic features (batch, feat_dim)
            labels: Optional ground truth labels
            
        Returns:
            Dictionary with logits, loss (if labels provided), and predictions
        """
        # IndicBERT encoding
        indicbert_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        if token_type_ids is not None:
            indicbert_kwargs['token_type_ids'] = token_type_ids
        
        outputs = self.indicbert(**indicbert_kwargs)
        hidden_states = outputs.last_hidden_state
        
        # Apply attention pooling for better representation
        pooled_output = self.attention_pool(hidden_states, attention_mask)
        pooled_output = self.hidden_dropout(pooled_output)
        
        # Combine with linguistic features if available
        if self.use_linguistic_features and linguistic_features is not None:
            ling_proj = self.linguistic_projection(linguistic_features)
            combined = torch.cat([pooled_output, ling_proj], dim=-1)
        else:
            combined = pooled_output
        
        # Main classification
        logits = self.classifier(combined)
        
        # Severity prediction
        severity_score = self.severity_head(pooled_output)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {
            'logits': logits,
            'loss': loss,
            'severity_score': severity_score,
            'pooled_output': pooled_output
        }
    
    def predict(
        self,
        input_ids: Any,
        attention_mask: Any,
        token_type_ids: Optional[Any] = None,
        linguistic_features: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """
        Get predictions with probabilities.
        
        Returns:
            Tuple of (predicted_labels, probabilities)
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids, attention_mask, token_type_ids, linguistic_features
            )
            probs = F.softmax(outputs['logits'], dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            
        return predictions, probs


# =============================================================================
# Trainer Class
# =============================================================================

class IndicBertTrainer:
    """
    Trainer class for IndicBERT cyberbullying classifier.
    
    Handles training, validation, early stopping, and model saving.
    """
    
    def __init__(
        self,
        model: 'IndicBertCyberbullyingClassifier',
        config: Dict[str, Any],
        device: Optional[Any] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: IndicBERT classifier model
            config: Training configuration
            device: Device to use for training
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required. Install with: pip install torch")
        
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        
        # Training state
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }
    
    def train(
        self,
        train_loader: Any,
        val_loader: Any,
        class_weights: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            class_weights: Optional class weights for imbalanced data
            
        Returns:
            Training history dictionary
        """
        # Optimizer with weight decay
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config['weight_decay']
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config['learning_rate'])
        
        # Learning rate scheduler
        total_steps = len(train_loader) * self.config['epochs']
        warmup_steps = int(total_steps * self.config['warmup_ratio'])
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss function with class weights
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Starting IndicBERT training on {self.device}")
        logger.info(f"Total steps: {total_steps}, Warmup: {warmup_steps}")
        
        for epoch in range(self.config['epochs']):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_steps = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
            
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                token_type_ids = None
                if 'token_type_ids' in batch:
                    token_type_ids = batch['token_type_ids'].to(self.device)
                
                linguistic_features = None
                if 'linguistic_features' in batch:
                    linguistic_features = batch['linguistic_features'].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    linguistic_features=linguistic_features,
                    labels=labels
                )
                
                loss = criterion(outputs['logits'], labels)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['max_grad_norm']
                )
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                train_steps += 1
                
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / train_steps
            
            # Validation phase
            val_metrics = self.evaluate(val_loader)
            
            # Log metrics
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['val_accuracy'].append(val_metrics['accuracy'])
            self.training_history['val_f1'].append(val_metrics['f1_macro'])
            
            logger.info(
                f"Epoch {epoch+1}: "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Val F1: {val_metrics['f1_macro']:.4f}"
            )
            
            # Early stopping check
            if val_metrics['f1_macro'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1_macro']
                self.patience_counter = 0
                # Save best model
                self._save_checkpoint('indicbert_best_model.pt')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config['early_stopping_patience']:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        return self.training_history
    
    def evaluate(self, data_loader: Any) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader: DataLoader for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                token_type_ids = None
                if 'token_type_ids' in batch:
                    token_type_ids = batch['token_type_ids'].to(self.device)
                
                linguistic_features = None
                if 'linguistic_features' in batch:
                    linguistic_features = batch['linguistic_features'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    linguistic_features=linguistic_features
                )
                
                loss = criterion(outputs['logits'], labels)
                total_loss += loss.item()
                num_batches += 1
                
                preds = torch.argmax(outputs['logits'], dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return {
            'loss': total_loss / max(num_batches, 1),
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall,
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels)
        }
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        save_dir = PROJECT_ROOT / '03_models' / 'saved_models' / 'transformer'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'best_val_f1': self.best_val_f1
        }
        torch.save(checkpoint, save_dir / filename)
        logger.info(f"Saved checkpoint to {save_dir / filename}")


# =============================================================================
# Main Classifier Interface
# =============================================================================

class IndicBertMessageClassifier:
    """
    High-level interface for IndicBERT-based cyberbullying classification.
    
    Provides easy-to-use methods for training and inference.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize classifier.
        
        Args:
            config: Configuration dictionary (uses defaults if not provided)
        """
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "PyTorch and Transformers required. Install with:\n"
                "pip install torch transformers"
            )
        
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.preprocessor = None
        self.linguistic_extractor = None
        
        # Initialize preprocessing
        if self.config['use_preprocessing'] and TextNormalizer:
            self.preprocessor = TextNormalizer()
        
        # Initialize feature extraction
        if self.config['use_linguistic_features'] and LinguisticFeatures:
            self.linguistic_extractor = LinguisticFeatures()
        
        logger.info(f"Initialized IndicBertMessageClassifier on {self.device}")
    
    def load_data(
        self,
        data_path: Optional[str] = None,
        text_column: str = 'message',
        label_column: str = 'label'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and split dataset.
        
        Args:
            data_path: Path to data directory (default: project data folder)
            text_column: Name of text column
            label_column: Name of label column
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if data_path is None:
            data_path = PROJECT_ROOT / '00_data' / 'processed'
        else:
            data_path = Path(data_path)
        
        # Load pre-split data if available
        train_path = data_path / 'train_data.csv'
        val_path = data_path / 'val_data.csv'
        test_path = data_path / 'test_data.csv'
        
        if train_path.exists() and val_path.exists() and test_path.exists():
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            test_df = pd.read_csv(test_path)
            logger.info(f"Loaded pre-split data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        else:
            # Load all data and split
            all_data = pd.read_csv(train_path if train_path.exists() else data_path / 'data.csv')
            
            train_df, temp_df = train_test_split(
                all_data, test_size=0.2, stratify=all_data[label_column], random_state=42
            )
            val_df, test_df = train_test_split(
                temp_df, test_size=0.5, stratify=temp_df[label_column], random_state=42
            )
            logger.info(f"Split data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        # Fit label encoder on training data
        self.label_encoder.fit(train_df[label_column])
        self.config['labels'] = list(self.label_encoder.classes_)
        self.config['num_labels'] = len(self.config['labels'])
        
        logger.info(f"Labels: {self.config['labels']}")
        
        return train_df, val_df, test_df
    
    def prepare_dataloaders(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        text_column: str = 'message',
        label_column: str = 'label'
    ) -> Tuple[Any, Any, Any]:
        """
        Prepare PyTorch DataLoaders.
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe
            text_column: Text column name
            label_column: Label column name
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        
        # Encode labels
        train_labels = self.label_encoder.transform(train_df[label_column])
        val_labels = self.label_encoder.transform(val_df[label_column])
        test_labels = self.label_encoder.transform(test_df[label_column])
        
        # Create datasets
        train_dataset = IndicCyberbullyingDataset(
            texts=train_df[text_column].tolist(),
            labels=train_labels.tolist(),
            tokenizer=self.tokenizer,
            max_length=self.config['max_length'],
            preprocessor=self.preprocessor,
            extract_features=self.config['use_linguistic_features'],
            linguistic_extractor=self.linguistic_extractor
        )
        
        val_dataset = IndicCyberbullyingDataset(
            texts=val_df[text_column].tolist(),
            labels=val_labels.tolist(),
            tokenizer=self.tokenizer,
            max_length=self.config['max_length'],
            preprocessor=self.preprocessor,
            extract_features=self.config['use_linguistic_features'],
            linguistic_extractor=self.linguistic_extractor
        )
        
        test_dataset = IndicCyberbullyingDataset(
            texts=test_df[text_column].tolist(),
            labels=test_labels.tolist(),
            tokenizer=self.tokenizer,
            max_length=self.config['max_length'],
            preprocessor=self.preprocessor,
            extract_features=self.config['use_linguistic_features'],
            linguistic_extractor=self.linguistic_extractor
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader, test_loader
    
    def build_model(self) -> IndicBertCyberbullyingClassifier:
        """
        Build and initialize the model.
        
        Returns:
            Initialized model
        """
        self.model = IndicBertCyberbullyingClassifier(
            model_name=self.config['model_name'],
            num_labels=self.config['num_labels'],
            hidden_dropout=self.config['hidden_dropout'],
            classifier_dropout=self.config['classifier_dropout'],
            use_linguistic_features=self.config['use_linguistic_features'],
            linguistic_feature_dim=self.config['linguistic_feature_dim']
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"IndicBERT model built: {total_params:,} total params, {trainable_params:,} trainable")
        
        return self.model
    
    def compute_class_weights(self, labels: np.ndarray) -> Any:
        """
        Compute class weights for imbalanced data.
        
        Args:
            labels: Array of label indices
            
        Returns:
            Tensor of class weights
        """
        classes = np.unique(labels)
        weights = compute_class_weight('balanced', classes=classes, y=labels)
        return torch.tensor(weights, dtype=torch.float)
    
    def train(
        self,
        train_loader: Any,
        val_loader: Any,
        use_class_weights: bool = True
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            use_class_weights: Whether to use class weights
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Compute class weights if needed
        class_weights = None
        if use_class_weights:
            # Get all training labels
            all_labels = []
            for batch in train_loader:
                all_labels.extend(batch['labels'].numpy())
            class_weights = self.compute_class_weights(np.array(all_labels))
        
        # Initialize trainer
        trainer = IndicBertTrainer(self.model, self.config, self.device)
        
        # Train
        history = trainer.train(train_loader, val_loader, class_weights)
        
        return history
    
    def evaluate(self, test_loader: Any) -> Dict[str, Any]:
        """
        Evaluate model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Evaluation metrics
        """
        trainer = IndicBertTrainer(self.model, self.config, self.device)
        metrics = trainer.evaluate(test_loader)
        
        # Generate classification report
        report = classification_report(
            metrics['labels'],
            metrics['predictions'],
            target_names=self.config['labels'],
            zero_division=0
        )
        
        logger.info("\n" + "="*60)
        logger.info("INDICBERT EVALUATION RESULTS")
        logger.info("="*60)
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1 Macro: {metrics['f1_macro']:.4f}")
        logger.info(f"F1 Weighted: {metrics['f1_weighted']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info("\nClassification Report:")
        logger.info("\n" + report)
        
        metrics['classification_report'] = report
        
        return metrics
    
    def predict(self, texts: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Predict labels for new texts.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            List of prediction dictionaries
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Create dataset for prediction
        dataset = IndicCyberbullyingDataset(
            texts=texts,
            tokenizer=self.tokenizer,
            max_length=self.config['max_length'],
            preprocessor=self.preprocessor,
            extract_features=self.config['use_linguistic_features'],
            linguistic_extractor=self.linguistic_extractor
        )
        
        loader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                token_type_ids = None
                if 'token_type_ids' in batch:
                    token_type_ids = batch['token_type_ids'].to(self.device)
                
                linguistic_features = None
                if 'linguistic_features' in batch:
                    linguistic_features = batch['linguistic_features'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    linguistic_features=linguistic_features
                )
                
                probs = F.softmax(outputs['logits'], dim=-1)
                preds = torch.argmax(probs, dim=-1)
                severity = outputs['severity_score']
                
                for i in range(len(preds)):
                    pred_idx = preds[i].item()
                    pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
                    confidence = probs[i][pred_idx].item()
                    
                    all_predictions.append({
                        'label': pred_label,
                        'confidence': confidence,
                        'severity_score': severity[i].item(),
                        'all_probabilities': {
                            label: probs[i][j].item()
                            for j, label in enumerate(self.config['labels'])
                        }
                    })
        
        return all_predictions
    
    def save(self, save_path: Optional[str] = None):
        """
        Save the complete model.
        
        Args:
            save_path: Path to save directory
        """
        if save_path is None:
            save_path = PROJECT_ROOT / '03_models' / 'saved_models' / 'transformer' / 'indicbert_classifier'
        else:
            save_path = Path(save_path)
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), save_path / 'model.pt')
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path / 'tokenizer')
        
        # Save label encoder
        with open(save_path / 'label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save config
        with open(save_path / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"IndicBERT model saved to {save_path}")
    
    @classmethod
    def load(cls, load_path: str) -> 'IndicBertMessageClassifier':
        """
        Load a saved model.
        
        Args:
            load_path: Path to saved model directory
            
        Returns:
            Loaded classifier instance
        """
        load_path = Path(load_path)
        
        # Load config
        with open(load_path / 'config.json', 'r') as f:
            config = json.load(f)
        
        # Create instance
        instance = cls(config)
        
        # Load tokenizer
        instance.tokenizer = AutoTokenizer.from_pretrained(load_path / 'tokenizer')
        
        # Load label encoder
        with open(load_path / 'label_encoder.pkl', 'rb') as f:
            instance.label_encoder = pickle.load(f)
        
        # Build and load model
        instance.build_model()
        instance.model.load_state_dict(
            torch.load(load_path / 'model.pt', map_location=instance.device)
        )
        instance.model.to(instance.device)
        instance.model.eval()
        
        logger.info(f"IndicBERT model loaded from {load_path}")
        
        return instance


# =============================================================================
# Training Script
# =============================================================================

def train_indicbert_classifier(
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    model_name: str = 'ai4bharat/indic-bert'
) -> IndicBertMessageClassifier:
    """
    Train the IndicBERT classifier.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        model_name: Pre-trained model name
        
    Returns:
        Trained classifier
    """
    logger.info("="*60)
    logger.info("IndicBERT Cyberbullying Classifier Training")
    logger.info("="*60)
    
    # Configuration
    config = {
        **DEFAULT_CONFIG,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'model_name': model_name
    }
    
    # Initialize classifier
    classifier = IndicBertMessageClassifier(config)
    
    # Load data
    logger.info("\n1. Loading data...")
    train_df, val_df, test_df = classifier.load_data()
    
    # Prepare data loaders
    logger.info("\n2. Preparing data loaders...")
    train_loader, val_loader, test_loader = classifier.prepare_dataloaders(
        train_df, val_df, test_df
    )
    
    # Build model
    logger.info("\n3. Building IndicBERT model...")
    classifier.build_model()
    
    # Train
    logger.info("\n4. Training...")
    history = classifier.train(train_loader, val_loader)
    
    # Evaluate on test set
    logger.info("\n5. Evaluating on test set...")
    metrics = classifier.evaluate(test_loader)
    
    # Save model
    logger.info("\n6. Saving model...")
    classifier.save()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("INDICBERT TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Final Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Final Test F1 Macro: {metrics['f1_macro']:.4f}")
    logger.info(f"Model saved to: 03_models/saved_models/transformer/indicbert_classifier/")
    
    return classifier


# =============================================================================
# Quick Test / Demo
# =============================================================================

def demo_predictions():
    """Demonstrate predictions on sample texts."""
    sample_texts = [
        "nee tumba stupid tara idiya",  # Kannada insult
        "You are such a loser",  # English insult
        "Did you finish the assignment?",  # Neutral
        "I will make you pay for this",  # Threat
        "Nobody wants you here, leave",  # Exclusion
        "Good morning everyone! 😊",  # Neutral
        "Shut up you idiot 🤬",  # Aggression
        "ನೀನು ಬಹಳ ಕೆಟ್ಟವನು",  # Kannada: You are very bad
        "என்னை விட்டு போ",  # Tamil: Leave me alone
    ]
    
    logger.info("\n" + "="*60)
    logger.info("DEMO: IndicBERT Sample Predictions")
    logger.info("="*60)
    
    # Try to load saved model
    model_path = PROJECT_ROOT / '03_models' / 'saved_models' / 'transformer' / 'indicbert_classifier'
    
    if model_path.exists():
        classifier = IndicBertMessageClassifier.load(str(model_path))
    else:
        logger.info("No saved model found. Please train first using train_indicbert_classifier()")
        return
    
    predictions = classifier.predict(sample_texts)
    
    for text, pred in zip(sample_texts, predictions):
        logger.info(f"\nText: '{text}'")
        logger.info(f"  Prediction: {pred['label']} (confidence: {pred['confidence']:.2%})")
        logger.info(f"  Severity: {pred['severity_score']:.2f}")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='IndicBERT Cyberbullying Classifier')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--demo', action='store_true', help='Run demo predictions')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--model', type=str, default='ai4bharat/indic-bert',
                        help='Pre-trained model name')
    
    args = parser.parse_args()
    
    if args.train:
        train_indicbert_classifier(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            model_name=args.model
        )
    elif args.demo:
        demo_predictions()
    else:
        print("\nIndicBERT Cyberbullying Classifier")
        print("="*60)
        print("\nUsage:")
        print("  Train model: python indicbert_classifier.py --train")
        print("  Run demo:    python indicbert_classifier.py --demo")
        print("\nOptions:")
        print("  --epochs N      Number of training epochs (default: 10)")
        print("  --batch-size N  Batch size (default: 16)")
        print("  --lr RATE       Learning rate (default: 2e-5)")
        print("  --model NAME    Pre-trained model (default: ai4bharat/indic-bert)")
        print("\nIndicBERT is optimized for Indian languages including:")
        print("  - Kannada, Hindi, Tamil, Telugu, Malayalam")
        print("  - Bengali, Marathi, Gujarati, Punjabi, Odia")
        print("  - And 12 major Indian languages + English")
