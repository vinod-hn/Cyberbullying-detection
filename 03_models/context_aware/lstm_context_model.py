# LSTM Context Model
"""
LSTM-based Context-Aware Model for Cyberbullying Detection

Implements bidirectional LSTM with attention for sequential context modeling:
- BiLSTM for capturing bidirectional context
- Attention mechanism for focusing on relevant parts
- Context encoding from conversation history
- Hierarchical structure for word→message→conversation

Optimized for Kannada-English code-mixed cyberbullying detection.

Author: Cyberbullying Detection Project Team
"""

import os
import sys
import logging
import pickle
from typing import List, Dict, Optional, Any, Union, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Deep learning imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Note: PyTorch features are optional for this module

# Import attention mechanisms
try:
    from attention_mechanism import (
        MultiHeadAttention, SelfAttention, AttentionPooling,
        create_padding_mask
    )
    ATTENTION_AVAILABLE = True
except ImportError:
    ATTENTION_AVAILABLE = False
    # Note: Attention mechanisms are optional


# =============================================================================
# LSTM Components
# =============================================================================

if TORCH_AVAILABLE:
    class BiLSTMEncoder(nn.Module):
        """
        Bidirectional LSTM encoder.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        
        def __init__(
            self,
            input_dim: int = 128,
            hidden_dim: int = 256,
            num_layers: int = 2,
            dropout: float = 0.3
        ):
            super().__init__()
            
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(
                input_dim,
                hidden_dim // 2,  # Divide by 2 because bidirectional
                num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0
            )
            
            self.dropout = nn.Dropout(dropout)
        
        def forward(
            self,
            x: Tensor,
            lengths: Optional[Tensor] = None
        ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
            """
            Forward pass.
            
            Args:
                x: Input tensor [batch, seq_len, input_dim]
                lengths: Sequence lengths for packing
                
            Returns:
                Tuple of (output, (hidden, cell))
            """
            if lengths is not None:
                # Pack padded sequence for efficiency
                x = pack_padded_sequence(
                    x, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
            
            output, (hidden, cell) = self.lstm(x)
            
            if lengths is not None:
                # Unpack sequence
                output, _ = pad_packed_sequence(output, batch_first=True)
            
            output = self.dropout(output)
            
            return output, (hidden, cell)


    class AttentionLSTM(nn.Module):
        """
        LSTM with attention mechanism.
        
        Combines BiLSTM encoding with attention pooling.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        
        def __init__(
            self,
            input_dim: int = 128,
            hidden_dim: int = 256,
            output_dim: int = 128,
            num_layers: int = 2,
            dropout: float = 0.3
        ):
            super().__init__()
            
            self.encoder = BiLSTMEncoder(input_dim, hidden_dim, num_layers, dropout)
            
            if ATTENTION_AVAILABLE:
                self.attention = AttentionPooling(hidden_dim, dropout)
            else:
                self.attention = None
            
            self.output_proj = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout)
        
        def forward(
            self,
            x: Tensor,
            lengths: Optional[Tensor] = None,
            mask: Optional[Tensor] = None
        ) -> Tuple[Tensor, Optional[Tensor]]:
            """
            Forward pass.
            
            Args:
                x: Input tensor [batch, seq_len, input_dim]
                lengths: Sequence lengths
                mask: Attention mask
                
            Returns:
                Tuple of (output, attention_weights)
            """
            # Encode with BiLSTM
            lstm_out, _ = self.encoder(x, lengths)
            
            # Apply attention if available
            if self.attention is not None:
                pooled, attn_weights = self.attention(lstm_out, mask)
            else:
                # Simple mean pooling
                if mask is not None:
                    mask = mask.unsqueeze(-1)
                    lstm_out = lstm_out * mask
                    pooled = lstm_out.sum(dim=1) / mask.sum(dim=1)
                else:
                    pooled = lstm_out.mean(dim=1)
                attn_weights = None
            
            # Project to output dimension
            output = self.output_proj(pooled)
            output = self.dropout(output)
            
            return output, attn_weights


    class HierarchicalLSTM(nn.Module):
        """
        Hierarchical LSTM for conversation modeling.
        
        Two-level hierarchy:
        1. Word-level LSTM: Encodes each message
        2. Message-level LSTM: Encodes conversation
        
        Args:
            word_embedding_dim: Word embedding dimension
            word_hidden_dim: Word-level LSTM hidden dimension
            message_hidden_dim: Message-level LSTM hidden dimension
            output_dim: Output dimension
            dropout: Dropout probability
        """
        
        def __init__(
            self,
            word_embedding_dim: int = 128,
            word_hidden_dim: int = 256,
            message_hidden_dim: int = 512,
            output_dim: int = 256,
            dropout: float = 0.3
        ):
            super().__init__()
            
            # Word-level encoder
            self.word_encoder = AttentionLSTM(
                input_dim=word_embedding_dim,
                hidden_dim=word_hidden_dim,
                output_dim=word_hidden_dim,
                num_layers=1,
                dropout=dropout
            )
            
            # Message-level encoder
            self.message_encoder = AttentionLSTM(
                input_dim=word_hidden_dim,
                hidden_dim=message_hidden_dim,
                output_dim=output_dim,
                num_layers=2,
                dropout=dropout
            )
        
        def forward(
            self,
            messages: Tensor,
            word_lengths: Optional[Tensor] = None,
            message_mask: Optional[Tensor] = None
        ) -> Tuple[Tensor, Dict[str, Any]]:
            """
            Forward pass.
            
            Args:
                messages: Message tensor [batch, num_messages, num_words, word_dim]
                word_lengths: Word lengths for each message
                message_mask: Message mask
                
            Returns:
                Tuple of (conversation representation, attention dict)
            """
            batch_size, num_messages, num_words, word_dim = messages.size()
            
            # Encode each message with word-level LSTM
            message_reps = []
            word_attentions = []
            
            for i in range(num_messages):
                message_words = messages[:, i, :, :]  # [batch, num_words, word_dim]
                
                word_lens = word_lengths[:, i] if word_lengths is not None else None
                
                message_rep, word_attn = self.word_encoder(message_words, word_lens)
                message_reps.append(message_rep)
                if word_attn is not None:
                    word_attentions.append(word_attn)
            
            # Stack message representations
            message_sequence = torch.stack(message_reps, dim=1)  # [batch, num_messages, word_hidden_dim]
            
            # Encode conversation with message-level LSTM
            conversation_rep, message_attn = self.message_encoder(
                message_sequence, mask=message_mask
            )
            
            attention_dict = {
                'word_level': word_attentions if word_attentions else None,
                'message_level': message_attn
            }
            
            return conversation_rep, attention_dict


# =============================================================================
# Context-Aware LSTM Model
# =============================================================================

if TORCH_AVAILABLE:
    class LSTMContextModel(nn.Module):
        """
        LSTM-based context-aware model for cyberbullying detection.
        
        Combines:
        - BiLSTM for message encoding
        - Context encoder for conversation history
        - Additional feature integration
        - Multi-task learning (classification + severity)
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension
            num_classes: Number of output classes
            num_additional_features: Number of additional features
            dropout: Dropout probability
            pretrained_embeddings: Optional pretrained embeddings
        """
        
        def __init__(
            self,
            vocab_size: int = 10000,
            embedding_dim: int = 128,
            hidden_dim: int = 256,
            num_classes: int = 11,
            num_additional_features: int = 0,
            dropout: float = 0.3,
            pretrained_embeddings: Optional[np.ndarray] = None
        ):
            super().__init__()
            
            self.vocab_size = vocab_size
            self.embedding_dim = embedding_dim
            self.hidden_dim = hidden_dim
            self.num_classes = num_classes
            
            # Embedding layer
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            if pretrained_embeddings is not None:
                self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            
            # Message encoder
            self.message_encoder = AttentionLSTM(
                input_dim=embedding_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim // 2,
                num_layers=2,
                dropout=dropout
            )
            
            # Context encoder (for previous messages)
            self.context_encoder = AttentionLSTM(
                input_dim=embedding_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim // 2,
                num_layers=1,
                dropout=dropout
            )
            
            # Combine message and context
            combined_dim = hidden_dim // 2 + hidden_dim // 2
            if num_additional_features > 0:
                combined_dim += num_additional_features
            
            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(combined_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes)
            )
            
            # Severity prediction head
            self.severity_head = nn.Sequential(
                nn.Linear(combined_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            
            self.dropout = nn.Dropout(dropout)
        
        def forward(
            self,
            message_ids: Tensor,
            context_ids: Optional[Tensor] = None,
            additional_features: Optional[Tensor] = None,
            message_lengths: Optional[Tensor] = None,
            context_lengths: Optional[Tensor] = None
        ) -> Dict[str, Tensor]:
            """
            Forward pass.
            
            Args:
                message_ids: Message token IDs [batch, max_msg_len]
                context_ids: Context token IDs [batch, max_ctx_len]
                additional_features: Additional features [batch, num_features]
                message_lengths: Message lengths
                context_lengths: Context lengths
                
            Returns:
                Dictionary with logits, probabilities, severity
            """
            # Embed message
            message_emb = self.embedding(message_ids)  # [batch, max_msg_len, emb_dim]
            message_emb = self.dropout(message_emb)
            
            # Encode message
            message_rep, msg_attn = self.message_encoder(
                message_emb, message_lengths
            )
            
            # Encode context if provided
            if context_ids is not None:
                context_emb = self.embedding(context_ids)
                context_emb = self.dropout(context_emb)
                context_rep, ctx_attn = self.context_encoder(
                    context_emb, context_lengths
                )
                # Combine message and context
                combined = torch.cat([message_rep, context_rep], dim=-1)
            else:
                combined = message_rep
                ctx_attn = None
            
            # Add additional features if provided
            if additional_features is not None:
                combined = torch.cat([combined, additional_features], dim=-1)
            
            # Classification
            logits = self.classifier(combined)
            probs = F.softmax(logits, dim=-1)
            
            # Severity prediction
            severity = self.severity_head(combined)
            
            return {
                'logits': logits,
                'probabilities': probs,
                'severity': severity,
                'message_attention': msg_attn,
                'context_attention': ctx_attn
            }
        
        def predict(self, *args, **kwargs) -> Dict[str, Any]:
            """Prediction method for inference."""
            self.eval()
            with torch.no_grad():
                outputs = self.forward(*args, **kwargs)
            
            # Get predicted class
            pred_class = outputs['logits'].argmax(dim=-1)
            confidence = outputs['probabilities'].max(dim=-1)[0]
            
            return {
                'predicted_class': pred_class,
                'confidence': confidence,
                'severity': outputs['severity'],
                'all_probabilities': outputs['probabilities']
            }


# =============================================================================
# Model Trainer
# =============================================================================

class LSTMContextTrainer:
    """
    Trainer for LSTM context model.
    """
    
    def __init__(
        self,
        model: 'LSTMContextModel',
        device: str = 'cpu',
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: LSTM context model
            device: Device to use ('cpu' or 'cuda')
            logger: Logger instance
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for training")
        
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.logger = logger or logging.getLogger(__name__)
    
    def train_step(
        self,
        batch: Dict[str, Tensor],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        # Move batch to device
        for key in batch:
            if isinstance(batch[key], Tensor):
                batch[key] = batch[key].to(self.device)
        
        # Forward pass
        outputs = self.model(
            batch['message_ids'],
            batch.get('context_ids'),
            batch.get('additional_features'),
            batch.get('message_lengths'),
            batch.get('context_lengths')
        )
        
        # Compute losses
        class_loss = criterion(outputs['logits'], batch['labels'])
        
        # Severity loss if labels provided
        if 'severity' in batch:
            severity_loss = F.mse_loss(
                outputs['severity'].squeeze(-1),
                batch['severity']
            )
            total_loss = class_loss + 0.3 * severity_loss
        else:
            severity_loss = 0.0
            total_loss = class_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'class_loss': class_loss.item(),
            'severity_loss': severity_loss if isinstance(severity_loss, float) else severity_loss.item()
        }
    
    def evaluate(
        self,
        dataloader,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                for key in batch:
                    if isinstance(batch[key], Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    batch['message_ids'],
                    batch.get('context_ids'),
                    batch.get('additional_features'),
                    batch.get('message_lengths'),
                    batch.get('context_lengths')
                )
                
                # Compute loss
                loss = criterion(outputs['logits'], batch['labels'])
                total_loss += loss.item()
                
                # Collect predictions
                preds = outputs['logits'].argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        # Compute metrics
        try:
            from sklearn.metrics import accuracy_score, f1_score  # type: ignore
            
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='weighted')
        except ImportError:
            accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
            f1 = 0.0
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy,
            'f1_weighted': f1
        }


# =============================================================================
# Utility Functions
# =============================================================================

def save_lstm_model(
    model: 'LSTMContextModel',
    save_path: str,
    vocab: Optional[Dict] = None,
    config: Optional[Dict] = None
) -> None:
    """Save LSTM model to disk."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': model.vocab_size,
            'embedding_dim': model.embedding_dim,
            'hidden_dim': model.hidden_dim,
            'num_classes': model.num_classes
        },
        'vocab': vocab,
        'config': config
    }
    
    torch.save(save_dict, save_path)
    logger.info(f"Model saved to: {save_path}")


def load_lstm_model(load_path: str, device: str = 'cpu') -> 'LSTMContextModel':
    """Load LSTM model from disk."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")
    
    checkpoint = torch.load(load_path, map_location=device)
    
    model_config = checkpoint['model_config']
    model = LSTMContextModel(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    logger.info(f"Model loaded from: {load_path}")
    return model


# =============================================================================
# Demo
# =============================================================================

def demo_lstm_context():
    """Demo LSTM context model."""
    if not TORCH_AVAILABLE:
        print("\nNote: PyTorch is required for LSTM models.")
        print("Install with: pip install torch")
        print("\nSkipping demo - PyTorch features are optional for baseline models.")
        return
    
    print("\n" + "="*60)
    print("LSTM CONTEXT MODEL DEMO")
    print("="*60)
    
    # Model parameters
    vocab_size = 10000
    embedding_dim = 128
    hidden_dim = 256
    num_classes = 11
    
    # Create model
    model = LSTMContextModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    )
    
    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Sample input
    batch_size = 4
    max_msg_len = 20
    max_ctx_len = 50
    
    message_ids = torch.randint(1, vocab_size, (batch_size, max_msg_len))
    context_ids = torch.randint(1, vocab_size, (batch_size, max_ctx_len))
    
    # Forward pass
    outputs = model(message_ids, context_ids)
    
    print(f"\nInput shapes:")
    print(f"  Message IDs: {message_ids.shape}")
    print(f"  Context IDs: {context_ids.shape}")
    print(f"\nOutput shapes:")
    print(f"  Logits: {outputs['logits'].shape}")
    print(f"  Probabilities: {outputs['probabilities'].shape}")
    print(f"  Severity: {outputs['severity'].shape}")
    
    print("\n" + "="*60)
    print("Demo complete!")


if __name__ == "__main__":
    if TORCH_AVAILABLE:
        demo_lstm_context()
    else:
        print("="*60)
        print("LSTM CONTEXT MODEL MODULE")
        print("="*60)
        print("\nStatus: PyTorch not installed")
        print("\nThis module provides LSTM-based context-aware models.")
        print("PyTorch is optional - baseline models work without it.")
        print("\nTo use LSTM models, install PyTorch:")
        print("  pip install torch")
        print("\nFor CPU-only (smaller):")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("="*60)
