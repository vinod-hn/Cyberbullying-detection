# Conversation Transformer
"""
Transformer-based Context-Aware Model for Cyberbullying Detection

Implements Transformer architecture for conversation-level cyberbullying detection:
- Multi-head self-attention for capturing dependencies
- Positional encoding for sequence information
- Hierarchical encoding (word → message → conversation)
- Context integration from conversation history

Optimized for Kannada-English code-mixed cyberbullying detection.

Author: Cyberbullying Detection Project Team
"""

import os
import sys
import logging
import math
from typing import List, Dict, Optional, Any, Union, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Deep learning imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Note: PyTorch features are optional for this module

# Import attention mechanisms
try:
    from attention_mechanism import (
        MultiHeadAttention, SelfAttention, create_padding_mask, create_causal_mask
    )
    ATTENTION_AVAILABLE = True
except ImportError:
    ATTENTION_AVAILABLE = False


# =============================================================================
# Positional Encoding
# =============================================================================

if TORCH_AVAILABLE:
    class PositionalEncoding(nn.Module):
        """
        Positional encoding for Transformer.
        
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        
        def __init__(self, d_model: int = 256, max_len: int = 512, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            
            # Create positional encoding matrix
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # [1, max_len, d_model]
            
            self.register_buffer('pe', pe)
        
        def forward(self, x: Tensor) -> Tensor:
            """
            Add positional encoding to input.
            
            Args:
                x: Input tensor [batch, seq_len, d_model]
                
            Returns:
                Tensor with positional encoding added
            """
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)


# =============================================================================
# Transformer Encoder
# =============================================================================

if TORCH_AVAILABLE:
    class TransformerEncoderLayer(nn.Module):
        """
        Single Transformer encoder layer.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
        """
        
        def __init__(
            self,
            d_model: int = 256,
            num_heads: int = 8,
            d_ff: int = 1024,
            dropout: float = 0.1
        ):
            super().__init__()
            
            # Multi-head attention
            if ATTENTION_AVAILABLE:
                self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
            else:
                self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
            
            # Feed-forward network
            self.feed_forward = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model)
            )
            
            # Layer normalization
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            
            self.dropout = nn.Dropout(dropout)
        
        def forward(
            self,
            x: Tensor,
            mask: Optional[Tensor] = None
        ) -> Tuple[Tensor, Optional[Tensor]]:
            """
            Forward pass.
            
            Args:
                x: Input tensor [batch, seq_len, d_model]
                mask: Attention mask
                
            Returns:
                Tuple of (output, attention weights)
            """
            # Self-attention with residual connection
            if ATTENTION_AVAILABLE:
                attn_out, attn_weights = self.self_attn(x, x, x, mask)
                x = self.norm1(x + attn_out)
            else:
                attn_out, attn_weights = self.self_attn(x, x, x, key_padding_mask=mask)
                x = self.norm1(x + self.dropout(attn_out))
            
            # Feed-forward with residual connection
            ff_out = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_out))
            
            return x, attn_weights


    class TransformerEncoder(nn.Module):
        """
        Transformer encoder stack.
        
        Args:
            num_layers: Number of encoder layers
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
        """
        
        def __init__(
            self,
            num_layers: int = 6,
            d_model: int = 256,
            num_heads: int = 8,
            d_ff: int = 1024,
            dropout: float = 0.1
        ):
            super().__init__()
            
            self.layers = nn.ModuleList([
                TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ])
            
            self.norm = nn.LayerNorm(d_model)
        
        def forward(
            self,
            x: Tensor,
            mask: Optional[Tensor] = None
        ) -> Tuple[Tensor, List[Tensor]]:
            """
            Forward pass through all encoder layers.
            
            Args:
                x: Input tensor [batch, seq_len, d_model]
                mask: Attention mask
                
            Returns:
                Tuple of (output, list of attention weights)
            """
            attention_weights = []
            
            for layer in self.layers:
                x, attn = layer(x, mask)
                if attn is not None:
                    attention_weights.append(attn)
            
            x = self.norm(x)
            
            return x, attention_weights


# =============================================================================
# Conversation Transformer Model
# =============================================================================

if TORCH_AVAILABLE:
    class ConversationTransformer(nn.Module):
        """
        Transformer-based model for conversation-level cyberbullying detection.
        
        Hierarchical architecture:
        1. Message encoder: Encodes individual messages
        2. Conversation encoder: Encodes message sequence
        3. Classification head: Predicts cyberbullying label and severity
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            num_heads: Number of attention heads
            num_encoder_layers: Number of encoder layers
            d_ff: Feed-forward dimension
            num_classes: Number of output classes
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            pretrained_embeddings: Optional pretrained embeddings
        """
        
        def __init__(
            self,
            vocab_size: int = 10000,
            d_model: int = 256,
            num_heads: int = 8,
            num_encoder_layers: int = 6,
            d_ff: int = 1024,
            num_classes: int = 11,
            max_seq_len: int = 512,
            dropout: float = 0.1,
            pretrained_embeddings: Optional[np.ndarray] = None
        ):
            super().__init__()
            
            self.vocab_size = vocab_size
            self.d_model = d_model
            self.num_classes = num_classes
            
            # Embedding layer
            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
            if pretrained_embeddings is not None:
                self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            
            # Positional encoding
            self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
            
            # Transformer encoder
            self.encoder = TransformerEncoder(
                num_layers=num_encoder_layers,
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            
            # Pooling
            self.pooling = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Tanh()
            )
            
            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_classes)
            )
            
            # Severity prediction head
            self.severity_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid()
            )
            
            self.dropout = nn.Dropout(dropout)
            
            # Initialize weights
            self._init_weights()
        
        def _init_weights(self):
            """Initialize model weights."""
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        
        def forward(
            self,
            input_ids: Tensor,
            attention_mask: Optional[Tensor] = None
        ) -> Dict[str, Tensor]:
            """
            Forward pass.
            
            Args:
                input_ids: Input token IDs [batch, seq_len]
                attention_mask: Attention mask [batch, seq_len]
                
            Returns:
                Dictionary with logits, probabilities, severity, and attention
            """
            # Embedding and positional encoding
            x = self.embedding(input_ids) * math.sqrt(self.d_model)
            x = self.pos_encoder(x)
            
            # Prepare mask for attention
            if attention_mask is not None:
                # Convert to format expected by attention
                # 1 = attend, 0 = mask
                attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            else:
                attn_mask = None
            
            # Transformer encoding
            encoded, attention_weights = self.encoder(x, attn_mask)
            
            # Pooling (use [CLS] token or mean pooling)
            if attention_mask is not None:
                # Mean pooling with mask
                mask_expanded = attention_mask.unsqueeze(-1).expand(encoded.size())
                sum_embeddings = torch.sum(encoded * mask_expanded, dim=1)
                sum_mask = mask_expanded.sum(dim=1)
                pooled = sum_embeddings / sum_mask.clamp(min=1e-9)
            else:
                pooled = encoded.mean(dim=1)
            
            pooled = self.pooling(pooled)
            
            # Classification
            logits = self.classifier(pooled)
            probs = F.softmax(logits, dim=-1)
            
            # Severity prediction
            severity = self.severity_head(pooled)
            
            return {
                'logits': logits,
                'probabilities': probs,
                'severity': severity,
                'attention_weights': attention_weights,
                'encoded': encoded
            }
        
        def predict(self, *args, **kwargs) -> Dict[str, Any]:
            """Prediction method for inference."""
            self.eval()
            with torch.no_grad():
                outputs = self.forward(*args, **kwargs)
            
            pred_class = outputs['logits'].argmax(dim=-1)
            confidence = outputs['probabilities'].max(dim=-1)[0]
            
            return {
                'predicted_class': pred_class,
                'confidence': confidence,
                'severity': outputs['severity'],
                'all_probabilities': outputs['probabilities']
            }


# =============================================================================
# Hierarchical Conversation Transformer
# =============================================================================

if TORCH_AVAILABLE:
    class HierarchicalTransformer(nn.Module):
        """
        Hierarchical Transformer for multi-message conversations.
        
        Two-level hierarchy:
        1. Message Transformer: Encodes individual messages
        2. Conversation Transformer: Encodes message sequence
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            num_heads: Number of attention heads
            num_message_layers: Number of message encoder layers
            num_conversation_layers: Number of conversation encoder layers
            d_ff: Feed-forward dimension
            num_classes: Number of output classes
            max_msg_len: Maximum message length
            max_conversation_len: Maximum conversation length
            dropout: Dropout probability
        """
        
        def __init__(
            self,
            vocab_size: int = 10000,
            d_model: int = 256,
            num_heads: int = 8,
            num_message_layers: int = 3,
            num_conversation_layers: int = 3,
            d_ff: int = 1024,
            num_classes: int = 11,
            max_msg_len: int = 100,
            max_conversation_len: int = 50,
            dropout: float = 0.1
        ):
            super().__init__()
            
            # Embedding
            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
            
            # Message-level encoder
            self.message_pos = PositionalEncoding(d_model, max_msg_len, dropout)
            self.message_encoder = TransformerEncoder(
                num_layers=num_message_layers,
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            
            # Conversation-level encoder
            self.conversation_pos = PositionalEncoding(d_model, max_conversation_len, dropout)
            self.conversation_encoder = TransformerEncoder(
                num_layers=num_conversation_layers,
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            
            # Classification
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_classes)
            )
            
            self.severity_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid()
            )
        
        def encode_messages(
            self,
            message_ids: Tensor,
            message_masks: Optional[Tensor] = None
        ) -> Tensor:
            """
            Encode multiple messages.
            
            Args:
                message_ids: [batch, num_messages, max_msg_len]
                message_masks: [batch, num_messages, max_msg_len]
                
            Returns:
                Message representations [batch, num_messages, d_model]
            """
            batch_size, num_messages, max_msg_len = message_ids.size()
            
            # Reshape to encode all messages together
            message_ids = message_ids.view(batch_size * num_messages, max_msg_len)
            
            # Embed and encode
            x = self.embedding(message_ids) * math.sqrt(self.d_model)
            x = self.message_pos(x)
            
            if message_masks is not None:
                message_masks = message_masks.view(batch_size * num_messages, max_msg_len)
                attn_mask = message_masks.unsqueeze(1).unsqueeze(2)
            else:
                attn_mask = None
            
            encoded, _ = self.message_encoder(x, attn_mask)
            
            # Pool each message representation
            if message_masks is not None:
                mask_expanded = message_masks.unsqueeze(-1).expand(encoded.size())
                sum_emb = torch.sum(encoded * mask_expanded, dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                pooled = sum_emb / sum_mask
            else:
                pooled = encoded.mean(dim=1)
            
            # Reshape back to conversations
            message_reps = pooled.view(batch_size, num_messages, self.d_model)
            
            return message_reps
        
        def forward(
            self,
            message_ids: Tensor,
            message_masks: Optional[Tensor] = None,
            conversation_mask: Optional[Tensor] = None
        ) -> Dict[str, Tensor]:
            """
            Forward pass.
            
            Args:
                message_ids: [batch, num_messages, max_msg_len]
                message_masks: [batch, num_messages, max_msg_len]
                conversation_mask: [batch, num_messages]
                
            Returns:
                Dictionary with outputs
            """
            # Encode messages
            message_reps = self.encode_messages(message_ids, message_masks)
            
            # Add positional encoding for conversation
            conversation_seq = self.conversation_pos(message_reps)
            
            # Encode conversation
            if conversation_mask is not None:
                conv_mask = conversation_mask.unsqueeze(1).unsqueeze(2)
            else:
                conv_mask = None
            
            conv_encoded, attn_weights = self.conversation_encoder(
                conversation_seq, conv_mask
            )
            
            # Pool conversation
            if conversation_mask is not None:
                mask_expanded = conversation_mask.unsqueeze(-1).expand(conv_encoded.size())
                sum_emb = torch.sum(conv_encoded * mask_expanded, dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                pooled = sum_emb / sum_mask
            else:
                pooled = conv_encoded.mean(dim=1)
            
            # Classification
            logits = self.classifier(pooled)
            probs = F.softmax(logits, dim=-1)
            severity = self.severity_head(pooled)
            
            return {
                'logits': logits,
                'probabilities': probs,
                'severity': severity,
                'attention_weights': attn_weights
            }


# =============================================================================
# Utility Functions
# =============================================================================

def save_transformer_model(
    model: Union['ConversationTransformer', 'HierarchicalTransformer'],
    save_path: str,
    vocab: Optional[Dict] = None,
    config: Optional[Dict] = None
) -> None:
    """Save Transformer model."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': model.vocab_size,
            'd_model': model.d_model if hasattr(model, 'd_model') else None,
            'num_classes': model.num_classes
        },
        'vocab': vocab,
        'config': config
    }
    
    torch.save(save_dict, save_path)
    logger.info(f"Model saved to: {save_path}")


def load_transformer_model(
    load_path: str,
    model_class: str = 'ConversationTransformer',
    device: str = 'cpu'
) -> Union['ConversationTransformer', 'HierarchicalTransformer']:
    """Load Transformer model."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")
    
    checkpoint = torch.load(load_path, map_location=device)
    model_config = checkpoint['model_config']
    
    if model_class == 'ConversationTransformer':
        model = ConversationTransformer(**model_config)
    else:
        model = HierarchicalTransformer(**model_config)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    logger.info(f"Model loaded from: {load_path}")
    return model


# =============================================================================
# Demo
# =============================================================================

def demo_conversation_transformer():
    """Demo Conversation Transformer."""
    if not TORCH_AVAILABLE:
        print("\nNote: PyTorch is required for Transformer models.")
        print("Install with: pip install torch")
        print("\nSkipping demo - PyTorch features are optional for baseline models.")
        return
    
    print("\n" + "="*60)
    print("CONVERSATION TRANSFORMER DEMO")
    print("="*60)
    
    # Parameters
    vocab_size = 10000
    d_model = 256
    num_classes = 11
    batch_size = 2
    seq_len = 50
    
    # Create model
    model = ConversationTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_classes=num_classes
    )
    
    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Sample input
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    outputs = model(input_ids, attention_mask)
    
    print(f"\nInput: {input_ids.shape}")
    print(f"Logits: {outputs['logits'].shape}")
    print(f"Probabilities: {outputs['probabilities'].shape}")
    print(f"Severity: {outputs['severity'].shape}")
    
    print("\n" + "="*60)
    print("Demo complete!")


if __name__ == "__main__":
    if TORCH_AVAILABLE:
        demo_conversation_transformer()
    else:
        print("="*60)
        print("CONVERSATION TRANSFORMER MODULE")
        print("="*60)
        print("\nStatus: PyTorch not installed")
        print("\nThis module provides Transformer-based conversation models.")
        print("PyTorch is optional - baseline models work without it.")
        print("\nTo use Transformer models, install PyTorch:")
        print("  pip install torch")
        print("\nFor CPU-only (smaller):")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("="*60)

