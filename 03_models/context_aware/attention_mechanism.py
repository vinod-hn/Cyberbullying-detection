# Attention Mechanism
"""
Attention Mechanism for Context-Aware Cyberbullying Detection

Implements various attention mechanisms for focusing on relevant parts of:
- Message sequences (temporal attention)
- Word sequences (word-level attention)
- Context features (feature attention)
- Multi-head attention for diverse patterns

Supports:
- Scaled Dot-Product Attention
- Multi-Head Attention
- Self-Attention
- Context Attention (message-to-context)
- Hierarchical Attention (word → message → conversation)

Optimized for Kannada-English code-mixed cyberbullying detection.

Author: Cyberbullying Detection Project Team
"""

import os
import sys
import logging
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
    # Install with: pip install torch (if needed)


# =============================================================================
# Attention Layers
# =============================================================================

if TORCH_AVAILABLE:
    class ScaledDotProductAttention(nn.Module):
        """
        Scaled Dot-Product Attention.
        
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        
        Args:
            dropout: Dropout probability
        """
        
        def __init__(self, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
        
        def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            mask: Optional[Tensor] = None
        ) -> Tuple[Tensor, Tensor]:
            """
            Forward pass.
            
            Args:
                query: Query tensor [batch, heads, seq_len, d_k]
                key: Key tensor [batch, heads, seq_len, d_k]
                value: Value tensor [batch, heads, seq_len, d_v]
                mask: Optional mask tensor
                
            Returns:
                Tuple of (attention output, attention weights)
            """
            d_k = query.size(-1)
            
            # Compute attention scores: QK^T / sqrt(d_k)
            scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
            
            # Apply mask if provided (set masked positions to -inf)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            # Apply softmax to get attention weights
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            # Apply attention weights to values
            output = torch.matmul(attention_weights, value)
            
            return output, attention_weights


    class MultiHeadAttention(nn.Module):
        """
        Multi-Head Attention mechanism.
        
        Allows the model to jointly attend to information from different
        representation subspaces at different positions.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        
        def __init__(
            self,
            d_model: int = 256,
            num_heads: int = 8,
            dropout: float = 0.1
        ):
            super().__init__()
            
            assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
            
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            
            # Linear projections for Q, K, V
            self.w_q = nn.Linear(d_model, d_model)
            self.w_k = nn.Linear(d_model, d_model)
            self.w_v = nn.Linear(d_model, d_model)
            
            # Output projection
            self.w_o = nn.Linear(d_model, d_model)
            
            # Attention
            self.attention = ScaledDotProductAttention(dropout)
            self.dropout = nn.Dropout(dropout)
            
            # Layer norm
            self.layer_norm = nn.LayerNorm(d_model)
        
        def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            mask: Optional[Tensor] = None
        ) -> Tuple[Tensor, Tensor]:
            """
            Forward pass.
            
            Args:
                query: Query tensor [batch, seq_len, d_model]
                key: Key tensor [batch, seq_len, d_model]
                value: Value tensor [batch, seq_len, d_model]
                mask: Optional mask tensor
                
            Returns:
                Tuple of (output, attention weights)
            """
            batch_size = query.size(0)
            residual = query
            
            # Linear projections and split into multiple heads
            # [batch, seq_len, d_model] -> [batch, num_heads, seq_len, d_k]
            q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            k = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            v = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            
            # Apply attention
            x, attention_weights = self.attention(q, k, v, mask)
            
            # Concatenate heads
            # [batch, num_heads, seq_len, d_k] -> [batch, seq_len, d_model]
            x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
            
            # Output projection
            x = self.w_o(x)
            x = self.dropout(x)
            
            # Residual connection and layer norm
            x = self.layer_norm(x + residual)
            
            return x, attention_weights


    class SelfAttention(nn.Module):
        """
        Self-Attention layer.
        
        Applies attention where query, key, and value all come from the same input.
        Useful for capturing dependencies within a single sequence.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        
        def __init__(
            self,
            d_model: int = 256,
            num_heads: int = 8,
            dropout: float = 0.1
        ):
            super().__init__()
            self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
            """
            Forward pass.
            
            Args:
                x: Input tensor [batch, seq_len, d_model]
                mask: Optional mask tensor
                
            Returns:
                Tuple of (output, attention weights)
            """
            return self.attention(x, x, x, mask)


    class ContextAttention(nn.Module):
        """
        Context Attention mechanism.
        
        Allows messages to attend to context information (previous messages,
        user history, conversation context).
        
        Args:
            d_model: Model dimension
            d_context: Context dimension
            dropout: Dropout probability
        """
        
        def __init__(
            self,
            d_model: int = 256,
            d_context: int = 128,
            dropout: float = 0.1
        ):
            super().__init__()
            
            self.d_model = d_model
            self.d_context = d_context
            
            # Project context to match model dimension
            self.context_proj = nn.Linear(d_context, d_model)
            
            # Attention scoring
            self.w_query = nn.Linear(d_model, d_model)
            self.w_key = nn.Linear(d_model, d_model)
            self.w_value = nn.Linear(d_model, d_model)
            
            # Output
            self.w_out = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(d_model)
        
        def forward(
            self,
            message: Tensor,
            context: Tensor,
            mask: Optional[Tensor] = None
        ) -> Tuple[Tensor, Tensor]:
            """
            Forward pass.
            
            Args:
                message: Message tensor [batch, d_model]
                context: Context tensor [batch, num_context, d_context]
                mask: Optional mask for context
                
            Returns:
                Tuple of (attended message, attention weights)
            """
            residual = message
            
            # Project context to model dimension
            context = self.context_proj(context)  # [batch, num_context, d_model]
            
            # Compute attention
            query = self.w_query(message).unsqueeze(1)  # [batch, 1, d_model]
            key = self.w_key(context)  # [batch, num_context, d_model]
            value = self.w_value(context)  # [batch, num_context, d_model]
            
            # Attention scores
            scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.d_model)
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            # Apply attention
            attended = torch.matmul(attention_weights, value).squeeze(1)
            
            # Output projection
            output = self.w_out(attended)
            output = self.dropout(output)
            
            # Residual connection and normalization
            output = self.layer_norm(output + residual)
            
            return output, attention_weights.squeeze(1)


    class HierarchicalAttention(nn.Module):
        """
        Hierarchical Attention mechanism.
        
        Applies attention at multiple levels:
        1. Word-level: Attention over words in a message
        2. Message-level: Attention over messages in a conversation
        
        Args:
            d_word: Word embedding dimension
            d_message: Message representation dimension
            d_output: Output dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        
        def __init__(
            self,
            d_word: int = 128,
            d_message: int = 256,
            d_output: int = 256,
            num_heads: int = 4,
            dropout: float = 0.1
        ):
            super().__init__()
            
            self.d_word = d_word
            self.d_message = d_message
            self.d_output = d_output
            
            # Word-level attention
            self.word_attention = MultiHeadAttention(d_word, num_heads, dropout)
            self.word_to_message = nn.Linear(d_word, d_message)
            
            # Message-level attention
            self.message_attention = MultiHeadAttention(d_message, num_heads, dropout)
            self.message_to_output = nn.Linear(d_message, d_output)
            
            self.dropout = nn.Dropout(dropout)
        
        def forward(
            self,
            words: Tensor,
            word_mask: Optional[Tensor] = None,
            message_mask: Optional[Tensor] = None
        ) -> Tuple[Tensor, Dict[str, Tensor]]:
            """
            Forward pass.
            
            Args:
                words: Word embeddings [batch, num_messages, num_words, d_word]
                word_mask: Mask for words
                message_mask: Mask for messages
                
            Returns:
                Tuple of (conversation representation, attention weights dict)
            """
            batch_size, num_messages, num_words, d_word = words.size()
            
            # Word-level attention (for each message)
            message_reps = []
            word_attention_weights = []
            
            for i in range(num_messages):
                message_words = words[:, i, :, :]  # [batch, num_words, d_word]
                
                # Apply word-level attention
                attended_words, word_attn = self.word_attention(
                    message_words, message_words, message_words, word_mask
                )
                
                # Pool to get message representation
                message_rep = attended_words.mean(dim=1)  # [batch, d_word]
                message_rep = self.word_to_message(message_rep)  # [batch, d_message]
                
                message_reps.append(message_rep)
                word_attention_weights.append(word_attn)
            
            # Stack message representations
            messages = torch.stack(message_reps, dim=1)  # [batch, num_messages, d_message]
            
            # Message-level attention
            conversation_rep, message_attn = self.message_attention(
                messages, messages, messages, message_mask
            )
            
            # Pool to get final representation
            output = conversation_rep.mean(dim=1)  # [batch, d_message]
            output = self.message_to_output(output)  # [batch, d_output]
            
            attention_weights = {
                'word_level': word_attention_weights,
                'message_level': message_attn
            }
            
            return output, attention_weights


    class FeatureAttention(nn.Module):
        """
        Feature Attention mechanism.
        
        Learns to weight different features based on their importance
        for the current prediction task.
        
        Args:
            num_features: Number of input features
            hidden_dim: Hidden dimension for attention scoring
            dropout: Dropout probability
        """
        
        def __init__(
            self,
            num_features: int,
            hidden_dim: int = 64,
            dropout: float = 0.1
        ):
            super().__init__()
            
            self.attention_score = nn.Sequential(
                nn.Linear(num_features, hidden_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
        
        def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
            """
            Forward pass.
            
            Args:
                features: Feature tensor [batch, num_features] or [batch, seq_len, num_features]
                
            Returns:
                Tuple of (weighted features, attention weights)
            """
            # Compute attention scores for each feature
            scores = self.attention_score(features)  # [batch, num_features, 1] or [batch, seq_len, num_features, 1]
            
            # Apply softmax to get attention weights
            attention_weights = F.softmax(scores.squeeze(-1), dim=-1)
            
            # Apply attention weights to features
            if features.dim() == 2:
                # [batch, num_features]
                weighted_features = features * attention_weights
            else:
                # [batch, seq_len, num_features]
                weighted_features = features * attention_weights.unsqueeze(-1)
            
            return weighted_features, attention_weights


# =============================================================================
# Attention Pooling
# =============================================================================

if TORCH_AVAILABLE:
    class AttentionPooling(nn.Module):
        """
        Attention-based pooling mechanism.
        
        Learns to pool sequence information using attention weights.
        Better than simple mean/max pooling for variable-length sequences.
        
        Args:
            d_model: Model dimension
            dropout: Dropout probability
        """
        
        def __init__(self, d_model: int = 256, dropout: float = 0.1):
            super().__init__()
            
            self.attention = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1)
            )
        
        def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
            """
            Forward pass.
            
            Args:
                x: Input tensor [batch, seq_len, d_model]
                mask: Optional mask tensor [batch, seq_len]
                
            Returns:
                Tuple of (pooled output, attention weights)
            """
            # Compute attention scores
            scores = self.attention(x).squeeze(-1)  # [batch, seq_len]
            
            # Apply mask if provided
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            # Softmax to get attention weights
            attention_weights = F.softmax(scores, dim=-1)  # [batch, seq_len]
            
            # Apply attention weights
            pooled = torch.bmm(
                attention_weights.unsqueeze(1),  # [batch, 1, seq_len]
                x  # [batch, seq_len, d_model]
            ).squeeze(1)  # [batch, d_model]
            
            return pooled, attention_weights


# =============================================================================
# Utility Functions
# =============================================================================

def create_padding_mask(seq_lens: Union[List[int], 'Tensor'], max_len: int = None) -> 'Tensor':
    """
    Create padding mask for sequences.
    
    Args:
        seq_lens: Sequence lengths [batch_size]
        max_len: Maximum sequence length
        
    Returns:
        Mask tensor [batch_size, max_len] where 1 = real token, 0 = padding
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for mask creation")
    
    if isinstance(seq_lens, list):
        seq_lens = torch.tensor(seq_lens)
    
    batch_size = seq_lens.size(0)
    max_len = max_len or seq_lens.max().item()
    
    # Create mask
    mask = torch.arange(max_len).expand(batch_size, max_len).to(seq_lens.device)
    mask = mask < seq_lens.unsqueeze(1)
    
    return mask.float()


def create_causal_mask(seq_len: int) -> 'Tensor':
    """
    Create causal mask for autoregressive attention.
    
    Prevents attending to future positions.
    
    Args:
        seq_len: Sequence length
        
    Returns:
        Causal mask [seq_len, seq_len]
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for mask creation")
    
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return (mask == 0).float()


def visualize_attention(
    attention_weights: np.ndarray,
    tokens: List[str],
    save_path: str = None
) -> None:
    """
    Visualize attention weights.
    
    Args:
        attention_weights: Attention weights [seq_len, seq_len] or [num_heads, seq_len, seq_len]
        tokens: List of token strings
        save_path: Path to save visualization
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import seaborn as sns  # type: ignore
        
        # Handle multi-head attention
        if len(attention_weights.shape) == 3:
            # Average across heads
            attention_weights = attention_weights.mean(axis=0)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            attention_weights,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='YlOrRd',
            ax=ax,
            cbar_kws={'label': 'Attention Weight'}
        )
        
        ax.set_title('Attention Weights Visualization')
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Attention visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        logger.warning("matplotlib/seaborn not available for visualization")


# =============================================================================
# Attention Wrapper
# =============================================================================

class AttentionModule:
    """
    High-level wrapper for attention mechanisms.
    
    Provides easy access to different attention types with common interface.
    """
    
    ATTENTION_TYPES = [
        'scaled_dot_product',
        'multi_head',
        'self_attention',
        'context_attention',
        'hierarchical',
        'feature_attention'
    ]
    
    def __init__(self, attention_type: str = 'multi_head', **kwargs):
        """
        Initialize attention module.
        
        Args:
            attention_type: Type of attention mechanism
            **kwargs: Arguments for specific attention type
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for attention mechanisms")
        
        if attention_type not in self.ATTENTION_TYPES:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        self.attention_type = attention_type
        self.config = kwargs
        
        # Create attention layer
        if attention_type == 'scaled_dot_product':
            self.attention = ScaledDotProductAttention(
                dropout=kwargs.get('dropout', 0.1)
            )
        elif attention_type == 'multi_head':
            self.attention = MultiHeadAttention(
                d_model=kwargs.get('d_model', 256),
                num_heads=kwargs.get('num_heads', 8),
                dropout=kwargs.get('dropout', 0.1)
            )
        elif attention_type == 'self_attention':
            self.attention = SelfAttention(
                d_model=kwargs.get('d_model', 256),
                num_heads=kwargs.get('num_heads', 8),
                dropout=kwargs.get('dropout', 0.1)
            )
        elif attention_type == 'context_attention':
            self.attention = ContextAttention(
                d_model=kwargs.get('d_model', 256),
                d_context=kwargs.get('d_context', 128),
                dropout=kwargs.get('dropout', 0.1)
            )
        elif attention_type == 'hierarchical':
            self.attention = HierarchicalAttention(
                d_word=kwargs.get('d_word', 128),
                d_message=kwargs.get('d_message', 256),
                d_output=kwargs.get('d_output', 256),
                num_heads=kwargs.get('num_heads', 4),
                dropout=kwargs.get('dropout', 0.1)
            )
        elif attention_type == 'feature_attention':
            self.attention = FeatureAttention(
                num_features=kwargs.get('num_features', 100),
                hidden_dim=kwargs.get('hidden_dim', 64),
                dropout=kwargs.get('dropout', 0.1)
            )
    
    def __call__(self, *args, **kwargs):
        """Forward pass through attention layer."""
        return self.attention(*args, **kwargs)
    
    def get_attention_layer(self):
        """Get the underlying attention layer."""
        return self.attention


# =============================================================================
# Testing and Demo
# =============================================================================

def demo_attention():
    """Demonstrate attention mechanisms."""
    if not TORCH_AVAILABLE:
        print("\nNote: PyTorch is required for attention mechanisms.")
        print("Install with: pip install torch")
        print("\nSkipping demo - PyTorch features are optional for baseline models.")
        return
    
    print("\n" + "="*60)
    print("ATTENTION MECHANISM DEMO")
    print("="*60)
    
    # Sample parameters
    batch_size = 2
    seq_len = 10
    d_model = 256
    num_heads = 8
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\nInput shape: {x.shape}")
    
    # 1. Self-Attention
    print("\n1. Self-Attention")
    self_attn = SelfAttention(d_model, num_heads)
    output, attn_weights = self_attn(x)
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")
    
    # 2. Multi-Head Attention
    print("\n2. Multi-Head Attention")
    mha = MultiHeadAttention(d_model, num_heads)
    output, attn_weights = mha(x, x, x)
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")
    
    # 3. Attention Pooling
    print("\n3. Attention Pooling")
    pool = AttentionPooling(d_model)
    output, attn_weights = pool(x)
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")
    
    # 4. Feature Attention
    print("\n4. Feature Attention")
    features = torch.randn(batch_size, 100)
    feat_attn = FeatureAttention(100)
    output, attn_weights = feat_attn(features)
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")
    
    print("\n" + "="*60)
    print("Demo complete!")


if __name__ == "__main__":
    if TORCH_AVAILABLE:
        demo_attention()
    else:
        print("="*60)
        print("ATTENTION MECHANISM MODULE")
        print("="*60)
        print("\nStatus: PyTorch not installed")
        print("\nThis module provides deep learning attention mechanisms.")
        print("PyTorch is optional - baseline models work without it.")
        print("\nTo use attention mechanisms, install PyTorch:")
        print("  pip install torch")
        print("\nFor CPU-only (smaller):")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("="*60)
