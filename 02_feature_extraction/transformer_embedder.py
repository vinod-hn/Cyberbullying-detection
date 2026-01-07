# Transformer Embedder
"""
TransformerEmbedder: Transformer-based embeddings for cyberbullying detection.
Supports BERT, mBERT, IndicBERT, and other transformer models.
Optimized for Kannada-English code-mixed text.
"""

import os
import logging
from typing import List, Dict, Optional, Any, Union
import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Transformer embeddings disabled.")

try:
    from transformers import (
        AutoModel, AutoTokenizer, 
        BertModel, BertTokenizer,
        AutoConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available. Transformer embeddings disabled.")


class TransformerEmbedder:
    """
    Transformer-based text embedding extractor.
    
    Supports multiple pre-trained models:
    - bert-base-multilingual-cased (mBERT): Good for code-mixed text
    - ai4bharat/indic-bert: Optimized for Indian languages
    - bert-base-uncased: English only
    - xlm-roberta-base: Cross-lingual
    
    Attributes:
        model_name: Name of the transformer model
        model: Loaded transformer model
        tokenizer: Loaded tokenizer
        config: Configuration dictionary
    """
    
    # Pre-defined model configurations
    MODEL_CONFIGS = {
        'mbert': {
            'name': 'bert-base-multilingual-cased',
            'max_length': 128,
            'hidden_size': 768
        },
        'indic-bert': {
            'name': 'ai4bharat/indic-bert',
            'max_length': 128,
            'hidden_size': 768
        },
        'bert-base': {
            'name': 'bert-base-uncased',
            'max_length': 128,
            'hidden_size': 768
        },
        'xlm-roberta': {
            'name': 'xlm-roberta-base',
            'max_length': 128,
            'hidden_size': 768
        },
        'distilbert': {
            'name': 'distilbert-base-multilingual-cased',
            'max_length': 128,
            'hidden_size': 768
        }
    }
    
    def __init__(
        self,
        model_name: str = 'bert-base-multilingual-cased',
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None
    ):
        """
        Initialize TransformerEmbedder.
        
        Args:
            model_name: Name of transformer model or key from MODEL_CONFIGS
            config: Configuration dictionary
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.config = config or {}
        
        # Resolve model name
        if model_name in self.MODEL_CONFIGS:
            model_config = self.MODEL_CONFIGS[model_name]
            self.model_name = model_config['name']
            self.max_length = model_config['max_length']
            self.hidden_size = model_config['hidden_size']
        else:
            self.model_name = model_name
            self.max_length = self.config.get('max_length', 128)
            self.hidden_size = self.config.get('hidden_size', 768)
        
        # Device setup
        if device:
            self.device = device
        elif TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        # Model and tokenizer
        self.model = None
        self.tokenizer = None
        
        # Pooling strategy
        self.pooling = self.config.get('pooling', 'cls')  # 'cls', 'mean', 'max'
        
        # Load model
        if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
            self._load_model()
    
    def _load_model(self) -> None:
        """Load transformer model and tokenizer."""
        try:
            logger.info(f"Loading transformer model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            self.model.to(self.device)
            self.model.eval()
            
            # Update hidden size from model config
            self.hidden_size = self.model.config.hidden_size
            
            logger.info(f"Model loaded successfully on {self.device}")
            logger.info(f"Hidden size: {self.hidden_size}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            self.tokenizer = None
    
    def is_available(self) -> bool:
        """Check if transformer model is available."""
        return self.model is not None and self.tokenizer is not None
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: Text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            Embedding matrix (n_samples, hidden_size)
        """
        if not self.is_available():
            raise RuntimeError("Transformer model not available")
        
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**encoded)
                
                # Get hidden states
                hidden_states = outputs.last_hidden_state
                
                # Apply pooling
                if self.pooling == 'cls':
                    # Use [CLS] token embedding
                    embeddings = hidden_states[:, 0, :]
                elif self.pooling == 'mean':
                    # Mean pooling over all tokens
                    attention_mask = encoded['attention_mask']
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                elif self.pooling == 'max':
                    # Max pooling over all tokens
                    embeddings = torch.max(hidden_states, dim=1)[0]
                else:
                    embeddings = hidden_states[:, 0, :]
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def extract(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Alias for encode method."""
        return self.encode(texts)
    
    def encode_with_attention(
        self,
        texts: Union[str, List[str]]
    ) -> Dict[str, np.ndarray]:
        """
        Encode texts and return attention weights.
        
        Args:
            texts: Text or list of texts
            
        Returns:
            Dictionary with embeddings and attention weights
        """
        if not self.is_available():
            raise RuntimeError("Transformer model not available")
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = self.model(**encoded, output_attentions=True)
            
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # Average attention across heads and layers
            attentions = outputs.attentions
            avg_attention = torch.stack(attentions).mean(dim=(0, 1))
            
        return {
            'embeddings': embeddings,
            'attention': avg_attention.cpu().numpy(),
            'tokens': [self.tokenizer.convert_ids_to_tokens(ids) for ids in encoded['input_ids']]
        }
    
    def get_token_embeddings(
        self,
        text: str,
        layer: int = -1
    ) -> Dict[str, Any]:
        """
        Get token-level embeddings.
        
        Args:
            text: Input text
            layer: Which layer to extract from (-1 for last)
            
        Returns:
            Dictionary with tokens and their embeddings
        """
        if not self.is_available():
            raise RuntimeError("Transformer model not available")
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = self.model(**encoded, output_hidden_states=True)
            
            # Get specified layer
            hidden_states = outputs.hidden_states[layer]
            
            # Get tokens
            tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
            
            # Get embeddings
            embeddings = hidden_states[0].cpu().numpy()
        
        return {
            'tokens': tokens,
            'embeddings': embeddings,
            'layer': layer
        }
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score
        """
        embeddings = self.encode([text1, text2])
        
        # Cosine similarity
        dot_product = np.dot(embeddings[0], embeddings[1])
        norm1 = np.linalg.norm(embeddings[0])
        norm2 = np.linalg.norm(embeddings[1])
        
        return float(dot_product / (norm1 * norm2))
    
    def batch_similarity(
        self,
        texts: List[str],
        query: str
    ) -> List[float]:
        """
        Calculate similarity of multiple texts to a query.
        
        Args:
            texts: List of texts
            query: Query text
            
        Returns:
            List of similarity scores
        """
        all_texts = [query] + texts
        embeddings = self.encode(all_texts)
        
        query_embedding = embeddings[0]
        text_embeddings = embeddings[1:]
        
        similarities = []
        for emb in text_embeddings:
            dot_product = np.dot(query_embedding, emb)
            norm1 = np.linalg.norm(query_embedding)
            norm2 = np.linalg.norm(emb)
            similarities.append(float(dot_product / (norm1 * norm2)))
        
        return similarities
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        path: str
    ) -> None:
        """Save embeddings to file."""
        np.save(path, embeddings)
        logger.info(f"Saved embeddings to {path}")
    
    def load_embeddings(self, path: str) -> np.ndarray:
        """Load embeddings from file."""
        return np.load(path)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_name': self.model_name,
            'hidden_size': self.hidden_size,
            'max_length': self.max_length,
            'pooling': self.pooling,
            'device': self.device,
            'available': self.is_available()
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TransformerEmbedder("
            f"model='{self.model_name}', "
            f"hidden_size={self.hidden_size}, "
            f"device='{self.device}')"
        )


# Convenience functions
def get_mbert_embeddings(texts: Union[str, List[str]]) -> np.ndarray:
    """Get embeddings using mBERT."""
    embedder = TransformerEmbedder('mbert')
    return embedder.encode(texts)


def get_similarity(text1: str, text2: str, model: str = 'mbert') -> float:
    """Calculate similarity between two texts."""
    embedder = TransformerEmbedder(model)
    return embedder.similarity(text1, text2)


if __name__ == "__main__":
    print("TransformerEmbedder Test")
    print("=" * 50)
    
    if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
        print("PyTorch or Transformers not available. Skipping test.")
    else:
        # Test with mBERT
        embedder = TransformerEmbedder('mbert')
        
        if embedder.is_available():
            test_texts = [
                "nee tumba irritating agthiya yaar",
                "This is very annoying behavior",
                "exam tumba tough aaytu",
            ]
            
            embeddings = embedder.encode(test_texts)
            print(f"Embeddings shape: {embeddings.shape}")
            print(f"Model info: {embedder.get_model_info()}")
            
            # Test similarity
            sim = embedder.similarity(test_texts[0], test_texts[1])
            print(f"Similarity between texts 0 and 1: {sim:.4f}")
        else:
            print("Model not available (may need to download)")
