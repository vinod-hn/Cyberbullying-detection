# 02_feature_extraction module
"""
02_feature_extraction: Feature Extraction Package for Cyberbullying Detection

This package provides comprehensive feature extraction capabilities for
Kannada-English code-mixed cyberbullying detection system.

Architecture Role:
------------------
As per the system architecture, this module is the second stage in the pipeline:

    Data → Preprocessing → [Feature Extraction] → Models → Severity → API → Dashboard

The feature extraction module handles:
- Text embeddings (TF-IDF, Word2Vec, FastText)
- Transformer embeddings (BERT, mBERT, IndicBERT)
- Contextual features (conversation context, user behavior)
- Linguistic features (syntax, semantics, sentiment)
- Behavioral features (user patterns, interaction history)
- Emoji features (sentiment, cyberbullying patterns)
- Sarcasm detection

Modules:
--------
- text_embedder: Traditional text embedding methods (TF-IDF, Word2Vec, FastText)
- transformer_embedder: Transformer-based embeddings (BERT, mBERT)
- contextual_features: Conversation and context-aware features
- linguistic_features: Linguistic analysis features
- behavioral_features: User behavior pattern features
- emoji_features: Emoji-based feature extraction
- sarcasm_detector: Sarcasm and irony detection

Usage:
------
    from feature_extraction import TextEmbedder, LinguisticFeatures
    
    embedder = TextEmbedder()
    features = embedder.extract_tfidf(texts)
    
    # Or use the combined pipeline
    from feature_extraction import FeatureExtractor
    extractor = FeatureExtractor()
    all_features = extractor.extract_all(texts)

Author: Cyberbullying Detection Project Team
Version: 1.0.0
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
import numpy as np

# =============================================================================
# Package Metadata
# =============================================================================
__version__ = "1.0.0"
__author__ = "Cyberbullying Detection Project Team"
__description__ = "Feature extraction for Kannada-English cyberbullying detection"
__all__ = [
    # Main Classes
    "TextEmbedder",
    "TransformerEmbedder",
    "ContextualFeatures",
    "LinguisticFeatures",
    "BehavioralFeatures",
    "EmojiFeatures",
    "SarcasmDetector",
    
    # Combined Pipeline
    "FeatureExtractor",
    "FeaturePipeline",
    
    # Convenience Functions
    "extract_features",
    "extract_embeddings",
    "extract_linguistic_features",
    "extract_emoji_features",
    
    # Configuration
    "load_config",
    "get_default_config",
    "FeatureConfig",
    
    # Utilities
    "get_version",
    "get_module_info",
]

# =============================================================================
# Logging Configuration
# =============================================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# =============================================================================
# Module Imports
# =============================================================================
try:
    from .text_embedder import TextEmbedder
    logger.debug("TextEmbedder loaded successfully")
except ImportError as e:
    logger.warning(f"Could not import TextEmbedder: {e}")
    TextEmbedder = None

try:
    from .transformer_embedder import TransformerEmbedder
    logger.debug("TransformerEmbedder loaded successfully")
except ImportError as e:
    logger.warning(f"Could not import TransformerEmbedder: {e}")
    TransformerEmbedder = None

try:
    from .contextual_features import ContextualFeatures
    logger.debug("ContextualFeatures loaded successfully")
except ImportError as e:
    logger.warning(f"Could not import ContextualFeatures: {e}")
    ContextualFeatures = None

try:
    from .linguistic_features import LinguisticFeatures
    logger.debug("LinguisticFeatures loaded successfully")
except ImportError as e:
    logger.warning(f"Could not import LinguisticFeatures: {e}")
    LinguisticFeatures = None

try:
    from .behavioral_features import BehavioralFeatures
    logger.debug("BehavioralFeatures loaded successfully")
except ImportError as e:
    logger.warning(f"Could not import BehavioralFeatures: {e}")
    BehavioralFeatures = None

try:
    from .emoji_features import EmojiFeatures
    logger.debug("EmojiFeatures loaded successfully")
except ImportError as e:
    logger.warning(f"Could not import EmojiFeatures: {e}")
    EmojiFeatures = None

try:
    from .sarcasm_detector import SarcasmDetector
    logger.debug("SarcasmDetector loaded successfully")
except ImportError as e:
    logger.warning(f"Could not import SarcasmDetector: {e}")
    SarcasmDetector = None

# =============================================================================
# Configuration
# =============================================================================
class FeatureConfig:
    """Configuration class for feature extraction."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize with optional config dictionary."""
        self._config = config_dict or self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "text_embedder": {
                "tfidf": {
                    "enabled": True,
                    "max_features": 5000,
                    "ngram_range": [1, 2],
                    "min_df": 2,
                    "max_df": 0.95
                },
                "word2vec": {
                    "enabled": True,
                    "vector_size": 100,
                    "window": 5,
                    "min_count": 2
                },
                "fasttext": {
                    "enabled": True,
                    "vector_size": 100,
                    "window": 5
                }
            },
            "transformer_embedder": {
                "model_name": "bert-base-multilingual-cased",
                "max_length": 128,
                "pooling": "cls"
            },
            "linguistic_features": {
                "enabled": True,
                "include_pos": True,
                "include_sentiment": True,
                "include_readability": True
            },
            "emoji_features": {
                "enabled": True,
                "include_sentiment": True,
                "include_counts": True
            },
            "contextual_features": {
                "enabled": True,
                "window_size": 5
            },
            "behavioral_features": {
                "enabled": True,
                "include_temporal": True
            },
            "sarcasm_detection": {
                "enabled": True,
                "threshold": 0.5
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self._config.copy()


def load_config(config_path: Optional[str] = None) -> FeatureConfig:
    """Load configuration from file."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'feature_config.json')
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            return FeatureConfig(config_dict)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
    
    return FeatureConfig()


def get_default_config() -> Dict[str, Any]:
    """Get default configuration dictionary."""
    return FeatureConfig().to_dict()


# =============================================================================
# Combined Feature Extractor
# =============================================================================
class FeatureExtractor:
    """
    Combined feature extractor that orchestrates all feature extraction modules.
    
    Provides unified interface for extracting all types of features from text.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize FeatureExtractor.
        
        Args:
            config: Feature extraction configuration
        """
        self.config = config or FeatureConfig()
        self._init_extractors()
    
    def _init_extractors(self) -> None:
        """Initialize individual feature extractors."""
        self.text_embedder = None
        self.transformer_embedder = None
        self.linguistic_features = None
        self.emoji_features = None
        self.contextual_features = None
        self.behavioral_features = None
        self.sarcasm_detector = None
        
        if TextEmbedder:
            try:
                self.text_embedder = TextEmbedder(
                    self.config.get('text_embedder', {})
                )
            except Exception as e:
                logger.warning(f"Could not initialize TextEmbedder: {e}")
        
        if LinguisticFeatures:
            try:
                self.linguistic_features = LinguisticFeatures(
                    self.config.get('linguistic_features', {})
                )
            except Exception as e:
                logger.warning(f"Could not initialize LinguisticFeatures: {e}")
        
        if EmojiFeatures:
            try:
                self.emoji_features = EmojiFeatures(
                    self.config.get('emoji_features', {})
                )
            except Exception as e:
                logger.warning(f"Could not initialize EmojiFeatures: {e}")
        
        if SarcasmDetector:
            try:
                self.sarcasm_detector = SarcasmDetector(
                    self.config.get('sarcasm_detection', {})
                )
            except Exception as e:
                logger.warning(f"Could not initialize SarcasmDetector: {e}")
    
    def extract_all(
        self,
        texts: Union[str, List[str]],
        include_embeddings: bool = True,
        include_linguistic: bool = True,
        include_emoji: bool = True,
        include_sarcasm: bool = True
    ) -> Dict[str, Any]:
        """
        Extract all features from texts.
        
        Args:
            texts: Input text or list of texts
            include_embeddings: Include text embeddings
            include_linguistic: Include linguistic features
            include_emoji: Include emoji features
            include_sarcasm: Include sarcasm detection
            
        Returns:
            Dictionary containing all extracted features
        """
        if isinstance(texts, str):
            texts = [texts]
        
        features = {
            'text_count': len(texts),
            'features': {}
        }
        
        if include_embeddings and self.text_embedder:
            try:
                features['features']['embeddings'] = self.text_embedder.extract(texts)
            except Exception as e:
                logger.warning(f"Embedding extraction failed: {e}")
        
        if include_linguistic and self.linguistic_features:
            try:
                features['features']['linguistic'] = self.linguistic_features.extract(texts)
            except Exception as e:
                logger.warning(f"Linguistic feature extraction failed: {e}")
        
        if include_emoji and self.emoji_features:
            try:
                features['features']['emoji'] = self.emoji_features.extract(texts)
            except Exception as e:
                logger.warning(f"Emoji feature extraction failed: {e}")
        
        if include_sarcasm and self.sarcasm_detector:
            try:
                features['features']['sarcasm'] = self.sarcasm_detector.detect(texts)
            except Exception as e:
                logger.warning(f"Sarcasm detection failed: {e}")
        
        return features


class FeaturePipeline:
    """Pipeline for sequential feature extraction."""
    
    def __init__(self, extractors: Optional[List] = None):
        """Initialize pipeline with list of extractors."""
        self.extractors = extractors or []
    
    def add_extractor(self, extractor) -> 'FeaturePipeline':
        """Add an extractor to the pipeline."""
        self.extractors.append(extractor)
        return self
    
    def extract(self, texts: List[str]) -> Dict[str, Any]:
        """Run all extractors and combine features."""
        combined_features = {}
        
        for extractor in self.extractors:
            try:
                features = extractor.extract(texts)
                combined_features[type(extractor).__name__] = features
            except Exception as e:
                logger.warning(f"Extractor {type(extractor).__name__} failed: {e}")
        
        return combined_features


# =============================================================================
# Convenience Functions
# =============================================================================
def extract_features(
    texts: Union[str, List[str]],
    feature_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convenience function to extract features.
    
    Args:
        texts: Input text or list of texts
        feature_types: Types of features to extract
        
    Returns:
        Dictionary of extracted features
    """
    extractor = FeatureExtractor()
    return extractor.extract_all(texts)


def extract_embeddings(
    texts: Union[str, List[str]],
    method: str = 'tfidf'
) -> np.ndarray:
    """
    Extract text embeddings.
    
    Args:
        texts: Input texts
        method: Embedding method ('tfidf', 'word2vec', 'fasttext')
        
    Returns:
        Embedding matrix
    """
    if TextEmbedder is None:
        raise ImportError("TextEmbedder not available")
    
    embedder = TextEmbedder()
    if isinstance(texts, str):
        texts = [texts]
    
    if method == 'tfidf':
        return embedder.extract_tfidf(texts)
    elif method == 'word2vec':
        return embedder.extract_word2vec(texts)
    elif method == 'fasttext':
        return embedder.extract_fasttext(texts)
    else:
        raise ValueError(f"Unknown method: {method}")


def extract_linguistic_features(texts: Union[str, List[str]]) -> List[Dict[str, Any]]:
    """Extract linguistic features from texts."""
    if LinguisticFeatures is None:
        raise ImportError("LinguisticFeatures not available")
    
    extractor = LinguisticFeatures()
    if isinstance(texts, str):
        texts = [texts]
    return extractor.extract(texts)


def extract_emoji_features(texts: Union[str, List[str]]) -> List[Dict[str, Any]]:
    """Extract emoji features from texts."""
    if EmojiFeatures is None:
        raise ImportError("EmojiFeatures not available")
    
    extractor = EmojiFeatures()
    if isinstance(texts, str):
        texts = [texts]
    return extractor.extract(texts)


# =============================================================================
# Utilities
# =============================================================================
def get_version() -> str:
    """Get package version."""
    return __version__


def get_module_info() -> Dict[str, Any]:
    """Get module information."""
    return {
        'name': '02_feature_extraction',
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'modules': {
            'TextEmbedder': TextEmbedder is not None,
            'TransformerEmbedder': TransformerEmbedder is not None,
            'ContextualFeatures': ContextualFeatures is not None,
            'LinguisticFeatures': LinguisticFeatures is not None,
            'BehavioralFeatures': BehavioralFeatures is not None,
            'EmojiFeatures': EmojiFeatures is not None,
            'SarcasmDetector': SarcasmDetector is not None,
        }
    }


if __name__ == "__main__":
    info = get_module_info()
    print(f"Feature Extraction Module v{info['version']}")
    print(f"Available modules:")
    for module, available in info['modules'].items():
        status = "✓" if available else "✗"
        print(f"  {status} {module}")
