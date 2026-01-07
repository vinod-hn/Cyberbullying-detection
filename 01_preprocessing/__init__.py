# 01_preprocessing module
"""
01_preprocessing: Text Preprocessing Package for Cyberbullying Detection

This package provides comprehensive text preprocessing capabilities for
Kannada-English code-mixed cyberbullying detection system.

Architecture Role:
------------------
As per the system architecture (Section 4, Page 7), this module is the first
stage in the data processing pipeline:

    Data → [Preprocessing] → Feature Extraction → Models → Severity → API → Dashboard

The preprocessing module handles:
- Text normalization (whitespace, case, unicode)
- Emoji processing and sentiment extraction
- Code-mix detection and processing (Kannada-English)
- Transliteration between Kannada script and Roman
- Slang expansion (Kannada and English)
- Conversation threading for context analysis

Modules:
--------
- text_normalizer: Text cleaning and normalization
- emoji_handler: Emoji detection, extraction, and sentiment
- code_mix_processor: Kannada-English code-mixing analysis
- transliterator: Bidirectional Kannada-Roman transliteration
- slang_expander: Slang and abbreviation expansion
- conversation_threader: Conversation context analysis

Usage:
------
    # Import main classes
    from preprocessing import TextNormalizer, EmojiHandler, CodeMixProcessor
    
    # Initialize preprocessors
    normalizer = TextNormalizer()
    emoji_handler = EmojiHandler()
    
    # Process text
    text = "nee tumba irritating agthiya 😡"
    normalized = normalizer.normalize(text)
    emojis = emoji_handler.extract_emojis(text)
    
    # Or use the convenience pipeline
    from preprocessing import preprocess_text
    processed = preprocess_text(text)

Author: Cyberbullying Detection Project Team
Version: 1.0.0
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union

# =============================================================================
# Package Metadata
# =============================================================================
__version__ = "1.0.0"
__author__ = "Cyberbullying Detection Project Team"
__description__ = "Text preprocessing for Kannada-English cyberbullying detection"
__all__ = [
    # Main Classes
    "TextNormalizer",
    "EmojiHandler",
    "CodeMixProcessor",
    "Transliterator",
    "SlangExpander",
    "ConversationThreader",
    
    # Convenience Functions
    "preprocess_text",
    "preprocess_batch",
    "normalize_text",
    "extract_emojis",
    "detect_language",
    "transliterate",
    "expand_slang",
    
    # Configuration
    "load_config",
    "get_default_config",
    "PreprocessingConfig",
    
    # Pipeline
    "PreprocessingPipeline",
    "create_pipeline",
    
    # Utilities
    "get_version",
    "get_module_info",
]

# =============================================================================
# Logging Configuration
# =============================================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler if no handlers exist
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# =============================================================================
# Module Imports - Core Classes
# =============================================================================
try:
    from .text_normalizer import TextNormalizer
    logger.debug("TextNormalizer loaded successfully")
except ImportError as e:
    logger.warning(f"Could not import TextNormalizer: {e}")
    TextNormalizer = None

try:
    from .emoji_handler import EmojiHandler
    logger.debug("EmojiHandler loaded successfully")
except ImportError as e:
    logger.warning(f"Could not import EmojiHandler: {e}")
    EmojiHandler = None

try:
    from .code_mix_processor import CodeMixProcessor
    logger.debug("CodeMixProcessor loaded successfully")
except ImportError as e:
    logger.warning(f"Could not import CodeMixProcessor: {e}")
    CodeMixProcessor = None

try:
    from .transliterator import Transliterator
    logger.debug("Transliterator loaded successfully")
except ImportError as e:
    logger.warning(f"Could not import Transliterator: {e}")
    Transliterator = None

try:
    from .slang_expander import SlangExpander
    logger.debug("SlangExpander loaded successfully")
except ImportError as e:
    logger.warning(f"Could not import SlangExpander: {e}")
    SlangExpander = None

try:
    from .conversation_threader import ConversationThreader
    logger.debug("ConversationThreader loaded successfully")
except ImportError as e:
    logger.warning(f"Could not import ConversationThreader: {e}")
    ConversationThreader = None


# =============================================================================
# Configuration Management
# =============================================================================
class PreprocessingConfig:
    """
    Configuration manager for preprocessing pipeline.
    
    Loads and manages configuration from preprocessing_config.json
    and provides default values for all preprocessing options.
    """
    
    _instance = None
    _config = None
    
    def __new__(cls):
        """Singleton pattern for configuration."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self) -> None:
        """Load configuration from JSON file."""
        config_path = os.path.join(
            os.path.dirname(__file__),
            'preprocessing_config.json'
        )
        
        self._config = self._get_defaults()
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    self._config.update(user_config)
                logger.debug(f"Configuration loaded from {config_path}")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error loading config: {e}. Using defaults.")
    
    def _get_defaults(self) -> Dict[str, Any]:
        """Return default configuration values."""
        return {
            # Text Normalizer Settings
            "normalizer": {
                "lowercase": True,
                "remove_urls": True,
                "remove_emails": True,
                "remove_hashtag_ids": True,
                "reduce_elongation": True,
                "max_char_repeat": 2,
                "normalize_unicode": True,
                "normalize_whitespace": True,
                "preserve_kannada_script": True,
                "reduce_punctuation": True,
                "max_punct_repeat": 1
            },
            
            # Emoji Handler Settings
            "emoji": {
                "extract_emojis": True,
                "compute_sentiment": True,
                "detect_bullying_patterns": True,
                "preserve_in_text": False,
                "replace_with_description": True
            },
            
            # Code-Mix Processor Settings
            "code_mix": {
                "detect_language": True,
                "calculate_ratio": True,
                "classify_tokens": True,
                "handle_romanized_kannada": True
            },
            
            # Transliterator Settings
            "transliterator": {
                "direction": "auto",  # auto, roman_to_kannada, kannada_to_roman
                "preserve_english": True,
                "preserve_numbers": True,
                "preserve_punctuation": True
            },
            
            # Slang Expander Settings
            "slang": {
                "expand_kannada": True,
                "expand_english": True,
                "use_lexicon": True,
                "lexicon_path": "../00_data/lexicon"
            },
            
            # Conversation Threader Settings
            "threader": {
                "enable_threading": True,
                "max_context_messages": 5,
                "detect_replies": True
            },
            
            # Pipeline Settings
            "pipeline": {
                "enabled_steps": [
                    "normalize",
                    "emoji",
                    "code_mix",
                    "slang"
                ],
                "parallel_processing": False,
                "batch_size": 100
            },
            
            # Data Paths
            "paths": {
                "lexicon_dir": "../00_data/lexicon",
                "profanity_kannada": "../00_data/lexicon/profanity_kannada.csv",
                "profanity_english": "../00_data/lexicon/profanity_english.csv",
                "kannada_slang": "../00_data/lexicon/kannada_slang.csv",
                "english_slang": "../00_data/lexicon/english_slang.csv",
                "emoji_semantics": "../00_data/lexicon/emoji_semantics.json"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key (supports dot notation)."""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load preprocessing configuration.
    
    Args:
        config_path: Optional path to configuration JSON file.
        
    Returns:
        Configuration dictionary.
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return PreprocessingConfig().to_dict()


def get_default_config() -> Dict[str, Any]:
    """
    Get default preprocessing configuration.
    
    Returns:
        Default configuration dictionary.
    """
    return PreprocessingConfig()._get_defaults()


# =============================================================================
# Preprocessing Pipeline
# =============================================================================
class PreprocessingPipeline:
    """
    Unified preprocessing pipeline for cyberbullying detection.
    
    Combines all preprocessing steps into a single configurable pipeline.
    Supports both single text and batch processing.
    
    Pipeline Steps:
    1. Text Normalization
    2. Emoji Processing
    3. Code-Mix Analysis
    4. Transliteration (optional)
    5. Slang Expansion
    6. Conversation Threading (optional)
    
    Usage:
        pipeline = PreprocessingPipeline()
        result = pipeline.process("nee tumba irritating 😡")
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize preprocessing pipeline.
        
        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or PreprocessingConfig().to_dict()
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize preprocessing components."""
        self.normalizer = None
        self.emoji_handler = None
        self.code_mix_processor = None
        self.transliterator = None
        self.slang_expander = None
        self.conversation_threader = None
        
        # Initialize available components
        if TextNormalizer:
            try:
                self.normalizer = TextNormalizer(
                    config=self.config.get('normalizer')
                )
            except Exception as e:
                logger.warning(f"Could not initialize TextNormalizer: {e}")
        
        if EmojiHandler:
            try:
                emoji_config = self.config.get('paths', {})
                self.emoji_handler = EmojiHandler(
                    semantics_path=emoji_config.get('emoji_semantics')
                )
            except Exception as e:
                logger.warning(f"Could not initialize EmojiHandler: {e}")
        
        if CodeMixProcessor:
            try:
                self.code_mix_processor = CodeMixProcessor()
            except Exception as e:
                logger.warning(f"Could not initialize CodeMixProcessor: {e}")
        
        if Transliterator:
            try:
                self.transliterator = Transliterator()
            except Exception as e:
                logger.warning(f"Could not initialize Transliterator: {e}")
        
        if SlangExpander:
            try:
                self.slang_expander = SlangExpander()
            except Exception as e:
                logger.warning(f"Could not initialize SlangExpander: {e}")
        
        if ConversationThreader:
            try:
                self.conversation_threader = ConversationThreader()
            except Exception as e:
                logger.warning(f"Could not initialize ConversationThreader: {e}")
    
    def process(
        self,
        text: str,
        steps: Optional[List[str]] = None,
        return_details: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Process text through the preprocessing pipeline.
        
        Args:
            text: Input text to process.
            steps: Optional list of steps to apply.
                   Default: ["normalize", "emoji", "code_mix", "slang"]
            return_details: If True, return detailed processing info.
            
        Returns:
            Processed text or dictionary with details.
        """
        if not text or not isinstance(text, str):
            return "" if not return_details else {"text": "", "steps": []}
        
        steps = steps or self.config.get('pipeline', {}).get('enabled_steps', [
            "normalize", "emoji", "code_mix", "slang"
        ])
        
        result = {
            "original": text,
            "text": text,
            "steps": [],
            "metadata": {}
        }
        
        processed_text = text
        
        # Step 1: Text Normalization
        if "normalize" in steps and self.normalizer:
            try:
                processed_text = self.normalizer.normalize(processed_text)
                result["steps"].append("normalize")
            except Exception as e:
                logger.error(f"Normalization error: {e}")
        
        # Step 2: Emoji Processing
        if "emoji" in steps and self.emoji_handler:
            try:
                emojis = self.emoji_handler.extract_emojis(processed_text)
                result["metadata"]["emojis"] = emojis
                result["metadata"]["emoji_count"] = len(emojis)
                
                if self.config.get('emoji', {}).get('compute_sentiment', True):
                    if hasattr(self.emoji_handler, 'get_text_emoji_sentiment'):
                        result["metadata"]["emoji_sentiment"] = \
                            self.emoji_handler.get_text_emoji_sentiment(processed_text)
                
                # Process text if configured
                if hasattr(self.emoji_handler, 'process_text'):
                    processed_text = self.emoji_handler.process_text(processed_text)
                
                result["steps"].append("emoji")
            except Exception as e:
                logger.error(f"Emoji processing error: {e}")
        
        # Step 3: Code-Mix Analysis
        if "code_mix" in steps and self.code_mix_processor:
            try:
                if hasattr(self.code_mix_processor, 'detect_language'):
                    result["metadata"]["language"] = \
                        self.code_mix_processor.detect_language(processed_text)
                
                if hasattr(self.code_mix_processor, 'calculate_code_mix_ratio'):
                    result["metadata"]["code_mix_ratio"] = \
                        self.code_mix_processor.calculate_code_mix_ratio(processed_text)
                
                if hasattr(self.code_mix_processor, 'normalize'):
                    processed_text = self.code_mix_processor.normalize(processed_text)
                
                result["steps"].append("code_mix")
            except Exception as e:
                logger.error(f"Code-mix processing error: {e}")
        
        # Step 4: Transliteration (optional)
        if "transliterate" in steps and self.transliterator:
            try:
                if hasattr(self.transliterator, 'auto_transliterate'):
                    processed_text = self.transliterator.auto_transliterate(processed_text)
                result["steps"].append("transliterate")
            except Exception as e:
                logger.error(f"Transliteration error: {e}")
        
        # Step 5: Slang Expansion
        if "slang" in steps and self.slang_expander:
            try:
                if hasattr(self.slang_expander, 'expand'):
                    processed_text = self.slang_expander.expand(processed_text)
                result["steps"].append("slang")
            except Exception as e:
                logger.error(f"Slang expansion error: {e}")
        
        result["text"] = processed_text
        
        if return_details:
            return result
        return processed_text
    
    def process_batch(
        self,
        texts: List[str],
        steps: Optional[List[str]] = None,
        return_details: bool = False
    ) -> List[Union[str, Dict[str, Any]]]:
        """
        Process multiple texts through the pipeline.
        
        Args:
            texts: List of texts to process.
            steps: Optional list of steps to apply.
            return_details: If True, return detailed processing info.
            
        Returns:
            List of processed texts or dictionaries.
        """
        return [
            self.process(text, steps, return_details)
            for text in texts
        ]
    
    def get_available_steps(self) -> List[str]:
        """Return list of available preprocessing steps."""
        steps = []
        if self.normalizer:
            steps.append("normalize")
        if self.emoji_handler:
            steps.append("emoji")
        if self.code_mix_processor:
            steps.append("code_mix")
        if self.transliterator:
            steps.append("transliterate")
        if self.slang_expander:
            steps.append("slang")
        if self.conversation_threader:
            steps.append("thread")
        return steps


def create_pipeline(config: Optional[Dict[str, Any]] = None) -> PreprocessingPipeline:
    """
    Create a preprocessing pipeline with optional configuration.
    
    Args:
        config: Optional configuration dictionary.
        
    Returns:
        PreprocessingPipeline instance.
    """
    return PreprocessingPipeline(config)


# =============================================================================
# Convenience Functions
# =============================================================================
# Global instances for convenience functions
_normalizer: Optional[Any] = None
_emoji_handler: Optional[Any] = None
_code_mix_processor: Optional[Any] = None
_transliterator: Optional[Any] = None
_slang_expander: Optional[Any] = None
_pipeline: Optional[Any] = None


def _get_normalizer() -> Optional[Any]:
    """Get or create global TextNormalizer instance."""
    global _normalizer
    if _normalizer is None and TextNormalizer:
        _normalizer = TextNormalizer()
    return _normalizer


def _get_emoji_handler() -> Optional[Any]:
    """Get or create global EmojiHandler instance."""
    global _emoji_handler
    if _emoji_handler is None and EmojiHandler:
        _emoji_handler = EmojiHandler()
    return _emoji_handler


def _get_code_mix_processor() -> Optional[Any]:
    """Get or create global CodeMixProcessor instance."""
    global _code_mix_processor
    if _code_mix_processor is None and CodeMixProcessor:
        _code_mix_processor = CodeMixProcessor()
    return _code_mix_processor


def _get_transliterator() -> Optional[Any]:
    """Get or create global Transliterator instance."""
    global _transliterator
    if _transliterator is None and Transliterator:
        _transliterator = Transliterator()
    return _transliterator


def _get_slang_expander() -> Optional[Any]:
    """Get or create global SlangExpander instance."""
    global _slang_expander
    if _slang_expander is None and SlangExpander:
        _slang_expander = SlangExpander()
    return _slang_expander


def _get_pipeline() -> PreprocessingPipeline:
    """Get or create global PreprocessingPipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = PreprocessingPipeline()
    return _pipeline


def preprocess_text(
    text: str,
    steps: Optional[List[str]] = None,
    return_details: bool = False
) -> Union[str, Dict[str, Any]]:
    """
    Preprocess text using the full pipeline.
    
    Convenience function for quick preprocessing without manual initialization.
    
    Args:
        text: Input text to preprocess.
        steps: Optional list of preprocessing steps.
        return_details: If True, return detailed info.
        
    Returns:
        Preprocessed text or dictionary with details.
        
    Example:
        >>> preprocess_text("nee tumba irritating agthiya 😡")
        'nee tumba irritating agthiya'
    """
    return _get_pipeline().process(text, steps, return_details)


def preprocess_batch(
    texts: List[str],
    steps: Optional[List[str]] = None,
    return_details: bool = False
) -> List[Union[str, Dict[str, Any]]]:
    """
    Preprocess multiple texts using the full pipeline.
    
    Args:
        texts: List of texts to preprocess.
        steps: Optional list of preprocessing steps.
        return_details: If True, return detailed info.
        
    Returns:
        List of preprocessed texts or dictionaries.
    """
    return _get_pipeline().process_batch(texts, steps, return_details)


def normalize_text(text: str) -> str:
    """
    Normalize text using TextNormalizer.
    
    Args:
        text: Input text to normalize.
        
    Returns:
        Normalized text.
    """
    normalizer = _get_normalizer()
    if normalizer:
        return normalizer.normalize(text)
    return text


def extract_emojis(text: str) -> List[str]:
    """
    Extract emojis from text.
    
    Args:
        text: Input text.
        
    Returns:
        List of extracted emojis.
    """
    handler = _get_emoji_handler()
    if handler:
        return handler.extract_emojis(text)
    return []


def detect_language(text: str) -> str:
    """
    Detect language of text (kannada, english, code-mixed).
    
    Args:
        text: Input text.
        
    Returns:
        Detected language string.
    """
    processor = _get_code_mix_processor()
    if processor and hasattr(processor, 'detect_language'):
        return processor.detect_language(text)
    return "unknown"


def transliterate(
    text: str,
    direction: str = "auto"
) -> str:
    """
    Transliterate text between Kannada and Roman script.
    
    Args:
        text: Input text to transliterate.
        direction: "auto", "roman_to_kannada", or "kannada_to_roman"
        
    Returns:
        Transliterated text.
    """
    trans = _get_transliterator()
    if trans:
        if direction == "auto" and hasattr(trans, 'auto_transliterate'):
            return trans.auto_transliterate(text)
        elif direction == "roman_to_kannada" and hasattr(trans, 'roman_to_kannada'):
            return trans.roman_to_kannada(text)
        elif direction == "kannada_to_roman" and hasattr(trans, 'kannada_to_roman'):
            return trans.kannada_to_roman(text)
    return text


def expand_slang(text: str) -> str:
    """
    Expand slang and abbreviations in text.
    
    Args:
        text: Input text with slang.
        
    Returns:
        Text with slang expanded.
    """
    expander = _get_slang_expander()
    if expander and hasattr(expander, 'expand'):
        return expander.expand(text)
    return text


# =============================================================================
# Utility Functions
# =============================================================================
def get_version() -> str:
    """
    Get package version.
    
    Returns:
        Version string.
    """
    return __version__


def get_module_info() -> Dict[str, Any]:
    """
    Get information about the preprocessing module.
    
    Returns:
        Dictionary with module information.
    """
    return {
        "name": "01_preprocessing",
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "components": {
            "TextNormalizer": TextNormalizer is not None,
            "EmojiHandler": EmojiHandler is not None,
            "CodeMixProcessor": CodeMixProcessor is not None,
            "Transliterator": Transliterator is not None,
            "SlangExpander": SlangExpander is not None,
            "ConversationThreader": ConversationThreader is not None,
        },
        "config_path": os.path.join(
            os.path.dirname(__file__),
            'preprocessing_config.json'
        ),
        "available_steps": _get_pipeline().get_available_steps() if _pipeline else [],
    }


# =============================================================================
# Package Initialization
# =============================================================================
def _initialize_package() -> None:
    """
    Initialize the preprocessing package.
    
    This function is called on package import to:
    1. Validate environment
    2. Check dependencies
    3. Log initialization status
    """
    logger.info(f"Initializing 01_preprocessing v{__version__}")
    
    # Check available components
    available = []
    unavailable = []
    
    components = [
        ("TextNormalizer", TextNormalizer),
        ("EmojiHandler", EmojiHandler),
        ("CodeMixProcessor", CodeMixProcessor),
        ("Transliterator", Transliterator),
        ("SlangExpander", SlangExpander),
        ("ConversationThreader", ConversationThreader),
    ]
    
    for name, component in components:
        if component is not None:
            available.append(name)
        else:
            unavailable.append(name)
    
    logger.debug(f"Available components: {available}")
    if unavailable:
        logger.debug(f"Unavailable components (not implemented): {unavailable}")
    
    logger.info("01_preprocessing initialized successfully")


# Run initialization on import
_initialize_package()

