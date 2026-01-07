# Emoji Features
"""
EmojiFeatures: Emoji-based feature extraction for cyberbullying detection.
Extracts sentiment, patterns, and cyberbullying indicators from emojis.
Optimized for Kannada-English code-mixed text with emoji usage.
"""

import re
import os
import json
import logging
from typing import List, Dict, Optional, Any, Union, Set
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)


class EmojiFeatures:
    """
    Emoji feature extractor for cyberbullying detection.
    
    Extracts:
    - Emoji counts and ratios
    - Emoji sentiment scores
    - Cyberbullying pattern indicators
    - Emoji category distribution
    
    Uses emoji_semantics.json for detailed emoji analysis.
    """
    
    # Unicode ranges for emoji detection
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols
        "\U00002600-\U000026FF"  # Misc symbols
        "\U00002700-\U000027BF"  # Dingbats
        "\U0001FA00-\U0001FAFF"  # Extended-A
        "]+",
        flags=re.UNICODE
    )
    
    # Emoji categories for cyberbullying detection
    OFFENSIVE_EMOJIS = {'🖕', '🖕🏻', '🖕🏼', '🖕🏽', '🖕🏾', '🖕🏿'}
    
    AGGRESSIVE_EMOJIS = {
        '😡', '🤬', '👊', '💢', '😤', '🔥', '💀', '☠️',
        '👊🏻', '👊🏼', '👊🏽', '👊🏾', '👊🏿'
    }
    
    THREATENING_EMOJIS = {
        '🔪', '🗡️', '⚔️', '🔫', '💣', '💥', '⚰️', '🪦',
        '☠️', '💀', '👹', '👺', '😈', '👿'
    }
    
    MOCKING_EMOJIS = {
        '🤡', '🤪', '😜', '🙄', '😏', '💅', '🤏', '🤣',
        '😂', '🥴', '🤓', '🥱'
    }
    
    BODY_SHAMING_EMOJIS = {
        '🐷', '🐖', '🐮', '🐄', '🦛', '🐘', '🤮', '🤢',
        '💩', '🦴', '🦷', '👃'
    }
    
    SEXUAL_HARASSMENT_EMOJIS = {
        '🍆', '🍑', '💦', '🍌', '👅', '👀', '😈', '🥵',
        '😘', '😍', '🤤', '💋'
    }
    
    POSITIVE_EMOJIS = {
        '😀', '😃', '😄', '😁', '😊', '🙂', '😇', '🥰',
        '😍', '🤗', '👍', '👏', '🙌', '❤️', '💕', '✨',
        '🌟', '💯', '✅', '🎉', '🎊'
    }
    
    NEUTRAL_EMOJIS = {
        '😐', '😑', '🤔', '🤷', '👋', '✋', '🖐️', '👌',
        '🙏', '📱', '💻', '📚', '📖', '✏️', '📝'
    }
    
    # All negative emojis combined
    ALL_NEGATIVE_EMOJIS = (
        OFFENSIVE_EMOJIS | AGGRESSIVE_EMOJIS | THREATENING_EMOJIS |
        MOCKING_EMOJIS | BODY_SHAMING_EMOJIS | SEXUAL_HARASSMENT_EMOJIS
    )
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize EmojiFeatures.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        
        # Load emoji semantics
        self.emoji_semantics = {}
        self._load_semantics()
        
        # Build emoji lookup
        self._build_emoji_lookup()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'include_counts': True,
            'include_sentiment': True,
            'include_categories': True,
            'include_patterns': True,
            'semantics_path': None
        }
    
    def _load_semantics(self) -> None:
        """Load emoji semantics from JSON file."""
        semantics_path = self.config.get('semantics_path')
        
        if semantics_path is None:
            # Try default locations
            possible_paths = [
                os.path.join(os.path.dirname(__file__), '..', '00_data', 'lexicon', 'emoji_semantics.json'),
                os.path.join(os.path.dirname(__file__), 'emoji_semantics.json'),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    semantics_path = path
                    break
        
        if semantics_path and os.path.exists(semantics_path):
            try:
                with open(semantics_path, 'r', encoding='utf-8') as f:
                    self.emoji_semantics = json.load(f)
                logger.info(f"Loaded emoji semantics from {semantics_path}")
            except Exception as e:
                logger.warning(f"Could not load emoji semantics: {e}")
    
    def _build_emoji_lookup(self) -> None:
        """Build lookup dictionaries from semantics."""
        self._emoji_sentiment = {}
        self._emoji_names = {}
        self._emoji_categories = {}
        self._emoji_cyberbullying = {}
        
        # Process negative emojis from semantics
        if 'negative_emojis' in self.emoji_semantics:
            for category, emojis in self.emoji_semantics['negative_emojis'].items():
                for emoji, data in emojis.items():
                    self._emoji_sentiment[emoji] = data.get('sentiment', -0.5)
                    self._emoji_names[emoji] = data.get('name', '')
                    self._emoji_categories[emoji] = f'negative_{category}'
                    self._emoji_cyberbullying[emoji] = data.get('cyberbullying_association', 'medium')
        
        # Process positive emojis from semantics
        if 'positive_emojis' in self.emoji_semantics:
            for category, emojis in self.emoji_semantics['positive_emojis'].items():
                for emoji, data in emojis.items():
                    self._emoji_sentiment[emoji] = data.get('sentiment', 0.5)
                    self._emoji_names[emoji] = data.get('name', '')
                    self._emoji_categories[emoji] = f'positive_{category}'
                    self._emoji_cyberbullying[emoji] = data.get('cyberbullying_association', 'none')
        
        # Add defaults for known emojis not in semantics
        for emoji in self.ALL_NEGATIVE_EMOJIS:
            if emoji not in self._emoji_sentiment:
                self._emoji_sentiment[emoji] = -0.7
                self._emoji_cyberbullying[emoji] = 'high'
        
        for emoji in self.POSITIVE_EMOJIS:
            if emoji not in self._emoji_sentiment:
                self._emoji_sentiment[emoji] = 0.6
                self._emoji_cyberbullying[emoji] = 'none'
    
    def extract_emojis(self, text: str) -> List[str]:
        """
        Extract all emojis from text.
        
        Args:
            text: Input text
            
        Returns:
            List of emojis found
        """
        emojis = []
        for match in self.EMOJI_PATTERN.finditer(text):
            # Split potential compound emojis
            for char in match.group():
                emojis.append(char)
        return emojis
    
    def count_emojis(self, text: str) -> Dict[str, int]:
        """
        Count emoji occurrences.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with emoji counts
        """
        emojis = self.extract_emojis(text)
        return dict(Counter(emojis))
    
    # =========================================================================
    # Feature Extraction Methods
    # =========================================================================
    def extract_count_features(self, text: str) -> Dict[str, Any]:
        """
        Extract emoji count features.
        
        Features:
        - Total emoji count
        - Unique emoji count
        - Emoji density (emojis per word)
        - Emoji position features
        """
        emojis = self.extract_emojis(text)
        words = text.split()
        word_count = max(len(words), 1)
        
        # Position features
        text_without_space = text.replace(' ', '')
        first_third = len(text_without_space) // 3
        last_third = 2 * first_third
        
        emojis_at_start = 0
        emojis_at_end = 0
        
        pos = 0
        for char in text:
            if char == ' ':
                continue
            if self.EMOJI_PATTERN.match(char):
                if pos < first_third:
                    emojis_at_start += 1
                elif pos >= last_third:
                    emojis_at_end += 1
            pos += 1
        
        return {
            'emoji_count': len(emojis),
            'unique_emoji_count': len(set(emojis)),
            'emoji_density': round(len(emojis) / word_count, 3),
            'emojis_at_start': emojis_at_start,
            'emojis_at_end': emojis_at_end,
            'has_emojis': len(emojis) > 0
        }
    
    def extract_sentiment_features(self, text: str) -> Dict[str, Any]:
        """
        Extract emoji sentiment features.
        
        Features:
        - Average emoji sentiment
        - Positive/negative emoji ratio
        - Sentiment variance
        """
        emojis = self.extract_emojis(text)
        
        if not emojis:
            return {
                'emoji_sentiment_avg': 0.0,
                'emoji_sentiment_sum': 0.0,
                'emoji_sentiment_variance': 0.0,
                'positive_emoji_count': 0,
                'negative_emoji_count': 0,
                'neutral_emoji_count': 0,
                'positive_negative_ratio': 0.0
            }
        
        sentiments = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for emoji in emojis:
            sentiment = self._emoji_sentiment.get(emoji, 0.0)
            sentiments.append(sentiment)
            
            if sentiment > 0.1:
                positive_count += 1
            elif sentiment < -0.1:
                negative_count += 1
            else:
                neutral_count += 1
        
        avg_sentiment = np.mean(sentiments) if sentiments else 0.0
        variance = np.var(sentiments) if len(sentiments) > 1 else 0.0
        
        pos_neg_ratio = (
            positive_count / negative_count
            if negative_count > 0 else positive_count
        )
        
        return {
            'emoji_sentiment_avg': round(float(avg_sentiment), 3),
            'emoji_sentiment_sum': round(float(sum(sentiments)), 3),
            'emoji_sentiment_variance': round(float(variance), 3),
            'positive_emoji_count': positive_count,
            'negative_emoji_count': negative_count,
            'neutral_emoji_count': neutral_count,
            'positive_negative_ratio': round(pos_neg_ratio, 3)
        }
    
    def extract_category_features(self, text: str) -> Dict[str, Any]:
        """
        Extract emoji category features.
        
        Features:
        - Count of emojis in each category
        - Dominant category
        """
        emojis = self.extract_emojis(text)
        
        categories = {
            'offensive': 0,
            'aggressive': 0,
            'threatening': 0,
            'mocking': 0,
            'body_shaming': 0,
            'sexual_harassment': 0,
            'positive': 0,
            'neutral': 0,
            'other': 0
        }
        
        for emoji in emojis:
            if emoji in self.OFFENSIVE_EMOJIS:
                categories['offensive'] += 1
            elif emoji in self.AGGRESSIVE_EMOJIS:
                categories['aggressive'] += 1
            elif emoji in self.THREATENING_EMOJIS:
                categories['threatening'] += 1
            elif emoji in self.MOCKING_EMOJIS:
                categories['mocking'] += 1
            elif emoji in self.BODY_SHAMING_EMOJIS:
                categories['body_shaming'] += 1
            elif emoji in self.SEXUAL_HARASSMENT_EMOJIS:
                categories['sexual_harassment'] += 1
            elif emoji in self.POSITIVE_EMOJIS:
                categories['positive'] += 1
            elif emoji in self.NEUTRAL_EMOJIS:
                categories['neutral'] += 1
            else:
                categories['other'] += 1
        
        # Dominant category
        if emojis:
            dominant = max(categories.items(), key=lambda x: x[1])[0]
        else:
            dominant = 'none'
        
        # Cyberbullying emoji count
        cyberbullying_count = (
            categories['offensive'] + categories['aggressive'] +
            categories['threatening'] + categories['mocking'] +
            categories['body_shaming'] + categories['sexual_harassment']
        )
        
        return {
            **{f'emoji_cat_{k}': v for k, v in categories.items()},
            'dominant_emoji_category': dominant,
            'cyberbullying_emoji_count': cyberbullying_count,
            'has_cyberbullying_emoji': cyberbullying_count > 0
        }
    
    def extract_pattern_features(self, text: str) -> Dict[str, Any]:
        """
        Extract emoji pattern features.
        
        Features:
        - Repeated emoji patterns
        - Emoji sequences
        - Emoji-text alternation
        """
        emojis = self.extract_emojis(text)
        
        # Check for repeated emojis (same emoji used multiple times)
        emoji_counts = Counter(emojis)
        max_repeat = max(emoji_counts.values()) if emoji_counts else 0
        
        # Check for emoji sequences (multiple emojis in a row)
        emoji_sequences = re.findall(
            r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
            r'\U0001F900-\U0001F9FF\U00002600-\U000026FF]{2,}',
            text
        )
        
        max_sequence_length = max(
            (len(seq) for seq in emoji_sequences), default=0
        )
        
        # Check for emphasis patterns (emoji at end of aggressive text)
        has_emphasis_pattern = (
            len(emojis) > 0 and
            any(e in self.ALL_NEGATIVE_EMOJIS for e in emojis[-2:] if len(emojis) >= 1)
        )
        
        # Mixed negative/positive (sarcasm indicator)
        has_positive = any(e in self.POSITIVE_EMOJIS for e in emojis)
        has_negative = any(e in self.ALL_NEGATIVE_EMOJIS for e in emojis)
        has_mixed_sentiment = has_positive and has_negative
        
        return {
            'max_emoji_repeat': max_repeat,
            'emoji_sequence_count': len(emoji_sequences),
            'max_sequence_length': max_sequence_length,
            'has_emphasis_pattern': has_emphasis_pattern,
            'has_mixed_emoji_sentiment': has_mixed_sentiment,
            'emoji_pattern_score': self._calculate_pattern_score(
                max_repeat, len(emoji_sequences), has_emphasis_pattern
            )
        }
    
    def _calculate_pattern_score(
        self,
        max_repeat: int,
        sequence_count: int,
        has_emphasis: bool
    ) -> float:
        """Calculate emoji pattern intensity score."""
        score = 0.0
        score += min(max_repeat * 0.1, 0.3)
        score += min(sequence_count * 0.15, 0.3)
        score += 0.2 if has_emphasis else 0
        return round(min(score, 1.0), 3)
    
    # =========================================================================
    # Combined Extraction
    # =========================================================================
    def extract(self, texts: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Extract all emoji features from texts.
        
        Args:
            texts: Text or list of texts
            
        Returns:
            List of feature dictionaries
        """
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        
        for text in texts:
            features = {}
            
            # Basic emoji extraction
            emojis = self.extract_emojis(text)
            features['emojis'] = emojis
            
            if self.config.get('include_counts', True):
                features.update(self.extract_count_features(text))
            
            if self.config.get('include_sentiment', True):
                features.update(self.extract_sentiment_features(text))
            
            if self.config.get('include_categories', True):
                features.update(self.extract_category_features(text))
            
            if self.config.get('include_patterns', True):
                features.update(self.extract_pattern_features(text))
            
            results.append(features)
        
        return results
    
    def extract_numeric_features(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Extract numeric emoji features as numpy array.
        
        Args:
            texts: Text or list of texts
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        all_features = self.extract(texts)
        
        # Select numeric features
        numeric_keys = [
            'emoji_count', 'unique_emoji_count', 'emoji_density',
            'emoji_sentiment_avg', 'emoji_sentiment_sum', 'emoji_sentiment_variance',
            'positive_emoji_count', 'negative_emoji_count', 'neutral_emoji_count',
            'emoji_cat_offensive', 'emoji_cat_aggressive', 'emoji_cat_threatening',
            'emoji_cat_mocking', 'emoji_cat_body_shaming', 'emoji_cat_sexual_harassment',
            'cyberbullying_emoji_count', 'max_emoji_repeat', 'emoji_pattern_score'
        ]
        
        matrix = []
        for features in all_features:
            row = [features.get(key, 0) for key in numeric_keys]
            matrix.append(row)
        
        return np.array(matrix)
    
    def get_feature_names(self) -> List[str]:
        """Get list of numeric feature names."""
        return [
            'emoji_count', 'unique_emoji_count', 'emoji_density',
            'emoji_sentiment_avg', 'emoji_sentiment_sum', 'emoji_sentiment_variance',
            'positive_emoji_count', 'negative_emoji_count', 'neutral_emoji_count',
            'emoji_cat_offensive', 'emoji_cat_aggressive', 'emoji_cat_threatening',
            'emoji_cat_mocking', 'emoji_cat_body_shaming', 'emoji_cat_sexual_harassment',
            'cyberbullying_emoji_count', 'max_emoji_repeat', 'emoji_pattern_score'
        ]
    
    def get_emoji_info(self, emoji: str) -> Dict[str, Any]:
        """Get information about a specific emoji."""
        return {
            'emoji': emoji,
            'sentiment': self._emoji_sentiment.get(emoji, 0.0),
            'name': self._emoji_names.get(emoji, 'unknown'),
            'category': self._emoji_categories.get(emoji, 'other'),
            'cyberbullying_association': self._emoji_cyberbullying.get(emoji, 'unknown'),
            'is_offensive': emoji in self.OFFENSIVE_EMOJIS,
            'is_threatening': emoji in self.THREATENING_EMOJIS,
            'is_positive': emoji in self.POSITIVE_EMOJIS
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"EmojiFeatures("
            f"semantics_loaded={bool(self.emoji_semantics)}, "
            f"sentiment_map_size={len(self._emoji_sentiment)})"
        )


if __name__ == "__main__":
    print("EmojiFeatures Test")
    print("=" * 50)
    
    extractor = EmojiFeatures()
    
    test_texts = [
        "You are so annoying 😡😡😡",
        "Great job! 👍👏🎉",
        "Stop messaging me 🖕",
        "nee tumba irritating agthiya 🤬💢",
        "Hello friend! 😊",
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        features = extractor.extract(text)[0]
        print(f"  Emojis: {features.get('emojis')}")
        print(f"  Sentiment avg: {features.get('emoji_sentiment_avg')}")
        print(f"  Cyberbullying count: {features.get('cyberbullying_emoji_count')}")
        print(f"  Dominant category: {features.get('dominant_emoji_category')}")
